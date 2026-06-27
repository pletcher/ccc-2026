"""
Epic (Homer) vs. tragedy leaning scores via the weighted log-odds-ratio with an
informative Dirichlet prior (Monroe, Colaresi & Quinn 2008, "Fightin' Words"),
plus per-speaker scoring.

INPUT  document-term matrices (DTMs) as pandas DataFrames or CSV paths:
         rows = documents, columns = lemmas, cells = integer counts.
       - the two reference DTMs (Homer, tragedy) are summed over rows to
         corpus-level count vectors;
       - the speaker DTM has one row per speaker.

OUTPUT - per-lemma table: delta (log-odds effect size; epic +, tragic -),
         var, and zeta (standardized score, for *ranking* distinctive lemmas);
       - per-speaker score S(s): mean per-token log-odds of epic vs. tragic in
         the speaker's word choice (epic +, tragic -), in nats/token.

Conventions: positive = Homeric/epic-leaning, negative = tragic-leaning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- I/O + alignment

def _as_frame(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x
    return pd.read_csv(x, index_col=0)


def align_vocab(*dtms):
    """Reindex every DTM onto the shared (union) vocabulary, zero-filling gaps."""
    dtms = [_as_frame(d) for d in dtms]
    vocab = sorted(set().union(*[d.columns for d in dtms]))
    return [d.reindex(columns=vocab, fill_value=0) for d in dtms], vocab


# ----------------------------------------------------------------------------- core statistic

def _delta(yE, yT, alpha, a0):
    """Smoothed log-odds-ratio per lemma (epic minus tragic), prior held fixed."""
    nE, nT = yE.sum(), yT.sum()
    logodds_E = np.log(yE + alpha) - np.log(nE + a0 - yE - alpha)
    logodds_T = np.log(yT + alpha) - np.log(nT + a0 - yT - alpha)
    return logodds_E - logodds_T


def fightin_words(homer_dtm, tragedy_dtm, alpha0=1000.0) -> pd.DataFrame:
    """
    Per-lemma log-odds-ratio with variance and z-score.

    alpha0 sets the strength of the informative prior (pooled relative
    frequency * alpha0); 500-1000 is a sensible range. Larger = more shrinkage
    of rare lemmas toward zero.

    Returns a DataFrame indexed by lemma: y_epic, y_trag, delta, var, zeta.
      delta > 0 -> epic-leaning,  delta < 0 -> tragic-leaning.
      zeta is delta standardized by its s.e. -- use it to RANK distinctive
      lemmas, not to aggregate per speaker (standardization breaks additivity).
    """
    (he, tr), vocab = align_vocab(homer_dtm, tragedy_dtm)
    yE = he.sum(axis=0).to_numpy(dtype=float)
    yT = tr.sum(axis=0).to_numpy(dtype=float)

    pooled = (yE + yT) / (yE.sum() + yT.sum())   # informative prior
    alpha = alpha0 * pooled
    a0 = alpha.sum()

    delta = _delta(yE, yT, alpha, a0)
    var = 1.0 / (yE + alpha) + 1.0 / (yT + alpha)
    zeta = delta / np.sqrt(var)

    return pd.DataFrame(
        {"y_epic": yE, "y_trag": yT, "delta": delta, "var": var, "zeta": zeta},
        index=vocab,
    )


def top_lemmas(fw: pd.DataFrame, n=20):
    """Most distinctively epic and tragic lemmas, ranked by zeta."""
    epic = fw.sort_values("zeta", ascending=False).head(n)
    tragic = fw.sort_values("zeta").head(n)
    return epic, tragic


# ----------------------------------------------------------------------------- speaker scoring

def score_speakers(speaker_dtm, fw: pd.DataFrame) -> pd.DataFrame:
    """
    S(s) = (1/N_s) * sum_w c_w * delta_w  -- mean per-token log-odds (epic +).

    Lemmas a speaker uses that are absent from the reference vocab have no delta
    and are dropped; 'coverage' reports the fraction of the speaker's tokens that
    were actually scored, so you can flag speakers scored on thin evidence.
    """
    sp_full = _as_frame(speaker_dtm)
    sp = sp_full.reindex(columns=fw.index, fill_value=0)
    C = sp.to_numpy(dtype=float)
    delta = fw["delta"].to_numpy()

    N_total = sp_full.to_numpy(dtype=float).sum(axis=1)   # incl. OOV lemmas
    N_scored = C.sum(axis=1)
    safe = np.where(N_scored > 0, N_scored, 1)
    scores = (C @ delta) / safe

    return pd.DataFrame(
        {
            "score": scores,
            "n_tokens": N_scored.astype(int),
            "coverage": np.divide(N_scored, np.where(N_total > 0, N_total, 1)),
        },
        index=sp.index,
    )


def score_speakers_loo(speaker_dtm, homer_dtm, tragedy_dtm, alpha0=1000.0) -> pd.DataFrame:
    """
    Leave-one-out scoring for when the speakers *are* the tragedy corpus.

    For each speaker, that speaker's own tokens are removed from the tragedy
    reference before delta is recomputed -- otherwise delta is estimated partly
    from the very text being scored, biasing tragic speakers toward tragic.
    (The prior is held at the full-corpus value; its LOO change is negligible.)
    """
    (he, tr, sp), vocab = align_vocab(homer_dtm, tragedy_dtm, speaker_dtm)
    yE = he.sum(axis=0).to_numpy(dtype=float)
    yT = tr.sum(axis=0).to_numpy(dtype=float)

    pooled = (yE + yT) / (yE.sum() + yT.sum())
    alpha = alpha0 * pooled
    a0 = alpha.sum()

    C = sp.to_numpy(dtype=float)
    scores = np.empty(C.shape[0])
    for i in range(C.shape[0]):
        c = C[i]
        yT_i = np.maximum(yT - c, 0.0)          # drop this speaker from tragedy
        d = _delta(yE, yT_i, alpha, a0)
        Ni = c.sum()
        scores[i] = (c @ d) / Ni if Ni > 0 else np.nan

    return pd.DataFrame(
        {"score_loo": scores, "n_tokens": C.sum(axis=1).astype(int)},
        index=sp.index,
    )


def bootstrap_ci(speaker_counts, fw: pd.DataFrame, n_boot=2000, ci=0.95, seed=0):
    """
    Percentile bootstrap CI for one speaker's score by resampling their tokens.
    Captures sampling noise in the speaker's own word draw (the dominant concern
    for short speakers); treats delta as fixed.

    speaker_counts: a Series/array aligned to fw.index, or a 1-row DataFrame.
    """
    if isinstance(speaker_counts, (pd.Series, pd.DataFrame)):
        c = _as_frame(speaker_counts.to_frame().T if isinstance(speaker_counts, pd.Series)
                      else speaker_counts).reindex(columns=fw.index, fill_value=0)
        c = c.to_numpy(dtype=float).ravel()
    else:
        c = np.asarray(speaker_counts, dtype=float)

    delta = fw["delta"].to_numpy()
    N = c.sum()
    if N == 0:
        return (np.nan, np.nan)

    nz = c > 0                                    # only categories the speaker uses
    cn, dn = c[nz], delta[nz]
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(int(N), cn / N, size=n_boot)   # (n_boot, |nz|)
    s = draws @ dn / N
    lo, hi = np.quantile(s, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


# ----------------------------------------------------------------------------- demo / self-test

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    vocab = [f"lemma_{i}" for i in range(40)]

    # synthetic corpora with deliberately different word preferences
    pE = rng.dirichlet(np.r_[np.full(20, 3.0), np.full(20, 0.5)])   # epic favors 0-19
    pT = rng.dirichlet(np.r_[np.full(20, 0.5), np.full(20, 3.0)])   # tragedy favors 20-39

    homer = pd.DataFrame(rng.multinomial(50_000, pE).reshape(1, -1),
                         index=["Homer"], columns=vocab)
    tragedy = pd.DataFrame(rng.multinomial(40_000, pT).reshape(1, -1),
                           index=["Tragedy"], columns=vocab)

    fw = fightin_words(homer, tragedy, alpha0=1000)
    epic, tragic = top_lemmas(fw, n=5)
    print("Most epic lemmas (zeta):\n", epic["zeta"].round(2), "\n")
    print("Most tragic lemmas (zeta):\n", tragic["zeta"].round(2), "\n")

    # three synthetic speakers: epic-ish, tragic-ish, and a short noisy one
    speakers = pd.DataFrame(
        np.vstack([
            rng.multinomial(3000, pE),
            rng.multinomial(3000, pT),
            rng.multinomial(60, 0.5 * pE + 0.5 * pT),
        ]),
        index=["epic_guy", "tragic_gal", "short_speaker"],
        columns=vocab,
    )

    print(score_speakers(speakers, fw).round(3), "\n")
    for name in speakers.index:
        lo, hi = bootstrap_ci(speakers.loc[name], fw)
        print(f"{name:14s} 95% CI = [{lo:+.3f}, {hi:+.3f}]")
