from pathlib import Path

import pandas as pd
import stanza

ROOT_DIR = Path(__file__).parent.parent
TRAGEDY_PARQUET = ROOT_DIR / "parquet" / "tragedy-with-years.parquet"
LOGLIKELIHOOD_PARQUET = ROOT_DIR / "parquet" / "epicness_log_ratio.parquet"
OUT_PARQUET = ROOT_DIR / "parquet" / "tragedy_line_epicness.parquet"

STOPWORDS = set(
    [
        w
        for w in (ROOT_DIR / "stopwords" / "grc_stopwords.txt").read_text().splitlines()
        if not w.startswith("#")
    ]
)
PUNCTUATION = set(",.·;·'ʼ῾᾿")


def _is_punctuation(lemma: str) -> bool:
    return all(c in PUNCTUATION for c in lemma)


def _is_stopword(lemma: str) -> bool:
    return lemma in STOPWORDS


def score_lines(df: pd.DataFrame, nlp: stanza.Pipeline, ll: pd.Series) -> pd.DataFrame:
    in_docs = [stanza.Document([], text=text) for text in df["text"]]
    out_docs = nlp.bulk_process(in_docs)

    epicness = []
    n_lemmata = []

    for doc in out_docs:
        scores = []
        for word in doc.iter_words():
            lemma = word.lemma
            if not lemma or _is_punctuation(lemma) or _is_stopword(lemma):
                continue
            if lemma in ll.index:
                scores.append(ll[lemma])

        n_lemmata.append(len(scores))
        epicness.append(sum(scores) if scores else float("nan"))

    return df.assign(line_epicness=epicness, n_lemmata_scored=n_lemmata)


def main():
    nlp = stanza.Pipeline("grc", processors="tokenize,lemma", verbose=False)
    df = pd.read_parquet(TRAGEDY_PARQUET)
    ll = pd.read_parquet(LOGLIKELIHOOD_PARQUET).set_index("lemma")["epicness"]

    scored = score_lines(df, nlp, ll)
    scored.to_parquet(OUT_PARQUET)
    print(f"wrote {len(scored)} scored lines to {OUT_PARQUET}")


if __name__ == "__main__":
    main()
