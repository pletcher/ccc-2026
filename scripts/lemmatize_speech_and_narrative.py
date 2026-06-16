### This file is no longer needed, as we use the
### Daphne treebanks in the ./conllu directory

from collections import defaultdict
from pathlib import Path

import pandas as pd
import stanza

ROOT_DIR = Path(__file__).parent.parent
NARRATIVE_PARQUET = ROOT_DIR / "parquet" / "homeric_narrative.parquet"
SPEECH_PARQUET = ROOT_DIR / "parquet" / "homeric_speeches.parquet"
STOPWORDS = set(
    [
        w
        for w in (ROOT_DIR / "stopwords" / "grc_stopwords.txt").read_text().splitlines()
        if not w.startswith("#")
    ]
)
PUNCTUATION = set(",.·;·'ʼ῾᾿")
TITLES = ["Iliad", "Odyssey"]


def _is_punctuation(lemma: str) -> bool:
    return all(c in PUNCTUATION for c in lemma)


def _is_stopword(lemma: str) -> bool:
    return lemma in STOPWORDS


def lemmatize_narrative(
    df: pd.DataFrame, nlp: stanza.Pipeline
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for _, row in df.iterrows():
        title = row["title"]
        doc = nlp(row["text"])

        for word in doc.iter_words():
            lemma = word.lemma

            if lemma and not _is_punctuation(lemma) and not _is_stopword(lemma):
                counts[lemma][title] += 1

    return counts


def lemmatize_speeches(
    df: pd.DataFrame, nlp: stanza.Pipeline
) -> dict[str, dict[tuple[str, str], int]]:
    counts: dict[str, dict[tuple[str, str], int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for _, row in df.iterrows():
        key = (row["title"], row["speaker_id"].split("@")[0])
        doc = nlp(row["content"])

        for word in doc.iter_words():
            lemma = word.lemma
            if lemma and not _is_punctuation(lemma) and not _is_stopword(lemma):
                counts[lemma][key] += 1

    return counts


def build_narrative_dtm(
    counts: dict[str, dict[str, int]], titles: list[str]
) -> pd.DataFrame:
    rows = {lemma: {t: counts[lemma].get(t, 0) for t in titles} for lemma in counts}

    df = pd.DataFrame.from_dict(rows, orient="index", columns=titles)

    df.index.name = "lemma"

    return df.sort_index()


def build_speech_dtm(
    counts: dict[str, dict[tuple[str, str], int]], speakers: list[tuple[str, str]]
):
    rows = {lemma: {k: counts[lemma].get(k, 0) for k in speakers} for lemma in counts}

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["title", "speaker_id"])

    df.index.name = "lemma"

    return df.sort_index()


def narrative_to_dtm(filename: Path):
    nlp = stanza.Pipeline("grc", processors="tokenize,lemma", verbose=False)
    df = pd.read_parquet(filename)
    counts = lemmatize_narrative(df, nlp)
    dtm = build_narrative_dtm(counts, TITLES)

    dtm.to_parquet(filename.with_stem(filename.stem + "_dtm"))


def speech_to_dtm(filename: Path):
    nlp = stanza.Pipeline("grc", processors="tokenize,lemma", verbose=False)
    df = pd.read_parquet(filename)
    speakers = [
        (s[0], s[1].split("@")[0])
        for s in df.groupby(["title", "speaker_id"]).size().index.tolist()
    ]
    counts = lemmatize_speeches(df, nlp)
    dtm = build_speech_dtm(counts, speakers)

    dtm.to_parquet(filename.with_stem(filename.stem + "_dtm"))


if __name__ == "__main__":
    narrative_to_dtm(NARRATIVE_PARQUET)
    speech_to_dtm(SPEECH_PARQUET)
