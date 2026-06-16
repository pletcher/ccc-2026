from collections import defaultdict
from pathlib import Path

import pandas as pd
import stanza

ROOT_DIR = Path(__file__).parent.parent
TRAGEDY_PARQUET = ROOT_DIR / "parquet" / "tragedy-with-years.parquet"
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


def lemmatize_tragedy(
    df: pd.DataFrame, nlp: stanza.Pipeline
) -> dict[str, dict[tuple[str, str, str], int]]:
    counts: dict[str, dict[tuple[str, str, str], int]] = defaultdict(
        lambda: defaultdict(int)
    )

    grouped = (
        df.groupby(["dramatist", "title", "speaker"])["text"]
        .apply(" ".join)
        .reset_index()
    )

    for _, row in grouped.iterrows():
        key = (row["dramatist"], row["title"], row["speaker"])
        doc = nlp(row["text"])

        for word in doc.iter_words():
            lemma = word.lemma
            if lemma and not _is_punctuation(lemma) and not _is_stopword(lemma):
                counts[lemma][key] += 1

    return counts


def build_tragedy_dtm(
    counts: dict[str, dict[tuple[str, str, str], int]],
    speakers: list[tuple[str, str, str]],
) -> pd.DataFrame:
    rows = {
        lemma: {k: counts[lemma].get(k, 0) for k in speakers} for lemma in counts
    }

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.columns = pd.MultiIndex.from_tuples(
        df.columns, names=["dramatist", "title", "speaker"]
    )
    df.index.name = "lemma"

    return df.sort_index()


def tragedy_to_dtm(filename: Path):
    nlp = stanza.Pipeline("grc", processors="tokenize,lemma", verbose=False)
    df = pd.read_parquet(filename)
    speakers = [
        (s[0], s[1], s[2])
        for s in df.groupby(["dramatist", "title", "speaker"]).size().index.tolist()
    ]
    counts = lemmatize_tragedy(df, nlp)
    dtm = build_tragedy_dtm(counts, speakers)

    dtm.to_parquet(filename.with_stem(filename.stem + "_dtm"))


if __name__ == "__main__":
    tragedy_to_dtm(TRAGEDY_PARQUET)
