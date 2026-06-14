import json

from collections import defaultdict
from pathlib import Path

import conllu
import pandas as pd


ROOT_DIR = Path(__file__).parent.parent
PUNCTUATION = set(",.·;·'ʼ῾᾿\"̓,")
STOPWORDS = set(
    [
        w
        for w in (ROOT_DIR / "stopwords" / "grc_stopwords.txt").read_text().splitlines()
        if not w.startswith("#")
    ]
)

FILES = {
    "iliad": ROOT_DIR / "conllu" / "tlg0012.tlg001.daphne_tb-grc1.conllu",
    "odyssey": ROOT_DIR / "conllu" / "tlg0012.tlg002.daphne_tb-grc1.conllu",
}

SPEECH_JSON = {
    "iliad": ROOT_DIR / "json" / "iliad_speeches.json",
    "odyssey": ROOT_DIR / "json" / "odyssey_speeches.json",
}


def build_speech_lines(work: str) -> set[tuple[str, str]]:
    speeches = json.loads(SPEECH_JSON[work].read_text())
    lines: set[tuple[str, str]] = set()

    for s in speeches:
        book, start = s["l_fi"].split(".")
        _, end = s["l_la"].split(".")

        for line in range(int(start), int(end) + 1):
            lines.add((book, str(line)))

    return lines


def build_counts() -> dict[tuple[str, str], dict[str, int]]:
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for work, path in FILES.items():
        speech_lines = build_speech_lines(work)

        for sent in conllu.parse_incr(path.open()):
            for token in sent:
                # we use an `or` here instead of a default value
                # because `misc` can be set to None — so the default
                # wouldn't trigger
                misc = token.get("misc") or {}
                ref = misc.get("Ref")

                if ref is None:
                    continue

                lemma = token["lemma"]

                if (
                    not lemma
                    or lemma in STOPWORDS
                    or all(c in PUNCTUATION for c in lemma)
                ):
                    continue

                book, line = ref.split(".")
                register = "speech" if (book, line) in speech_lines else "narrative"

                counts[(work, register)][lemma] += 1

    return counts


def build_dataframe(counts: dict[tuple[str, str], dict[str, int]]) -> pd.DataFrame:
    keys = [
        ("iliad", "speech"),
        ("iliad", "narrative"),
        ("odyssey", "speech"),
        ("odyssey", "narrative"),
    ]

    lemmata = sorted({lemma for cell in counts.values() for lemma in cell})

    columns = pd.MultiIndex.from_tuples(keys, names=["work", "register"])
    df = pd.DataFrame(0, index=lemmata, columns=columns)
    df.index.name = "lemma"

    for key, cell in counts.items():
        df[key] = df.index.map(cell).fillna(0).astype(int)

    return df


if __name__ == "__main__":
    counts = build_counts()
    df = build_dataframe(counts)

    df.to_parquet(ROOT_DIR / "parquet" / "homer_speech_narrative.parquet")
