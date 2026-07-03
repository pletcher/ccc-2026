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


def ordered_refs(path: Path) -> list[tuple[str, str]]:
    refs: list[tuple[str, str]] = []
    for sent in conllu.parse_incr(path.open()):
        for token in sent:
            misc = token.get("misc") or {}
            ref = misc.get("Ref")
            if not ref:
                continue
            book, line = ref.split(".")
            if not refs or refs[-1] != (book, line):
                refs.append((book, line))
    return refs


def build_speech_lines(work: str, refs: list[tuple[str, str]]) -> set[tuple[str, str]]:
    """Expand each speech's line range into (book, line) tuples.

    A handful of speeches (e.g. Odysseus's tale to the Phaeacians) span
    multiple books, so we can't always assume `l_fi` and `l_la` share a book
    and do int arithmetic on the line numbers. For those we look up
    where each endpoint falls in the treebank's actual line sequence and
    slice between them. Same-book ranges use plain int arithmetic instead,
    since a handful of line numbers (e.g. Odyssey 10.456) are simply absent
    from this edition's text and wouldn't resolve to an index.
    """
    speeches = json.loads(SPEECH_JSON[work].read_text())
    ref_index = {ref: i for i, ref in enumerate(refs)}
    lines: set[tuple[str, str]] = set()

    for s in speeches:
        book_fi, line_fi = s["l_fi"].split(".")
        book_la, line_la = s["l_la"].split(".")

        if book_fi == book_la:
            for line in range(int(line_fi), int(line_la) + 1):
                lines.add((book_fi, str(line)))
            continue

        start_idx = ref_index[(book_fi, line_fi)]
        end_idx = ref_index[(book_la, line_la)]
        lines.update(refs[start_idx : end_idx + 1])

    return lines


def build_ngrams(n=2) -> dict[str, dict[tuple, int]]:
    counts: dict[str, dict[tuple, int]] = defaultdict(lambda: defaultdict(int))

    for work, path in FILES.items():
        for sent in conllu.parse_incr(path.open()):
            no_punct = [t for t in sent if not all(c in PUNCTUATION for c in t["lemma"])]
            no_stops = [t for t in no_punct if not t["lemma"] in STOPWORDS]
            tokens = [t for t in no_stops if t["lemma"]]
            lemmata = [t["lemma"] for t in tokens]
            ngrams = [tuple(lemmata[i:i+n]) for i in range(len(lemmata) - n + 1)]

            for gram in ngrams:
                counts[work][gram] += 1

    return counts


def build_counts() -> dict[tuple[str, str], dict[str, int]]:
    counts: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for work, path in FILES.items():
        refs = ordered_refs(path)
        speech_lines = build_speech_lines(work, refs)

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


def build_epic_counts() -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for work, path in FILES.items():
        for sent in conllu.parse_incr(path.open()):
            for token in sent:
                lemma = token["lemma"]

                if (
                    not lemma
                    or lemma in STOPWORDS
                    or all(c in PUNCTUATION for c in lemma)
                ):
                    continue

                counts[work][lemma] += 1

    return counts


def build_ngram_with_register_counts(n=2) -> dict[tuple[str, str], dict[tuple, int]]:
    """N-gram analogue of `build_counts`: counts n-grams by (work, register).

    A gram is only counted if every token in its window has a resolvable
    `Ref` and all of them fall in the same register; grams that straddle a
    speech/narrative boundary (or contain a token with no ref) are dropped
    rather than assigned to either side.
    """
    counts: dict[tuple[str, str], dict[tuple, int]] = defaultdict(
        lambda: defaultdict(int)
    )

    for work, path in FILES.items():
        refs = ordered_refs(path)
        speech_lines = build_speech_lines(work, refs)

        for sent in conllu.parse_incr(path.open()):
            no_punct = [t for t in sent if not all(c in PUNCTUATION for c in t["lemma"])]
            no_stops = [t for t in no_punct if t["lemma"] not in STOPWORDS]
            tokens = [t for t in no_stops if t["lemma"]]
            lemmata = [t["lemma"] for t in tokens]

            for i in range(len(lemmata) - n + 1):
                window = tokens[i:i + n]
                registers = set()

                for t in window:
                    misc = t.get("misc") or {}
                    ref = misc.get("Ref")
                    if ref is None:
                        registers.add(None)
                        break
                    book, line = ref.split(".")
                    registers.add(
                        "speech" if (book, line) in speech_lines else "narrative"
                    )

                if len(registers) != 1 or None in registers:
                    continue

                register = registers.pop()
                gram = tuple(lemmata[i:i + n])
                counts[(work, register)][gram] += 1

    return counts


def build_bigram_register_dataframe(
    counts: dict[tuple[str, str], dict[tuple, int]]
) -> pd.DataFrame:
    keys = [
        ("iliad", "speech"),
        ("iliad", "narrative"),
        ("odyssey", "speech"),
        ("odyssey", "narrative"),
    ]

    bigrams = sorted({gram for cell in counts.values() for gram in cell})

    columns = pd.MultiIndex.from_tuples(keys, names=["work", "register"])
    df = pd.DataFrame(0, index=bigrams, columns=columns)
    df.index.name = "bigram"

    for key, cell in counts.items():
        df[key] = df.index.map(cell).fillna(0).astype(int)

    return df


def build_bigram_dataframe(counts: dict[str, dict[tuple, int]]) -> pd.DataFrame:
    works = list(FILES.keys())
    bigrams = sorted({gram for cell in counts.values() for gram in cell})

    df = pd.DataFrame(0, index=bigrams, columns=works)
    df.index.name = "bigram"

    for work, cell in counts.items():
        df[work] = df.index.map(cell).fillna(0).astype(int)

    return df



def build_epic_dataframe(counts: dict[str, dict[str, int]]) -> pd.DataFrame:
    works = list(FILES.keys())
    lemmata = sorted({lemma for cell in counts.values() for lemma in cell})

    df = pd.DataFrame(0, index=lemmata, columns=works)
    df.index.name = "lemma"

    for work, cell in counts.items():
        df[work] = df.index.map(cell).fillna(0).astype(int)

    return df


if __name__ == "__main__":
    # counts = build_counts()
    # df = build_dataframe(counts)
    # df.to_parquet(ROOT_DIR / "parquet" / "homer_speech_narrative.parquet")

    # epic_counts = build_epic_counts()
    # epic_df = build_epic_dataframe(epic_counts)
    # epic_df.to_parquet(ROOT_DIR / "parquet" / "homer_epic.parquet")

    bigram_counts = build_ngrams()
    bigram_df = build_bigram_dataframe(bigram_counts)
    bigram_df.to_parquet(ROOT_DIR / "parquet" / "bigrams_by_epic.parquet")

    bigram_register_counts = build_ngram_with_register_counts()
    bigram_register_df = build_bigram_register_dataframe(bigram_register_counts)
    bigram_register_df.to_parquet(ROOT_DIR / "parquet" / "bigrams_by_epic_and_register.parquet")
