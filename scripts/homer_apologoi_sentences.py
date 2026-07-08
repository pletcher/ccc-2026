import argparse
import json

from pathlib import Path

import conllu
import pandas as pd


ROOT_DIR = Path(__file__).parent.parent
PUNCTUATION = set(",.·;·'ʼ῾᾿\"̓,")

FILES = {
    "iliad": ROOT_DIR / "conllu" / "tlg0012.tlg001.daphne_tb-grc1.conllu",
    "odyssey": ROOT_DIR / "conllu" / "tlg0012.tlg002.daphne_tb-grc1.conllu",
}

SPEECH_JSON = {
    "iliad": ROOT_DIR / "json" / "iliad_speeches.json",
    "odyssey": ROOT_DIR / "json" / "odyssey_speeches.json",
}

OUT_FILE = ROOT_DIR / "csv" / "homer_speech_and_narrative_by_sentence-APOLOGOI_ONLY.csv"


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


def in_apologoi(book, line):
    book = int(book)

    if book >= 10 and book <= 12:
        return True

    line = int(line)

    if book == 9 and line >= 2:
        return True

    return False


def build_sentences():
    sentences = []

    for work, path in FILES.items():
        refs = ordered_refs(path)
        speech_lines = build_speech_lines(work, refs)

        for sent in conllu.parse_incr(path.open()):
            tokens = []
            registers = []

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
                    or all(c in PUNCTUATION for c in lemma)
                    or token.get("upos") == "PUNCT"
                ):
                    continue

                book, line = ref.split(".")
                if work == "odyssey" and in_apologoi(book, line):
                    register = "speech" if (book, line) in speech_lines else "narrative"

                    tokens.append(token)
                    registers.append(register)

            if len(registers) == 0:
                continue

            if len(set(registers)) > 1:
                print(
                    f"More than one register for {tokens[0].get('misc').get('Ref')}: {registers}"
                )

            text = " ".join([t.get("form") for t in tokens])

            sentences.append(
                dict(
                    text=text,
                    register=("speech" if "speech" in registers else "narrative"),
                )
            )

    return sentences


def main():
    sentences = build_sentences()

    df = pd.DataFrame(sentences)

    df.to_csv(OUT_FILE, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CoNLL-U treebanks to lines, optionally excluding Odysseus' _Apologoi_"
    )

    parser.add_argument(
        "--exclude-apologoi",
        type=bool,
        default=False,
        help="Exclude _Odyssey_ 9.2 to 12.453",
    )

    args = parser.parse_args()

    main()
