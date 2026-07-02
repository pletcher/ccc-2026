import json

from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
INPUT_ILIAD = ROOT_DIR / "json" / "iliad_speeches.json"
INPUT_ODYSSEY = ROOT_DIR / "json" / "odyssey_speeches.json"
INPUT_PARQUET = ROOT_DIR / "parquet" / "old_homer.parquet"
OUTPUT_PARQUET = ROOT_DIR / "parquet" / "homeric_narrative.parquet"
OUTPUT_SPEECH_PARQUET = ROOT_DIR / "parquet" / "homeric_speeches.parquet"


def make_speech_set(title, speeches, ordered_refs, ref_index):
    """Expand each speech's line range into (title, book, line) tuples.

    A handful of speeches (e.g. Odysseus's tale to the Phaeacians) span
    multiple books, so we can't always assume `l_fi` and `l_la` share a book
    and do naive int arithmetic on the line numbers. For those we look up
    where each endpoint falls in the work's actual line sequence and slice
    between them. Same-book ranges use plain int arithmetic instead, since
    a handful of line numbers (e.g. Odyssey 10.456) are simply absent from
    this edition's text and wouldn't resolve to an index.
    """
    speech_lines = set()
    for s in speeches:
        book_fi, line_fi = s["l_fi"].split(".")
        book_la, line_la = s["l_la"].split(".")

        if book_fi == book_la:
            for line in range(int(line_fi), int(line_la) + 1):
                speech_lines.add((title, book_fi, str(line)))
            continue

        start_idx = ref_index[(book_fi, line_fi)]
        end_idx = ref_index[(book_la, line_la)]
        for book, line in ordered_refs[start_idx : end_idx + 1]:
            speech_lines.add((title, book, line))
    return speech_lines


def main():
    iliad_speeches = json.load(INPUT_ILIAD.open())
    odyssey_speeches = json.load(INPUT_ODYSSEY.open())
    all_homer_df = pd.read_parquet(INPUT_PARQUET)

    all_speech_lines = set()
    for title, speeches in (("Iliad", iliad_speeches), ("Odyssey", odyssey_speeches)):
        ordered_refs = list(
            all_homer_df.loc[all_homer_df["title"] == title, ["book_n", "n"]].itertuples(
                index=False, name=None
            )
        )
        ref_index = {ref: i for i, ref in enumerate(ordered_refs)}
        all_speech_lines |= make_speech_set(title, speeches, ordered_refs, ref_index)

    narrative_df = all_homer_df[
        ~all_homer_df.apply(
            lambda row: (row["title"], row["book_n"], row["n"]) in all_speech_lines,
            axis=1,
        )
    ]

    speech_df = all_homer_df[
        all_homer_df.apply(
            lambda row: (row["title"], row["book_n"], row["n"]) in all_speech_lines,
            axis=1,
        )
    ]

    OUTPUT_PARQUET.write_bytes(narrative_df.to_parquet())


if __name__ == "__main__":
    main()
