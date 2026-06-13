import json

from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
INPUT_ILIAD = ROOT_DIR / "json" / "iliad_speeches.json"
INPUT_ODYSSEY = ROOT_DIR / "json" / "odyssey_speeches.json"
INPUT_PARQUET = ROOT_DIR / "parquet" / "homer.parquet"
OUTPUT_PARQUET = ROOT_DIR / "parquet" / "homeric_narrative.parquet"


def make_speech_set(expanded_refs):
    speech_lines = set()
    for book, line_range in expanded_refs:
        for line in line_range:
            speech_lines.add((book, str(line)))
    return speech_lines


def main():
    iliad_speeches = json.load(INPUT_ILIAD.open())
    odyssey_speeches = json.load(INPUT_ODYSSEY.open())
    all_homer_df = pd.read_parquet(INPUT_PARQUET)

    iliad_refs = [(s["l_fi"], s["l_la"]) for s in iliad_speeches]
    odyssey_refs = [(s["l_fi"], s["l_la"]) for s in odyssey_speeches]

    iliad_expanded_refs = [
        (
            r[0].split(".")[0],
            range(int(r[0].split(".")[1]), int(r[1].split(".")[1]) + 1),
        )
        for r in iliad_refs
    ]
    odyssey_expanded_refs = [
        (
            r[0].split(".")[0],
            range(int(r[0].split(".")[1]), int(r[1].split(".")[1]) + 1),
        )
        for r in odyssey_refs
    ]

    iliad_speech_lines = make_speech_set(iliad_expanded_refs)
    odyssey_speech_lines = make_speech_set(odyssey_expanded_refs)
    all_speech_lines = iliad_speech_lines | odyssey_speech_lines

    narrative_df = all_homer_df[
        ~all_homer_df.apply(
            lambda row: (row["book_n"], row["n"]) in all_speech_lines, axis=1
        )
    ]

    OUTPUT_PARQUET.write_bytes(narrative_df.to_parquet())


if __name__ == "__main__":
    main()
