"""One-off repair for parquet/old_homer.parquet.

Odyssey rows 3979-6210 (real books 9-12) were mislabeled: book 10 and part
of book 11 were stamped with book_n="9", and the `n` (line-within-book)
counter never reset at the true book boundaries. The underlying text is in
the correct reading order, so we recover the correct book_n/n/urn by
positional alignment against the Daphne CoNLL-U treebank, which has the
same 2232 lines correctly labeled for that span.
"""

from pathlib import Path

import conllu
import pandas as pd

ROOT_DIR = Path(__file__).parent.parent
OLD_HOMER_PARQUET = ROOT_DIR / "parquet" / "old_homer.parquet"
ODYSSEY_CONLLU = ROOT_DIR / "conllu" / "tlg0012.tlg002.daphne_tb-grc1.conllu"

CORRUPT_START = 3979
CORRUPT_END = 6210  # inclusive


def ordered_refs(path: Path) -> list[str]:
    refs: list[str] = []
    for sent in conllu.parse_incr(path.open()):
        for token in sent:
            misc = token.get("misc") or {}
            ref = misc.get("Ref")
            if ref and (not refs or refs[-1] != ref):
                refs.append(ref)
    return refs


def main():
    df = pd.read_parquet(OLD_HOMER_PARQUET)
    od_mask = df["title"] == "Odyssey"
    od = df[od_mask].reset_index(drop=True)

    refs = ordered_refs(ODYSSEY_CONLLU)
    corrupt_refs = refs[CORRUPT_START : CORRUPT_END + 1]
    assert len(corrupt_refs) == CORRUPT_END - CORRUPT_START + 1

    books, lines = zip(*(r.split(".") for r in corrupt_refs))

    urn_prefix = od.loc[0, "urn"].rsplit(":", 1)[0]
    od.loc[CORRUPT_START : CORRUPT_END, "book_n"] = books
    od.loc[CORRUPT_START : CORRUPT_END, "n"] = lines
    od.loc[CORRUPT_START : CORRUPT_END, "urn"] = [
        f"{urn_prefix}:{b}.{n}" for b, n in zip(books, lines)
    ]

    assert not od.duplicated(subset=["book_n", "n"]).any()

    df.loc[od_mask, ["urn", "book_n", "n"]] = od[["urn", "book_n", "n"]].values
    df.to_parquet(OLD_HOMER_PARQUET)


if __name__ == "__main__":
    main()
