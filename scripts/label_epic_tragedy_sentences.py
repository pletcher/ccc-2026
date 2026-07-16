from pathlib import Path

import conllu
import pandas as pd


ROOT_DIR = Path(__file__).parent.parent
CONLLU_DIR = ROOT_DIR / "conllu"
PUNCTUATION = set(",.·;·'ʼ῾᾿\"̓,")

OUT_FILE = ROOT_DIR / "csv" / "epic_tragedy_sentences.csv"

EPIC_URN_PREFIX = "tlg0012"


def label_for(path: Path) -> str:
    return "epic" if path.name.startswith(EPIC_URN_PREFIX) else "tragedy"


def in_odyssey_apologoi(book: str, line: str) -> bool:
    """Odyssey 9.2-12.453."""
    book_i = int(book)

    if 10 <= book_i <= 12:
        return True

    return book_i == 9 and int(line) >= 2


def in_iliad_nestor(book: str, line: str) -> bool:
    """Iliad 11.656-803."""
    return int(book) == 11 and 656 <= int(line) <= 803


def homer_exclusion(fn):
    def check(ref: str) -> bool:
        book, line = ref.split(".")
        return fn(book, line)

    return check


def first_line_number(ref: str) -> int:
    """Parse a Ref like '203', '138-139' (shared line), or '203a' into an int."""
    digits = ""
    for c in ref:
        if c.isdigit():
            digits += c
        elif digits:
            break
    return int(digits)


def line_range_exclusion(start: int, end: int):
    def check(ref: str) -> bool:
        return start <= first_line_number(ref) <= end

    return check


# Held-out passages, dropped from the training data. Sophocles' Electra
# (tlg0011.tlg005) is the only one of the three tragedy passages the user
# asked for that actually has a treebank in conllu/ -- Aeschylus' Persians
# (tlg0085.tlg002) and Euripides' Hippolytus (tlg0006.tlg005) aren't present,
# so there's nothing to exclude them from.
EXCLUSIONS = {
    "tlg0012.tlg002.daphne_tb-grc1": homer_exclusion(in_odyssey_apologoi),
    "tlg0012.tlg001.daphne_tb-grc1": homer_exclusion(in_iliad_nestor),
    "tlg0011.tlg005.daphne_tb-grc1": line_range_exclusion(680, 763),
}


def build_sentences() -> list[dict]:
    sentences = []

    for path in sorted(CONLLU_DIR.glob("*.conllu")):
        label = label_for(path)
        exclude = EXCLUSIONS.get(path.stem)

        for sent in conllu.parse_incr(path.open()):
            tokens = []

            for token in sent:
                if not (
                    token["lemma"]
                    and token["lemma"] != "_"
                    and token.get("upos") not in (None, "_", "PUNCT")
                    and not all(c in PUNCTUATION for c in token["lemma"])
                ):
                    continue

                if exclude is not None:
                    misc = token.get("misc") or {}
                    ref = misc.get("Ref")
                    if ref is not None and exclude(ref):
                        continue

                tokens.append(token)

            if not tokens:
                continue

            text = " ".join(token.get("form") for token in tokens)

            sentences.append(
                dict(
                    text=text,
                    label=label,
                    source=path.stem,
                    sent_id=sent.metadata.get("sent_id"),
                )
            )

    return sentences


def main():
    sentences = build_sentences()
    df = pd.DataFrame(sentences)
    df.to_csv(OUT_FILE, index=False)
    print(f"Wrote {len(df)} sentences to {OUT_FILE}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
