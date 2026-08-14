"""Input parsing for the local MSA pipeline.

Two inputs are required to start a search:

1. **`<target>.fasta`** — one record per *unique* entity. Header order
   (top-down) assigns chain letters A, B, C, ... in order. Wrapped
   sequences, CRLF line endings, blank lines, and inline whitespace
   inside sequences are tolerated.

2. **stoi string** — the legacy ``AnBn`` form passed as ``--stoi`` (e.g.
   ``"A1"`` monomer, ``"A2"`` homomer, ``"A1B1"`` heteromer; ``"UNK"`` → ``A1``).

The output of `parse_inputs` is a `ParsedInputs` dataclass holding
three index-aligned lists (`unique_seqs`, `cardinality`, `chain_letters`)
plus convenience properties (`n_unique`, `total_chains`, `is_complex`).

`parse_stoi` / `format_stoi` convert between the legacy CLI
stoi string and the dict form used internally; `normalize_stoi` validates and
canonicalizes a CLI stoi string in one call.
"""

import re
import string
from dataclasses import dataclass
from pathlib import Path

from MSA.script.colab_msa_template_search.parse_fasta import read_fasta_records


@dataclass(frozen=True)
class ParsedInputs:
    """Normalized form of (fasta, stoi string). All three lists are
    aligned by index — `unique_seqs[i]` is the sequence of the entity
    assigned chain letter `chain_letters[i]` with copy count
    `cardinality[i]`.
    """

    unique_seqs: list[str]
    cardinality: list[int]
    chain_letters: list[str]

    @property
    def n_unique(self) -> int:
        return len(self.unique_seqs)

    @property
    def total_chains(self) -> int:
        return sum(self.cardinality)

    @property
    def is_complex(self) -> bool:
        return self.total_chains > 1


_LEGACY_STOI_TOKEN = re.compile(r"([A-Z])(\d+|n)")


def parse_stoi(stoi: str) -> dict[str, int]:
    """Convert legacy stoi string (e.g. ``"A1B1"``, ``"A2"``, ``"UNK"``)
    to a dict keyed by chain letter. ``"UNK"`` maps to ``{"A": 1}``
    (monomer fallback used by Thal-Kak's CLI). ``"n"`` counts collapse
    to 1, matching
    `MSA/script/colab_msa_template_search/parse_fasta.py`.

    Raises:
        ValueError if the string contains content not matched by
        ``[A-Z](\\d+|n)`` tokens, has duplicate chain letters, or has a
        non-positive count.
    """
    if stoi.upper() == "UNK":
        return {"A": 1}
    parsed = _LEGACY_STOI_TOKEN.findall(stoi)
    if not parsed:
        raise ValueError(f"unrecognized stoi string {stoi!r}")
    if "".join(f"{c}{n}" for c, n in parsed) != stoi:
        raise ValueError(
            f"stoi {stoi!r} contains content not matched by "
            f"[A-Z](\\d+|n) tokens (got tokens {parsed})"
        )
    out: dict[str, int] = {}
    for letter, n_str in parsed:
        if letter in out:
            raise ValueError(f"stoi {stoi!r}: chain {letter} appears more than once")
        n = 1 if n_str == "n" else int(n_str)
        if n < 1:
            raise ValueError(f"stoi {stoi!r}: count for {letter} must be ≥ 1")
        out[letter] = n
    return out


def format_stoi(stoi: dict[str, int]) -> str:
    """Inverse of `parse_stoi`. Output is sorted by chain letter
    so it's deterministic regardless of insertion order.
    """
    return "".join(f"{k}{v}" for k, v in sorted(stoi.items()))


def normalize_stoi(stoi: str) -> str:
    """Validate a legacy CLI stoi string and return its canonical form
    (``"UNK"`` → ``"A1"``, chain letters sorted). Raises ``ValueError`` on
    malformed input (delegated to `parse_stoi`).

    Single entry point for CLI callers that need a validated, canonical stoi
    string (parse_fasta, the re-run skip-key, method_log, and the ``--stoi``
    handed to the MSA.cssb_msa subprocess).
    """
    return format_stoi(parse_stoi(stoi))


def parse_inputs(fasta: Path | str, stoi: str) -> ParsedInputs:
    """Parse the input pair (fasta + legacy stoi string) into normalized form.

    Validation:
    - fasta record count must equal stoi key count
    - stoi keys must be exactly ``A``, ``B``, ..., contiguous starting from A
    - every sequence must be non-empty
    - every cardinality must be ≥ 1 (enforced by `parse_stoi`)

    Returns a `ParsedInputs` whose three lists are index-aligned:
    `unique_seqs[i]` is the sequence for chain letter `chain_letters[i]`
    with copy count `cardinality[i]`.
    """
    fasta = Path(fasta)
    records = read_fasta_records(fasta)
    stoi_map = parse_stoi(stoi)

    n_records = len(records)
    n_stoi_keys = len(stoi_map)
    if n_records != n_stoi_keys:
        raise ValueError(
            f"{fasta} has {n_records} record(s) but the stoi declares "
            f"{n_stoi_keys} chain(s); these must match"
        )

    expected_letters = list(string.ascii_uppercase[:n_records])
    stoi_letters = sorted(stoi_map.keys())
    if stoi_letters != expected_letters:
        raise ValueError(
            f"the stoi declares chains {stoi_letters} but for {n_records} "
            f"fasta record(s) we expected {expected_letters} "
            f"(single-letter, contiguous starting from A)"
        )

    unique_seqs: list[str] = []
    cardinality: list[int] = []
    chain_letters: list[str] = []
    for letter, (_header, seq) in zip(expected_letters, records):
        if not seq:
            raise ValueError(
                f"{fasta}: record for chain {letter} has empty sequence"
            )
        unique_seqs.append(seq)
        cardinality.append(stoi_map[letter])
        chain_letters.append(letter)
    return ParsedInputs(
        unique_seqs=unique_seqs,
        cardinality=cardinality,
        chain_letters=chain_letters,
    )
