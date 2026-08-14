# Thal-Kak
# Copyright 2026 CSSB, Seoul National University
#
# Licensed under the Apache License, Version 2.0 (see LICENSE).
#
# ---------------------------------------------------------------------------
# Third-party attribution (see also NOTICE)
#
# The assembly routines in this file are a Python port of ColabFold's
# msa_to_str(), pair_msa(), pair_sequences() and pad_sequences()
# (colabfold/batch.py, https://github.com/sokrypton/ColabFold):
#
#   MIT License
#
#   Copyright (c) 2021 Sergey Ovchinnikov
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense,
#   and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.
#
# MODIFIED by CSSB Thal-Kak (2026): rewritten against this pipeline's
# per-chain a3m files and ParsedInputs rather than ColabFold's in-memory
# structures. The ColabFold complex-a3m conventions it reproduces — numeric
# `>{101+n}` chain ids, the `#L1,L2<TAB>C1,C2` cardinality header, and the
# homomer short-circuit that drops the paired block — are unchanged.
# ---------------------------------------------------------------------------

"""Phase 6 — assemble per-chain a3m into ColabFold complex format.

Pure-Python (no mmseqs). Takes the per-chain unpaired text a3m (Phase 4
merge output) plus optionally per-chain paired a3m (Phase 5 output for
true heteromers) plus a `ParsedInputs`, and emits the single
ColabFold-complex-format a3m this pipeline produces.

Faithful replica of colabfold `msa_to_str` + `pair_msa` +
`pair_sequences` + `pad_sequences` (`batch.py`). Output passes
`unserialize_msa` round-trip and is parsed correctly by
`MSA/script/colab_msa_template_search/colab_a3m_to_yaml.py`.

Cardinality semantics (colabfold convention):
- Header carries the **original** cardinality (`#L1,L2\\tC1,C2`).
- Body is always built with cardinality reset to all-1s — the body has
  one copy of each unique sequence's hits, the model replicates at
  inference time.
- Homomers (n_unique=1, cardinality>1): caller passes `paired=None`
  even if Phase 5 produced output. Colabfold's main() does the same:
  paired_msa is gated on `len(query_seqs_cardinality) > 1`, which is
  False for homomers since a homomer has a single-element cardinality
  list.
"""

from pathlib import Path

from MSA.cssb_msa.common.input import ParsedInputs


def _read_text(p: Path | str) -> str:
    return Path(p).read_text()


def _rewrite_query_header(text: str, new_header: str) -> str:
    """Replace the first record's header line with `new_header`.

    Used to conform per-chain a3m to colabfold's `>{101+n}` numeric-ID
    convention. Dedup output preserves the original query name from
    each per-DB result2msa (e.g. `>H0208`), but the downstream splitter
    `colab_a3m_to_yaml.split_colab_a3m_write_yaml` only treats digit-only
    first tokens as chain-boundary markers — without the rewrite,
    heteromer unpaired blocks all get merged into the paired buffer.
    Reference: ColabFold `colabfold/batch.py` (its synthetic-MSA case uses
    `>{101+sequence_index}`).
    """
    if not text:
        return text
    parts = text.split("\n", 1)
    if not parts[0].startswith(">"):
        raise ValueError(
            f"a3m text does not start with '>' header: {parts[0][:80]!r}"
        )
    if len(parts) == 1:
        return new_header
    return new_header + "\n" + parts[1]


def _pair_sequences(chain_a3m_texts: list[str], cardinality_reset: list[int]) -> str:
    """Build paired block. Each `chain_a3m_texts[n]` is the per-chain
    paired a3m. All chains must have equal line count — paired rows
    align positionally across chains.

    Faithful to colabfold's `pair_sequences`. Headers in chain `n>0`
    get their first `>` replaced by `\\t` (tab-join semantics).
    """
    if not chain_a3m_texts:
        return ""
    line_count = len(chain_a3m_texts[0].splitlines())
    a3m_line_paired = [""] * line_count
    for n, text in enumerate(chain_a3m_texts):
        lines = text.splitlines()
        if len(lines) != line_count:
            raise ValueError(
                f"paired chain {n} has {len(lines)} lines; chain 0 has "
                f"{line_count} — paired rows must align positionally"
            )
        for i, line in enumerate(lines):
            if line.startswith(">"):
                if n != 0:
                    line = line.replace(">", "\t", 1)
                a3m_line_paired[i] += line
            else:
                a3m_line_paired[i] += line * cardinality_reset[n]
    return "\n".join(a3m_line_paired)


def _pad_sequences(
    chain_a3m_texts: list[str],
    query_seqs_unique: list[str],
    cardinality_reset: list[int],
) -> str:
    """Build unpaired block with per-chain gap padding.

    Faithful to colabfold's `pad_sequences`. Padding lengths are
    `len(query_seqs_unique[m])` for each chain m — that's the
    match-state length (queries have no insertions).
    """
    blank_seq = [
        "-" * len(seq)
        for n, seq in enumerate(query_seqs_unique)
        for _ in range(cardinality_reset[n])
    ]
    out: list[str] = []
    pos = 0
    for n, _ in enumerate(query_seqs_unique):
        for _j in range(cardinality_reset[n]):
            for line in chain_a3m_texts[n].split("\n"):
                if not line:
                    continue
                if line.startswith(">"):
                    out.append(line)
                else:
                    out.append("".join(blank_seq[:pos] + [line] + blank_seq[pos + 1 :]))
            pos += 1
    return "\n".join(out)


def _pair_msa(
    query_seqs_unique: list[str],
    cardinality_reset: list[int],
    paired_a3m_texts: list[str] | None,
    unpaired_a3m_texts: list[str] | None,
) -> str:
    """Inner assembler — mirrors colabfold's `batch.py` branching."""
    has_paired = paired_a3m_texts is not None
    has_unpaired = unpaired_a3m_texts is not None and len(unpaired_a3m_texts) > 0
    if not has_paired and has_unpaired:
        return _pad_sequences(unpaired_a3m_texts, query_seqs_unique, cardinality_reset)
    if has_paired and has_unpaired:
        return (
            _pair_sequences(paired_a3m_texts, cardinality_reset)
            + "\n"
            + _pad_sequences(unpaired_a3m_texts, query_seqs_unique, cardinality_reset)
        )
    if has_paired and not has_unpaired:
        return _pair_sequences(paired_a3m_texts, cardinality_reset)
    return ""


def assemble_complex_a3m(
    parsed: ParsedInputs,
    unpaired_paths: list[Path],
    paired_paths: list[Path] | None,
) -> str:
    """Assemble per-chain a3m files into a single ColabFold complex a3m.

    Args:
        parsed: ParsedInputs from `MSA.cssb_msa.common.input.parse_inputs`.
        unpaired_paths: per-chain unpaired a3m. Length == n_unique.
        paired_paths: per-chain paired a3m, or None. Length == n_unique
            when not None. Pass None for monomer / homomer (n_unique=1
            cases) — even if Phase 5 was run, colabfold convention is
            to ignore paired for homomers.

    Returns:
        ColabFold complex format a3m text. Always starts with the
        `#L1,L2,...\\tC1,C2,...\\n` header. No trailing newline beyond
        what individual lines contribute.
    """
    if len(unpaired_paths) != parsed.n_unique:
        raise ValueError(
            f"unpaired_paths length {len(unpaired_paths)} != n_unique {parsed.n_unique}"
        )
    if paired_paths is not None and len(paired_paths) != parsed.n_unique:
        raise ValueError(
            f"paired_paths length {len(paired_paths)} != n_unique {parsed.n_unique}"
        )

    unpaired_texts = [
        _rewrite_query_header(_read_text(p), f">{101 + n}")
        for n, p in enumerate(unpaired_paths)
    ]
    paired_texts = (
        [
            _rewrite_query_header(_read_text(p), f">{101 + n}")
            for n, p in enumerate(paired_paths)
        ]
        if paired_paths is not None else None
    )

    header = (
        "#"
        + ",".join(str(len(s)) for s in parsed.unique_seqs)
        + "\t"
        + ",".join(str(c) for c in parsed.cardinality)
        + "\n"
    )

    cardinality_reset = [1 for _ in parsed.cardinality]
    body = _pair_msa(parsed.unique_seqs, cardinality_reset, paired_texts, unpaired_texts)
    return header + body


def assemble_complex_a3m_to_file(
    parsed: ParsedInputs,
    unpaired_paths: list[Path],
    paired_paths: list[Path] | None,
    out_path: Path,
) -> None:
    text = assemble_complex_a3m(parsed, unpaired_paths, paired_paths)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if text and not text.endswith("\n"):
        text += "\n"
    out_path.write_text(text)
