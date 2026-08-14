"""Shared engine-agnostic core for cssb_template's local template engines.

Holds the dataclasses, default BioMolDB/seqres paths, query-a3m resolution,
cif gunzip, and the multi-chain orchestrator (`run_for_msa_dir_generic`)
shared by the two production template engines:

  - ``engine_mmseqs.run_mmseqs_for_msa_dir``  (mmseqs vs BioMolDB; ``mmseqs_cssb``)
  - ``engine_hmmer.run_hmmer_for_msa_dir``    (local hmmbuild+hmmsearch; ``hhblits_cssb``)

BioMolDB = the local PDB-structure template database snapshot (seqres FASTA +
mmCIF files) both engines search for templates. Neither engine needs an
AlphaFold3 Singularity image.

Output contract (both engines): ``<msa_dir>/<target>_env/pdb70.m8`` (13-col TSV)
+ ``templates_<101+cid>/<pdb>.cif``. See ``m8_writer.py`` for the m8 row format.
"""

from __future__ import annotations

import datetime
import gzip
import logging
import os
import shutil
import string
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

from MSA.db_paths import TEMPLATE_ROOT, TEMPLATE_SNAPSHOT

logger = logging.getLogger(__name__)

QueryA3mSource = Literal["uniref30", "uniref100", "merged"]
DEFAULT_QUERY_A3M_SOURCE: QueryA3mSource = "uniref30"

DEFAULT_SEQRES_FASTA = TEMPLATE_ROOT / "fasta" / "pdb_seqres_protein.fasta"
DEFAULT_MMCIF_DIR = TEMPLATE_ROOT / "cif" / "raw"
DEFAULT_MAX_TEMPLATE_DATE = datetime.date(3000, 1, 1)
DEFAULT_MAX_HITS = 20

TemplateEngine = Literal["mmseqs", "hmmer", "mmseqs+hmmer"]


def resolve_template_engine(mode: str, template_cfg: dict | None) -> str:
    """Resolve the effective template engine for a cssb MSA mode.

    ``mode`` ∈ {mmseqs_cssb, hhblits_cssb, mmseqs_hhblits_cssb}; ``template_cfg``
    is the EFFECTIVE merged template block
    (``{**cfg["template"], **(cfg[mode].get("template") or {})}``).

      - combined (``mmseqs_hhblits_cssb``): ALWAYS runs both engines + a
        mmseqs-first merge; the ``engine`` knob is ignored → ``"mmseqs+hmmer"``.
      - else ``engine: auto`` → ``"mmseqs"`` (mmseqs_cssb) / ``"hmmer"``
        (hhblits_cssb); an explicit ``"mmseqs"``/``"hmmer"`` is used verbatim
        (validated upstream by ``__main__._validate``, which checks the merged
        effective engine — this function does NOT re-validate).
    """
    if mode == "mmseqs_hhblits_cssb":
        return "mmseqs+hmmer"
    engine = (template_cfg or {}).get("engine", "auto")
    if engine == "auto":
        return "hmmer" if mode == "hhblits_cssb" else "mmseqs"
    return engine


@dataclass(frozen=True)
class TemplateEntry:
    """One row in the data-yaml `templates:` list (one entry per hit)."""

    path: Path
    chain_template: list[str]
    chain_query: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "chain_template": list(self.chain_template),
            "chain_query": list(self.chain_query),
        }


@dataclass(frozen=True)
class TemplateSearchResult:
    out_dir: Path
    m8_path: Path
    template_entries: list[TemplateEntry]
    log_path: Path
    hits_json_path: Path
    num_hits: int


def _resolve_a3m_root(msa_dir: Path, source: QueryA3mSource) -> Path:
    """Map a `query_a3m_source` key to the dir holding per-cid a3m files.

    - ``"uniref30"`` (default) → ``<msa_dir>/_workdir/raw/uniref30_2302/``,
      the raw UniRef30 search output before the per-chain merge.
    - ``"uniref100"`` → ``<msa_dir>/_workdir/raw/uniref100_2026_01/``, the
      hhblits primary UniRef pass and the template seed on that path.
    - ``"merged"`` → ``<msa_dir>/_workdir/merged/``, the per-chain merge of
      every selected DB (deduplicated and capped).

    All three live under ``_workdir``, so the builder must have run with
    ``keep_intermediate=True`` — the default for the Thal-Kak MSA methods, but
    not for direct library callers. `run_for_msa_dir_generic` checks that the
    returned dir exists.
    """
    if source == "uniref30":
        return msa_dir / "_workdir" / "raw" / "uniref30_2302"
    if source == "uniref100":
        return msa_dir / "_workdir" / "raw" / "uniref100_2026_01"
    if source == "merged":
        return msa_dir / "_workdir" / "merged"
    raise ValueError(
        f"unknown query_a3m_source={source!r}; expected one of "
        f"{{'uniref30', 'uniref100', 'merged'}}"
    )


def _gunzip_cif(src_gz: Path, dst_cif: Path) -> None:
    """Read `<pdb>.cif.gz` and write the decompressed `.cif`.

    A hit whose cif is missing means the snapshot's sequence DB and its cif tree
    disagree, which is a broken install rather than a property of the target. Both
    engines therefore stop here: continuing would silently hand the models fewer
    templates than the search found, and there is no way to tell that apart from
    a genuinely template-poor target after the fact.

    Use a tmp + rename so partial writes don't leave half-gunzipped files.
    """
    if not src_gz.exists():
        raise FileNotFoundError(
            f"template cif missing: {src_gz}\n"
            "  The template DB's sequence DB has a hit whose structure file is not\n"
            "  in cif/raw/ — the snapshot is incomplete. Reinstall it with\n"
            "  ./install_db.sh --family template"
        )
    tmp = dst_cif.with_suffix(dst_cif.suffix + ".part")
    with gzip.open(src_gz, "rb") as f_in, tmp.open("wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.replace(tmp, dst_cif)


@dataclass(frozen=True)
class MsaDirResult:
    """Aggregate result of a multi-chain template search.

    `template_entries` is the merged list across all per-chain results,
    with each entry's `chain_query` patched to the **full** alphabet
    letter list that the chain represents (e.g., A2B2 → cid 0 entries
    have `chain_query=["A","B"]`, cid 1 entries have `chain_query=["C","D"]`).
    `per_chain` retains the raw per-cid `TemplateSearchResult`s for
    callers that need pre-patch state.
    """

    out_dir: Path
    m8_path: Path
    template_entries: list[TemplateEntry]
    per_chain: list[TemplateSearchResult]
    chain_letters: dict[int, list[str]]


def _alphabet_letters(n: int) -> list[str]:
    """Sequential alphabet letters: 0→A, 1→B, ..., 25→Z, 26→AA, ..."""
    chars = string.ascii_uppercase
    out: list[str] = []
    for i in range(n):
        if i < 26:
            out.append(chars[i])
        else:
            first = chars[(i // 26) - 1]
            second = chars[i % 26]
            out.append(f"{second}{first}")
    return out


def _read_first_fasta_record(a3m_path: Path) -> str:
    """Return the sequence of the first FASTA record in `a3m_path`.

    Skips comment lines starting with `#`. Reads everything between the
    first `>` and the next `>` (or EOF). Returns the raw sequence (no
    case/dash normalization — the engine handles its own input).
    """
    seq_parts: list[str] = []
    saw_header = False
    with Path(a3m_path).open() as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith("#"):
                continue
            if line.startswith(">"):
                if saw_header:
                    break
                saw_header = True
                continue
            if saw_header and line:
                seq_parts.append(line.strip())
    return "".join(seq_parts)


def _emit_zero_hits_banner(cid: int, letters: list[str], a3m_path: Path) -> None:
    """Loud-banner warning when a chain has 0 template hits."""
    banner = "=" * 72
    logger.warning(
        "\n%s\n0 template hits for chain cid=%d (letters=%s, a3m=%s)\n"
        "  This may be normal for divergent queries — continuing without templates "
        "for this chain.\n%s",
        banner,
        cid,
        letters,
        a3m_path,
        banner,
    )


# Per-chain search callable shared by `run_for_msa_dir_generic`. Invoked with
# keyword args query_sequence/query_a3m_path/out_dir/query_id/query_chain_id/
# append_m8; returns a TemplateSearchResult. Each engine
# (`engine_mmseqs`, `engine_hmmer`) binds its own per-chain driver.
PerChainSearch = Callable[..., TemplateSearchResult]


def run_for_msa_dir_generic(
    msa_dir: Path,
    *,
    query_a3m_source: QueryA3mSource,
    per_chain_search: PerChainSearch,
) -> MsaDirResult:
    """Engine-agnostic multi-chain orchestrator shared by the template engines.

    Parses the master a3m header, resolves the per-cid query a3m dir for
    ``query_a3m_source``, then calls ``per_chain_search`` once per unique
    chain into a shared ``<msa_dir>/<target>_env/`` out_dir (so `pdb70.m8`
    accumulates across chains and per-cid cifs land under their own
    ``templates_<query_id>/`` subdir), patching each returned
    ``TemplateEntry.chain_query`` to the full alphabet letter list for its
    cid. 0-hit chains emit a loud-banner warning and continue.

    Output layout (auto-detected by `split_colab_a3m_write_yaml`):

        <msa_dir>/<target>_env/
        ├── pdb70.m8                  (merged, all chains)
        └── templates_<101+cid>/<pdb_id>.cif × N
    """
    msa_dir = Path(msa_dir).resolve()

    # Locate the master a3m: only *.a3m directly under msa_dir whose name
    # does NOT match the per-chain split patterns.
    candidates = [
        p
        for p in msa_dir.glob("*.a3m")
        if "_paired_msa_chains_" not in p.name and "_unpaired_msa_chains_" not in p.name
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one master a3m under {msa_dir}, found {candidates}"
        )
    master_a3m = candidates[0]
    target = master_a3m.stem

    # Parse the ColabFold-complex header: `#L1,L2,...\tC1,C2,...`.
    with master_a3m.open() as fh:
        header = fh.readline().rstrip("\r\n")
    if not header.startswith("#"):
        raise RuntimeError(
            f"master a3m {master_a3m} missing ColabFold-complex header (first line: {header!r})"
        )
    header_body = header[1:]
    parts = header_body.split("\t")
    chain_lengths = [int(x) for x in parts[0].split(",")]
    if len(parts) > 1:
        cardinality = [int(x) for x in parts[1].split(",")]
    else:
        cardinality = [1] * len(chain_lengths)
    if len(cardinality) != len(chain_lengths):
        raise RuntimeError(
            f"header L/C length mismatch in {master_a3m}: "
            f"L={chain_lengths} C={cardinality}"
        )

    # Resolve per-cid a3m dir from the source key, then discover
    # <a3m_root>/<cid>.a3m sorted by integer cid.
    a3m_root = _resolve_a3m_root(msa_dir, query_a3m_source)
    if not a3m_root.is_dir():
        raise FileNotFoundError(
            f"query_a3m_source={query_a3m_source!r} expects {a3m_root} to exist. "
            f"Re-run the builder with keep_intermediate=True (default for "
            f"Thal-Kak CSSB MSA methods; not guaranteed for direct library "
            f"callers)."
        )
    a3ms = sorted(
        a3m_root.glob("*.a3m"),
        key=lambda p: int(p.stem),
    )
    if len(a3ms) != len(chain_lengths):
        raise RuntimeError(
            f"{a3m_root.name}/ a3m count {len(a3ms)} != unique chain count "
            f"{len(chain_lengths)} (master={master_a3m})"
        )

    # Per-cid full alphabet letter list: walk cardinality assigning
    # sequential letters. A2B2 → {0: ["A","B"], 1: ["C","D"]}.
    total_chains = sum(cardinality)
    flat_letters = _alphabet_letters(total_chains)
    chain_letters: dict[int, list[str]] = {}
    cursor = 0
    for cid, count in enumerate(cardinality):
        chain_letters[cid] = flat_letters[cursor : cursor + count]
        cursor += count

    out_dir = msa_dir / f"{target}_env"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_chain: list[TemplateSearchResult] = []
    merged_entries: list[TemplateEntry] = []

    for cid, a3m_path in enumerate(a3ms):
        query_sequence = _read_first_fasta_record(a3m_path)
        if not query_sequence:
            raise RuntimeError(f"could not read query sequence from {a3m_path}")
        letters = chain_letters[cid]
        result = per_chain_search(
            query_sequence=query_sequence,
            query_a3m_path=a3m_path,
            out_dir=out_dir,
            query_id=str(101 + cid),
            query_chain_id=letters[0],
            append_m8=(cid > 0),
        )

        # Patch chain_query → full letter list for this cid (frozen
        # dataclass, so use dataclasses.replace).
        patched = [
            replace(entry, chain_query=list(letters))
            for entry in result.template_entries
        ]
        result = replace(result, template_entries=patched)
        per_chain.append(result)
        merged_entries.extend(patched)

        if result.num_hits == 0:
            _emit_zero_hits_banner(cid, letters, a3m_path)

    return MsaDirResult(
        out_dir=out_dir,
        m8_path=out_dir / "pdb70.m8",
        template_entries=merged_entries,
        per_chain=per_chain,
        chain_letters=chain_letters,
    )
