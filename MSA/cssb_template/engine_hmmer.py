"""Local hmmer template search — reimplements the AF3 template algorithm
WITHOUT the AlphaFold3 Singularity image.

Why this exists
---------------
Running AF3's own ``Templates.from_seq_and_a3m`` would mean depending on the
AlphaFold 3 code, which we cannot redistribute. This module reproduces the
same pipeline (AF3 supplement §2.4) using only locally-installed HMMER
(`hmmbuild` + `hmmsearch`):

    query a3m (the primary UniRef profile MSA; AF3 seeds this with UniRef90)
      → hmmbuild (HMM, query columns forced to be the model match states)
      → hmmsearch vs the PDB protein seqres FASTA (AlphaFold's flags)
      → release-date cutoff + AlphaFold's hit filters
      → e-value sort, dedup, cap to max_hits
      → synthesize pdb70.m8 (via `m8_writer`) + gunzip per-hit cifs

Output layout matches the other engine (`engine_mmseqs`), so the downstream
`Structure/script/{protenix,boltz,chai}` parsers read either engine's output:

    <msa_dir>/<target>_env/
    ├── pdb70.m8                  (merged, all chains)
    └── templates_<101+cid>/<pdb_id>.cif × N

Coordinate / date conventions:
- hmmsearch run against the seqres FASTA reports target coords that ARE
  the seqres (`_entity_poly_seq`) 1-based numbering — exactly what the m8
  `t_start/t_end` columns and downstream `parse_m8_cigar` expect. No AF3
  internal remapping needed.
- `hmmbuild --hand` with a reference (`#=GC RF`) line marking every column
  as a match state guarantees the model has exactly ``len(query)`` match
  states == query positions, so the k-th match column in the hmmsearch
  ``-A`` alignment is query position k (all match columns are emitted,
  even all-gap ones).
- Release dates come from ``<TEMPLATE_ROOT>/release_dates.tsv``; any pdb it
  lacks falls back to ``<TEMPLATE_ROOT>/metadata/cif_metadata.tsv``
  (`deposition_date`, YYYY-MM-DD). Deposition date is <= release date, so
  the fallback is a strictly-more-conservative cutoff and never lets a
  too-new template through.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from .m8_writer import build_m8_row, synthesize_cigar

# Shared engine-agnostic core: dataclasses, the multi-chain orchestrator,
# query-a3m resolution, cif gunzip, and default DB paths (single source of
# truth shared with the mmseqs engine).
from ._common import (
    DEFAULT_MAX_HITS,
    DEFAULT_MAX_TEMPLATE_DATE,
    DEFAULT_MMCIF_DIR,
    DEFAULT_QUERY_A3M_SOURCE,
    DEFAULT_SEQRES_FASTA,
    MsaDirResult,
    QueryA3mSource,
    TemplateEntry,
    TemplateSearchResult,
    TEMPLATE_ROOT,
    _gunzip_cif,
    run_for_msa_dir_generic,
)

logger = logging.getLogger(__name__)

DEFAULT_CIF_METADATA = TEMPLATE_ROOT / "metadata" / "cif_metadata.tsv"

# AlphaFold's hmmsearch flags, as its own `data/tools/hmmsearch.py` passes
# them. AF3 uses the same set (supplement §2.4).
_HMMSEARCH_FLAGS = [
    "--noali",
    "--F1", "0.1",
    "--F2", "0.1",
    "--F3", "0.1",
    "-E", "100",
    "--incE", "100",
    "--domE", "100",
    "--incdomE", "100",
]

# Template hit filters. The first two are AlphaFold's own defaults (its
# `data/templates.py` prefilter); AF3 carries the same values forward and
# adds the minimum hit length.
MAX_SUBSEQUENCE_RATIO = 0.95   # drop hits that are ~the query itself
MIN_ALIGN_RATIO = 0.1          # aligned residues / query length
MIN_HIT_LENGTH = 10            # aligned residue count


# binary resolution (mirrors cssb_msa/hhblits/runner.resolve_hhblits)
def _resolve_bin(name: str, explicit: Path | None = None) -> Path:
    """Resolve a HMMER binary: an `explicit` path if given, else `name`
    from the active env's PATH. Fails loud if it is missing — no
    fallback."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"{name} binary {path} not found")
        return path
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(
            f"{name} not found on PATH; activate the project env, which "
            f"provides HMMER (`conda activate thalkak`), or pass an explicit path"
        )
    return Path(found)


def resolve_hmmbuild(explicit: Path | None = None) -> Path:
    return _resolve_bin("hmmbuild", explicit)


def resolve_hmmsearch(explicit: Path | None = None) -> Path:
    return _resolve_bin("hmmsearch", explicit)


# a3m → Stockholm (match-only, RF forces query columns as match states)
def _read_a3m(path: Path) -> list[tuple[str, str]]:
    recs: list[tuple[str, str]] = []
    name: str | None = None
    parts: list[str] = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if line.startswith("#"):
                continue
            if line.startswith(">"):
                if name is not None:
                    recs.append((name, "".join(parts)))
                name = line[1:].split()[0]
                parts = []
            elif line.strip():
                parts.append(line.strip())
    if name is not None:
        recs.append((name, "".join(parts)))
    return recs


def _strip_inserts(seq: str) -> str:
    """Drop a3m insert states (lowercase letters / '.') → match columns only."""
    return "".join(c for c in seq if not (c.islower() or c == "."))


def a3m_to_match_stockholm(query_a3m_path: Path, out_sto: Path) -> str:
    """Convert a query a3m into a match-only Stockholm MSA and return the
    query sequence (first record).

    Every column of the resulting MSA is a query match column, and the
    written ``#=GC RF`` line marks them all 'x' so ``hmmbuild --hand``
    builds exactly ``len(query)`` match states aligned 1:1 to the query.
    """
    recs = _read_a3m(query_a3m_path)
    if not recs:
        raise ValueError(f"empty a3m: {query_a3m_path}")
    aln = [(n, _strip_inserts(s)) for n, s in recs]
    width = len(aln[0][1])
    # Keep only rows that match the query width (defensive: a malformed
    # downstream row shouldn't abort the whole search).
    clean = [(n, s) for n, s in aln if len(s) == width]
    query_seq = clean[0][1].replace("-", "")
    out_sto = Path(out_sto)
    out_sto.parent.mkdir(parents=True, exist_ok=True)
    with out_sto.open("w") as fh:
        fh.write("# STOCKHOLM 1.0\n")
        for i, (name, seq) in enumerate(clean):
            # Stockholm names must be unique + whitespace-free.
            fh.write(f"{f'seq{i}_{name}'[:60]:<63} {seq}\n")
        fh.write(f"{'#=GC RF':<63} {'x' * width}\n")
        fh.write("//\n")
    return query_seq


# hmmsearch output parsing
def _parse_stockholm_alignment(path: Path) -> list[tuple[str, str]]:
    """Parse an hmmsearch ``-A`` Stockholm file into (seq_name, aligned_seq).

    Concatenates wrapped blocks per sequence; ignores #=GF/#=GS/#=GR/#=GC
    annotation lines. Sequence names are ``<target>/<ali_from>-<ali_to>``.
    """
    seqs: dict[str, list[str]] = {}
    order: list[str] = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            sp = line.split()
            if len(sp) != 2:
                continue
            name, chunk = sp
            if name not in seqs:
                seqs[name] = []
                order.append(name)
            seqs[name].append(chunk)
    return [(n, "".join(seqs[n])) for n in order]


def _parse_domtbl_evalues(path: Path) -> dict[tuple[str, int, int], float]:
    """Map (target, ali_from, ali_to) → i-Evalue from an hmmsearch domtbl.

    domtbl columns (0-based, whitespace-split): target=0, i-Evalue=12,
    ali_from=17, ali_to=18.
    """
    out: dict[tuple[str, int, int], float] = {}
    with Path(path).open() as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            c = line.split()
            if len(c) < 19:
                continue
            try:
                target = c[0]
                ievalue = float(c[12])
                ali_from = int(c[17])
                ali_to = int(c[18])
            except (ValueError, IndexError):
                continue
            out[(target, ali_from, ali_to)] = ievalue
    return out


_NAME_RE = re.compile(r"^(?P<target>.+)/(?P<start>\d+)-(?P<end>\d+)$")


def _build_query_to_hit(
    aligned_seq: str, ali_from: int, query_seq: str
) -> tuple[dict[int, int], int, int]:
    """Walk an hmmsearch ``-A`` aligned hit row → 0-based query→hit mapping.

    Match columns (uppercase / '-') advance the query index; insert columns
    (lowercase / '.') do not. Residues (any case) advance the hit index,
    which starts at ``ali_from - 1`` (0-based seqres coord). All model match
    states are emitted by hmmsearch, so the k-th match column is query
    position k.

    Returns (mapping, n_aligned, n_identical) where n_identical counts
    matched positions whose query residue equals the hit residue.
    """
    mapping: dict[int, int] = {}
    q_idx = 0
    h_idx = ali_from - 1
    n_identical = 0
    for ch in aligned_seq:
        if ch == "-":                       # match column, deletion in hit
            q_idx += 1
        elif ch == ".":                      # insert column, gap
            continue
        elif ch.isupper():                   # match column, residue
            if 0 <= q_idx < len(query_seq):
                mapping[q_idx] = h_idx
                if query_seq[q_idx] == ch:
                    n_identical += 1
            q_idx += 1
            h_idx += 1
        elif ch.islower():                   # insert column, residue in hit
            h_idx += 1
        # anything else (e.g. '*') is ignored
    return mapping, len(mapping), n_identical


# release-date metadata (module-cached; ~1.5M rows, loaded once)
_CIF_DATE_CACHE: dict[str, dict[str, datetime.date]] = {}


def _load_cif_dates(metadata_tsv: Path) -> dict[str, datetime.date]:
    """Load pdb_id_lower → deposition_date from cif_metadata.tsv.

    cif_id is ``<pdb>_<model>_<poly>_<chain>`` where ``<chain>`` is most
    often the ``.`` "all-chains" placeholder, so we key by the PDB id only
    — deposition_date is an entry-level property (identical across a PDB's
    chains/models). Keeping the earliest date per id is defensive against
    any inconsistency. Cached per path for the process lifetime.
    """
    key = str(Path(metadata_tsv).resolve())
    cached = _CIF_DATE_CACHE.get(key)
    if cached is not None:
        return cached
    out: dict[str, datetime.date] = {}
    with Path(metadata_tsv).open() as fh:
        header = fh.readline().rstrip("\n").split("\t")
        try:
            cid_i = header.index("cif_id")
            date_i = header.index("deposition_date")
        except ValueError:
            cid_i, date_i = 0, 2
        for line in fh:
            cols = line.rstrip("\n").split("\t")
            if len(cols) <= max(cid_i, date_i):
                continue
            pdb = cols[cid_i].split("_", 1)[0].lower()
            if not pdb:
                continue
            try:
                d = datetime.date.fromisoformat(cols[date_i])
            except ValueError:
                continue
            prev = out.get(pdb)
            if prev is None or d < prev:
                out[pdb] = d
    _CIF_DATE_CACHE[key] = out
    return out


DEFAULT_RELEASE_DATES = TEMPLATE_ROOT / "release_dates.tsv"
_RELEASE_DATE_CACHE: dict[tuple[str, str], dict[str, datetime.date]] = {}


def _load_release_dates(
    release_tsv: Path = DEFAULT_RELEASE_DATES,
    cif_metadata: Path = DEFAULT_CIF_METADATA,
) -> dict[str, datetime.date]:
    """Load pdb_id_lower -> initial RELEASE date for template leakage control.

    The initial release date (earliest `_pdbx_audit_revision_history.revision_date`,
    AF3's `min(...)` convention) is the correct cutoff field: a structure deposited
    before but RELEASED after the cutoff (embargo) must still be excluded —
    deposition_date alone leaks it (e.g. 8fg6: deposited 2022-12-12, released
    2024-03-20). Release dates come from the `release_dates.tsv` sidecar that
    ships with the template DB snapshot; deposition_date
    (`cif_metadata.tsv`) fills any pdb the sidecar lacks. Shared by both local
    engines (this module + `engine_mmseqs`). Cached per (release_tsv, cif_metadata).
    """
    key = (str(Path(release_tsv).resolve()), str(Path(cif_metadata).resolve()))
    cached = _RELEASE_DATE_CACHE.get(key)
    if cached is not None:
        return cached
    # deposition_date base (fallback), then override with release dates where known.
    out: dict[str, datetime.date] = dict(_load_cif_dates(cif_metadata))
    rp = Path(release_tsv)
    if rp.is_file():
        n = 0
        with rp.open() as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                try:
                    out[parts[0].strip().lower()] = datetime.date.fromisoformat(parts[1].strip())
                    n += 1
                except ValueError:
                    continue
        logger.info(
            "template date filter: loaded %d release dates from %s "
            "(deposition_date fallback for the rest).",
            n, rp,
        )
    else:
        logger.warning(
            "release-date sidecar %s not found -> falling back to "
            "deposition_date, which LEAKS embargoed structures (deposited before but "
            "released after the cutoff). It ships with the template DB snapshot — "
            "reinstall the snapshot to get it.",
            rp,
        )
    _RELEASE_DATE_CACHE[key] = out
    return out


# per-chain search
def run_local_hmmer_template_search(
    *,
    query_sequence: str,
    query_a3m_path: Path,
    out_dir: Path,
    query_id: str = "101",
    query_chain_id: str = "A",
    seqres_fasta: Path = DEFAULT_SEQRES_FASTA,
    mmcif_dir: Path = DEFAULT_MMCIF_DIR,
    cif_metadata: Path = DEFAULT_CIF_METADATA,
    max_template_date: datetime.date = DEFAULT_MAX_TEMPLATE_DATE,
    max_hits: int = DEFAULT_MAX_HITS,
    hmmbuild_bin: Path | None = None,
    hmmsearch_bin: Path | None = None,
    threads: int = 8,
    append_m8: bool = False,
) -> TemplateSearchResult:
    """Single-chain local-hmmer template search (AF3 algorithm).

    Implements the per-chain `_common.PerChainSearch` contract so it
    plugs into `run_for_msa_dir_generic` as the per-chain callable. Writes
    pdb70.m8 (append or overwrite) + gunzipped per-hit cifs under
    ``out_dir/templates_<query_id>/`` and a debug ``hits.json``.

    The ``query_sequence`` argument is authoritative for filtering/identity;
    ``query_a3m_path`` supplies the profile MSA for hmmbuild (first record
    is expected to be the query).
    """
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    template_dir = out_dir / f"templates_{query_id}"
    template_dir.mkdir(parents=True, exist_ok=True)

    sto_in = template_dir / "query_match.sto"
    hmm_path = template_dir / "query.hmm"
    hits_sto = template_dir / "hits.sto"
    dom_tbl = template_dir / "hits.domtbl"
    hits_json = template_dir / "hits.json"
    log_path = template_dir / "template_search.log"

    # 1. a3m → match-only Stockholm. Prefer the explicit query_sequence
    #    (e.g. from the master a3m) over whatever the a3m's first row holds.
    a3m_query = a3m_to_match_stockholm(query_a3m_path, sto_in)
    query_seq = query_sequence or a3m_query
    qlen = len(query_seq)

    hmmbuild = resolve_hmmbuild(hmmbuild_bin)
    hmmsearch = resolve_hmmsearch(hmmsearch_bin)

    def _run(cmd: list[str], logf) -> None:
        logf.write(f"cmd: {' '.join(cmd)}\n")
        logf.flush()
        subprocess.run(cmd, check=True, stdout=logf, stderr=subprocess.STDOUT)

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("w") as logf:
        logf.write(f"=== hmmer template search ({query_id}) at {started} ===\n")
        try:
            # 2. hmmbuild (--hand keeps RF columns = query positions).
            _run(
                [str(hmmbuild), "--hand", "--amino", "--cpu", str(threads),
                 str(hmm_path), str(sto_in)],
                logf,
            )
            # 3. hmmsearch vs seqres FASTA (AF3 flags) → -A alignment + domtbl.
            _run(
                [str(hmmsearch), *_HMMSEARCH_FLAGS, "--cpu", str(threads),
                 "-A", str(hits_sto), "--domtblout", str(dom_tbl),
                 str(hmm_path), str(Path(seqres_fasta).resolve())],
                logf,
            )
        except subprocess.CalledProcessError as exc:
            logf.write(f"\n=== FAILED exit {exc.returncode} ===\n")
            raise RuntimeError(
                f"hmmer template search failed (exit {exc.returncode}). "
                f"See log: {log_path}"
            ) from exc

    # 4. parse alignment + e-values.
    evalues = _parse_domtbl_evalues(dom_tbl) if dom_tbl.exists() else {}
    aligned = (
        _parse_stockholm_alignment(hits_sto) if hits_sto.exists() else []
    )
    cif_dates = _load_release_dates(cif_metadata=cif_metadata)

    # 5. build candidate hits with filters.
    candidates: list[dict[str, Any]] = []
    for name, seq in aligned:
        m = _NAME_RE.match(name)
        if not m:
            continue
        target = m.group("target")
        ali_from = int(m.group("start"))
        ali_to = int(m.group("end"))
        if "_" not in target:
            continue
        pdb_id, auth_chain = target.rsplit("_", 1)
        pdb_id = pdb_id.lower()

        # release-date cutoff (initial release date; deposition_date fallback for
        # pdbs absent from the sidecar). Unknown id → kept (no date to filter on).
        rel = cif_dates.get(pdb_id)
        if rel is not None and rel > max_template_date:
            continue

        mapping, n_aln, n_ident = _build_query_to_hit(seq, ali_from, query_seq)
        if n_aln < MIN_HIT_LENGTH:
            continue
        if qlen == 0 or (n_aln / qlen) < MIN_ALIGN_RATIO:
            continue
        # near-identical-to-query (avoid trivially copying the query).
        if qlen and (n_ident / qlen) > MAX_SUBSEQUENCE_RATIO:
            continue

        cd = synthesize_cigar({k: v for k, v in mapping.items()})
        if cd is None:
            continue
        candidates.append({
            "pdb_id": pdb_id,
            "auth_chain_id": auth_chain,
            "ali_from": ali_from,
            "ali_to": ali_to,
            "evalue": evalues.get((target, ali_from, ali_to), 1e2),
            "cigar_data": cd,
            "hit_residues": "".join(
                c for c in seq if c.isupper()
            ),
            "release_date": rel.isoformat() if rel else None,
            "n_aligned": n_aln,
        })

    # 6. sort by e-value, dedup by aligned-residue string, cap to max_hits.
    candidates.sort(key=lambda h: h["evalue"])
    deduped: list[dict[str, Any]] = []
    seen_seqs: set[str] = set()
    for h in candidates:
        if h["hit_residues"] in seen_seqs:
            continue
        seen_seqs.add(h["hit_residues"])
        deduped.append(h)
        if len(deduped) >= max_hits:
            break

    # 7. emit m8 rows + gunzip cifs.
    template_entries: list[TemplateEntry] = []
    m8_lines: list[str] = []
    seen_pdbs: set[str] = set()
    mmcif_root = Path(mmcif_dir).resolve()
    for h in deduped:
        pdb_id = h["pdb_id"]
        auth = h["auth_chain_id"]
        m8_lines.append(
            build_m8_row(query_id, pdb_id, auth, h["cigar_data"],
                         evalue=h["evalue"])
        )
        cif_dst = template_dir / f"{pdb_id}.cif"
        if pdb_id not in seen_pdbs:
            cif_src = mmcif_root / pdb_id[1:3] / f"{pdb_id}.cif.gz"
            _gunzip_cif(cif_src, cif_dst)
            seen_pdbs.add(pdb_id)
        template_entries.append(
            TemplateEntry(
                path=cif_dst,
                chain_template=[auth],
                chain_query=[query_chain_id],
            )
        )

    m8_path = out_dir / "pdb70.m8"
    mode = "a" if append_m8 and m8_path.exists() else "w"
    with m8_path.open(mode) as fh:
        if m8_lines:
            fh.write("\n".join(m8_lines) + "\n")

    with hits_json.open("w") as fh:
        json.dump(
            {
                "query_id": query_id,
                "query_sequence": query_seq,
                "seqres_fasta": str(seqres_fasta),
                "max_template_date": max_template_date.isoformat(),
                "max_hits": max_hits,
                "num_candidates": len(candidates),
                "num_hits": len(deduped),
                "hits": [
                    {k: v for k, v in h.items() if k != "cigar_data"}
                    for h in deduped
                ],
            },
            fh,
            indent=2,
        )

    return TemplateSearchResult(
        out_dir=out_dir,
        m8_path=m8_path,
        template_entries=template_entries,
        log_path=log_path,
        hits_json_path=hits_json,
        num_hits=len(template_entries),
    )


def run_hmmer_for_msa_dir(
    msa_dir: Path,
    *,
    query_a3m_source: QueryA3mSource = DEFAULT_QUERY_A3M_SOURCE,
    seqres_fasta: Path = DEFAULT_SEQRES_FASTA,
    mmcif_dir: Path = DEFAULT_MMCIF_DIR,
    cif_metadata: Path = DEFAULT_CIF_METADATA,
    max_template_date: datetime.date = DEFAULT_MAX_TEMPLATE_DATE,
    max_hits: int = DEFAULT_MAX_HITS,
    threads: int = 8,
) -> MsaDirResult:
    """Multi-chain local-hmmer template search for a `cssb_msa` ``<msa_dir>``.

    Thin wrapper over `_common.run_for_msa_dir_generic` (shared master-a3m
    parsing, per-cid loop, chain_query patching) binding the per-chain
    callable to `run_local_hmmer_template_search`. For the hhblits path the
    default ``query_a3m_source`` is ``"uniref100"`` (set in
    msa_config.hhblits.yaml), i.e. the profile hhblits found against
    UniRef100; it takes the place of the UniRef90 seed AF3 uses here.
    """

    def _search(**kw: Any) -> TemplateSearchResult:
        return run_local_hmmer_template_search(
            seqres_fasta=seqres_fasta,
            mmcif_dir=mmcif_dir,
            cif_metadata=cif_metadata,
            max_template_date=max_template_date,
            max_hits=max_hits,
            threads=threads,
            **kw,
        )

    return run_for_msa_dir_generic(
        msa_dir,
        query_a3m_source=query_a3m_source,
        per_chain_search=_search,
    )
