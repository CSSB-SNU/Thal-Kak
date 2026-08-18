"""Per-source adapters that expose a finished search's per-DB a3ms by DB kind.

`local_source` has one consumer: `combined/build.py`'s Stage-2 (the
`mmseqs_hhblits_local` builder) calls it whenever `merge.mode: leveled`, the
default — it is how the two sources' per-kind texts are collected before the
single hhfilter pass.

`local_source` yields, per unique chain (keyed by query `raw_seq_key`), the per-kind
unpaired a3m text (uniref / env) + the paired a3m text, plus the source's
``<target>_env`` template dir, so mmseqs and hhblits reach the merge in one uniform
shape. It reads `_workdir/raw/<db>/<cid>.a3m` grouped by DB role (`kind_of(db)`),
keyed by the chain's query (`merged/<cid>.a3m`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from MSA.local_msa.common.db_registry import DEFAULT_REGISTRY, MERGE_ORDER
from MSA.local_msa.common.dedup import iter_a3m_records, raw_seq_key


@dataclass(frozen=True)
class SourceChains:
    """One source's per-chain texts, keyed by query raw_seq_key."""
    label: str
    uniref: dict[str, str] = field(default_factory=dict)   # query_key -> uniref a3m text
    env: dict[str, str] = field(default_factory=dict)      # query_key -> env a3m text
    paired: dict[str, str] = field(default_factory=dict)   # query_key -> paired a3m text ("" if none)
    env_dir: Path | None = None                            # <target>_env (templates)
    chain_order: list[str] = field(default_factory=list)   # query_key by source-local cid (env qid=101+cid)


def _read(p: Path) -> str:
    p = Path(p)
    if not p.is_file():
        return ""
    t = p.read_text()
    return t if (not t or t.endswith("\n")) else t + "\n"


def _query_key(a3m_path: Path) -> str | None:
    text = Path(a3m_path).read_text() if Path(a3m_path).is_file() else ""
    for _h, body in iter_a3m_records(text):
        return raw_seq_key(body)
    return None


def local_source(workdir: Path, env_dir: Path, kind_of, label: str) -> SourceChains:
    """mmseqs/hhblits precomputed `_workdir` → SourceChains.

    ``kind_of(db_key) -> "uniref"|"env"`` sets each raw DB's role (mmseqs uses
    db_registry.kind; hhblits uses dbs[0]=uniref, rest=env). Chains keyed by the
    query of `merged/<cid>.a3m`.
    """
    workdir = Path(workdir)
    raw, merged, pair = workdir / "raw", workdir / "merged", workdir / "raw" / "pair"
    dbs = [d.name for d in raw.iterdir()
           if d.is_dir() and d.name != "pair" and d.name in DEFAULT_REGISTRY] if raw.is_dir() else []
    sc = SourceChains(label=label, env_dir=Path(env_dir) if env_dir else None)
    if not merged.is_dir():
        return sc
    # iterate chains in source-local cid order (env qid = 101 + cid)
    for cid_i in sorted(int(f.stem) for f in merged.glob("*.a3m") if f.stem.isdigit()):
        cid = str(cid_i)
        qk = _query_key(merged / f"{cid}.a3m")
        if qk is None:
            continue
        sc.uniref[qk] = "".join(_read(raw / db / f"{cid}.a3m")
                                for db in MERGE_ORDER if db in dbs and kind_of(db) == "uniref")
        sc.env[qk] = "".join(_read(raw / db / f"{cid}.a3m")
                             for db in MERGE_ORDER if db in dbs and kind_of(db) == "env")
        sc.paired[qk] = _read(pair / f"{cid}.paired.a3m")
        sc.chain_order.append(qk)
    return sc
