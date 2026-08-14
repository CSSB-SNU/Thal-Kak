"""Template-search engine selection + single-engine dispatch for cssb MSA modes.

The one entry `cssb_msa` calls to run local template search on a finished
msa_dir. Engine modules are imported lazily (inside the function) so
`resolve_template_engine` stays cheap to import for the skip-key path in
`msa_generation`.

  mmseqs        -> engine_mmseqs.run_mmseqs_for_msa_dir  (vs BioMolDB; mmseqs_cssb)
  hmmer         -> engine_hmmer.run_hmmer_for_msa_dir    (local hmmbuild+hmmsearch; hhblits_cssb)
  mmseqs+hmmer  -> combined mode; dispatched by cssb_msa/combined/build.py, NOT here.

Both engines emit the same <target>_env/pdb70.m8 +
templates_<101+cid>/<pdb>.cif contract.
"""
from __future__ import annotations

import datetime
import logging
from pathlib import Path

from ._common import MsaDirResult, resolve_template_engine

logger = logging.getLogger(__name__)


def run_template_search_for_msa_dir(
    msa_dir, *, mode: str, template_cfg: dict
) -> MsaDirResult:
    """Resolve the engine for (mode, template_cfg) and run it on msa_dir.

    ``template_cfg`` is the EFFECTIVE merged template block. Caller gates on
    ``template_cfg["enable"]``. Single-engine modes only (combined is dispatched
    by ``combined/build.py:run_combined_template_search``).
    """
    engine = resolve_template_engine(mode, template_cfg)
    max_date = datetime.date.fromisoformat(template_cfg["max_date"])
    max_hits = template_cfg["max_hits"]
    if engine == "mmseqs":
        from .engine_mmseqs import run_mmseqs_for_msa_dir
        logger.info("Running local template search (engine=mmseqs, vs BioMolDB)...")
        return run_mmseqs_for_msa_dir(
            msa_dir=Path(msa_dir), max_template_date=max_date, max_hits=max_hits,
        )
    if engine == "hmmer":
        from .engine_hmmer import run_hmmer_for_msa_dir
        logger.info(
            "Running local template search (engine=hmmer, local "
            "hmmbuild+hmmsearch, query a3m source = %s)...",
            template_cfg["query_a3m"],
        )
        return run_hmmer_for_msa_dir(
            msa_dir=Path(msa_dir),
            query_a3m_source=template_cfg["query_a3m"],
            max_template_date=max_date,
            max_hits=max_hits,
        )
    raise ValueError(
        f"engine {engine!r} (mode={mode!r}) is not single-engine dispatchable here; "
        f"combined mode runs via cssb_msa/combined/build.py."
    )
