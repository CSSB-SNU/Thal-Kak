"""Stage A — UniRef30 monomer search (colabfold-equivalent procedure).

Independent implementation of the UniRef block of colabfold
`mmseqs/search.py:mmseqs_search_monomer` plus its per-chain `unpackdb`
— same mmseqs commands in the same order, no colabfold source
reproduced. This implementation deliberately differs in two places:

1. We do NOT clean up `prof_res` at function end. The caller (`build_a3m`)
   keeps it alive so Stage C (env DB) and pair search can reuse the
   UniRef-derived HMM profile.
2. We do NOT inline-run env DB or template stages. Those are separate
   modules (`search_envdb`, `MSA.cssb_template`).

Otherwise equivalent: same flag values, same call order, same
filter-override behavior.

Reference: ColabFold's colabfold/mmseqs/search.py
"""

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from MSA.cssb_msa.mmseqs.runner import (
    lndb,
    mvdb,
    rmdb,
    run_mmseqs,
    unpackdb,
)


@dataclass(frozen=True)
class UnirefSearchResult:
    """What `search_uniref30_monomer` leaves behind in `workdir`."""

    prof_res: Path  # workdir/prof_res — caller cleans up after Stage C / pair done
    a3m_dir: Path   # workdir/uniref — text a3m per query id (0.a3m, 1.a3m, ...)
    a3m_db: Path    # workdir/uniref.a3m — mmseqs result DB; build_a3m rmdb's it


def search_uniref30_monomer(
    qdb: Path,
    uniref_dbbase: Path,
    uniref_basename: str,
    workdir: Path,
    *,
    log_dir: Path | None = None,
    threads: int = 16,
    db_load_mode: int = 2,
    sensitivity: float = 8.0,
    expand_eval: float = math.inf,
    align_eval: int = 10,
    diff: int = 3000,
    qsc: float = -20.0,
    max_accept: int = 1000000,
    filter: bool = True,
    num_iterations: int = 3,
    max_seqs: int = 10000,
    initial_eval: float = 0.1,
    expand_max_seq_id: float = 0.95,
    alt_ali: int = 10,
    filter_min_enable: int = 1000,
    mmseqs: Path | None = None,
) -> UnirefSearchResult:
    """Run UniRef Stage A faithfully. See module docstring for flag basis.

    Args:
        qdb: mmseqs sequence DB (output of `createdb` on the query fasta).
            For complex queries, qdb has N entries (one per unique seq).
        uniref_dbbase: directory holding the UniRef DB files.
        uniref_basename: filename stem (e.g. ``"uniref30_2302"``).
        workdir: scratch + output dir for this stage. Must already exist
            or be creatable.
        log_dir: per-step log dir; default `workdir/_logs`.
        threads, db_load_mode: passthrough to mmseqs. `db_load_mode` is
            forced to 0 if no `.idx` is present (matches colabfold).
        sensitivity, expand_eval, align_eval, diff, qsc, max_accept,
            filter: same defaults and override semantics as colabfold's
            `mmseqs_search_monomer`. With `filter=True` (the default),
            `align_eval/qsc/max_accept` are silently overridden to
            `10/0.8/100000` regardless of caller values, matching
            upstream.
        mmseqs: binary override.

    Returns:
        UnirefSearchResult(prof_res, a3m_dir, a3m_db).
    """
    qdb = Path(qdb)
    uniref_dbbase = Path(uniref_dbbase)
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if log_dir is None:
        log_dir = workdir / "_logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Filter override — matches colabfold/mmseqs/search.py.
    if filter:
        align_eval = 10
        qsc = 0.8
        max_accept = 100000

    # DB index detection — matches colabfold/mmseqs/search.py.
    db_dbtype = uniref_dbbase / f"{uniref_basename}.dbtype"
    if not db_dbtype.is_file():
        raise FileNotFoundError(f"UniRef DB missing: {db_dbtype}")
    has_idx = (uniref_dbbase / f"{uniref_basename}.idx").is_file() or (
        uniref_dbbase / f"{uniref_basename}.idx.index"
    ).is_file()
    if has_idx:
        db_suffix1 = ".idx"
        db_suffix2 = ".idx"
        effective_load_mode = db_load_mode
    else:
        db_suffix1 = "_seq"
        db_suffix2 = "_aln"
        effective_load_mode = 0

    db = uniref_dbbase / uniref_basename
    db_idx1 = uniref_dbbase / f"{uniref_basename}{db_suffix1}"
    db_idx2 = uniref_dbbase / f"{uniref_basename}{db_suffix2}"

    res = workdir / "res"
    res_exp = workdir / "res_exp"
    res_exp_realign = workdir / "res_exp_realign"
    res_exp_realign_filter = workdir / "res_exp_realign_filter"
    uniref_a3m_db = workdir / "uniref.a3m"
    prof_res = workdir / "prof_res"
    tmp = workdir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    a3m_dir = workdir / "uniref"
    a3m_dir.mkdir(parents=True, exist_ok=True)

    # Param sets — the same flag values colabfold's search.py passes.
    # --expand-filter-clusters/--filter-msa want INT (0/1). Colabfold uses
    # str(filter), which works there only because main() parses filter as
    # int (type=int) — when imported as a function with a bool, str(True)=
    # 'True' fails with "Error in argument --expand-filter-clusters".
    # Convert ourselves.
    filter_str = "1" if filter else "0"
    threads_arg = ["--threads", str(threads)]
    dlm_arg = ["--db-load-mode", str(effective_load_mode)]
    search_param = [
        "--num-iterations", str(num_iterations),
        *dlm_arg,
        "-a",
        "-s", str(sensitivity),
        "-e", str(initial_eval),
        "--max-seqs", str(max_seqs),
        # pair.sh search step uses the sensitivity tuple.
        "--k-score", "seq:96,prof:80",
    ]
    expand_param = [
        "--expansion-mode", "0",
        "-e", str(expand_eval),
        "--expand-filter-clusters", filter_str,
        "--max-seq-id", str(expand_max_seq_id),
    ]
    filter_param = [
        "--filter-msa", filter_str,
        "--filter-min-enable", str(filter_min_enable),
        "--diff", str(diff),
        "--qid", "0.0,0.2,0.4,0.6,0.8,1.0",
        "--qsc", "0",
        "--max-seq-id", str(expand_max_seq_id),
    ]

    # 1. Iterative search → builds tmp/latest/profile_1
    run_mmseqs(
        [
            "search",
            str(qdb), str(db), str(res), str(tmp),
            *threads_arg,
            *search_param,
        ],
        log_path=log_dir / "01_search.log",
        mmseqs=mmseqs,
    )

    # 2. Cluster expansion
    run_mmseqs(
        [
            "expandaln",
            str(qdb), str(db_idx1), str(res), str(db_idx2), str(res_exp),
            *dlm_arg,
            *threads_arg,
            *expand_param,
        ],
        log_path=log_dir / "02_expandaln.log",
        mmseqs=mmseqs,
    )

    # 3. Persist iter-3 profile
    mvdb(
        tmp / "latest" / "profile_1", prof_res,
        log_path=log_dir / "03_mvdb_prof.log",
        mmseqs=mmseqs,
    )

    # 4. Link query header DB onto prof_res so headers stay readable
    qdb_h = qdb.parent / f"{qdb.name}_h"
    prof_res_h = workdir / "prof_res_h"
    lndb(
        qdb_h, prof_res_h,
        log_path=log_dir / "04_lndb_qdb_h.log",
        mmseqs=mmseqs,
    )

    # 5. Profile-based realign
    run_mmseqs(
        [
            "align",
            str(prof_res), str(db_idx1), str(res_exp), str(res_exp_realign),
            *dlm_arg,
            "-e", str(align_eval),
            "--max-accept", str(max_accept),
            *threads_arg,
            "--alt-ali", str(alt_ali),
            "-a",
        ],
        log_path=log_dir / "05_align.log",
        mmseqs=mmseqs,
    )

    # 6. qsc filter
    run_mmseqs(
        [
            "filterresult",
            str(qdb), str(db_idx1),
            str(res_exp_realign), str(res_exp_realign_filter),
            *dlm_arg,
            "--qid", "0",
            "--qsc", str(qsc),
            "--diff", "0",
            *threads_arg,
            "--max-seq-id", "1.0",
            "--filter-min-enable", "100",
        ],
        log_path=log_dir / "06_filterresult.log",
        mmseqs=mmseqs,
    )

    # 7. Result DB → a3m DB
    run_mmseqs(
        [
            "result2msa",
            str(qdb), str(db_idx1),
            str(res_exp_realign_filter), str(uniref_a3m_db),
            "--msa-format-mode", "6",
            *dlm_arg,
            *threads_arg,
            *filter_param,
        ],
        log_path=log_dir / "07_result2msa.log",
        mmseqs=mmseqs,
    )

    # 8. Cleanup intermediate result DBs
    rmdb(res_exp_realign, log_path=log_dir / "08_rmdb_realign.log", mmseqs=mmseqs)
    rmdb(res_exp,         log_path=log_dir / "09_rmdb_exp.log",     mmseqs=mmseqs)
    rmdb(res,             log_path=log_dir / "10_rmdb_res.log",     mmseqs=mmseqs)
    rmdb(res_exp_realign_filter,
         log_path=log_dir / "11_rmdb_filter.log", mmseqs=mmseqs)

    # 9. Unpack uniref.a3m → per-chain text a3m. We unpack here (rather
    # than after merge with env, like colabfold does) so the per-DB
    # outputs stay isolated for downstream textual provenance + plotting.
    # The uniref.a3m result DB is left in place for the caller to remove.
    unpackdb(
        uniref_a3m_db, a3m_dir,
        suffix=".a3m",
        name_mode=0,
        log_path=log_dir / "12_unpackdb.log",
        mmseqs=mmseqs,
    )

    # 10. Remove tmp dir. prof_res deliberately
    # KEPT — caller cleans it up after Stage C / pair done.
    if tmp.is_dir():
        shutil.rmtree(tmp)

    return UnirefSearchResult(prof_res=prof_res, a3m_dir=a3m_dir, a3m_db=uniref_a3m_db)


