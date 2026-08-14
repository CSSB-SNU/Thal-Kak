"""Stage C — env DB search using a UniRef-derived HMM profile.

Independent implementation of the colabfold `mmseqs_search_monomer` env
block. Critical differences from Stage A (`search_uniref`):

1. **search query is `prof_res`**, not `qdb`. This is the "profile
   chain" — UniRef's iter-3 HMM is the search query, so the prefilter
   uses position-specific scores from UniRef-deep evolutionary signal.
   The env DB hits are filtered through that lens, finding remote
   homologs that a raw-sequence query couldn't find.

2. **expandaln** here drops `--expand-filter-clusters` and
   `--max-seq-id 0.95` (Stage A passes both via `expand_param`). Only
   `-e <expand_eval> --expansion-mode 0 --db-load-mode --threads`.

3. **align query is `tmp/latest/profile_1`** — but THIS IS NOT
   `prof_res` from Phase 2. It's the profile that *this* Stage C call's
   own search step just produced (env-iter-3 profile). Each env DB
   invocation creates its own `tmp/latest/profile_1`, so each call
   needs its own `tmp/` dir to avoid clobber.

4. **filterresult / result2msa** revert to using `qdb` (original
   query), so the output a3m's first row is the user's actual sequence.

Same flag values as Stage A (filter override semantics included): with
`filter=True` (default), `align_eval/qsc/max_accept` get silently
overridden to `10/0.8/100000` regardless of caller values.

Reference: ColabFold's colabfold/mmseqs/search.py
"""

import math
import shutil
from dataclasses import dataclass
from pathlib import Path

from MSA.cssb_msa.mmseqs.runner import (
    rmdb,
    run_mmseqs,
    unpackdb,
)


@dataclass(frozen=True)
class EnvdbSearchResult:
    a3m_dir: Path  # workdir/env — per-chain text a3m (0.a3m, 1.a3m, ...)
    a3m_db: Path   # workdir/env.a3m — mmseqs result DB; the caller removes it


def search_envdb_with_profile(
    qdb: Path,
    prof_res: Path,
    env_dbbase: Path,
    env_basename: str,
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
    expandable: bool = True,
    num_iterations: int = 3,
    max_seqs: int = 10000,
    initial_eval: float = 0.1,
    expand_max_seq_id: float = 0.95,
    alt_ali: int = 10,
    filter_min_enable: int = 1000,
    mmseqs: Path | None = None,
) -> EnvdbSearchResult:
    """Run env-DB Stage C faithful to colabfold's search.py env block.

    Args:
        qdb: original query DB (createdb output). Used by filterresult
            and result2msa to keep the output a3m's query row as the
            user's sequence.
        prof_res: UniRef-derived HMM profile DB (output of `search_uniref`).
            Must remain readable here — its sidecars (`prof_res.lookup`
            etc.) may be symlinks to qdb sidecars; both qdb and prof_res
            workdirs must still exist when this runs.
        env_dbbase, env_basename: env DB location + basename.
        workdir: scratch + output root for this call. **Each env DB
            invocation needs its OWN workdir** because Stage C's search
            step writes `tmp/latest/profile_1` which is consumed
            immediately by align — concurrent or unsegregated calls
            would clobber.
        log_dir: per-step logs; default `workdir/_logs`.
        threads, db_load_mode, sensitivity, expand_eval, align_eval,
            diff, qsc, max_accept, filter: same defaults + override
            semantics as `search_uniref`.
        expandable: True (default) runs `expandaln` (cluster DB with
            `_aln` members). False skips it for a flat seq DB (e.g.
            envhog) and aligns the raw search hits directly.
        mmseqs: binary override.

    Returns:
        EnvdbSearchResult(a3m_dir=workdir/"env", a3m_db=workdir/"env.a3m").
        Per-chain a3m at `0.a3m`, `1.a3m`, ...

        `build_a3m` currently discards this return value and rebuilds
        ``<workdir>/env.a3m`` itself to rmdb it, so that name is written down in
        two places — change both together.
    """
    qdb = Path(qdb)
    prof_res = Path(prof_res)
    env_dbbase = Path(env_dbbase)
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
    db_dbtype = env_dbbase / f"{env_basename}.dbtype"
    if not db_dbtype.is_file():
        raise FileNotFoundError(f"env DB missing: {db_dbtype}")
    has_idx = (env_dbbase / f"{env_basename}.idx").is_file() or (
        env_dbbase / f"{env_basename}.idx.index"
    ).is_file()
    if has_idx:
        db_suffix1 = ".idx"
        db_suffix2 = ".idx"
        effective_load_mode = db_load_mode
    else:
        db_suffix1 = "_seq"
        db_suffix2 = "_aln"
        effective_load_mode = 0

    db = env_dbbase / env_basename
    db_idx1 = env_dbbase / f"{env_basename}{db_suffix1}"
    db_idx2 = env_dbbase / f"{env_basename}{db_suffix2}"

    res_env = workdir / "res_env"
    res_env_exp = workdir / "res_env_exp"
    res_env_exp_realign = workdir / "res_env_exp_realign"
    res_env_exp_realign_filter = workdir / "res_env_exp_realign_filter"
    env_a3m_db = workdir / "env.a3m"
    tmp = workdir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    a3m_dir = workdir / "env"
    a3m_dir.mkdir(parents=True, exist_ok=True)

    filter_str = "1" if filter else "0"  # INT (0/1); str(bool) emits 'True'/'False'
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
    filter_param = [
        "--filter-msa", filter_str,
        "--filter-min-enable", str(filter_min_enable),
        "--diff", str(diff),
        "--qid", "0.0,0.2,0.4,0.6,0.8,1.0",
        "--qsc", "0",
        "--max-seq-id", str(expand_max_seq_id),
    ]

    # 1. search prof_res vs env DB → res_env + tmp/latest/profile_1
    run_mmseqs(
        [
            "search",
            str(prof_res), str(db), str(res_env), str(tmp),
            *threads_arg,
            *search_param,
        ],
        log_path=log_dir / "01_search.log",
        mmseqs=mmseqs,
    )

    # 2. expandaln (note: NO --expand-filter-clusters, NO --max-seq-id 0.95).
    # Skipped for a FLAT (non-clustered) env DB (expandable=False): it has
    # no _aln cluster members to expand, and expandaln there dies with
    # "getData: local id (...) >= db size". Align the raw search hits
    # (res_env) directly — for a flat DB every hit is already a full
    # member sequence, so there is nothing to expand.
    if expandable:
        run_mmseqs(
            [
                "expandaln",
                str(prof_res), str(db_idx1), str(res_env), str(db_idx2), str(res_env_exp),
                "-e", str(expand_eval),
                "--expansion-mode", "0",
                *dlm_arg,
                *threads_arg,
            ],
            log_path=log_dir / "02_expandaln.log",
            mmseqs=mmseqs,
        )
        align_input = res_env_exp
    else:
        align_input = res_env

    # 3. align with the env-iter-3 profile (NOT prof_res)
    env_iter3_profile = tmp / "latest" / "profile_1"
    run_mmseqs(
        [
            "align",
            str(env_iter3_profile), str(db_idx1),
            str(align_input), str(res_env_exp_realign),
            *dlm_arg,
            "-e", str(align_eval),
            "--max-accept", str(max_accept),
            *threads_arg,
            "--alt-ali", str(alt_ali),
            "-a",
        ],
        log_path=log_dir / "03_align.log",
        mmseqs=mmseqs,
    )

    # 4. filterresult — query reverts to original qdb
    run_mmseqs(
        [
            "filterresult",
            str(qdb), str(db_idx1),
            str(res_env_exp_realign), str(res_env_exp_realign_filter),
            *dlm_arg,
            "--qid", "0",
            "--qsc", str(qsc),
            "--diff", "0",
            "--max-seq-id", "1.0",
            *threads_arg,
            "--filter-min-enable", "100",
        ],
        log_path=log_dir / "04_filterresult.log",
        mmseqs=mmseqs,
    )

    # 5. result2msa — qdb again
    run_mmseqs(
        [
            "result2msa",
            str(qdb), str(db_idx1),
            str(res_env_exp_realign_filter), str(env_a3m_db),
            "--msa-format-mode", "6",
            *dlm_arg,
            *threads_arg,
            *filter_param,
        ],
        log_path=log_dir / "05_result2msa.log",
        mmseqs=mmseqs,
    )

    # 6. Cleanup intermediates.
    rmdb(res_env_exp_realign_filter, log_path=log_dir / "06_rmdb_filter.log", mmseqs=mmseqs)
    rmdb(res_env_exp_realign,        log_path=log_dir / "07_rmdb_realign.log", mmseqs=mmseqs)
    if expandable:
        rmdb(res_env_exp,            log_path=log_dir / "08_rmdb_exp.log",     mmseqs=mmseqs)
    rmdb(res_env,                    log_path=log_dir / "09_rmdb_res_env.log", mmseqs=mmseqs)

    # 7. Unpack env a3m DB → per-chain text a3m. The result DB is left in
    # place for the caller to remove.
    unpackdb(
        env_a3m_db, a3m_dir,
        suffix=".a3m",
        name_mode=0,
        log_path=log_dir / "10_unpackdb.log",
        mmseqs=mmseqs,
    )

    # 8. Drop tmp (the env-iter-3 profile is gone with it — fine, we
    # already consumed it via align step 3).
    if tmp.is_dir():
        shutil.rmtree(tmp)

    return EnvdbSearchResult(a3m_dir=a3m_dir, a3m_db=env_a3m_db)


