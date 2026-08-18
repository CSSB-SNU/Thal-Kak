"""Phase 5 — UniRef pair search for multimer queries.

Independent implementation. The flag set is the one the public ColabFold
MSA server actually uses for multimer pairing, read off the `pair.sh` that the
server emits inside its own `out.tar.gz` response. That script is what the
server does; ColabFold's local Python `mmseqs_search_pair` is a stale mirror
that misses the pairaln flags. No source from the server implementation is
reproduced here. The flags that matter: `--k-score seq:96,prof:80`,
`--pairing-mode 0`, `--pairing-dummy-mode {0,1}`.

Runs only against a UniRef DB — env/template DBs lack the NCBI
taxonomy headers that `pairaln` needs to match hits across chains.
Caller invokes this only when `n_unique > 1` (true heteromer); homomers
(`n_unique == 1, cardinality > 1`) skip pair search entirely — colabfold
itself short-circuits homomer pairing in `main()` by setting
`paired_msa=None`.

Pipeline (10 mmseqs commands):

  | step | command | note |
  |------|---------|------|
  | 1    | search                    | iter-3 PSSM (-a -s 8 -e 0.1) + `--k-score seq:96,prof:80` |
  | 1b   | mvdb tmp/latest/profile_1 → prof_res | persist iter3 PSSM |
  | 1c   | lndb qdb_h → prof_res_h   | attach query headers to profile |
  | 2    | expandaln                 | --expand-filter-clusters 0 |
  | 3    | align (strict)            | **prof_res** (NOT qdb), -e 0.001, max-accept 1M, NO cov gate |
  | 4    | pairaln (1st)             | taxonomy match + `--pairing-mode 0 --pairing-dummy-mode 0` |
  | 5    | align (back-translate)    | **prof_res**, -e inf, -a |
  | 6    | pairaln (2nd)             | final refinement + `--pairing-mode 0 --pairing-dummy-mode 1` (gap-fill) |
  | 7    | result2msa                | --msa-format-mode 6 (rich headers, matches colab) |
  | 8    | unpackdb                  | --unpack-suffix .paired.a3m |

Design notes:
- align steps 3, 5 use `prof_res` (PSSM), not raw `qdb`, to keep recall on
  short / motif-driven chains (profile-based align retains partial motif hits
  via position-specific weights). Step 3 omits the coverage gate
  (`-c 0.5 --cov-mode 1`) since the colab API doesn't gate coverage here, and
  step 5 adds `-a` (backtrace, as in colab pair.sh).
- result2msa uses `--msa-format-mode 6` (rich headers, matches colab output).
- `--pairing-mode` / `--pairing-dummy-mode` are pair-only; `--k-score
  seq:96,prof:80` is also used by `search_uniref.py` / `search_envdb.py`.

Important runtime contract:
- Caller passes `qdb` from elsewhere; we do **not** rmdb it (colabfold
  L316-317 does, because pair is its last stage; here the caller reuses it).
- `prof_res` is built INSIDE this stage from THIS stage's iterative
  search (independent of Phase 2's monomer prof_res). We mvdb it from
  `tmp/latest/profile_1` and rmdb it at the end.
- `pairaln` requires UniRef DB taxonomy. Default is `uniref30_2302`.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path

from MSA.local_msa.mmseqs.runner import (
    lndb,
    mvdb,
    rmdb,
    run_mmseqs,
    unpackdb,
)


@dataclass(frozen=True)
class PairSearchResult:
    paired_a3m_dir: Path  # workdir/paired — per-chain N.paired.a3m


def search_pair_uniref30(
    qdb: Path,
    uniref_dbbase: Path,
    uniref_basename: str,
    workdir: Path,
    *,
    log_dir: Path | None = None,
    threads: int = 16,
    db_load_mode: int = 2,
    sensitivity: float = 8.0,
    num_iterations: int = 3,
    max_seqs: int = 10000,
    initial_eval: float = 0.1,
    expand_max_seq_id: float = 0.95,
    pair_align_eval: float = 0.001,
    mmseqs: Path | None = None,
) -> PairSearchResult:
    """Run UniRef pair search, following the ColabFold API's pair.sh.

    Args:
        qdb: query DB (output of `createdb`). Must contain ≥ 2 entries
            for pairaln to produce non-degenerate output. Caller is
            responsible for the multimer gate.
        uniref_dbbase, uniref_basename: UniRef DB location + basename.
        workdir: scratch + output dir for this stage. Each invocation
            needs its own — `tmp/latest/profile_1` is rebuilt inside.
        log_dir: per-step logs; default `workdir/_logs`.
        threads, db_load_mode, sensitivity: passthrough.
        mmseqs: binary override.

    Returns:
        PairSearchResult(paired_a3m_dir=workdir/"paired"). Per-chain
        paired a3m at `0.paired.a3m`, `1.paired.a3m`, ... Row counts
        are positionally aligned across chains (row k of chain 0 paired
        with row k of chain 1, etc.).
    """
    qdb = Path(qdb)
    uniref_dbbase = Path(uniref_dbbase)
    workdir = Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if log_dir is None:
        log_dir = workdir / "_logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    db_dbtype = uniref_dbbase / f"{uniref_basename}.dbtype"
    if not db_dbtype.is_file():
        raise FileNotFoundError(f"UniRef DB missing: {db_dbtype}")
    # `pairaln` (steps 4 and 6) matches hits across chains by NCBI taxon, which it
    # reads from these two sidecars. They are not part of the database proper --
    # for uniref30_2302 they ship separately, so a DB that searches perfectly well
    # can still be unable to pair. Monomers never reach this function, so this
    # only gates multimer runs.
    missing = [
        p.name
        for p in (
            uniref_dbbase / f"{uniref_basename}_mapping",
            uniref_dbbase / f"{uniref_basename}_taxonomy",
        )
        if not p.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"UniRef DB {uniref_dbbase} cannot pair: missing {', '.join(missing)}\n"
            "  Multimer pairing needs the taxonomy sidecars, which install\n"
            "  separately from the database. Run ./install_db.sh --family mmseqs uniref30_2302"
        )
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
    prof_res = workdir / "prof_res"
    prof_res_h = workdir / "prof_res_h"
    qdb_h = Path(str(qdb) + "_h")
    res_exp = workdir / "res_exp"
    res_exp_realign = workdir / "res_exp_realign"
    res_exp_realign_pair = workdir / "res_exp_realign_pair"
    res_exp_realign_pair_bt = workdir / "res_exp_realign_pair_bt"
    res_final = workdir / "res_final"
    pair_a3m_db = workdir / "pair.a3m"
    tmp = workdir / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    paired_dir = workdir / "paired"
    paired_dir.mkdir(parents=True, exist_ok=True)

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
    # NB: --expand-filter-clusters is hardcoded "0" in colabfold's pair
    # path, unlike Stage A which passes str(filter).
    expand_param = [
        "--expansion-mode", "0",
        "-e", "inf",
        "--expand-filter-clusters", "0",
        "--max-seq-id", str(expand_max_seq_id),
    ]

    # 1. Iterative search → builds tmp/latest/profile_1 (iter3 PSSM)
    run_mmseqs(
        ["search", str(qdb), str(db), str(res), str(tmp), *threads_arg, *search_param],
        log_path=log_dir / "01_search.log",
        mmseqs=mmseqs,
    )

    # 1b. Persist iter3 profile to a stable path (mvdb); subsequent align
    # steps use this PSSM as the query DB instead of raw qdb. Matches
    # colab API pair.sh.
    mvdb(
        tmp / "latest" / "profile_1", prof_res,
        log_path=log_dir / "01b_mvdb_prof_res.log", mmseqs=mmseqs,
    )

    # 1c. Attach the query header DB (qdb_h) to prof_res via lndb so
    # downstream result2msa knows the original query name. Profile DBs
    # don't carry their own headers.
    lndb(
        qdb_h, prof_res_h,
        log_path=log_dir / "01c_lndb_prof_res_h.log", mmseqs=mmseqs,
    )

    # 2. Cluster expansion (--expand-filter-clusters 0)
    run_mmseqs(
        [
            "expandaln",
            str(qdb), str(db_idx1), str(res), str(db_idx2), str(res_exp),
            *dlm_arg, *threads_arg, *expand_param,
        ],
        log_path=log_dir / "02_expandaln.log",
        mmseqs=mmseqs,
    )

    # 3. Strict align using PROFILE (NOT raw qdb): -e 0.001, no cov gate.
    # PSSM picks up distant homologs that raw seq align would drop;
    # removing -c 0.5 --cov-mode 1 lets partial motif alignments through.
    # Matches pair.sh.
    run_mmseqs(
        [
            "align",
            str(prof_res), str(db_idx1), str(res_exp), str(res_exp_realign),
            *dlm_arg,
            "-e", str(pair_align_eval),
            "--max-accept", "1000000",
            *threads_arg,
        ],
        log_path=log_dir / "03_align.log",
        mmseqs=mmseqs,
    )

    # 4. First pairaln — taxonomy-based pair matching.
    # explicit --pairing-mode 0 (greedy maximal per-species)
    # + --pairing-dummy-mode 0 (no gap-fill rows here; gap-fill is for the
    # 2nd pairaln only). Matches API pair.sh.
    run_mmseqs(
        [
            "pairaln",
            str(qdb), str(db),
            str(res_exp_realign), str(res_exp_realign_pair),
            *dlm_arg, *threads_arg,
            "--pairing-mode", "0",
            "--pairing-dummy-mode", "0",
        ],
        log_path=log_dir / "04_pairaln.log",
        mmseqs=mmseqs,
    )

    # 5. Back-translate align using PROFILE: -e inf -a (compute
    # backtrace). Matches pair.sh; raw qdb would lose profile info.
    run_mmseqs(
        [
            "align",
            str(prof_res), str(db_idx1),
            str(res_exp_realign_pair), str(res_exp_realign_pair_bt),
            *dlm_arg,
            "-e", "inf",
            "-a",
            *threads_arg,
        ],
        log_path=log_dir / "05_align_bt.log",
        mmseqs=mmseqs,
    )

    # 6. Second pairaln — final refinement.
    # --pairing-mode 0 + --pairing-dummy-mode 1 (gap-only
    # DUMMY rows for species missing one chain, equalising row count
    # across chains). Matches API pair.sh.
    run_mmseqs(
        [
            "pairaln",
            str(qdb), str(db),
            str(res_exp_realign_pair_bt), str(res_final),
            *dlm_arg, *threads_arg,
            "--pairing-mode", "0",
            "--pairing-dummy-mode", "1",
        ],
        log_path=log_dir / "06_pairaln_final.log",
        mmseqs=mmseqs,
    )

    # 7. Result → a3m DB. --msa-format-mode 6 matches the colab API: hit
    # headers carry `>id1\t<aln_len>\t<pident>\t<evalue>\t<qstart>\t<qend>\t
    # <qlen>\t<tstart>\t<tend>\t<tlen>\tid2\t...` rather than a bare
    # `>id1\tid2`. The downstream splitter accepts either form.
    run_mmseqs(
        [
            "result2msa",
            str(qdb), str(db_idx1), str(res_final), str(pair_a3m_db),
            *dlm_arg,
            "--msa-format-mode", "6",
            *threads_arg,
        ],
        log_path=log_dir / "07_result2msa.log",
        mmseqs=mmseqs,
    )

    # 8. Unpack to per-chain text paired a3m
    unpackdb(
        pair_a3m_db, paired_dir,
        suffix=".paired.a3m",
        name_mode=0,
        log_path=log_dir / "08_unpackdb.log",
        mmseqs=mmseqs,
    )

    # 9. Cleanup our intermediates. NB we do NOT rmdb qdb (caller's).
    rmdb(res,                     log_path=log_dir / "09_rmdb_res.log",          mmseqs=mmseqs)
    rmdb(prof_res,                log_path=log_dir / "10_rmdb_prof_res.log",     mmseqs=mmseqs)
    rmdb(prof_res_h,              log_path=log_dir / "11_rmdb_prof_res_h.log",   mmseqs=mmseqs)
    rmdb(res_exp,                 log_path=log_dir / "12_rmdb_exp.log",          mmseqs=mmseqs)
    rmdb(res_exp_realign,         log_path=log_dir / "13_rmdb_realign.log",      mmseqs=mmseqs)
    rmdb(res_exp_realign_pair,    log_path=log_dir / "14_rmdb_pair.log",         mmseqs=mmseqs)
    rmdb(res_exp_realign_pair_bt, log_path=log_dir / "15_rmdb_pair_bt.log",      mmseqs=mmseqs)
    rmdb(res_final,               log_path=log_dir / "16_rmdb_final.log",        mmseqs=mmseqs)
    rmdb(pair_a3m_db,             log_path=log_dir / "17_rmdb_pair_a3m.log",     mmseqs=mmseqs)
    if tmp.is_dir():
        shutil.rmtree(tmp)

    return PairSearchResult(paired_a3m_dir=paired_dir)


