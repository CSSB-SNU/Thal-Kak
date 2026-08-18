"""Low-level mmseqs subprocess wrapper + management-subcommand helpers.

This module is the only place in `MSA.local_msa` that calls mmseqs as a
subprocess. Higher-level modules — `search_uniref`, `search_envdb`,
`search_pair`, ... — compose pipelines using these helpers plus the
generic `run_mmseqs(args, ...)` for the search/expandaln/align/...
subcommands whose call-sites have too many params to wrap thinly.

Every invocation logs to a per-step file:

    === mmseqs <subcommand> at YYYY-MM-DD HH:MM:SS ===
    cmd: <full command line>
    [combined stdout+stderr]
    === exit <code> in <wall>s ===

On non-zero exit the `subprocess.CalledProcessError` is wrapped into a
`RuntimeError` mentioning the log path so the caller can grep without
hunting for which file got the output.

The default mmseqs binary is `shutil.which("mmseqs")` on the active conda
env's PATH — whatever env the pipeline runs in supplies mmseqs (the
project env `thalkak`, from `environment.yml`). Override per-call via
the `mmseqs=` kwarg.
"""

import logging
import shutil
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_mmseqs(explicit: Path | None = None) -> Path:
    """Resolve the mmseqs binary: an `explicit` path if given, else `mmseqs`
    from the active env's PATH. Fails loud if it is missing — no fallback."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"mmseqs binary {path} not found")
        return path
    found = shutil.which("mmseqs")
    if found is None:
        raise FileNotFoundError(
            "mmseqs not found on PATH; activate the project env "
            "(`conda activate thalkak`), or pass an explicit `mmseqs`"
        )
    return Path(found)


def mmseqs_version(mmseqs: Path | None = None) -> str | None:
    """Best-effort version capture for method_log. Never raises."""
    try:
        binary = resolve_mmseqs(mmseqs)
        out = subprocess.run(
            [str(binary), "version"],
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip() or None
    except Exception as exc:
        logger.warning("could not capture mmseqs version: %s", exc)
        return None


def run_mmseqs(
    args: list[str],
    *,
    log_path: Path,
    mmseqs: Path | None = None,
    cwd: Path | None = None,
) -> None:
    """Run `mmseqs <args[0]> <args[1:]>`, log to `log_path`, raise on
    non-zero exit. The caller assembles the full subcommand + flags;
    this wrapper only prepends the binary path and handles logging.
    """
    if not args:
        raise ValueError("run_mmseqs: empty args list")
    binary = resolve_mmseqs(mmseqs)
    cmd = [str(binary), *args]
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_iso = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("w") as log_f:
        log_f.write(f"=== mmseqs {args[0]} at {started_iso} ===\n")
        log_f.write(f"cmd: {' '.join(cmd)}\n")
        if cwd is not None:
            log_f.write(f"cwd: {cwd}\n")
        log_f.write("\n")
        log_f.flush()
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(cwd) if cwd else None,
            )
        except subprocess.CalledProcessError as exc:
            elapsed = time.monotonic() - started
            log_f.write(f"\n=== exit {exc.returncode} in {elapsed:.2f}s ===\n")
            raise RuntimeError(
                f"mmseqs {args[0]} failed (exit {exc.returncode}). "
                f"See log: {log_path}"
            ) from exc
        elapsed = time.monotonic() - started
        log_f.write(f"\n=== exit 0 in {elapsed:.2f}s ===\n")


# Management-subcommand helpers (thin args wrappers).


def createdb(
    input_fa: Path,
    out_db: Path,
    *,
    log_path: Path,
    mmseqs: Path | None = None,
    shuffle: bool = False,
    dbtype: int = 1,
) -> None:
    """`mmseqs createdb <input_fa> <out_db> --shuffle 0 --dbtype 1` —
    fasta → mmseqs sequence DB. Default `shuffle=False` matches colabfold
    `mmseqs/search.py` (colabfold always passes `--shuffle 0` so the
    qdb sequence order tracks the input fasta record order, which we
    rely on for chain id assignment downstream). Default `dbtype=1`
    (amino acid) matches MMseqs2-App `pair.sh` (the actual ColabFold API
    server template). local_msa is protein-only so this is always 1.
    """
    args = [
        "createdb", str(input_fa), str(out_db),
        "--shuffle", "1" if shuffle else "0",
        "--dbtype", str(dbtype),
    ]
    run_mmseqs(args, log_path=log_path, mmseqs=mmseqs)


def mvdb(
    src: Path,
    dst: Path,
    *,
    log_path: Path,
    mmseqs: Path | None = None,
) -> None:
    """`mmseqs mvdb <src> <dst>` — move an mmseqs DB and all its sidecars
    to a new path. Used to persist `tmp/latest/profile_1` → `prof_res` so
    Stage C can chain off the Stage A profile.
    """
    run_mmseqs(
        ["mvdb", str(src), str(dst)],
        log_path=log_path,
        mmseqs=mmseqs,
    )


def lndb(
    src: Path,
    dst: Path,
    *,
    log_path: Path,
    mmseqs: Path | None = None,
) -> None:
    """`mmseqs lndb <src> <dst>` — symlink an mmseqs DB (sidecars too).
    Colabfold uses this in Stage A to attach the query header lookup
    (`qdb_h`) to the profile DB so result a3ms keep the original query
    name.
    """
    run_mmseqs(
        ["lndb", str(src), str(dst)],
        log_path=log_path,
        mmseqs=mmseqs,
    )


def rmdb(
    target: Path,
    *,
    log_path: Path,
    mmseqs: Path | None = None,
) -> None:
    """`mmseqs rmdb <target>` — remove an mmseqs DB and its sidecars
    (`<target>`, `<target>.dbtype`, `<target>.index`, `<target>.lookup`,
    `<target>.source`).

    NB: companion DBs like `<target>_h` (header DB created alongside a
    sequence DB by `createdb`) are NOT cleaned by this call. Issue a
    second `rmdb(target_h_path, ...)` if you need that, or rely on the
    workdir teardown.
    """
    run_mmseqs(
        ["rmdb", str(target)],
        log_path=log_path,
        mmseqs=mmseqs,
    )


def mergedbs(
    qdb: Path,
    out_db: Path,
    parts: list[Path],
    *,
    log_path: Path,
    mmseqs: Path | None = None,
) -> None:
    """`mmseqs mergedbs <qdb> <out_db> <part_1> ... <part_N>` —
    column-wise concat of N result DBs (entries with same query id are
    appended). Currently unused: the per-chain merge works on unpacked
    text a3ms instead (`common/leveled_merge.py`, `common/cap_merge.py`).

    `qdb` is the lookup source for query IDs; can also be a result DB
    used as a "lookup borrow" trick (see colabfold's
    `mmseqs/merge_and_split_msas.py:merge_msa`).
    """
    if not parts:
        raise ValueError("mergedbs requires at least one input part")
    run_mmseqs(
        ["mergedbs", str(qdb), str(out_db), *(str(p) for p in parts)],
        log_path=log_path,
        mmseqs=mmseqs,
    )


def unpackdb(
    src_db: Path,
    out_dir: Path,
    *,
    suffix: str,
    name_mode: int = 0,
    log_path: Path,
    mmseqs: Path | None = None,
) -> None:
    """`mmseqs unpackdb <src_db> <out_dir> --unpack-suffix <suffix>
    --unpack-name-mode <mode>` — unpack a result DB into per-entry text
    files at `out_dir/<id><suffix>`.

    `name_mode=0` (default, matches colabfold) names files by sequential
    int id — that is what turns `uniref.a3m` / `env.a3m` into `0.a3m`,
    `1.a3m`, ... and `pair.a3m` into `0.paired.a3m`, ... .
    """
    run_mmseqs(
        [
            "unpackdb",
            str(src_db),
            str(out_dir),
            "--unpack-suffix", suffix,
            "--unpack-name-mode", str(name_mode),
        ],
        log_path=log_path,
        mmseqs=mmseqs,
    )
