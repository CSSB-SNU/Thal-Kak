"""Low-level hhblits subprocess wrapper.

Mirrors `MSA.local_msa.mmseqs.runner.run_mmseqs` in shape: one helper
that runs the binary, logs cmd + stdout/stderr + exit code to a per-step
file, and raises `RuntimeError` referencing the log on non-zero exit. No
retries — keep the wrapper minimal and let real errors surface.

The binary comes from the active env's PATH; HHsuite is part of the
project environment (`environment.yml`).
"""

import logging
import shutil
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_hhblits(explicit: Path | None = None) -> Path:
    """Resolve the hhblits binary: an `explicit` path if given, else
    `hhblits` from the active env's PATH. Fails loud if it is missing —
    no fallback."""
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"hhblits binary {path} not found")
        return path
    found = shutil.which("hhblits")
    if found is None:
        raise FileNotFoundError(
            "hhblits not found on PATH; activate the project env "
            "(`conda activate thalkak`), or pass an explicit `hhblits`"
        )
    return Path(found)


def hhblits_version(hhblits: Path | None = None) -> str | None:
    """Best-effort version capture for method_log. Never raises."""
    try:
        binary = resolve_hhblits(hhblits)
        # `hhblits -h` prints "HHblits 3.3.0:" on the first line.
        out = subprocess.run(
            [str(binary), "-h"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        first = (out.stdout or out.stderr).splitlines()[:1]
        return first[0].strip() if first else None
    except Exception as exc:
        logger.warning("could not capture hhblits version: %s", exc)
        return None


def run_hhblits(
    input_path: Path,
    output_a3m: Path,
    db_stem: Path,
    *,
    log_path: Path,
    output_hhr: Path | None = None,
    n_iter: int = 3,
    evalue: float = 0.001,
    threads: int = 16,
    hhblits: Path | None = None,
    extra_args: list[str] | None = None,
) -> None:
    """Run hhblits once. Logs cmd + stdout/stderr to log_path; raises
    RuntimeError on non-zero exit referencing the log.

    Args:
        input_path: query (FASTA for Stage A; a3m for Stage C iterating
            on a previously-built MSA — hhblits auto-detects).
        output_a3m: -oa3m destination.
        db_stem: -d argument; hhblits discovers
            `<stem>_a3m.ffindex/.ffdata`, `_hhm.*`, `_cs219.*`.
        log_path: per-call log file (created; parent dirs created).
        output_hhr: -o destination for hhr report. Default: `output_a3m`
            with its suffix replaced by `.hhr`. Pass `Path("/dev/null")` to
            skip writing one.
        n_iter: -n iterations. 3 for both UniRef Stage A and env Stage C
            (matches AF2/ColabFold convention).
        evalue: -e threshold.
        threads: -cpu.
        hhblits: binary path override.
        extra_args: appended verbatim (escape hatch for one-off flags).
    """
    binary = resolve_hhblits(hhblits)
    output_a3m = Path(output_a3m)
    output_a3m.parent.mkdir(parents=True, exist_ok=True)
    if output_hhr is None:
        output_hhr = output_a3m.with_suffix(".hhr")
    # AlphaFold's HHblits invocation, as its own `data/tools/hhblits.py`
    # defaults it:
    #   -n 3 -e 0.001 -realign_max 100000 -maxfilt 100000
    #   -min_prefilter_hits 1000 -p 20 -Z 500
    # The latter five loosen HHblits's default filter ladder so more
    # hits survive prefilter → maxfilt → realign and end up in the
    # output a3m. -p and -Z match HHblits defaults but are pinned so the
    # invocation stays explicit.
    args = [
        str(binary),
        "-i", str(input_path),
        "-d", str(db_stem),
        "-oa3m", str(output_a3m),
        "-o", str(output_hhr),
        "-n", str(n_iter),
        "-e", str(evalue),
        "-realign_max", "100000",
        "-maxfilt", "100000",
        "-min_prefilter_hits", "1000",
        "-p", "20",
        "-Z", "500",
        "-cpu", str(threads),
    ]
    if extra_args:
        args.extend(extra_args)

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_iso = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("w") as log_f:
        log_f.write(f"=== hhblits at {started_iso} ===\n")
        log_f.write(f"cmd: {' '.join(args)}\n\n")
        log_f.flush()
        try:
            subprocess.run(
                args,
                check=True,
                stdout=log_f,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as exc:
            elapsed = time.monotonic() - started
            log_f.write(f"\n=== exit {exc.returncode} in {elapsed:.2f}s ===\n")
            raise RuntimeError(
                f"hhblits failed (exit {exc.returncode}). See log: {log_path}"
            ) from exc
        elapsed = time.monotonic() - started
        log_f.write(f"\n=== exit 0 in {elapsed:.2f}s ===\n")
