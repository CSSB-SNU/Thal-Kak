import argparse, json, logging, subprocess, yaml, os, sys, glob, shutil, shlex, tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import gemmi

from thalkak import get_logger, run_logged, log_lines

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMMON_DIR = os.path.join(ROOT, "Structure", "script", "common")
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)
from chain_utils import PDB_CHAIN_CHARS

log = get_logger("relax")


def cif_to_pdb(cif_path):
    """Convert an mmCIF model to PDB, keeping the chain ids as they are.

    The output path is the input path with the ``.cif`` suffix replaced by
    ``.pdb``, and is returned.
    """
    if not os.path.isfile(cif_path):
        raise FileNotFoundError(f"File not found: {cif_path}")

    structure = gemmi.read_structure(str(cif_path))
    if len(structure) == 0:
        raise ValueError(f"No model found in {cif_path}")

    n_chains = len(structure[0])
    if n_chains > len(PDB_CHAIN_CHARS):
        raise ValueError(
            f"Too many chains to convert to pdb: {n_chains} > "
            f"{len(PDB_CHAIN_CHARS)} single-char PDB chain labels ({cif_path})"
        )

    pdb_path = str(Path(cif_path).with_suffix(".pdb"))
    structure.write_pdb(pdb_path)

    return pdb_path


def convert_cif_decoys(decoy_dir, log=log):
    """Convert ``*.cif`` decoys in ``decoy_dir`` into same-basename ``*.pdb``
    files, so they are picked up downstream -- QA, the validation pre-pass and
    all three relax paths read ``*.pdb`` only. A cif that fails to convert is
    warned about and left out of the batch."""
    cifs = sorted(glob.glob(os.path.join(decoy_dir, "*.cif")))
    if not cifs:
        return
    log.info(f"Found {len(cifs)} cif decoy(s) in {decoy_dir}; converting to pdb...")
    for cif in cifs:
        out_pdb = str(Path(cif).with_suffix(".pdb"))
        if os.path.exists(out_pdb):
            log.info(
                f"pdb already exists: {os.path.basename(out_pdb)}, skipping conversion."
            )
            continue
        try:
            cif_to_pdb(cif)
        except Exception as e:
            log.warning(
                f"{os.path.basename(cif)}: cif->pdb conversion failed ({e}); "
                f"skipping this structure."
            )
            continue
        log.info(f"{os.path.basename(cif)} -> {os.path.basename(out_pdb)}")


def build_ligand_specs(data_yaml=None, ligand_ccd=None, ligand_smiles=None):
    """Ligand spec list ({ccd|smiles} dicts) for a standalone `relax`: the data
    config's 'ligand' list (--data_yaml) plus any --ligand_ccd/--ligand_smiles."""
    specs = []
    if data_yaml:
        with open(data_yaml) as f:
            specs = list((yaml.safe_load(f) or {}).get("ligand") or [])
    specs += [{"ccd": c} for c in ligand_ccd or []]
    specs += [{"smiles": s} for s in ligand_smiles or []]
    return specs


def _merge_per_job_energies(relax_dir):
    """Aggregate per-job ``*.energy.yaml`` files (written by each relax worker)
    into a single ``energies.yaml`` keyed by the output basename. Done in the
    orchestrator after the parallel batch so workers don't race on a shared
    file."""
    energies_path = os.path.join(relax_dir, "energies.yaml")
    merged = {}
    if os.path.exists(energies_path):
        with open(energies_path) as f:
            merged = yaml.safe_load(f) or {}
    per_job_files = glob.glob(os.path.join(relax_dir, "*.energy.yaml"))
    for per_job in per_job_files:
        with open(per_job) as f:
            merged.update(yaml.safe_load(f) or {})
    if merged:
        with open(energies_path, "w") as f:
            yaml.dump(merged, f, sort_keys=False)
    for per_job in per_job_files:
        os.remove(per_job)


def _available_cpus():
    """Number of CPUs to use.

    Inside a SLURM job submitted with ``--cpus-per-task`` (e.g. sbatch with
    ``-c N``), respects that allocation via ``SLURM_CPUS_PER_TASK``. Otherwise
    (interactive ``srun --pty`` with no ``-c``, or no SLURM at all) falls back
    to the OS's view -- the process CPU affinity mask, then ``os.cpu_count()``.
    """
    n = os.environ.get("SLURM_CPUS_PER_TASK")
    if n:
        try:
            return max(1, int(n))
        except ValueError:
            pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def _validate_decoys(decoy_dir, staging, log):
    """Run ``Relax/script/validate.py`` over ``decoy_dir``, writing normalized
    copies (prochiral methyl naming + C-terminal sp2 carboxylate; same basenames,
    B-factors/numbering preserved) into ``staging`` — a scratch dir the caller
    removes once the batch is done. Both relax methods then read the copies."""
    os.makedirs(staging, exist_ok=True)
    log.info(f"Validating decoy inputs before relaxation -> {staging}")
    run_logged(
        f"python {ROOT}/Relax/script/validate.py "
        f"-in_dir {shlex.quote(decoy_dir)} -out_dir {shlex.quote(staging)}",
        log,
    )
    return staging


def relaxation(decoy_dir, relax_method, relax_config=None, ligand_specs=None):
    relax_dir = os.path.join(decoy_dir, "relaxed", relax_method)
    os.makedirs(relax_dir, exist_ok=True)

    # Inherit the decoy_dir method_log and record the relax method on top, so
    # the relaxed method_log keeps upstream provenance rather than only the
    # `relax:` entry.
    relax_log = {}
    _decoy_ml = os.path.join(decoy_dir, "method_log.yaml")
    if os.path.exists(_decoy_ml):
        with open(_decoy_ml) as f:
            relax_log = yaml.safe_load(f) or {}
    relax_log["relax"] = relax_method

    # Ligands are kept/parametrized only by openmm.
    specs = list(ligand_specs or [])
    if specs and relax_method != "openmm":
        log.warning(
            f"{len(specs)} ligand spec(s) ignored for method '{relax_method}' "
            f"(only 'openmm' keeps and parametrizes ligands)."
        )

    cfg = {}
    config_flag = ""
    if relax_method != "none":
        if relax_config is None:
            relax_config = os.path.join(ROOT, "examples", f"{relax_method}.yaml")
        with open(relax_config) as f:
            cfg = yaml.safe_load(f) or {}
        config_flag = f"-relax_config {shlex.quote(relax_config)}"
        log.info(f"Using {relax_method} relax config: {relax_config}")

    # cif decoys (e.g. straight from a predictor) are converted in place first,
    # so every path below sees them as ordinary *.pdb inputs.
    convert_cif_decoys(decoy_dir)

    # Pre-relaxation validation: normalize inputs once for the actual relax
    # methods into a scratch dir removed after the batch (see the finally below).
    # `none` is a pass-through of the original structure, so it is left untouched.
    src_dir = decoy_dir
    staging = None
    if relax_method == "openmm":
        staging = tempfile.mkdtemp(prefix="thalkak_validate_")
        src_dir = _validate_decoys(decoy_dir, staging, log)

    try:
        match relax_method:
            case "none":
                log.info("No relaxation.")
                log.info(
                    f"Copying unrelaxed structures from {decoy_dir} to relax directory..."
                )
                decoys = glob.glob(os.path.join(decoy_dir, "*.pdb"))
                for pdb in decoys:
                    out_prefix = os.path.join(
                        relax_dir,
                        os.path.splitext(os.path.basename(pdb))[0] + "_unrelaxed",
                    )
                    if os.path.exists(f"{out_prefix}.pdb"):
                        log.info(
                            f"Unrelaxed structure already exists: {out_prefix}.pdb, skipping."
                        )
                        continue
                    shutil.copy(pdb, f"{out_prefix}.pdb")
                log.info("Copying complete.")

            case "openmm":
                solvent = cfg.get("implicit_solvent", "obc2")
                log.info(
                    f"Running all-atom OpenMM (amber19-all + GLYCAM + ions, "
                    f"{solvent.upper()}) relaxation..."
                )
                decoys = glob.glob(os.path.join(src_dir, "*.pdb"))
                log.info(f"found {len(decoys)} pdb files in {src_dir}")
                lig_flags = ""
                if specs:
                    lig_flags = f" -ligand_specs {shlex.quote(json.dumps(specs))}"
                # Serial (GPU-bound); each job streams its output live. Collect
                # failures and keep going -- a bad decoy shouldn't abort the rest of
                # the batch -- then raise once at the end.
                failures = []
                for pdb in decoys:
                    out_prefix = os.path.join(
                        relax_dir,
                        os.path.splitext(os.path.basename(pdb))[0] + "_relaxed_openmm",
                    )
                    if os.path.exists(f"{out_prefix}.pdb"):
                        log.info(
                            f"Relaxed structure already exists: {out_prefix}.pdb, skipping."
                        )
                        continue
                    returncode = run_logged(
                        f"python {ROOT}/Relax/script/openmm/relax.py "
                        f"-pdb_fn {pdb} -out_prefix {out_prefix} {config_flag}{lig_flags}",
                        log,
                        check=False,
                    )
                    if returncode != 0:
                        failures.append(pdb)
                        log.error(f"FAILED: {os.path.basename(pdb)}")
                    else:
                        log.info(f"done: {os.path.basename(pdb)}")
                    _merge_per_job_energies(relax_dir)
                _merge_per_job_energies(relax_dir)  # sweep strays (e.g. all-skip case)
                if failures:
                    raise RuntimeError(
                        f"OpenMM relaxation failed for {len(failures)} "
                        f"structure(s): {[os.path.basename(f) for f in failures]}"
                    )
                log.info("Relaxation complete.")

        with open(os.path.join(relax_dir, "method_log.yaml"), "w") as f:
            yaml.dump(relax_log, f, sort_keys=False)

        return relax_dir
    finally:
        if staging:
            shutil.rmtree(staging, ignore_errors=True)


if __name__ == "__main__":
    from thalkak import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--decoy_dir",
        type=str,
        required=True,
        help="Path to directory containing decoy PDB files",
    )
    parser.add_argument(
        "--relax",
        type=str,
        required=True,
        choices=["none", "openmm"],
        help="Relaxation method to use",
    )
    parser.add_argument(
        "--relax_config",
        type=str,
        default=None,
        help="Relax config yaml (default: examples/{relax}.yaml)",
    )
    parser.add_argument(
        "--data_yaml",
        type=str,
        default=None,
        help="Path to the same data config yaml used for prediction; its "
        "'ligand' list is read and mapped onto the structure (openmm only). "
        "Robust for placeholder-resname ligands.",
    )
    parser.add_argument(
        "--ligand_ccd",
        type=str,
        nargs="+",
        default=None,
        metavar="CCD",
        help="Explicit wwPDB CCD code(s) for ligand(s) (openmm only)",
    )
    parser.add_argument(
        "--ligand_smiles",
        type=str,
        nargs="+",
        default=None,
        metavar="SMILES",
        help="Explicit SMILES for ligand(s) (openmm only)",
    )
    args = parser.parse_args()
    relaxation(
        args.decoy_dir,
        args.relax,
        args.relax_config,
        ligand_specs=build_ligand_specs(
            args.data_yaml, args.ligand_ccd, args.ligand_smiles
        ),
    )
