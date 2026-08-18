import argparse, yaml, os, glob, sys
from datetime import datetime

from thalkak import get_logger, log_stream, run_logged

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

log = get_logger("structure")

# Protenix ships several model generations; Thal-Kak exposes two of them as
# separate structure methods. protenix_v1 runs a v1-generation checkpoint
# (default protenix_base_default_v1.0.0, 368M base), protenix_v2 the 464M
# scaled-up protenix-v2. Both support MSA, RNA MSA and templates; only
# protenix_v2 accepts Training-Free Guidance. The checkpoint stays a config
# key (`model_name`) so a section can pick another checkpoint of its own
# generation, e.g. protenix_base_20250630_v1.0.0 for protenix_v1.

# The protenix-v2 checkpoint is no longer served by the official endpoint
# (it returns HTTP 403 AccessDenied for everyone). It is fetched from a
# community mirror and verified against this SHA-256 before use: protenix
# loads checkpoints with torch.load(weights_only=False), so an unverified
# file could execute arbitrary code. A digest mismatch aborts the run.
_PROTENIX_V2_MIRROR_URL = (
    "https://huggingface.co/TMF001/pxdesign-weights/resolve/main/checkpoint/protenix-v2.pt"
)
_PROTENIX_V2_SHA256 = (
    "8f931f9774a396b67033d0e58628e1834f4a1448165e04254b40a780b0c0d599"
)


def _sha256_of(path, chunk=1 << 20):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download_protenix_v2(checkpoint_path):
    """Download the protenix-v2 checkpoint from the mirror, verify its
    SHA-256, then move it into place. Raises RuntimeError on mismatch."""
    import urllib.request

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    tmp_path = checkpoint_path + ".download"

    # urlretrieve fires the hook once per block (thousands of times); log at
    # 10% steps so the download leaves a handful of lines, not a flood.
    logged_pct = 0

    def _progress(block_num, block_size, total_size):
        nonlocal logged_pct
        if total_size <= 0:
            return
        pct = min(100.0, block_num * block_size * 100.0 / total_size)
        if pct >= logged_pct + 10:
            logged_pct = int(pct) // 10 * 10
            log.info(f"  downloading protenix-v2.pt ... {logged_pct}%")

    log.info(
        f"protenix-v2 checkpoint not found; downloading from mirror to {checkpoint_path}"
    )
    urllib.request.urlretrieve(_PROTENIX_V2_MIRROR_URL, tmp_path, reporthook=_progress)

    digest = _sha256_of(tmp_path)
    if digest != _PROTENIX_V2_SHA256:
        os.remove(tmp_path)
        raise RuntimeError(
            f"protenix-v2 checkpoint failed SHA-256 verification (got {digest}, "
            f"expected {_PROTENIX_V2_SHA256}); refusing to use it."
        )
    os.replace(tmp_path, checkpoint_path)
    log.info("protenix-v2 checkpoint verified (SHA-256 match).")


def _resolve_model_config(model, model_config):
    """Extract ``model``'s section from a model-keyed config yaml (one section
    per model, keyed by name -- see examples/model_config.yaml) and write it to
    a temp {model}.yaml for the predictor. Raises if there is no section for
    ``model``."""
    if not model_config:
        raise SystemExit("structure prediction requires a model config yaml.")
    with open(model_config) as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict) or model not in cfg:
        raise SystemExit(
            f"{model_config}: no '{model}' section. The model config is keyed by "
            f"model name (see examples/model_config.yaml)."
        )
    import tempfile

    out = os.path.join(tempfile.mkdtemp(prefix="thalkak_mc_"), f"{model}.yaml")
    with open(out, "w") as f:
        yaml.safe_dump(cfg[model], f, sort_keys=False)
    return out


def structure_prediction(args):
    with open(args.data_config) as f:
        data_yaml = yaml.safe_load(f)
    output_dir = data_yaml["output_dir"]
    target_name = os.path.basename(args.data_config).split(".")[0]
    job_name = data_yaml["job_name"]
    # The model config is keyed by model name; hand the predictor this model's
    # section as a flat per-model yaml.
    model_config = _resolve_model_config(args.model, args.model_config)

    match args.model:
        case "boltz2":
            log.info("Running inference with Boltz2...")
            from Structure.script.boltz.run_boltz import main as run_boltz
            from Structure.script.boltz.boltz_confidence import main as boltz_confidence
            with log_stream(log):
                result_root = run_boltz(args.data_config, model_config)
                boltz_confidence(result_root, target_name)

        case "chai1":
            log.info("Running inference with Chai-1...")
            from Structure.script.chai.convert_yaml_to_json import convert_yaml_to_json
            from Structure.script.chai.run_chai import main as run_chai
            os.makedirs("temp/", exist_ok=True)

            data_json_path = f"temp/{target_name}.json"
            convert_yaml_to_json(args.data_config, data_json_path)

            model_name = os.path.basename(model_config).split(".")[0]
            model_json_path = f"temp/{target_name}_{model_name}.json"
            convert_yaml_to_json(model_config, model_json_path)

            with log_stream(log):
                result_root = run_chai(data_json_path, model_json_path)

            # move json files to result directory and clean up temp
            os.rename(data_json_path, f"{result_root}/{target_name}.json")
            os.rename(model_json_path, f"{result_root}/{model_name}.json")
            try:
                os.rmdir("temp/")
            except:
                pass

        case "esmfold2":
            log.info("Running inference with ESMFold2...")
            from Structure.script.esmfold2.run_esmfold2 import main as run_esmfold2
            with log_stream(log):
                result_root = run_esmfold2(args.data_config, model_config)

        case "protenix_v1" | "protenix_v2":
            with open(model_config) as f:
                protenix_yaml = yaml.safe_load(f)
            protenix_model_name = protenix_yaml["model_name"]
            # TFG is wired for v2 only; reject the key in a v1 section before
            # doing any work, rather than silently dropping it.
            if args.model == "protenix_v1" and "use_tfg_guidance" in protenix_yaml:
                raise SystemExit(
                    "protenix_v1 does not take use_tfg_guidance; it is a "
                    "protenix_v2 option. Remove it from the protenix_v1 "
                    "section of the model config."
                )
            log.info(f"Running inference with Protenix ({protenix_model_name})...")
            protenix_root = f"{ROOT}/Structure/submodules/protenix"
            seed = data_yaml["seed"]
            seed = ",".join(map(str, seed if isinstance(seed, list) else [seed]))

            result_root = f"{output_dir}/{args.model}_results_{target_name}_{job_name}"
            if os.path.exists(result_root):
                result_root += datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
            common_dir = f"{result_root}/common"
            os.makedirs(common_dir)

            os.environ["PROTENIX_ROOT_DIR"] = protenix_root
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            os.environ["TQDM_DISABLE"] = "1"
            os.environ["LAYERNORM_TYPE"] = "torch"

            from argparse import Namespace
            from Structure.script.protenix.process_msa_to_json import main as protenix_msa_to_json
            from Structure.script.protenix.protenix_confidence import process_protenix_results

            with log_stream(log):
                protenix_msa_to_json(
                    Namespace(
                        data=args.data_config,
                        protenix=model_config,
                        save_path=result_root,
                        name=target_name,
                    )
                )

            if protenix_root not in sys.path:
                sys.path.insert(0, protenix_root)
            # protenix runs in its own process. It merges a model's overrides
            # into its module-level config dicts in place -- runner/inference.py
            # builds base_configs as a shallow {**configs_base}, so the nested
            # sections are the module's own objects and deep_update writes
            # straight through them. Two generations in one interpreter would
            # therefore build the second model with the first one's
            # architecture: after protenix_v2, msa_module.c_z holds the literal
            # 256 and hidden_scale_up is True (the GlobalConfigValue sentinels
            # that would resolve them back are gone), and the protenix_v1
            # checkpoint does not fit that shape. A fresh interpreter starts
            # from pristine defaults. protenix reads sys.argv[1:], so the
            # argument list below is the same either way.
            inference_argv = [
                sys.executable,
                os.path.join(protenix_root, "runner", "inference.py"),
                "--model_name", protenix_model_name,
                "--seeds", seed,
                "--dump_dir", result_root,
                "--input_json_path", f"{result_root}/input.json",
                "--model.N_cycle", str(protenix_yaml["N_cycle"]),
                "--sample_diffusion.N_sample", str(protenix_yaml["N_sample"]),
                "--sample_diffusion.N_step", str(protenix_yaml["N_step"]),
                "--triangle_attention", "triattention",
                "--triangle_multiplicative", "cuequivariance",
                "--use_rna_msa", "true",
                "--use_template", "true",
            ]

            # protenix loads {load_checkpoint_dir}/{model_name}.pt. Default to
            # the submodule's checkpoint dir (protenix's own default);
            # PROTENIX_CHECKPOINT_DIR (e.g. a persistent cache) overrides it.
            protenix_ckpt_dir = os.environ.get(
                "PROTENIX_CHECKPOINT_DIR"
            ) or os.path.join(protenix_root, "checkpoint")
            inference_argv += ["--load_checkpoint_dir", protenix_ckpt_dir]

            # protenix-v2 weights are no longer downloadable from the official
            # endpoint (403); fetch + verify them from the mirror if absent.
            # Every other checkpoint is still served, so protenix downloads
            # those itself on first run.
            if protenix_model_name == "protenix-v2":
                v2_path = os.path.join(protenix_ckpt_dir, "protenix-v2.pt")
                if not os.path.exists(v2_path):
                    _download_protenix_v2(v2_path)

            min_size_test = protenix_yaml.get("data.msa.min_size.test")
            if min_size_test is not None:
                inference_argv += ["--data.msa.min_size.test", str(min_size_test)]

            if protenix_yaml.get("use_tfg_guidance"):
                # TFG's VinaStericPotential crashes on single-chain inputs:
                # potentials.py:1206 calls a closure with 2 positional args that
                # is defined to take 1. Skip TFG for monomers until upstream fixes.
                total_chains = sum(e.get("copy", 1) for e in data_yaml.get("a3m") or [])
                total_chains += sum(l.get("copy", 1) for l in data_yaml.get("ligand") or [])
                if total_chains == 1:
                    log.info(
                        "Skipping TFG: monomer input triggers protenix VinaSteric bug."
                    )
                else:
                    inference_argv += ["--sample_diffusion.guidance.enable", "true"]

            # protenix imports `configs` / `protenix` / `runner` as top-level
            # packages from its own root, which running the script by path does
            # not put on the path.
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join(
                p for p in (protenix_root, env.get("PYTHONPATH")) if p
            )
            run_logged(inference_argv, log, env=env)

            # Confidence scoring
            protenix_output = f"{result_root}/{target_name}"
            with log_stream(log):
                process_protenix_results(protenix_output, job_name, args.model)

            # copy to common
            for file in glob.glob(f"{protenix_output}/seed_*/predictions/*.pdb"):
                os.system(f"cp {file} {result_root}/common/")
            for file in glob.glob(f"{protenix_output}/*.png"):
                os.system(f"mv {file} {result_root}/common/")
            for file in glob.glob(f"{protenix_output}/*.csv"):
                os.system(f"mv {file} {result_root}/common/")

    # Write method log (inherit from MSA)
    method_log_path = data_yaml.get("method_log")
    if method_log_path and os.path.exists(method_log_path):
        with open(method_log_path) as f:
            method_log = yaml.safe_load(f)
    else:
        method_log = {"msa": None}
    method_log["structure"] = args.model
    with open(os.path.join(result_root, "common", "method_log.yaml"), "w") as f:
        yaml.dump(method_log, f)

    return result_root


if __name__ == "__main__":
    from thalkak import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["boltz2", "chai1", "protenix_v1", "protenix_v2", "esmfold2"],
        help="The model to use for inference.",
    )
    parser.add_argument(
        "--data_config",
        type=str,
        required=True,
        help="Path to the data configuration yaml file.",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to the model configuration yaml file.",
    )

    args = parser.parse_args()
    structure_prediction(args)
