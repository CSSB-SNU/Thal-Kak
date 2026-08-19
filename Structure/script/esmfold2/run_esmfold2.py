"""ESMFold2 inference runner for the Thal-Kak pipeline.

Reads a Thal-Kak data yaml (the one produced by MSA generation) plus an
ESMFold2 config yaml, builds an ``esm`` ``StructurePredictionInput`` from the
per-chain a3m files (paired + unpaired merged with ``key=`` taxonomy markers
for cross-chain pairing), then loops over ``data_yaml['seed']`` and writes
each diffusion sample as ``{target}_seed_{N}_sample_{M}.pdb`` into the
result directory. A ``{target}_results_summary.csv`` mirroring the schema of
the other model runners is also written, and a ``RESULT_DIR:<path>`` line is
emitted on stdout so the orchestrator can pick it up.
"""

import argparse
import csv
import os
import shutil
import string
import sys
from datetime import datetime
from pathlib import Path

import gemmi
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

# Silence rdkit's pickle-version warning spammed while loading ccd.pkl
# (built with a newer rdkit than the pinned 2024.9.6).
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

from esm.models.esmfold2.processor import ESMFold2InputBuilder
from esm.utils.msa.msa import MSA
from esm.utils.parsing import FastaEntry, read_sequences
from esm.utils.structure.input_builder import (
    DNAInput,
    LigandInput,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
)
from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:  # sibling imports when run as a script
    sys.path.insert(0, SCRIPT_DIR)
COMMON_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "common")
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)
from chain_utils import assign_chain_indices
from chain_utils import PDB_CHAIN_CHARS, CIF_CHAIN_CHARS
from thalkak import get_logger

log = get_logger("structure")


def _enable_cueq_ops() -> None:
    """Preload cuequivariance's ``libcue_ops.so`` (``RTLD_GLOBAL``) so the cueq
    kernel backend is loadable without ``LD_LIBRARY_PATH``. Must run after torch
    is imported (torch pulls the CUDA libs the .so links). No-op if
    cuequivariance isn't installed / CPU-only -- backend selection then falls
    back to pure PyTorch.
    """
    try:
        import ctypes

        import cuequivariance_ops

        lib = os.path.join(
            os.path.dirname(cuequivariance_ops.__file__), "lib", "libcue_ops.so"
        )
        if os.path.exists(lib):
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
    except Exception:
        pass


def _set_msa_trimul_backend(model, backend: str) -> int:
    """Route the MSA encoder's trimul blocks through ``backend``.

    ``model.set_kernel_backend()`` does not reach ``msa_encoder``, so flip the
    backend on each block's ``TriangleMultiplicativeUpdate`` directly. Returns
    the number of trimul modules switched.
    """
    msa = getattr(model, "msa_encoder", None)
    n = 0
    if msa is not None and hasattr(msa, "blocks"):
        for blk in msa.blocks:
            for attr in ("tri_mul_out", "tri_mul_in"):
                t = getattr(blk, attr, None)
                if t is not None and hasattr(t, "set_kernel_backend"):
                    t.set_kernel_backend(backend)
                    n += 1
    return n


class _LMOffloader:
    """Parks ``model._esmc`` (the ESM-C backbone) on CPU during the
    trunk/diffusion phase, moving it to the GPU only for its one forward per
    fold.

    The backbone runs once at the start of ``forward()`` and is never touched by
    the folding trunk or diffusion sampler, yet it otherwise stays resident on
    the GPU for the whole folding phase. Forward hooks move it in just before
    the LM forward and evict it (+``empty_cache``) right after -- non-invasive,
    no edits to the model. Trade-off: a CPU<->GPU transfer per fold, so enable
    only when GPU memory is the constraint. Numerically identical -- same
    compute, the module is only relocated.
    """

    def __init__(self, model, device: "torch.device") -> None:
        self.esmc = getattr(model, "_esmc", None)
        self.device = device
        self._handles: list = []

    @property
    def active(self) -> bool:
        return self.esmc is not None and self.device.type == "cuda"

    def install(self) -> "_LMOffloader":
        if not self.active:
            return self
        # Park immediately so the first trunk pass already runs lean.
        self.esmc.to("cpu")
        torch.cuda.empty_cache()

        # Hooks must return None -- a non-None pre-hook return replaces forward's
        # args (``.to()`` returns the module, which would clobber input_ids).
        def _to_gpu(_m, _a) -> None:
            self.esmc.to(self.device)

        def _evict(_m, _a, _o) -> None:
            self.esmc.to("cpu")
            torch.cuda.empty_cache()

        self._handles.append(self.esmc.register_forward_pre_hook(_to_gpu))
        self._handles.append(self.esmc.register_forward_hook(_evict))
        return self

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        if self.esmc is not None and self.device.type == "cuda":
            self.esmc.to(self.device)  # restore so the model is left as found


def _set_msa_opm_chunk(model, chunk: int) -> int:
    """Enable chunking on the MSA encoder's OuterProductMean. Returns #set.

    OPM forms a [B, L, L, d_hidden, d_hidden] outer product -- O(L^2 * d_hidden^2),
    the model's largest transient. It has a chunked path that tiles along the
    i-axis, but ``model.set_chunk_size()`` never reaches ``msa_encoder``, so OPM
    runs unchunked by default. Set it per block, as with the trimul backends.
    """
    msa = getattr(model, "msa_encoder", None)
    n = 0
    if msa is not None and hasattr(msa, "blocks"):
        for blk in msa.blocks:
            opm = getattr(blk, "outer_product_mean", None)
            if opm is not None and hasattr(opm, "set_chunk_size"):
                opm.set_chunk_size(chunk)
                n += 1
    return n


def _configure_acceleration(model, esm_cfg: dict):
    """Apply optional inference accelerators from the config.

    Each lever is read from ``esm_cfg`` (absent -> off):

    - ``tf32``: allow TF32 on the fp32 matmul path (faster, lower precision;
      changes numerics). Matches the ESMFold2 paper's inference config.
    - ``kernel_backend``: ``"cuequivariance"`` | ``"fused"`` | null -- kernel
      path for the trunk/diffusion/confidence blocks (changes numerics).
    - ``cueq_msa``: also route the MSA encoder trimul through cueq (requires
      ``kernel_backend: cuequivariance``).
    - ``chunk_size``: cap L^2 transients (triangle / OPM / attention) by tiling
      -- the memory lever for large complexes. int | null. Note this does not
      reach the MSA encoder; see ``msa_opm_chunk``.
    - ``msa_opm_chunk``: tile the MSA encoder's OuterProductMean, which
      ``chunk_size`` leaves unchunked. int | null.
    - ``offload_lm``: park the ESM-C backbone on CPU during folding;
      numerically identical.
    - ``confidence_chunk``: run the confidence head over this many diffusion
      samples at a time instead of all at once -- the head, not the trunk, sets
      the memory peak. Mathematically equivalent, but the smaller batch reorders
      float reductions, so scores move within noise. int | null.

    Returns ``(label, keepalive)``; keep ``keepalive`` alive for the whole
    folding loop (the offloader's forward hooks stay installed until GC /
    ``remove()``, and the confidence patch until its revert thunk is dropped).
    """
    tf32 = bool(esm_cfg.get("tf32", False))
    kernel_backend = esm_cfg.get("kernel_backend", None)
    cueq_msa = bool(esm_cfg.get("cueq_msa", False))
    chunk_size = esm_cfg.get("chunk_size", None)
    offload_lm = bool(esm_cfg.get("offload_lm", False))
    confidence_chunk = esm_cfg.get("confidence_chunk", None)
    msa_opm_chunk = esm_cfg.get("msa_opm_chunk", None)

    # Kernel backends (cuequivariance / fused) use Triton kernels that need
    # native bf16 -- compute capability >= 8.0 (Ampere+). On older GPUs (e.g.
    # V100 cc 7.0, Turing 7.5) `fused` crashes and `cuequivariance` falls back
    # per operation to a slow chunked path that also floods the log with MLIR
    # dumps; the pure-PyTorch path gives the same result (rankings identical,
    # confidence within ~0.004). Demote when the device can't run the kernels.
    if kernel_backend and (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] < 8
    ):
        cap = (
            ".".join(map(str, torch.cuda.get_device_capability()))
            if torch.cuda.is_available()
            else "no CUDA device"
        )
        print(
            f"[esmfold2] note: kernel_backend={kernel_backend!r} needs compute "
            f"capability >= 8.0 (bf16); this GPU is {cap} -- using the "
            f"pure-PyTorch path instead.",
            flush=True,
        )
        kernel_backend = None
        cueq_msa = False

    parts = []
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        parts.append("tf32")

    if kernel_backend:
        if kernel_backend == "cuequivariance":
            _enable_cueq_ops()
        model.set_kernel_backend(kernel_backend)
        parts.append(f"backend={kernel_backend}")
        if cueq_msa and kernel_backend == "cuequivariance":
            n = _set_msa_trimul_backend(model, kernel_backend)
            parts.append(f"cueq-msa x{n}")

    if chunk_size is not None:
        model.set_chunk_size(int(chunk_size))
        parts.append(f"chunk_size={int(chunk_size)}")

    if msa_opm_chunk is not None:
        n = _set_msa_opm_chunk(model, int(msa_opm_chunk))
        if n:
            parts.append(f"msa_opm_chunk={int(msa_opm_chunk)} x{n}")

    offloader = None
    if offload_lm:
        device = next(model.parameters()).device
        offloader = _LMOffloader(model, device).install()
        if offloader.active:
            parts.append("offload_lm")

    revert_confidence = None
    if confidence_chunk is not None and int(confidence_chunk) >= 1:
        from chunk_confidence import install_chunked_confidence

        revert_confidence = install_chunked_confidence(model, int(confidence_chunk))
        parts.append(f"confidence_chunk={int(confidence_chunk)}")

    return (
        ", ".join(parts) if parts else "reference (pure-pytorch)",
        (offloader, revert_confidence),
    )



def _cif_to_pdb(cif_path, pdb_path):
    """Convert an mmCIF model to PDB, relabeling chains to single-character ids
    in ascending chain-index order: A-Z, then a-z, then 0-9 ("figures").

    ESMFold2 writes chains grouped by input entity, so the cif's physical order
    ('A', 'C', 'B', 'D', ...) does not match its chain ids. Chains are sorted by
    chain index before relabeling -- relabeling by position alone would rename
    'C' to 'B' and collapse the interleaved assignment from
    assign_chain_indices() into per-entity blocks. Returns ``pdb_path`` on
    success, or ``None`` when the model has more chains than the 62 single-char
    labels can hold: PDB format cannot represent multi-character chain ids, so
    no PDB is written in that case (the caller keeps the cif).
    """
    structure = gemmi.read_structure(cif_path)
    if len(structure) == 0:
        raise ValueError(f"No model found in {cif_path}")

    model = structure[0]
    chains = list(model)
    if len(chains) > len(PDB_CHAIN_CHARS):
        return None
    chains.sort(key=lambda chain: CIF_CHAIN_CHARS.index(chain.name))

    ordered = gemmi.Model(model.num)
    for chain in chains:
        ordered.add_chain(chain)
    for new_name, chain in zip(PDB_CHAIN_CHARS, ordered):
        chain.name = new_name
    structure[0] = ordered

    structure.write_pdb(pdb_path)
    return pdb_path


def _chain_breaks(chain_ids):
    """Token positions where the chain id transitions, matching the convention
    used by the other model wrappers' pAE/pLDDT plots."""
    arr = np.asarray(chain_ids)
    return [i + 1 for i in range(len(arr) - 1) if arr[i] != arr[i + 1]]


def _plot_pae_matrix(ax, pae, breaks, title):

    pae = np.asarray(pae)
    num_res = pae.shape[0]
    im = ax.imshow(pae, cmap="bwr", vmin=0, vmax=30, extent=(0, num_res, num_res, 0))
    ax.set_xlim(0, num_res)
    ax.set_ylim(num_res, 0)
    ax.set_aspect("equal")
    for b in breaks:
        ax.axvline(x=b, color="black", linewidth=1.0)
        ax.axhline(y=b, color="black", linewidth=1.0)
    ax.set_title(title, fontsize=9)
    return im


def _save_confidence_plots(common_dir, target, plot_data):
    """Save PAE matrix grid + pLDDT comparison PNGs for the top-5 samples."""
    if not plot_data:
        return
    plot_data = sorted(plot_data, key=lambda d: d["rank_key"], reverse=True)[:5]

    fig_pae, axes = plt.subplots(1, len(plot_data), figsize=(4 * len(plot_data) + 1, 4))
    if len(plot_data) == 1:
        axes = [axes]
    im = None
    for ax, d in zip(axes, plot_data):
        if d["pae"] is None:
            ax.set_axis_off()
            continue
        im = _plot_pae_matrix(ax, d["pae"], d["breaks"], f"{d['rank_label']}\n{d['stem']}")
    if im is not None:
        fig_pae.subplots_adjust(right=0.85)
        cbar_ax = fig_pae.add_axes([0.88, 0.15, 0.02, 0.7])
        fig_pae.colorbar(im, cax=cbar_ax, label="predicted Expected Error")
        fig_pae.savefig(
            os.path.join(common_dir, f"{target}_pae_rankings.png"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig_pae)

    plt.figure(figsize=(12, 5))
    drew = False
    for d in plot_data:
        if d["plddt"] is None:
            continue
        plt.plot(d["plddt"], label=d["rank_label"], alpha=0.8)
        if d["rank_label"] == "rank_001":
            for b in d["breaks"]:
                plt.axvline(x=b, color="black", linestyle="--", linewidth=0.7, alpha=0.4)
        drew = True
    if drew:
        plt.title(f"{target} pLDDT Comparison")
        plt.xlabel("Residue Index")
        plt.ylabel("pLDDT")
        plt.ylim(0, 100)
        plt.legend()
        plt.grid(True, axis="y", alpha=0.2)
        plt.savefig(
            os.path.join(common_dir, f"{target}_plddt_comparison.png"),
            dpi=300,
            bbox_inches="tight",
        )
    plt.close("all")


def _build_chain_msa(paired_path, unpaired_path) -> MSA:
    """Merge our split paired/unpaired a3m files into one ESMFold2 MSA with
    ``key=`` taxonomy markers. Both source files include the query at row 0,
    so we keep it once and drop the duplicate."""
    paired_rows = list(read_sequences(paired_path)) if paired_path else []
    unpaired_rows = list(read_sequences(unpaired_path)) if unpaired_path else []

    if not paired_rows and not unpaired_rows:
        raise ValueError("Both paired_path and unpaired_path are empty")

    entries: list[FastaEntry] = []
    query_header, query_seq = (paired_rows or unpaired_rows)[0]
    entries.append(FastaEntry(query_header, query_seq))

    # Paired rows past the query: same row index across chains → cross-chain pair
    for i, (header, seq) in enumerate(paired_rows[1:], start=1):
        entries.append(FastaEntry(f"{header} key={i}", seq))

    # Unpaired rows past the query: key=-1 → block-diagonal unpaired region
    for header, seq in unpaired_rows[1:]:
        entries.append(FastaEntry(f"{header} key=-1", seq))

    return MSA(entries)


def _make_polymer_input(entry, chain_ids, use_msa):
    """Build a ProteinInput / RNAInput / DNAInput from one a3m entry."""
    mol_type = entry.get("type", "protein")
    paired = entry.get("paired_path")
    unpaired = entry.get("unpaired_path")

    # Sequence: first row of either a3m (both have query at row 0)
    src = paired or unpaired
    if not src:
        raise ValueError(f"a3m entry has no paired_path or unpaired_path: {entry}")
    _, query_seq = next(iter(read_sequences(src)))

    if mol_type == "protein":
        msa = _build_chain_msa(paired, unpaired) if use_msa else None
        return ProteinInput(id=chain_ids, sequence=query_seq, msa=msa)
    if mol_type == "rna":
        return RNAInput(id=chain_ids, sequence=query_seq)
    if mol_type == "dna":
        return DNAInput(id=chain_ids, sequence=query_seq)
    raise ValueError(f"Unsupported a3m entry type: {mol_type!r}")


def _make_ligand_input(entry, chain_ids):
    if entry.get("smiles"):
        return LigandInput(id=chain_ids, smiles=entry["smiles"])
    if entry.get("ccd"):
        ccd = entry["ccd"]
        return LigandInput(id=chain_ids, ccd=ccd if isinstance(ccd, list) else [ccd])
    raise ValueError(f"Ligand entry has neither smiles nor ccd: {entry}")


def _result_root(output_dir, target, job_name):
    root = os.path.join(output_dir, f"esmfold2_results_{target}_{job_name}")
    if os.path.exists(root):
        root += datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    os.makedirs(os.path.join(root, "common"))
    return root


def main(data_yaml_path, esm_yaml_path):
    with open(data_yaml_path) as f:
        data_cfg = yaml.safe_load(f)
    with open(esm_yaml_path) as f:
        esm_cfg = yaml.safe_load(f)

    target = data_cfg.get("name") or Path(data_yaml_path).stem
    job_name = data_cfg["job_name"]
    output_dir = data_cfg["output_dir"]
    seeds = data_cfg["seed"]
    if not isinstance(seeds, list):
        seeds = [seeds]

    a3m_entries = data_cfg.get("a3m", [])
    ligand_entries = data_cfg.get("ligand", [])

    # ESMFold2 has no template channel; a `templates` block in the shared data
    # yaml (consumed by boltz/chai/af3) is silently ignored here -- flag it.
    if data_cfg.get("templates"):
        print(
            "[esmfold2] note: 'templates' in the data yaml is ignored "
            "(ESMFold2 is MSA-only and has no template input).",
            flush=True,
        )

    model_variant = esm_cfg.get("model_variant", "biohub/ESMFold2")
    num_loops = int(esm_cfg.get("num_loops", 3))
    num_sampling_steps = int(esm_cfg.get("num_sampling_steps", 200))
    num_diffusion_samples = int(esm_cfg.get("num_diffusion_samples", 5))
    # MSA inference-diversity knobs (esm >= #342/#343). msa_max_depth=null
    # disables row subsampling (msa_subsample_at_inference is derived from it);
    # msa_column_mask_rate=0.0 disables column masking. Both off reproduces the
    # pre-diversity behavior.
    msa_max_depth = esm_cfg.get("msa_max_depth", None)
    if msa_max_depth is not None:
        msa_max_depth = int(msa_max_depth)
    msa_column_mask_rate = float(esm_cfg.get("msa_column_mask_rate", 0.0))
    # MSA is only used by the full ESMFold2 variant; the -Fast variant skips it.
    is_fast = model_variant.endswith("-Fast")
    use_msa = esm_cfg.get("use_msa", True) and not is_fast

    # Chain id assignment: protein round-robin first, NA next, ligands last
    # (matches other models; keeps template chain_query (protein-only at MSA
    # time) aligned with the runner's chain pool).
    polymer_copies = [entry.get("copy", 1) for entry in a3m_entries]
    polymer_types = [
        0 if entry.get("type", "protein") == "protein" else 1
        for entry in a3m_entries
    ]
    chains_per_entity = assign_chain_indices(polymer_copies, polymer_types)
    n_polymer = sum(polymer_copies)
    chains_per_ligand = []
    k = n_polymer
    for entry in ligand_entries:
        c = entry.get("copy", 1)
        chains_per_ligand.append(list(range(k, k + c)))
        k += c

    sequences = []
    for idx, entry in enumerate(a3m_entries):
        ids = [CIF_CHAIN_CHARS[i] for i in chains_per_entity[idx]]
        sequences.append(_make_polymer_input(entry, ids, use_msa))
    for j, entry in enumerate(ligand_entries):
        ids = [CIF_CHAIN_CHARS[i] for i in chains_per_ligand[j]]
        sequences.append(_make_ligand_input(entry, ids))

    spi = StructurePredictionInput(sequences=sequences)

    print(f"[esmfold2] loading {model_variant} ...", flush=True)
    model = ESMFold2Model.from_pretrained(model_variant).cuda().eval()
    # Held for the whole folding loop: the offloader's hooks and the
    # confidence-head patch stay in place only while this is alive.
    accel_label, _accel_keepalive = _configure_acceleration(model, esm_cfg)
    print(f"[esmfold2] acceleration: {accel_label}", flush=True)
    builder = ESMFold2InputBuilder()

    result_root = _result_root(output_dir, target, job_name)
    common = os.path.join(result_root, "common")
    cif_dir = os.path.join(result_root, "cif_outputs")
    os.makedirs(cif_dir, exist_ok=True)

    summary_rows = []
    plot_data = []
    for seed in seeds:
        print(f"[esmfold2] seed={seed} folding ...", flush=True)
        results = builder.fold(
            model,
            spi,
            num_loops=num_loops,
            num_sampling_steps=num_sampling_steps,
            num_diffusion_samples=num_diffusion_samples,
            msa_max_depth=msa_max_depth,
            msa_column_mask_rate=msa_column_mask_rate,
            seed=int(seed),
            complex_id=target,
        )
        if not isinstance(results, list):
            results = [results]

        for m, res in enumerate(results):
            stem = f"{target}_seed_{seed}_sample_{m}"
            cif_path = os.path.join(cif_dir, f"{stem}.cif")
            pdb_path = os.path.join(common, f"{stem}.pdb")
            with open(cif_path, "w") as f:
                f.write(res.complex.to_mmcif())
            if _cif_to_pdb(cif_path, pdb_path) is None:
                fallback_cif = os.path.join(common, f"{stem}.cif")
                shutil.copyfile(cif_path, fallback_cif)
                log.warning(
                    f"[esmfold2] {stem}: model has more than "
                    f"{len(PDB_CHAIN_CHARS)} chains; cif->pdb conversion not "
                    f"possible. Copied cif to {fallback_cif} instead."
                )
            # ESMFold2 emits plddt in [0, 1]; rescale to the 0-100 convention
            # the rest of the pipeline (and the comparison plot) expects.
            plddt_arr = (
                res.plddt.detach().cpu().numpy() * 100.0
                if res.plddt is not None
                else None
            )
            pae_arr = res.pae.detach().cpu().numpy() if res.pae is not None else None
            breaks = _chain_breaks(res.complex.chain_id)
            mean_plddt = float(plddt_arr.mean()) if plddt_arr is not None else None
            ptm = float(res.ptm) if res.ptm is not None else None
            iptm = float(res.iptm) if res.iptm is not None else None
            # Ranking score: AF3-style 0.8*iptm + 0.2*ptm when available, else ptm
            ranking = (
                0.8 * iptm + 0.2 * ptm
                if (iptm is not None and ptm is not None)
                else (ptm if ptm is not None else mean_plddt)
            )
            summary_rows.append(
                {
                    "target": target,
                    "option": job_name,
                    "model": "esmfold2",
                    "seed-sample": f"seed_{seed}_sample_{m}",
                    "ranking_score": (
                        round(ranking, 4) if ranking is not None else ""
                    ),
                    "mean_plddt": round(mean_plddt, 3) if mean_plddt is not None else "",
                    "ptm": round(ptm, 3) if ptm is not None else "",
                    "iptm": round(iptm, 3) if iptm is not None else "",
                }
            )
            plot_data.append(
                {
                    "stem": stem,
                    "rank_key": ranking if ranking is not None else float("-inf"),
                    "plddt": plddt_arr,
                    "pae": pae_arr,
                    "breaks": breaks,
                }
            )

    summary_path = os.path.join(common, f"{target}_results_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "target",
                "option",
                "model",
                "seed-sample",
                "ranking_score",
                "mean_plddt",
                "ptm",
                "iptm",
            ],
        )
        writer.writeheader()
        # Sort by ranking_score desc so consumers that read top-N get best first
        summary_rows.sort(
            key=lambda r: float(r["ranking_score"]) if r["ranking_score"] != "" else -1,
            reverse=True,
        )
        writer.writerows(summary_rows)

    plot_data.sort(key=lambda d: d["rank_key"], reverse=True)
    for i, d in enumerate(plot_data, 1):
        d["rank_label"] = f"rank_{i:03d}"
    _save_confidence_plots(common, target, plot_data)

    print(f"RESULT_DIR:{result_root}")
    return result_root


if __name__ == "__main__":
    from thalkak import setup_logging

    setup_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_yaml", type=str, required=True)
    parser.add_argument("--esmfold2_yaml", type=str, required=True)
    args = parser.parse_args()
    main(args.data_yaml, args.esmfold2_yaml)
