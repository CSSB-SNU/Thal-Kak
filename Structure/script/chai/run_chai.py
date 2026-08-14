import os, sys, glob

# Hide chai's per-recycle / per-diffusion-step tqdm bars (one log line per
# update). Must precede the tqdm-using chai_lab import below.
os.environ.setdefault("TQDM_DISABLE", "1")

import json, argparse, shutil, string
from pathlib import Path
import pandas as pd
import subprocess
import requests
from functools import lru_cache
from collections import defaultdict
import gc
import numpy as np
from Bio.PDB import MMCIFParser, PDBIO, Select, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException
# from chai_lab.chai1 import run_inference
from chai_lab.chai1 import (
    make_all_atom_feature_context,
    run_folding_on_context,
    StructureCandidates,
)
import matplotlib.pyplot as plt
import torch
from Bio.SeqUtils import seq1

COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common")
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)
from process_template import generate_m8_from_hhsearch
from chain_utils import assign_chain_indices
from chain_utils import PDB_CHAIN_CHARS, CIF_CHAIN_CHARS

@lru_cache(maxsize=None)
def ccd_to_smiles(ccd_id):
    """
    Convert a PDB Chemical Component Dictionary (CCD) id (e.g. "ATP", "HEM")
    to a SMILES string using the RCSB data API.

    The stereochemistry-aware SMILES is preferred when available, otherwise the
    plain SMILES is returned. Results are cached so each id is fetched only once.
    ccd_id: CCD identifier (case-insensitive)
    """
    comp_id = ccd_id.strip().upper()
    url = f"https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status == 404:
            raise ValueError(
                f"CCD id '{comp_id}' was not found in the RCSB chemical component "
                f"dictionary. Check the id or provide a 'smiles' string instead."
            ) from e
        raise
    except requests.RequestException as e:
        raise RuntimeError(
            f"Failed to fetch SMILES for CCD id '{comp_id}' from RCSB ({url}). "
            f"Check the network connection or provide a 'smiles' string instead. "
            f"Original error: {e}"
        ) from e

    descriptor = response.json().get("rcsb_chem_comp_descriptor", {})
    smiles = descriptor.get("SMILES_stereo") or descriptor.get("SMILES")
    if not smiles:
        raise ValueError(
            f"No SMILES descriptor available for CCD id '{comp_id}'. "
            f"Please provide a 'smiles' string instead."
        )
    return smiles

def resolve_ligand_smiles(ligand):
    """
    Resolve a ligand entry from the data json to a SMILES string.

    A ligand may be specified either by:
      - "smiles": a SMILES string (used as-is), or
      - "ccd":    a PDB Chemical Component Dictionary id (e.g. "ATP"), which is
                  converted to SMILES via ccd_to_smiles().
    If both are provided, "smiles" takes precedence.
    ligand: a single ligand dict from the data json "ligand" list
    """
    smiles = ligand.get("smiles")
    if smiles:
        return smiles

    ccd = ligand.get("ccd")
    if ccd:
        smiles = ccd_to_smiles(ccd)
        print(f"[Process] Resolved CCD id '{ccd}' to SMILES: {smiles}")
        return smiles

    raise ValueError("Each ligand must define either a 'smiles' or a 'ccd' field")

def write_query_fasta(target, a3m_list, ligand_list, query_fp):
    """
    extract query sequence from unpaired MSAs
    target: name of the target
    a3m_list: a3m part in data.json
    query_fp: path to output query fasta file
    """
    query = []
    num_map = defaultdict(list)

    # Pre-load entity sequences
    seqs = []
    for a3m in a3m_list:
        if a3m["type"] not in ["protein", "rna", "dna"]:
            raise ValueError("Entity must be one of protein, rna, or dna")
        with open(a3m["unpaired_path"], "r") as f:
            seqs.append(f.readlines()[1])

    # Chain id assignment: protein round-robin first, NA next, ligands last
    # (matches the other runners and the protein-only chain_query written at
    # MSA time). chai assigns output PDB chains in fasta order, so emit in
    # chain-index order.
    a3m_copies = [a3m["copy"] for a3m in a3m_list]
    a3m_types = [0 if a3m["type"] == "protein" else 1 for a3m in a3m_list]
    chains_per_entity = assign_chain_indices(a3m_copies, a3m_types)
    chain_to_entity = {ci: j for j, idxs in enumerate(chains_per_entity) for ci in idxs}
    n_polymer = sum(a3m_copies)

    for ci in range(n_polymer):
        j = chain_to_entity[ci]
        chain_letter = CIF_CHAIN_CHARS[ci]
        query.append(f">{a3m_list[j]['type']}|name={target}_{chain_letter}\n")
        query.append(seqs[j])
        num_map[f"{101 + j}"].append(f"{target}_{chain_letter}")

    k = n_polymer
    for ligand in ligand_list:
        smiles = resolve_ligand_smiles(ligand)
        for _ in range(ligand["copy"]):
            chain_letter = CIF_CHAIN_CHARS[k]
            query.append(f">ligand|name={target}_{chain_letter}\n")
            query.append(smiles + "\n")
            k += 1
    query[-1] = query[-1].rstrip("\n")

    with open(query_fp, "w") as f:
        f.writelines(query)

    print("[Process] Created query fasta at", query_fp)
    return num_map

def convert_a3m_to_pqt(target, a3m_list, output_dir):
    """
    convert unpaired a3m to pqt format using chai-lab a3m-to-pqt
    target: name of the target
    a3m_list: a3m part in data.json
    output_dir: output directory path
    """
    msa_parent_dir = output_dir / "msas"

    for i, a3m in enumerate(a3m_list):
        unpaired = a3m["unpaired_path"]
        entity_type = a3m["type"]

        if entity_type == "protein":
            chain_id = CIF_CHAIN_CHARS[i]
            copy_num = a3m["copy"]
            chain_msa_dir =  msa_parent_dir / f"{target}_{chain_id}"
            chain_msa_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy(unpaired, chain_msa_dir / "uniref90.a3m")
            command = ["chai-lab", "a3m-to-pqt", "--output-directory", chain_msa_dir, chain_msa_dir]

            subprocess.run(command, check=True)
            pqt_fp = glob.glob(str(chain_msa_dir / "*.pqt"))[0]
            pqt_fn = os.path.basename(pqt_fp)
            print("[Process] Converted a3m to pqt for <", unpaired, "> to <", pqt_fp, ">")

            shutil.move(pqt_fp, output_dir / "msas" / pqt_fn)
            shutil.rmtree(chain_msa_dir)

    return msa_parent_dir

def extract_ca_data_from_cif(cif_path):
    """
    Parse the loop_ section of a CIF file (field-order independent) and
    extract per-CA pLDDT (B_iso_or_equiv) and chain ids.
    """
    if not os.path.exists(cif_path):
        return None, []

    # Map each _atom_site column tag to its index.
    tag_to_idx = {}
    current_idx = 0
    in_atom_site_loop = False

    plddts, chains = [], []

    with open(cif_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line: continue

        # Collect the _atom_site column definitions.
        if line.startswith("_atom_site."):
            in_atom_site_loop = True
            tag_name = line.split()[0]
            tag_to_idx[tag_name] = current_idx
            current_idx += 1
            continue

        # Data rows (ATOM / HETATM).
        if in_atom_site_loop and (line.startswith("ATOM") or line.startswith("HETATM")):
            parts = line.split()

            try:
                atom_id_idx = tag_to_idx["_atom_site.label_atom_id"]
                asym_id_idx = tag_to_idx["_atom_site.label_asym_id"]
                bfactor_idx = tag_to_idx["_atom_site.B_iso_or_equiv"]

                atom_id = parts[atom_id_idx]

                # Keep only CA (protein) / C4' (nucleic) representative atoms.
                if atom_id == "\"C4'\"" or atom_id == "CA":
                    plddts.append(float(parts[bfactor_idx]))
                    chains.append(parts[asym_id_idx])

            except (KeyError, IndexError):
                # Missing field or short data row -- skip.
                continue

        # A new loop_ ends the atom_site block.
        elif in_atom_site_loop and line.startswith("loop_"):
            in_atom_site_loop = False

    if not plddts:
        return None, []

    # Chain-break positions (where the chain id changes).
    breaks = [i + 1 for i in range(len(chains) - 1) if chains[i] != chains[i + 1]]
    
    return np.array(plddts), breaks

def plot_pae_matrix(ax, pae_data, breaks, title):
    pae = np.array(pae_data)
    num_res = pae.shape[0]
    im = ax.imshow(pae, cmap="bwr", vmin=0, vmax=30, extent=(0, num_res, num_res, 0))
    ax.set_xlim(0, num_res)
    ax.set_ylim(num_res, 0)
    ax.set_aspect("equal")
    # Draw chain-break lines at residue boundaries.
    for b in breaks:
        ax.axvline(x=b, color="black", linewidth=0.8)
        ax.axhline(y=b, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=9)
    return im

def output_adapter(target, option, empty_output_dir, result_data, result_file):
    '''
    process output files
    target: target name
    output_dir: path to root output directory
    '''
    class AcceptAll(Select):
        def accept_model(self, model):
            return True
        def accept_chain(self, chain):
            return True
        def accept_residue(self, residue):
            return True
        def accept_atom(self, atom):
            return True
    
    output_dir = empty_output_dir.parent

    # seed
    seed = os.path.basename(empty_output_dir).split('_seed')[-1]

    # model
    cif_files = sorted(
        glob.glob(f"{empty_output_dir}/*.cif"),
        key=lambda fp: int(os.path.basename(fp)[:-4].split('_')[-1]),
    )
    parser = MMCIFParser()
    io = PDBIO()
    success_count = 0
    mean_plddts = []
    for i, fp in enumerate(cif_files):
        filename = f"seed{seed}_{os.path.basename(fp)}"
        sample = os.path.basename(fp)[:-4].split('_')[-1]
        pdb_filename = f'{target}_seed_{seed}_sample_{sample}.pdb'
        output_path = f"{output_dir}/common/{pdb_filename}"
        ca_plddts, _ = extract_ca_data_from_cif(fp)
        mean_ca_plddt = np.mean(ca_plddts) if ca_plddts is not None else 0
        mean_plddts.append(mean_ca_plddt)

        try:
            structure = parser.get_structure(pdb_filename[:-4], fp)
            model = structure[0]
            # Relabel chains to single-char ids in order (A-Z, a-z, 0-9) so
            # they fit PDB's single chain column; BioPython's PDBIO raises on
            # multi-char ids. A model with more chains than the 62 labels can
            # hold cannot be written as PDB -- keep the cif instead.
            chains = list(model)
            if len(chains) > len(PDB_CHAIN_CHARS):
                fallback_cif = f"{output_dir}/common/{pdb_filename[:-4]}.cif"
                shutil.copyfile(fp, fallback_cif)
                print(
                    f"[Warning] {pdb_filename[:-4]}: model has more than "
                    f"{len(PDB_CHAIN_CHARS)} chains; cif->pdb conversion not "
                    f"possible. Copied cif to {fallback_cif} instead."
                )
                continue
            for new_chain_id, chain in zip(PDB_CHAIN_CHARS, chains):
                chain.id = new_chain_id
            # Sanitize ligand names so they fit PDB's fixed columns
            # (residue name: 3 chars, atom name: 4 chars)
            for chain in model:
                for residue in chain:
                    if not residue.id[0].startswith("H_"):
                        continue
                    if len(residue.resname) > 3:
                        # e.g. LIG2 -> LG2: keep entity index if present
                        shortened = residue.resname.replace("LIG", "LG", 1)[:3]
                        residue.resname = shortened
                    for atom in list(residue):
                        if len(atom.name) <= 4:
                            continue
                        # e.g. C10_1 -> C10, CL1_1 -> CL1
                        new_name = atom.name.split("_", 1)[0][:4]
                        residue.detach_child(atom.id)
                        atom.name = new_name
                        atom.id = new_name
                        atom.fullname = new_name.ljust(4)
                        residue.add(atom)
            io.set_structure(model)
            io.save(output_path, select=AcceptAll())
            
            print(f"[Process] cif conversion to pdb: {filename} -> {pdb_filename}")
            success_count += 1
            
        except PDBConstructionException as e:
            print(f"[Warning] Could not parse {filename}. Skipping. Error: {e}")
        except Exception as e:
            print(f"Error processing {filename}. Skipping. Error: {e}")

    print(f"\n--- Conversion Complete ---")
    print(f"Total files processed: {len(cif_files)}")
    print(f"Successfully converted: {success_count}")

    # scores
    for i, res in enumerate(result_data.ranking_data):
        ptm = res.ptm_scores.complex_ptm[0]
        iptm = res.ptm_scores.interface_ptm[0]
        plddt = mean_plddts[i]
        chai_aggregate_score = res.aggregate_score[0]

        with open(result_file, "a") as f:
            f.write(f"{target},{option},chai-1,seed_{seed}_sample_{i},{plddt:.3f},{ptm:.3f},{iptm:.3f},{chai_aggregate_score:.3f}\n")
    pae = result_data.pae
    for i in range(pae.shape[0]):
        pae_filename = f'pae_{target}_seed_{seed}_sample_{i}.pt'
        output_path = f"{output_dir}/common/{pae_filename}"
        torch.save(pae[i], output_path)

    df = pd.read_csv(result_file)
    df = df.sort_values(by="ranking_score", ascending=False).reset_index(drop=True)
    df.to_csv(result_file, index=False)

def draw_plots(result_file, output_dir):
    df = pd.read_csv(result_file)
    top_five = df.sort_values(by=df.columns[-1], ascending=False).head(5)

    # plots
    all_plddts_data = []
    fig_pae, axes_pae = plt.subplots(1, 5, figsize=(5 * 4, 4))
    for i, (_, row) in enumerate(top_five.iterrows()):
        target = row['target']
        parts = row['seed-sample'].split('_')
        seed, sample = parts[1], parts[3]
        pae_fp = output_dir / "common" / f"pae_{target}_seed_{seed}_sample_{sample}.pt"
        cif_file = output_dir / f"output_seed{seed}" / f"pred.model_idx_{sample}.cif"
        ca_plddts, breaks = extract_ca_data_from_cif(cif_file)
        # A sample whose pae tensor or cif is missing (e.g. its seed failed
        # mid-adapter) leaves its panel blank rather than killing the figure.
        if not pae_fp.is_file() or ca_plddts is None:
            print(
                f"[Warning] Missing plot inputs for seed_{seed}_sample_{sample}; "
                f"skipping panel."
            )
            continue
        pae = torch.load(pae_fp)
        rank_label = f"rank_{i + 1:03d}"
        all_plddts_data.append((rank_label, ca_plddts, breaks))
        plot_pae_matrix(
                        axes_pae[i],
                        pae,
                        breaks,
                        f"rank_{i}\nseed_{seed}_sample_{sample}",
                    )
    plt.tight_layout()
    plt.savefig(output_dir / "common" / f"pae.png", dpi=300)
    plt.close(fig_pae)

    fig_plddt = plt.figure(figsize=(12, 5))
    for label, plddts, breaks in all_plddts_data:
        plt.plot(plddts, label=label, alpha=0.8)
        if label == "rank_001":
            for b in breaks:
                plt.axvline(
                    x=b, color="black", linestyle="--", linewidth=0.7, alpha=0.4
                )

    plt.title(f"pLDDT")
    plt.xlabel("Residue Index")
    plt.ylabel("pLDDT")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.2)
    plt.savefig(output_dir / "common" / f"plddt.png", dpi=300)
    plt.close(fig_plddt)

def main(data_json, chai_json):
    with open(data_json, "r") as file:
        data_config = json.load(file)
    with open(chai_json, "r") as file:
        chai_config = json.load(file)
    
    # data_config
    if "job_name" not in data_config.keys() or data_config["job_name"] is None:
        raise ValueError("Please provide job_name in data json")
    if "output_dir" not in data_config.keys() or data_config["output_dir"] is None:
        raise ValueError("Please provide output_dir in data json")
    if "seed" not in data_config.keys() or data_config["seed"] is None:
        raise ValueError("Please provide seed in data json")
    target = os.path.basename(data_json)[:-5]
    output_parent_dir = Path(data_config["output_dir"])
    seed = data_config["seed"]
    job_name = data_config["job_name"]

    # chai_config
    num_trunk_samples = chai_config.get("num-trunk-samples", 1)
    num_diffn_samples = chai_config.get("num-diffn-samples", 5)
    num_diffn_timesteps = chai_config.get("num-diffn-timesteps", 200)
    recycle_msa_subsample = chai_config.get("recycle-msa-subsample", 0)
    num_trunk_recycles = chai_config.get("num-trunk-recycles", 3)
    constraint_path = chai_config.get("constraint-path", None)
    use_esm_embeddings = chai_config.get("use-esm-embeddings", True)
    fasta_names_as_cif_chains = chai_config.get("fasta-names-as-cif-chains", False)
    
    print("[Process] Starting Chai-Lab inference for target:", target)

    # create output directory
    output_dir = output_parent_dir / f"chai-1_results_{target}_{job_name}"
    if output_dir.exists():
        from datetime import datetime
        output_dir = output_parent_dir / f"chai-1_results_{target}_{job_name}_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
    output_dir.mkdir(parents=True)

    # create common result directory
    common_dir = output_dir / "common"
    common_dir.mkdir(exist_ok=True)
    result_file = output_dir / "common" / f"{target}_results_summary.csv"
    with open(result_file, "w") as f:
        f.write("target,option,model,seed-sample,mean_plddt,ptm,iptm,ranking_score\n")

    # create msas directory
    output_msa_dir = output_dir / "msas"
    output_msa_dir.mkdir(exist_ok=True)

    # generate query fasta
    a3m_list = data_config["a3m"]
    ligand_list = data_config.get("ligand", [])
    query_fp = output_dir / "query.fasta"
    num_map = write_query_fasta(target, a3m_list, ligand_list, query_fp)

    # convert a3m to pqt
    msa_parent_dir = convert_a3m_to_pqt(target, a3m_list, output_dir)

    # leave hits if template_hits_m8 is provided
    if "templates" in data_config.keys():
        hits_fp = generate_m8_from_hhsearch(num_map, a3m_list, data_config["templates"], output_dir)
    else:
        hits_fp = Path()
        print("[Warning] No custom templates are detected")
    
    # run inference
    if msa_parent_dir.is_dir():
        msas_dir = msa_parent_dir
    else:
        msas_dir = None

    if hits_fp.is_file():
        use_templates_server = False
        template_hits_m8_fp = hits_fp
    else:
        template_hits_m8_fp = None

    # ---- featurize ONCE, then reuse it for every seed -----------------------
    # make_all_atom_feature_context() does all the seed-INDEPENDENT CPU work:
    # parse inputs, load MSAs/templates, and -- the slow part for large branched
    # glycan ligands -- RDKit ligand-conformer generation. The original run_chai.py
    # paid this inside run_inference() on EVERY seed; here we build it a single
    # time and reuse the same context across all seeds (chai already reuses one
    # context across its own num_trunk_samples loop, so this is safe).
    # NOTE esm_device must stay on the GPU to match run_inference()'s behaviour
    # (it passes esm_device=cuda:0); defaulting to CPU would silently slow ESM.
    torch_device = torch.device("cuda:0")
    feature_context = make_all_atom_feature_context(
        fasta_file=query_fp,
        output_dir=output_dir,
        entity_name_as_subchain=fasta_names_as_cif_chains,
        use_esm_embeddings=use_esm_embeddings,
        use_msa_server=False,
        msa_directory=msas_dir,
        constraint_path=constraint_path,
        use_templates_server=False,
        templates_path=template_hits_m8_fp,
        esm_device=torch_device,
    )

    for i, s in enumerate(seed):
        try: 
            print(f"[Process] Prediction info\nOutput dir {output_dir}\nMSA dir {msas_dir}\nTotal Samples {num_diffn_samples * num_trunk_samples}\n(Number of trunk samples {num_trunk_samples}; Number of diffn samples {num_diffn_samples})\nNumber of diffn time steps {num_diffn_timesteps}\nNumber of recycle msa subsample {recycle_msa_subsample}\nNumber of trunk recycles {num_trunk_recycles}\nTemplate hits {template_hits_m8_fp}\nSeed {s}\nConstraint path {constraint_path}\nUse ESM embeddings {use_esm_embeddings}\nFasta names as cif chains {fasta_names_as_cif_chains}")
            empty_output_dir = output_dir / f"output_seed{s}"            
            # Only the GPU folding varies per seed. Mirror run_inference()'s
            # num_trunk_samples loop, but on the pre-built feature_context.
            all_candidates = []
            for trunk_idx in range(num_trunk_samples):
                cand = run_folding_on_context(
                    feature_context,
                    output_dir=(
                        empty_output_dir / f"trunk_{trunk_idx}"
                        if num_trunk_samples > 1
                        else empty_output_dir
                    ),
                    num_trunk_recycles=num_trunk_recycles,
                    num_diffn_timesteps=num_diffn_timesteps,
                    num_diffn_samples=num_diffn_samples,
                    recycle_msa_subsample=recycle_msa_subsample,
                    seed=int(s) + trunk_idx,
                    device=torch_device,
                    low_memory=True,
                    entity_names_as_chain_names_in_output_cif=fasta_names_as_cif_chains,
                )
                all_candidates.append(cand)
            prediction_result = StructureCandidates.concat(all_candidates)
            output_adapter(target, job_name, empty_output_dir, prediction_result, result_file)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
            print("Skipping the current prediction...")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")
        gc.collect()
    # Plots are cosmetic: never let them cost the caller the result directory.
    try:
        draw_plots(result_file, output_dir)
    except Exception as e:
        print(f"[Warning] Plot generation failed: {e}")

    print(f"RESULT_DIR:{output_dir}")

    return output_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_json", type=str, required=True)
    parser.add_argument("--chai_json", type=str, required=True)
    args = parser.parse_args()

    main(args.data_json, args.chai_json)
