import os, sys
import json
import shutil
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import string
import gemmi

COMMON_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common")
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)
from chain_utils import PDB_CHAIN_CHARS


def load_json(file_path):
    if not os.path.exists(file_path):
        return None
    with open(file_path, "r") as f:
        return json.load(f)
    
def cif_to_pdb_with_seedname(cif_path):
    """
    Convert CIF → PDB and rename as:
    {target}_{seed}_{sample}.pdb
    
    Example:
    H1106_sample_0.cif
    → H1106_seed_101_sample_0.pdb

    Chains are relabeled to single-character ids in the order they appear
    (A-Z, a-z, 0-9); chain order is preserved.
    Models with more than 62 chains cannot be written as PDB, so the cif is
    copied to the unified name instead and None is returned.
    """

    if not os.path.isfile(cif_path):
        raise FileNotFoundError(f"File not found: {cif_path}")

    # split dirname and filename
    dir_path = os.path.dirname(cif_path)
    filename = os.path.basename(cif_path)

    # get seed
    seed_name = os.path.basename(os.path.dirname(dir_path))


    # get base
    base = filename.replace(".cif", "")

    # get target name
    target_name = base.split("_sample")[0]

    # get sample number
    sample_part = base.split(target_name + "_")[-1]

    # make new pdb name
    new_name = f"{target_name}_{seed_name}_{sample_part}.pdb"
    pdb_path = os.path.join(dir_path, new_name)

    # relabel chains to single-char ids (A-Z, a-z, 0-9) in chain order, then
    # save pdb. A PDB has a single chain column, so a model with >62 chains
    # cannot be written -- copy the cif to the unified name instead.
    structure = gemmi.read_structure(str(cif_path))
    if len(structure) == 0:
        raise ValueError(f"No model found in {cif_path}")

    chains = list(structure[0])
    if len(chains) > len(PDB_CHAIN_CHARS):
        fallback_cif = os.path.join(
            dir_path, f"{target_name}_{seed_name}_{sample_part}.cif"
        )
        shutil.copyfile(cif_path, fallback_cif)
        print(
            f"[WARN] {target_name}_{seed_name}_{sample_part}: model has more "
            f"than {len(PDB_CHAIN_CHARS)} chains; cif->pdb conversion not "
            f"possible. Copied cif to {fallback_cif} instead."
        )
        return None

    for new_chain_name, chain in zip(PDB_CHAIN_CHARS, chains):
        chain.name = new_chain_name

    structure.write_pdb(pdb_path)

    return pdb_path

def extract_ca_data_from_cif(cif_path):
    if not os.path.exists(cif_path):
        return None, []

    PROTEIN_RES = {
        "ALA","GLY","VAL","LEU","ILE","SER","THR","CYS",
        "MET","ASP","GLU","ASN","GLN","LYS","ARG",
        "HIS","PHE","TYR","TRP","PRO"
    }
    DNA_RES = {"DA", "DT", "DG", "DC"}
    RNA_RES = {"A", "U", "G", "C"}
    NA_RES = DNA_RES | RNA_RES

    tag_to_idx = {}
    in_atom_site_loop = False
    current_idx = 0

    values = []
    chains = []

    with open(cif_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # loop start
            if line == "loop_":
                tag_to_idx = {}
                current_idx = 0
                in_atom_site_loop = False
                continue

            # atom_site header
            if line.startswith("_atom_site."):
                in_atom_site_loop = True
                tag_to_idx[line.split()[0]] = current_idx
                current_idx += 1
                continue

            # loop end
            if in_atom_site_loop and line == "#":
                in_atom_site_loop = False
                continue

            # atom rows
            if in_atom_site_loop and (line.startswith("ATOM") or line.startswith("HETATM")):
                parts = line.split()
                try:
                    atom = parts[tag_to_idx["_atom_site.label_atom_id"]]
                    resn = parts[tag_to_idx["_atom_site.label_comp_id"]]
                    bfac = float(parts[tag_to_idx["_atom_site.B_iso_or_equiv"]])

                    chain_idx = (
                        tag_to_idx.get("_atom_site.auth_asym_id")
                        or tag_to_idx.get("_atom_site.label_asym_id")
                    )
                    chain = parts[chain_idx]
                    if resn in PROTEIN_RES and atom == "CA":
                        values.append(bfac)
                        chains.append(chain)
                    elif resn in NA_RES and atom == "\"C4'\"":
                        values.append(bfac)
                        chains.append(chain)

                except Exception:
                    continue

    if not values:
        return None, []

    breaks = [
        i + 1
        for i in range(len(chains) - 1)
        if chains[i] != chains[i + 1]
    ]
    return np.array(values), breaks

def plot_pae_matrix(ax, pae_data, breaks, title):
    pae = np.array(pae_data)
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

def process_protenix_results(base_dir, method, model):
    base_path = Path(base_dir)
    job_name = base_path.name  # e.g. H1204_abag

    summary_files = list(base_path.glob("**/predictions/*_summary_confidence_sample_*.json"))
    if not summary_files:
        print(f"Error: No summary files found in {base_dir}")
        return

    data_list = []
    for sf in summary_files:
        sample_num = sf.stem.split("_")[-1]
        seed_val = sf.parents[1].name.replace("seed_", "")
        summary = load_json(sf)
        if not summary: continue

        cif_name = sf.name.replace("_summary_confidence", "").replace(".json", ".cif")
        cif_path = sf.parent / cif_name
        
        data_list.append({
            "seed": seed_val,
            "sample": sample_num,
            "ranking_score": summary.get("ranking_score", 0),
            "ptm": summary.get("ptm", 0),
            "iptm": summary.get("iptm", 0),
            "full_path": sf,
            "cif_path": cif_path,
        })

    df = pd.DataFrame(data_list).sort_values(by="ranking_score", ascending=False).reset_index(drop=True)

    csv_rows = []
    all_plddts_data = []
    num_plots = min(len(df), 5)
    
    fig_pae, axes_pae = plt.subplots(1, num_plots, figsize=(num_plots * 4 + 1, 4))
    if num_plots == 1: axes_pae = [axes_pae]

    for i, row in df.iterrows():
        rank_label = f"rank_{i + 1:03d}"
        sample_id = f"seed_{row['seed']}_sample_{row['sample']}"
        cif_to_pdb_with_seedname(row["cif_path"])
        ca_plddts, breaks = extract_ca_data_from_cif(row["cif_path"])
        mean_plddt = np.mean(ca_plddts) if ca_plddts is not None else 0

        output_str = f"{rank_label}_{model}_{sample_id} pLDDT={mean_plddt:.1f} pTM={row['ptm']:.3f} ipTM={row['iptm']:.3f}"
        print(output_str)

        save_summary={
            "target": job_name,
            "option": method,
            "model": model,
            "seed-sample": sample_id,
            "ranking_score": round(row['ranking_score'], 3), 
            "mean_plddt": round(mean_plddt, 1),
            "ptm": round(row['ptm'], 3),
        }
        if row['iptm'] != None:
            save_summary["iptm"]= round(row['iptm'], 3)
        csv_rows.append(save_summary)

        if i < 5:
            full_conf = load_json(row["full_path"])
            if full_conf and "pae_matrix" in full_conf:
                im = plot_pae_matrix(axes_pae[i], full_conf["pae_matrix"], breaks, f"{rank_label}\n{sample_id}")
            if ca_plddts is not None:
                all_plddts_data.append((rank_label, ca_plddts, breaks))

    # 1. save CSV 
    csv_df = pd.DataFrame(csv_rows)
    csv_out_path = base_path / f"{job_name}_results_summary.csv"
    csv_df.to_csv(csv_out_path, index=False)

    # 2. PAE chart
    if num_plots > 0:
        # Shared colorbar to the right of the last subplot.
        fig_pae.subplots_adjust(right=0.85)
        cbar_ax = fig_pae.add_axes([0.88, 0.15, 0.02, 0.7])
        fig_pae.colorbar(im, cax=cbar_ax, label='predicted Expected Error')
        fig_pae.savefig(base_path / f"{job_name}_pae_rankings.png", dpi=300, bbox_inches='tight')

    # 3. save pLDDT 
    plt.figure(figsize=(12, 5))
    for label, plddts, breaks in all_plddts_data[:5]:
        plt.plot(plddts, label=label, alpha=0.8)
        if label == "rank_001":
            for b in breaks:
                plt.axvline(x=b, color="black", linestyle="--", linewidth=0.7, alpha=0.4)
    plt.title(f"{job_name} pLDDT Comparison")
    plt.xlabel("Residue Index")
    plt.ylabel("pLDDT")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.2)
    plt.savefig(base_path / f"{job_name}_plddt_comparison.png", dpi=300)
    plt.close('all')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protenix Result Visualizer")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to Protenix output directory")
    parser.add_argument("--option", type=str, required=True, help="Method to run Protenix prediction")
    parser.add_argument("--model", type=str, required=True,
                        choices=["protenix_v1", "protenix_v2"],
                        help="Protenix variant recorded in the summary")
    args = parser.parse_args()
    process_protenix_results(args.input, args.option, args.model)
