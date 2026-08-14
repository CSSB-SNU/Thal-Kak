import numpy as np
import pandas as pd
import json, argparse
import matplotlib.pyplot as plt
from pathlib import Path


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


def main(base_dir, target_name):
    if not Path(base_dir).name.startswith("boltz_results_"):
        raise ValueError("Input directory name should start with 'boltz_results_'")

    with open(f"{base_dir}/chain_info.json", "r") as f:
        chain_info = json.load(f)
    breaks = np.cumsum(
        [
            chain_len
            for chain_len, chain_copy in zip(
                chain_info["chain_len"], chain_info["chain_copy"]
            )
            for _ in range(chain_copy)
        ]
    )[:-1]
    metrics_df = pd.read_csv(f"{base_dir}/common/{target_name}_results_summary.csv")

    # Plot PAE matrices for top 5 models
    num_samples = metrics_df.shape[0]
    num_plots = min(num_samples, 5)
    fig_pae, axes_pae = plt.subplots(1, num_plots, figsize=(4 * num_plots + 1, 4))
    if num_plots == 1:
        axes_pae = [axes_pae]

    for i in range(num_plots):
        parts = metrics_df.iloc[i]["seed-sample"].split("_")
        seed, sample = parts[1], parts[3]
        if seed != "None":
            npz_path = f"{base_dir}/predictions/{target_name}_seed_{seed}/pae_{target_name}_model_{sample}.npz"
        else:
            npz_path = f"{base_dir}/predictions/{target_name}/pae_{target_name}_model_{sample}.npz"
        plot_pae_matrix(
            axes_pae[i],
            np.load(npz_path)["pae"],
            breaks,
            f"rank_{i+1:03d}",
        )

    plt.tight_layout()
    plt.savefig(f"{base_dir}/common/{target_name}_pae_rankings.png", dpi=300)

    # Plot pLDDT line for top 5 models
    plt.figure(figsize=(12, 5))
    for i in range(num_plots):
        parts = metrics_df.iloc[i]["seed-sample"].split("_")
        seed, sample = parts[1], parts[3]
        if seed != "None":
            npz_path = f"{base_dir}/predictions/{target_name}_seed_{seed}/plddt_{target_name}_model_{sample}.npz"
        else:
            npz_path = f"{base_dir}/predictions/{target_name}/plddt_{target_name}_model_{sample}.npz"
        plt.plot(np.load(npz_path)["plddt"], label=f"rank_{i+1:03d}", alpha=0.8)
        if i == 0:
            for b in breaks:
                plt.axvline(
                    x=b, color="black", linestyle="--", linewidth=0.7, alpha=0.4
                )

    plt.title(f"{target_name} pLDDT Comparison")
    plt.xlabel("Residue Index")
    plt.ylabel("pLDDT")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, axis="y", alpha=0.2)
    plt.savefig(f"{base_dir}/common/{target_name}_plddt_comparison.png", dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--name", type=str, required=True)
    args = parser.parse_args()

    main(args.dir, args.name)
