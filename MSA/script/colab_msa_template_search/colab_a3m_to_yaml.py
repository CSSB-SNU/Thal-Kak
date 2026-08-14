import yaml
import argparse
import string
from pathlib import Path
from collections import defaultdict


class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True


def get_chain_names(n):
    chars = string.ascii_uppercase
    names = []
    for i in range(n):
        if i < 26:
            names.append(chars[i])
        else:
            first = chars[(i // 26) - 1]
            second = chars[i % 26]
            names.append(f"{second}{first}")
    return names


def parse_m8_top_templates(m8_path, env_dir, n_chains, top_n=4):
    """Parse pdb70.m8 and return top_n templates per chain kind by lowest e-value."""
    hits = defaultdict(list)
    with open(m8_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            query_id = int(parts[0])
            target_name = parts[1]
            evalue = float(parts[10])
            hits[query_id].append((target_name, evalue))

    templates_per_chain = {}
    for idx in range(n_chains):
        query_id = 101 + idx
        if query_id not in hits:
            templates_per_chain[idx] = []
            continue

        sorted_hits = sorted(hits[query_id], key=lambda x: x[1])
        template_dir = env_dir / f"templates_{query_id}"

        chain_templates = []
        for target_name, evalue in sorted_hits:
            if len(chain_templates) >= top_n:
                break
            pdb_id, chain_id = target_name.split("_", 1)
            cif_path = template_dir / f"{pdb_id}.cif"
            if not cif_path.exists():
                continue
            chain_templates.append({
                "path": str(cif_path.absolute()),
                "chain_template": [chain_id],
            })
        templates_per_chain[idx] = chain_templates

    return templates_per_chain


def split_colab_a3m_write_yaml(a3m_path, top_n_templates=4):
    a3m_path = Path(a3m_path)
    dir_path = a3m_path.parent
    file_name = a3m_path.stem

    with open(a3m_path, "r") as f:
        lines = f.readlines()

    header_line = lines[0][1:].strip().split("\t")
    chain_lengths = list(map(int, header_line[0].split(",")))
    if len(header_line) > 1:
        chain_counts = list(map(int, header_line[1].split(",")))
    else:
        chain_counts = [1] * len(chain_lengths)

    n_kind = len(chain_lengths)
    chain_kind_names = get_chain_names(n_kind)
    offsets = [0]
    for l in chain_lengths:
        offsets.append(offsets[-1] + l)

    total_chains_needed = sum(chain_counts)
    all_alphabet_names = get_chain_names(total_chains_needed)

    if len(chain_counts) == 1:
        sequence_chains = {0: [c for c in all_alphabet_names]}
        if lines[0].startswith("#"):
            header_data = lines[0][1:].strip().split("\t")
            clean_lines = lines[1:]
        chain_str = "_".join([c.lower() for c in all_alphabet_names])
        u_name = f"{file_name}_unpaired_msa_chains_{chain_str}.a3m"
        u_path = dir_path / u_name
        with open(u_path, "w") as f:
            f.writelines(clean_lines)

        # A homomer has nothing to pair across.
        a3m_yaml_list = []
        for idx in range(len(chain_lengths)):
            a3m_yaml_list.append(
                {
                    "paired_path": None,
                    "unpaired_path": str(u_path.absolute()),
                    "copy": chain_counts[idx],
                    "type": "protein",
                }
            )

    else:
        sequence_chains = defaultdict(list)
        current_idx = 0
        max_copies = max(chain_counts, default=0)
        for r in range(max_copies):
            for i, n in enumerate(chain_counts):
                if r < n:
                    sequence_chains[i].append(all_alphabet_names[current_idx])
                    current_idx += 1
        header_indices = defaultdict(list)
        for i, line in enumerate(lines):
            if line.startswith(">"):
                parts = line[1:].strip().split()
                if parts and parts[0].isdigit() and i > 1:
                    header_indices[int(parts[0])].append(i)

        paired_buffers = {c: [] for c in chain_kind_names}
        unpaired_buffers = {c: [] for c in chain_kind_names}

        unpaired_start_id = 101

        if unpaired_start_id in header_indices:
            paired_end_line = header_indices[unpaired_start_id][0]
        else:
            paired_end_line = len(lines)

        # Paired MSA
        for i in range(1, paired_end_line):
            line = lines[i]
            if line.startswith(">"):
                for c in chain_kind_names:
                    paired_buffers[c].append(line)
            else:
                seq = line.strip()

                split_seqs = []
                current_chain_seq = []
                chain_idx = 0
                match_count = 0

                for ch in seq:
                    # Chains are cut by match-state count: once this chain has
                    # taken chain_lengths[chain_idx] match states (uppercase or
                    # '-'), the next character starts the following chain.
                    if (
                        chain_idx < len(chain_lengths)
                        and match_count == chain_lengths[chain_idx]
                    ):
                        split_seqs.append("".join(current_chain_seq))
                        current_chain_seq = []
                        chain_idx += 1
                        match_count = 0

                    current_chain_seq.append(ch)
                    if ch.isupper() or ch == "-":
                        match_count += 1

                if current_chain_seq:
                    split_seqs.append("".join(current_chain_seq))

                for idx, c in enumerate(chain_kind_names):
                    paired_buffers[c].append(split_seqs[idx] + "\n")

        # Unpaired MSA
        for idx, c in enumerate(chain_kind_names):
            start_id = 101 + idx
            if start_id not in header_indices:
                continue

            start_line = header_indices[start_id][0]
            if header_indices[start_id + 1]:
                end_line = header_indices[start_id + 1][0]
            else:
                end_line = len(lines)
            for line in lines[start_line:end_line]:
                if line.startswith(">"):
                    unpaired_buffers[c].append(line)
                else:
                    seq = line.strip()
                    unp_start = sum(chain_lengths[:idx])
                    if idx + 1 == len(chain_lengths):
                        unpaired_buffers[c].append(seq[unp_start:] + "\n")
                    else:
                        unp_end = -sum(chain_lengths[-(len(chain_lengths) - idx) + 1 :])
                        unpaired_buffers[c].append(seq[unp_start:unp_end] + "\n")

        # Save A3M and build the YAML 'a3m' section.
        a3m_yaml_list = []
        for idx, c in enumerate(chain_kind_names):
            current_chains = sequence_chains[idx]
            chain_str = "_".join([c.lower() for c in current_chains])

            p_name = f"{file_name}_paired_msa_chains_{chain_str}.a3m"
            u_name = f"{file_name}_unpaired_msa_chains_{chain_str}.a3m"

            p_path = dir_path / p_name
            u_path = dir_path / u_name

            with open(p_path, "w") as f:
                f.writelines(paired_buffers[c])
            with open(u_path, "w") as f:
                f.writelines(unpaired_buffers[c])

            a3m_yaml_list.append(
                {
                    "paired_path": str(p_path.absolute()),
                    "unpaired_path": str(u_path.absolute()),
                    "copy": chain_counts[idx],
                    "type": "protein",
                }
            )

    final_yaml = {
        "a3m": a3m_yaml_list,
    }

    # Parse templates from pdb70.m8 if available
    env_dir = dir_path / f"{file_name}_env"
    m8_path = env_dir / "pdb70.m8"
    if top_n_templates > 0 and m8_path.exists():
        templates_per_chain = parse_m8_top_templates(
            m8_path, env_dir, n_kind, top_n=top_n_templates
        )
        templates_yaml_list = []
        for idx in range(n_kind):
            query_chains = sequence_chains[idx]
            for templ in templates_per_chain.get(idx, []):
                templ["chain_query"] = query_chains
                templates_yaml_list.append(templ)
        if templates_yaml_list:
            final_yaml["templates"] = templates_yaml_list

    with open(dir_path / f"{file_name}.yaml", "w") as f:
        yaml.dump(final_yaml, f, sort_keys=False, indent=2, Dumper=NoAliasDumper)

    return dir_path / f"{file_name}.yaml"


def main(args):
    input_a3m = args.input_a3m
    split_colab_a3m_write_yaml(input_a3m, top_n_templates=args.top_n_templates)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument("--input_a3m", default=None, help="Path to input a3m")
    argparser.add_argument(
        "--top_n_templates",
        type=int,
        default=4,
        help="Number of top templates (by lowest e-value) to include in YAML. 0 to disable.",
    )
    args = argparser.parse_args()
    main(args)
