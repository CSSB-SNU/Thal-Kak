import yaml, os, json, argparse, shutil, time, sys, string
from datetime import datetime
from pathlib import Path
import pandas as pd
from collections import Counter
import shlex

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMMON_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "common")
for _p in (SCRIPT_DIR, COMMON_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from chain_utils import assign_chain_indices
from template_cleaner import clean_template_for_boltz
from chain_utils import PDB_CHAIN_CHARS, CIF_CHAIN_CHARS
from thalkak import get_logger, run_logged

log = get_logger("structure")


def auth_to_label(cif_path, auth_chain):
    # Convert auth_chain to label_chain using cif file
    tags = []
    i_auth = i_label = None
    in_atom_loop = False
    cnt = Counter()

    with open(cif_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            if s == "loop_":
                tags = []
                in_atom_loop = False
                i_auth = i_label = None
                continue

            # collect atom_site tags
            if s.startswith("_atom_site."):
                tags.append(s.split()[0])
                continue

            # first data line after atom_site tags
            if tags and not in_atom_loop:
                try:
                    i_auth = tags.index("_atom_site.auth_asym_id")
                    i_label = tags.index("_atom_site.label_asym_id")
                    in_atom_loop = True
                except ValueError:
                    tags = []  # not the loop we want
                    continue

            if in_atom_loop:
                # end of atom_site loop
                if (
                    s.startswith("_")
                    or s == "loop_"
                    or s.startswith("data_")
                    or s.startswith("#")
                ):
                    break
                row = shlex.split(s)
                if len(row) > max(i_auth, i_label) and row[i_auth] == auth_chain:
                    cnt[row[i_label]] += 1

    if not cnt:
        raise ValueError(f"auth chain {auth_chain} not found")
    return cnt.most_common(1)[0][0]


def fix_pdb_chain_ids(pdb_path):
    """Collapse boltz's 2-character chain IDs (e.g. AA, BA) to single chars,
    continuing the label sequence after 'Z' as a-z then 0-9 (the shared
    PDB_CHAIN_CHARS scheme). Single-char A-Z chains are left untouched.

    Returns True on success (including when there is nothing to fix). Returns
    False without modifying the file when the model has more chains than the
    62 single-char labels can hold -- such a model cannot be written as a
    standard PDB, and the caller falls back to the cif.
    """
    with open(pdb_path, "r") as f:
        lines = f.readlines()

    # Collect 2-char chain IDs in order of first appearance
    seen = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM")) and line[22].isalpha():
            chain_id = line[21:23]
            if chain_id not in seen:
                seen.append(chain_id)

    if not seen:
        return True

    # boltz emits 2-char ids only for chains past the 26 single-char A-Z ones,
    # so they continue the label sequence at 'a': PDB_CHAIN_CHARS[26], [27], ...
    if 26 + len(seen) > len(PDB_CHAIN_CHARS):
        return False

    chain_map = {cid: PDB_CHAIN_CHARS[26 + i] for i, cid in enumerate(seen)}

    # Remap ATOM/HETATM *and* TER records -- boltz puts the chain id in the
    # same columns on TER lines, and leaving them with the old 2-char id both
    # looks wrong and misleads reorder_pdb_chains (which groups by column 22).
    fixed_lines = []
    for line in lines:
        if line.startswith(("ATOM", "HETATM", "TER")) and line[22:23].isalpha():
            chain_id = line[21:23]
            if chain_id in chain_map:
                line = line[:21] + chain_map[chain_id] + line[23:]
        fixed_lines.append(line)

    with open(pdb_path, "w") as f:
        f.writelines(fixed_lines)
    return True


def _boltz_cif_to_pdb(cif_path, pdb_path):
    """Convert a boltz mmCIF model to PDB, collapsing 2-char chain names to
    single chars the same way fix_pdb_chain_ids does for boltz's PDB output, so
    both output_format modes yield identical chain labels. Single-char A-Z
    chains are left as-is; 2-char chains (boltz chains past 'Z') continue the
    label sequence at 'a' in chain order (a-z, 0-9).

    Returns pdb_path on success, or None if the model has more chains than the
    62 single-char labels can hold (the caller keeps the cif instead).
    """
    import gemmi

    structure = gemmi.read_structure(cif_path)
    if len(structure) == 0:
        raise ValueError(f"No model found in {cif_path}")

    chains = list(structure[0])
    if len(chains) > len(PDB_CHAIN_CHARS):
        return None

    # Relabel the multi-char chains (boltz names them AA, BA, ...) to a-z/0-9,
    # in chain order, leaving single-char A-Z chains untouched.
    two_char = [c for c in chains if len(c.name) > 1]
    for i, chain in enumerate(two_char):
        chain.name = PDB_CHAIN_CHARS[26 + i]

    structure.write_pdb(pdb_path)
    return pdb_path


def _make_ter(serial, last_atom):
    """Build a well-formed TER record terminating the chain of ``last_atom``.

    Copies resName/chainID/resSeq/iCode (cols 18-27) from the chain's final
    ATOM/HETATM line and gives the TER its own serial.
    """
    return f"TER   {serial:>5}      {last_atom[17:27]}\n"


def reorder_pdb_chains(pdb_path):
    """Reorder atom records so chains appear alphabetically (A, B, C, D) and
    terminate every polymer chain with exactly one TER.

    boltz groups chain copies by entity, so an A2B2 complex comes out as
    A, C, B, D, whereas the other models emit A, B, C, D. Reorder the chain
    blocks to match and renumber atom serials so they stay monotonic. Chain
    labels are unchanged -- only record order and serials.

    boltz's PDB writer also omits the TER after the last chain in the file;
    once chain blocks are relocated that gap ends up in the middle of the
    model. So drop the inherited TER records entirely and regenerate one per
    polymer chain, guaranteeing a TER between chains and at the terminal chain
    regardless of what boltz emitted (or of whether a reorder was needed).
    """
    with open(pdb_path, "r") as f:
        lines = f.readlines()

    # TER lines are dropped here and regenerated below.
    blocks, order, header, trailer = {}, [], [], []
    seen_atom = False
    for line in lines:
        if line.startswith(("ATOM", "HETATM", "ANISOU")):
            seen_atom = True
            chain_id = line[21:22]
            if chain_id not in blocks:
                blocks[chain_id] = []
                order.append(chain_id)
            blocks[chain_id].append(line)
        elif line.startswith("TER"):
            seen_atom = True
        elif not seen_atom:
            header.append(line)
        else:
            trailer.append(line)

    out = list(header)
    serial = 0
    for chain_id in sorted(blocks, key=PDB_CHAIN_CHARS.find):
        last_atom = None
        for line in blocks[chain_id]:
            if line.startswith("ANISOU"):
                s = serial  # ANISOU shares the serial of its preceding atom
            else:
                serial += 1
                s = serial
                last_atom = line
            out.append(f"{line[:6]}{s:>5}{line[11:]}")
        # Terminate polymer chains (those carrying ATOM records) with a TER.
        if last_atom is not None and last_atom.startswith("ATOM"):
            serial += 1
            out.append(_make_ter(serial, last_atom))
    out.extend(trailer)

    with open(pdb_path, "w") as f:
        f.writelines(out)


def reorder_cif_chains(cif_path):
    """Reorder atom records so chains appear alphabetically (A, B, C, D) and
    renumber _atom_site.id so the serials stay monotonic.

    boltz groups chain copies by entity, so an A2B2 complex comes out as
    A, C, B, D, whereas the other models emit A, B, C, D. Reorder the chain
    blocks to match. The mmcif counterpart of reorder_pdb_chains, minus the
    TER regeneration -- mmcif has no TER records. Chain labels are left exactly
    as boltz wrote them (A-Z, then AA, BA, ... past 'Z'); only record order and
    atom ids change.

    Only the atom records move: every other chain-keyed category
    (_struct_asym, _pdbx_poly_seq_scheme, _ma_qa_metric_local, ...) is an
    unordered set per the mmcif spec, and nothing in a boltz model points back
    at _atom_site.id, so renumbering the serials breaks no cross-reference.
    """
    import gemmi

    # Edit the raw cif document rather than a parsed gemmi.Structure: boltz
    # writes ModelCIF, and a Structure round-trip would drop its model-specific
    # categories (the _ma_qa_metric_local pLDDTs among them).
    doc = gemmi.cif.read_file(cif_path)
    block = doc.sole_block()

    # boltz writes auth_asym_id and label_asym_id identically; group on auth,
    # the one that becomes the chain column if the model is later made a pdb.
    loop = chain_col = None
    for tag in ("_atom_site.auth_asym_id", "_atom_site.label_asym_id"):
        candidate = block.find_loop(tag).get_loop()
        if candidate is not None:
            loop, chain_col = candidate, candidate.tags.index(tag)
            break
    if loop is None:
        # No _atom_site loop -- a single-atom model is written as key/value
        # pairs instead. Either way there is nothing to reorder.
        return

    width = loop.width()
    rows = [loop.values[i * width : (i + 1) * width] for i in range(loop.length())]

    # Sort on boltz's own label sequence (A-Z, then AA, BA, ...) rather than
    # lexicographically, so chain AA follows Z instead of landing after A. The
    # sort is stable, so atoms keep their original order within a chain.
    def chain_key(chain_id):
        try:
            return (0, CIF_CHAIN_CHARS.index(chain_id))
        except ValueError:
            return (1, chain_id)

    rows.sort(key=lambda row: chain_key(row[chain_col]))

    if "_atom_site.id" in loop.tags:
        id_col = loop.tags.index("_atom_site.id")
        for serial, row in enumerate(rows, start=1):
            row[id_col] = str(serial)

    # set_all_values takes one list per column, so transpose back.
    loop.set_all_values([list(column) for column in zip(*rows)])
    doc.write_file(cif_path)


def rename_output_dir(output_dir):
    if os.path.exists(f"{output_dir}"):
        output_dir += datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
    return output_dir


def mv_output_dir(temp_dir, output_dir):
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    shutil.move(temp_dir, output_dir)
    os.makedirs(f"{output_dir}/common", exist_ok=True)
    try:
        os.rmdir("./temp/")
    except OSError:
        pass


def main(data_yaml, boltz2_yaml):
    with open(data_yaml, "r") as file:
        data_config = yaml.safe_load(file)
    with open(boltz2_yaml, "r") as file:
        boltz_config = yaml.safe_load(file)

    ### Make input.yaml and csv files
    ## initialize
    name = os.path.basename(data_yaml).split(".")[0]
    temp_dir = f"temp/boltz_results_{name}"
    # Preserve leftover state from a previous failed run -- stale msa CSVs,
    # cleaned templates, or input.yaml under temp_dir would otherwise leak
    # into this job. Rename with a timestamp so debug evidence is kept.
    if os.path.exists(temp_dir):
        failed_dir = f"{temp_dir}_failed_{datetime.now():%Y%m%d_%H%M%S}"
        os.rename(temp_dir, failed_dir)
        print(f"[boltz] preserved previous temp at {failed_dir}")
    os.makedirs(f"{temp_dir}/msa")
    yaml_output = "version: 1\nsequences:\n"  # Initialize YAML output
    n_chain = 0

    ## a3m parsing
    chain_len, chain_copy = [], []
    protein_chain_ids = set()
    a3m_chain_indices = assign_chain_indices(
        [e["copy"] for e in data_config["a3m"]],
        [0 if e.get("type", "protein") == "protein" else 1 for e in data_config["a3m"]],
    )
    n_chain = sum(e["copy"] for e in data_config["a3m"])
    for i, entity in enumerate(data_config["a3m"]):
        paired_a3m, unpaired_a3m = [], []
        if entity["paired_path"] != None:
            with open(entity["paired_path"], "r") as f:
                paired_a3m = f.readlines()
        if entity["unpaired_path"] != None:
            with open(entity["unpaired_path"], "r") as f:
                unpaired_a3m = f.readlines()
        sequence = unpaired_a3m[1].strip()
        chain_len.append(len(sequence))
        chain_copy.append(entity["copy"])

        if entity["type"] == "protein":
            ## Write msa.csv files
            with open(f"{temp_dir}/msa/{name}_msa_{i}.csv", "w") as csv_file:
                write_data = "key,sequence\n"
                for j, seq in enumerate(paired_a3m[1::2]):
                    write_data += f"{j},{seq.strip()}\n"
                for seq in unpaired_a3m[1::2]:
                    write_data += f"-1,{seq.strip()}\n"
                csv_file.write(write_data[:-1])  # remove last newline

        ## Append input.yaml
        chain_ids = [
            chr(65 + k) if k < 26 else chr(65 + k % 26) + chr(64 + k // 26)
            for k in a3m_chain_indices[i]
        ]
        yaml_output += f"  - {entity['type']}:\n"
        yaml_output += f"      id: [{','.join(chain_ids)}]\n"
        yaml_output += f"      sequence: {sequence}\n"
        if entity["type"] == "protein":
            protein_chain_ids.update(chain_ids)
            yaml_output += (
                f"      msa: {Path.cwd() / Path(temp_dir)}/msa/{name}_msa_{i}.csv\n"
            )

    ## ligand parsing
    if "ligand" in data_config and data_config["ligand"]:
        for entity in data_config["ligand"]:
            yaml_output += f"  - ligand:\n"
            yaml_output += f"      id: [{','.join([chr(65 + k) if k < 26 else chr(65 + k % 26) + chr(64 + k // 26) for k in range(n_chain, n_chain + entity['copy'])])}]\n"
            if "smiles" in entity and entity["smiles"]:
                yaml_output += f"      smiles: '{entity['smiles']}'\n"
            elif "ccd" in entity and entity["ccd"]:
                yaml_output += f"      ccd: '{entity['ccd']}'\n"
            n_chain += entity["copy"]

    ## template parsing
    if "templates" in data_config and data_config["templates"]:
        cleaned_tmpl_dir = Path(temp_dir) / "cleaned_templates"
        cleaned_tmpl_dir.mkdir(parents=True, exist_ok=True)
        template_yaml = ""
        for template in data_config["templates"]:
            # boltz only supports templates on protein chains. Drop any
            # non-protein (RNA/DNA) query chains, else boltz raises
            # "Chain X assigned for template ... is not one of the protein chains!"
            query_chains = [
                c for c in template["chain_query"] if c in protein_chain_ids
            ]
            if not query_chains:
                print(
                    f"[boltz] skipping non-protein template "
                    f"{Path(template['path']).name} "
                    f"(query chains {list(template['chain_query'])})"
                )
                continue
            src = Path(template["path"])
            # Assume chain_template has only one unique value
            chain = template["chain_template"][0]
            cleaned = cleaned_tmpl_dir / f"{src.stem}_{chain}.cif"
            label_chain, n_renamed = clean_template_for_boltz(src, cleaned, chain)
            if n_renamed:
                print(
                    f"[boltz] {src.name}/{chain}: replaced {n_renamed} unknown "
                    f"residue(s) with UNK/N/DN"
                )
            template_yaml += f"  - cif: {cleaned}\n"
            template_yaml += (
                f"    template_id: [{','.join([label_chain] * len(query_chains))}]\n"
            )
            template_yaml += f"    chain_id: [{','.join(query_chains)}]\n"
        if template_yaml:
            yaml_output += "templates:\n" + template_yaml

    ## constraint parsing
    if "constraints" in boltz_config and boltz_config["constraints"]:
        yaml_output += "constraints:\n  "
        yaml_dump = yaml.dump(boltz_config["constraints"], default_flow_style=False)
        yaml_output += yaml_dump.replace("\n", "\n  ")

    with open(f"{temp_dir}/{name}.yaml", "w") as input_file:
        input_file.write(yaml_output)

    ### Run boltz2

    if boltz_config["no_kernels"]:
        command = f"boltz predict {temp_dir}/{name}.yaml --out_dir {Path(temp_dir).parents[0]} --output_format {boltz_config['output_format']} --diffusion_samples {boltz_config['n_samples']} --recycling_steps {boltz_config['recycling_steps']} --sampling_steps {boltz_config['sampling_steps']} --no_kernels"
    else:
        command = f"boltz predict {temp_dir}/{name}.yaml --out_dir {Path(temp_dir).parents[0]} --output_format {boltz_config['output_format']} --diffusion_samples {boltz_config['n_samples']} --recycling_steps {boltz_config['recycling_steps']} --sampling_steps {boltz_config['sampling_steps']}"
    # boltz's native MSA subsampler (random num_subsampled_msa=1024 rows by
    # default). Off/absent = full MSA.
    if boltz_config.get("subsample_msa", False):
        command += " --subsample_msa"
    output_dir = (
        f"{data_config['output_dir']}/boltz_results_{name}_{data_config['job_name']}"
    )
    # boltz skips inputs it cannot parse and still exits 0, so check=True below
    # proves nothing about whether predictions were written.
    def _require_predictions(seed=None):
        src = Path(temp_dir) / "predictions" / name
        if src.is_dir():
            return src
        target = repr(name) + (f" (seed {seed})" if seed is not None else "")
        raise RuntimeError(
            f"boltz exited 0 but wrote no predictions for {target}: {src} is "
            f"missing. boltz skips inputs it cannot parse and still exits 0 -- "
            f"look for 'Failed to process ... Skipping' earlier in this log for "
            f"the real cause."
        )

    if data_config["seed"] != None:
        base_command = command
        for seed in data_config["seed"]:
            run_logged(f"{base_command} --seed {seed}", log=log)
            os.rename(
                _require_predictions(seed),
                f"{temp_dir}/predictions/{name}_seed_{seed}/",
            )
    else:
        run_logged(command, log=log)
        _require_predictions()
    output_dir = rename_output_dir(output_dir)
    mv_output_dir(temp_dir, output_dir)

    ### Organize output metrics
    ## Create metrics csv
    csv_rows = []
    for seed in data_config["seed"] if data_config["seed"] != None else [None]:
        for i in range(boltz_config["n_samples"]):
            if seed is not None:
                prediction_dir = f"{output_dir}/predictions/{name}_seed_{seed}"
            else:
                prediction_dir = f"{output_dir}/predictions/{name}"

            with open(
                f"{prediction_dir}/confidence_{name}_model_{i}.json", "r"
            ) as json_file:
                confidence_json = json.load(json_file)
            csv_rows.append(
                {
                    "target": name,
                    "option": data_config["job_name"],
                    "model": "boltz",
                    "seed-sample": f"seed_{seed}_sample_{i}",
                    "mean_plddt": f"{confidence_json['complex_plddt'] * 100:.3f}",
                    "ptm": f"{confidence_json['ptm']:.3f}",
                    "iptm": f"{confidence_json['iptm']:.3f}",
                    "ranking_score": f"{confidence_json['confidence_score']:.3f}",
                }
            )

            ## Copy the output model into the common dir with seed info in the
            ## filename. boltz writes .pdb or .cif depending on output_format;
            ## either way the chains are reordered alphabetically. A .cif keeps
            ## boltz's own chain labels, while a .pdb also needs its 2-char
            ## labels collapsed to fit the single chain column (>62 chains
            ## cannot be written as a standard PDB at all).
            stem = f"{name}_seed_{seed}_sample_{i}"
            if boltz_config["output_format"] == "mmcif":
                pred_cif = f"{prediction_dir}/{name}_model_{i}.cif"
                common_cif = f"{output_dir}/common/{stem}.cif"
                shutil.copy(pred_cif, common_cif)
                reorder_cif_chains(common_cif)
            else:  # output_format == "pdb"
                common_pdb = f"{output_dir}/common/{stem}.pdb"
                shutil.copy(f"{prediction_dir}/{name}_model_{i}.pdb", common_pdb)
                if fix_pdb_chain_ids(common_pdb):
                    reorder_pdb_chains(common_pdb)
                else:
                    os.remove(common_pdb)
                    log.warning(
                        f"[boltz] {stem}: model has more than "
                        f"{len(PDB_CHAIN_CHARS)} chains; cannot write a standard "
                        f"PDB and boltz was run with output_format=pdb (no cif to "
                        f"keep). Skipped."
                    )
    df = pd.DataFrame(csv_rows)
    df.sort_values(by=["ranking_score"], inplace=True, ascending=False)
    df.to_csv(f"{output_dir}/common/{name}_results_summary.csv", index=False)

    # for plotting
    chain_info = {"chain_len": chain_len, "chain_copy": chain_copy}
    with open(f"{output_dir}/chain_info.json", "w") as json_file:
        json.dump(chain_info, json_file)

    print(f"RESULT_DIR:{output_dir}")
    return output_dir


if __name__ == "__main__":
    from thalkak import setup_logging

    setup_logging()
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_yaml", type=str, required=True)
    parser.add_argument("--boltz2_yaml", type=str, required=True)
    args = parser.parse_args()

    main(args.data_yaml, args.boltz2_yaml)

    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")
