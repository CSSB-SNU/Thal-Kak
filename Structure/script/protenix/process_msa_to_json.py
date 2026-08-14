import yaml
import argparse
import json
import re
import os
import sys
import tempfile
from pathlib import Path

from Bio.PDB.MMCIF2Dict import MMCIF2Dict
import gemmi

COMMON_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "common"
)
if COMMON_DIR not in sys.path:
    sys.path.insert(0, COMMON_DIR)
from process_template import generate_m8_from_hhsearch
from chain_utils import assign_chain_indices, CIF_CHAIN_CHARS


# ============================================================================
# 1. CIGAR
# ============================================================================


def parse_m8_cigar(query_start, hit_start, cigar):
    """Return {query_pos: hit_pos} mapping (1-based) from an m8 CIGAR string."""
    mapping = {}
    q, h = query_start, hit_start
    for length, op in re.findall(r"(\d+)([MIDNSHP=X])", cigar):
        length = int(length)
        if op in ("M", "=", "X"):
            for _ in range(length):
                mapping[q] = h
                q += 1
                h += 1
        elif op == "I":
            q += length
        elif op == "D":
            h += length
    return mapping


# ============================================================================
# 2. m8 loading
# ============================================================================


def load_m8(m8_path):
    m8_map = {}
    with open(m8_path) as f:
        for line in f:
            cols = line.strip().split("\t")
            if len(cols) < 12:
                continue
            m8_map[cols[1]] = {
                "q_start": int(cols[6]),
                "h_start": int(cols[8]),
                "cigar": cols[-1],
            }
    print(f"[INFO] Loaded existing m8: {m8_path} ({len(m8_map)} entries)")
    return m8_map


# ============================================================================
# 3. Sequence helpers
# ============================================================================

def detect_sequence_type(s_type):
    if s_type == "rna":
        return "rnaSequence"
    elif s_type == "dna":
        return "dnaSequence"
    elif s_type == "protein":
        return "proteinChain"

def extract_query_sequence(a3m_path):
    """Read the first (query) sequence from a FASTA or a3m file."""
    path = Path(a3m_path)
    if not path.exists():
        return ""
    seq = ""
    in_seq = False
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line.startswith(">"):
                if in_seq:
                    break
                in_seq = True
                continue
            if in_seq:
                seq += line.split("\t")[0].strip()
    return seq.replace("-", "").replace(".", "").upper()


# ============================================================================
# 4. Template CIF extraction (AF3-compatible, full SEQRES preserved)
# ============================================================================


def _pdb_to_cif(pdb_path):
    structure = gemmi.read_structure(str(pdb_path))
    structure.setup_entities()
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".cif", delete=False)
    tmp.close()
    structure.make_mmcif_document().write_file(tmp.name)
    return tmp.name


def _is_polymer_residue(comp_id):
    try:
        ent = gemmi.find_tabulated_residue(comp_id)
    except Exception:
        return False
    if ent is None:
        return False
    return ent.is_amino_acid() or ent.is_nucleic_acid()


def _read_seqres_from_raw_cif(cif_path, chain_id):
    """Read the FULL polymer SEQRES (including unresolved residues) from a raw cif.

    This is what m8 CIGAR hit positions are aligned against. Using only
    resolved residues (e.g. via gemmi chain iteration) gives a shorter
    sequence and causes spurious OOR filtering of valid alignment pairs.
    """
    try:
        mmcif = MMCIF2Dict(cif_path)
    except Exception:
        return []

    auth_to_entity = {}
    if "_atom_site.auth_asym_id" in mmcif:
        n = len(mmcif["_atom_site.auth_asym_id"])
        for i in range(n):
            a = mmcif["_atom_site.auth_asym_id"][i]
            e = mmcif["_atom_site.label_entity_id"][i]
            if a not in auth_to_entity:
                auth_to_entity[a] = e

    target_entity = auth_to_entity.get(chain_id)
    if target_entity is None:
        return []

    if "_entity_poly_seq.entity_id" not in mmcif:
        return []

    n_rows = len(mmcif["_entity_poly_seq.entity_id"])
    seqres = []
    for i in range(n_rows):
        if mmcif["_entity_poly_seq.entity_id"][i] != target_entity:
            continue
        try:
            num = int(mmcif["_entity_poly_seq.num"][i])
        except (ValueError, TypeError):
            continue
        mon = mmcif["_entity_poly_seq.mon_id"][i]
        seqres.append((num, mon))

    seqres.sort(key=lambda x: x[0])
    return seqres


def extract_template_chain(
    input_path, output_cif, chain_id, fallback_date="2020-01-01"
):
    """
    Extract a single chain from a PDB/mmCIF into an AF3-compatible mmCIF.

    - entity_poly_seq copied directly from raw cif (FULL SEQRES, including
      unresolved residues / rare modifications gemmi cannot recognize),
      so label_seq_id 1..L is preserved and matches m8 hit indices.
    - atom_site contains only polymer ATOM records that gemmi recognizes
      as amino acid / nucleic acid. Non-polymer / rare modifications are
      excluded from atom_site but remain as gaps in entity_poly_seq.
    - chain renamed to A and entity_id renamed to 1 (Protenix/AF3 expect
      a single-chain template cif).

    Returns: (output_cif_path, polymer_length, dropped_residues)
    """
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    cleanup_path = None
    if suffix in (".pdb", ".ent"):
        cif_for_parsing = _pdb_to_cif(input_path)
        cleanup_path = cif_for_parsing
    elif suffix in (".cif", ".mmcif"):
        cif_for_parsing = str(input_path)
    else:
        raise ValueError(f"Unsupported structure file extension: {suffix}")

    try:
        seqres = _read_seqres_from_raw_cif(cif_for_parsing, chain_id)
        if not seqres:
            raise ValueError(
                f"Could not read _entity_poly_seq for chain '{chain_id}' "
                f"in {input_path}."
            )
        polymer_length = max(n for n, _ in seqres)

        mmcif = MMCIF2Dict(cif_for_parsing)

        atom_site_fields = [
            "group_PDB",
            "id",
            "type_symbol",
            "label_atom_id",
            "label_alt_id",
            "label_comp_id",
            "label_asym_id",
            "label_entity_id",
            "label_seq_id",
            "pdbx_PDB_ins_code",
            "Cartn_x",
            "Cartn_y",
            "Cartn_z",
            "occupancy",
            "B_iso_or_equiv",
            "auth_seq_id",
            "auth_asym_id",
            "pdbx_PDB_model_num",
        ]

        n_atoms = len(mmcif["_atom_site.auth_asym_id"])
        for f in atom_site_fields:
            key = f"_atom_site.{f}"
            if key not in mmcif:
                mmcif[key] = ["?"] * n_atoms

        rows = []
        dropped_residues = []
        seen_dropped = set()
        _is_polymer_cache = {}
        first_model = None

        for i in range(n_atoms):
            if mmcif["_atom_site.auth_asym_id"][i] != chain_id:
                continue

            # Keep a single model / conformation. Downstream mmCIF parsers
            # (e.g. boltz) treat extra models as an ensemble and assert that
            # every atom's presence matches across them, which multi-model NMR
            # templates violate; alternate conformations cause the same mismatch.
            model_num = mmcif["_atom_site.pdbx_PDB_model_num"][i]
            if first_model is None:
                first_model = model_num
            elif model_num != first_model:
                continue
            if mmcif["_atom_site.label_alt_id"][i] not in ("", ".", "?", "A"):
                continue

            comp = mmcif["_atom_site.label_comp_id"][i]
            if comp not in _is_polymer_cache:
                _is_polymer_cache[comp] = _is_polymer_residue(comp)
            if not _is_polymer_cache[comp]:
                auth_seq = mmcif["_atom_site.auth_seq_id"][i]
                key = (auth_seq, comp)
                if key not in seen_dropped:
                    seen_dropped.add(key)
                    dropped_residues.append(key)
                continue

            raw_label_seq = mmcif["_atom_site.label_seq_id"][i]
            if raw_label_seq in ("", ".", "?"):
                continue

            row = []
            for f in atom_site_fields:
                val = mmcif[f"_atom_site.{f}"][i]
                row.append(val if val not in ("", ".", "?") else "?")

            idx_label = atom_site_fields.index("label_asym_id")
            idx_auth = atom_site_fields.index("auth_asym_id")
            idx_entity = atom_site_fields.index("label_entity_id")
            idx_group = atom_site_fields.index("group_PDB")

            row[idx_label] = "A"
            row[idx_auth] = "A"
            row[idx_entity] = "1"
            row[idx_group] = "ATOM"

            rows.append(row)

        if not rows:
            raise ValueError(
                f"No polymer atoms recognized for chain '{chain_id}' in {input_path}"
            )

        with open(output_cif, "w") as out:
            out.write(f"data_TEMPLATE_{chain_id}\n#\n")
            out.write(
                "loop_\n_pdbx_audit_revision_history.revision_date\n"
                "_pdbx_audit_revision_history.ordinal\n"
            )
            out.write(f"{fallback_date} 1\n#\n")
            out.write("loop_\n_entity.id\n_entity.type\n1 polymer\n#\n")
            out.write(
                "loop_\n_entity_poly_seq.entity_id\n_entity_poly_seq.num\n"
                "_entity_poly_seq.mon_id\n"
            )
            for num, mon in seqres:
                out.write(f"1 {num} {mon}\n")
            out.write("#\n")
            out.write(
                "loop_\n_struct_asym.id\n_struct_asym.entity_id\n"
                "_struct_asym.pdbx_PDB_id\n"
            )
            out.write("A 1 A\n#\n")
            out.write("loop_\n")
            out.write("_entity_poly.entity_id\n")
            out.write("_entity_poly.type\n")
            out.write("_entity_poly.nstd_linkage\n")
            out.write("_entity_poly.nstd_monomer\n")
            out.write("_entity_poly.pdbx_seq_one_letter_code\n")
            out.write("_entity_poly.pdbx_seq_one_letter_code_can\n")
            out.write("_entity_poly.pdbx_strand_id\n")
            out.write("1 polypeptide(L) no yes ? ? A\n")
            out.write("#\n")
            out.write("loop_\n")
            for f in atom_site_fields:
                out.write(f"_atom_site.{f}\n")
            for r in rows:
                out.write(" ".join(r) + "\n")
            out.write("#\n")

    finally:
        if cleanup_path is not None:
            Path(cleanup_path).unlink(missing_ok=True)

    return output_cif, polymer_length, dropped_residues


# ============================================================================
# 5. Ligand helpers (Protenix dialect)
# ============================================================================


def _build_ligand_entry(lig_yaml_entry):
    """Convert yaml ligand entry into Protenix-format ligand object.

    Schema: {"ligand": {"ligand": "<value>", "count": <int>}}
    Where <value> is:
      - "CCD_<code>" for CCD codes
      - "FILE_<path>" for 3D-conformer files
      - "<smiles>" for raw SMILES (no prefix)
    """
    count = int(lig_yaml_entry.get("copy", 1))

    if "ccd" in lig_yaml_entry and lig_yaml_entry["ccd"] is not None:
        ccd_raw = str(lig_yaml_entry["ccd"]).strip()
        ccd_value = ccd_raw if ccd_raw.upper().startswith("CCD_") else f"CCD_{ccd_raw}"
        return {"ligand": {"ligand": ccd_value, "count": count}}

    if "smiles" in lig_yaml_entry and lig_yaml_entry["smiles"] is not None:
        smiles = str(lig_yaml_entry["smiles"]).strip()
        return {"ligand": {"ligand": smiles, "count": count}}

    if "file" in lig_yaml_entry and lig_yaml_entry["file"] is not None:
        file_path = str(lig_yaml_entry["file"]).strip()
        file_value = file_path if file_path.startswith("FILE_") else f"FILE_{file_path}"
        return {"ligand": {"ligand": file_value, "count": count}}

    return None


# ============================================================================
# 6. Main
# ============================================================================


def main(args):
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    tpl_out_dir = save_dir / "templates"
    tpl_out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)

    final_name = args.name if args.name else save_dir.name
    a3m_list = data_cfg.get("a3m", [])
    tmpl_list = data_cfg.get("templates", [])
    ligand_list = data_cfg.get("ligand", [])

    # ---- m8 map -----------------------------------------------------------
    m8_map = {}
    if tmpl_list:
        tpl_dir = Path(tmpl_list[0]["path"]).parent.parent
        m8_path = tpl_dir / "pdb70.m8"

        if m8_path.exists():
            print(f"[INFO] Found existing pdb70.m8 at {m8_path}; skipping hhsearch")
            m8_map = load_m8(m8_path)
        else:
            print(f"[INFO] No pdb70.m8 found at {m8_path}; running hhsearch...")
            target_name = data_cfg.get("job_name")
            start_idx = 101
            num_map = {
                str(start_idx + i): [f"{target_name}_{item['chain_template'][0]}"]
                for i, item in enumerate(tmpl_list)
            }
            result = generate_m8_from_hhsearch(num_map, a3m_list, tmpl_list, save_dir)
            if isinstance(result, (str, Path)):
                m8_map = load_m8(result)
            elif isinstance(result, dict):
                m8_map = result
            else:
                raise TypeError(
                    f"Unexpected return type from generate_m8_from_hhsearch: "
                    f"{type(result)}"
                )

    # ---- Pre-map template hits keyed by query chain ID --------------------
    tpl_lookup = {}
    for t_entry in tmpl_list:
        cif_p = Path(t_entry["path"])
        for q_id, t_id in zip(t_entry["chain_query"], t_entry["chain_template"]):
            tpl_lookup.setdefault(q_id, []).append((cif_p, t_id))

    # ---- Build per-entity JSON entries ------------------------------------
    a3m_copies = [item.get("copy", 1) for item in a3m_list]
    a3m_types = [
        0 if item.get("type", "protein") == "protein" else 1 for item in a3m_list
    ]
    chains_per_entity = assign_chain_indices(a3m_copies, a3m_types)

    data_stem = Path(args.data).stem
    entity_info = []

    for entity_idx, item in enumerate(a3m_list):
        unpaired_path = item.get("unpaired_path")
        query_seq = extract_query_sequence(unpaired_path)
        if not query_seq:
            entity_info.append(None)
            continue

        base_chain_id = CIF_CHAIN_CHARS[chains_per_entity[entity_idx][0]]
        entity_type = detect_sequence_type(item["type"])
        query_len = len(query_seq)

        # ---- Build AF3-style template objects for this chain --------------
        template_objs = []
        if base_chain_id in tpl_lookup:
            for hit_idx, (cif_path, t_id) in enumerate(tpl_lookup[base_chain_id]):
                hit_key = f"{cif_path.stem}_{t_id}"
                out_cif_name = (
                    f"{data_stem}_entity{entity_idx}_hit{hit_idx}_chain_{t_id}.cif"
                )
                out_cif_path = tpl_out_dir / out_cif_name

                try:
                    _, polymer_len, dropped = extract_template_chain(
                        cif_path, out_cif_path, t_id
                    )
                except Exception as e:
                    print(f"[WARN] {hit_key}: extract_template_chain failed: {e}")
                    continue

                if dropped:
                    sample = dropped[:10]
                    more = f" (+{len(dropped) - 10} more)" if len(dropped) > 10 else ""
                    print(
                        f"[INFO] {hit_key}: {len(dropped)} non-polymer residues "
                        f"left as gaps in entity_poly_seq: {sample}{more}"
                    )

                if hit_key not in m8_map:
                    print(f"[WARN] {hit_key} not in m8; skipping template")
                    continue

                m_info = m8_map[hit_key]
                init_mapping = parse_m8_cigar(
                    m_info["q_start"], m_info["h_start"], m_info["cigar"]
                )

                valid_pairs = [
                    (q, h)
                    for q, h in init_mapping.items()
                    if 1 <= q <= query_len and 1 <= h <= polymer_len
                ]

                n_total = len(init_mapping)
                n_drop_q = sum(1 for q in init_mapping if not (1 <= q <= query_len))
                n_drop_h = sum(
                    1
                    for q, h in init_mapping.items()
                    if (1 <= q <= query_len) and not (1 <= h <= polymer_len)
                )
                if n_drop_q or n_drop_h:
                    print(
                        f"[WARN] {hit_key}: {n_total} alignment pairs -> "
                        f"{len(valid_pairs)} valid "
                        f"(query_OOR={n_drop_q}, hit_OOR={n_drop_h}, "
                        f"query_len={query_len}, polymer_len={polymer_len})"
                    )

                if not valid_pairs:
                    print(
                        f"[WARN] {hit_key}: no valid alignment pairs; skipping template"
                    )
                    continue

                valid_pairs.sort()
                q_indices = [q - 1 for q, _ in valid_pairs]
                t_indices = [h - 1 for _, h in valid_pairs]

                # Protenix's InferenceTemplateFeaturizer expects the cif
                # body as an inline 'mmcif' string, NOT 'mmcifPath' (which
                # is AF3-dialect-only). The on-disk cif file is kept for
                # debugging / external use, but only its contents go into
                # the json.
                with open(out_cif_path, "r") as cif_fh:
                    mmcif_str = cif_fh.read()

                template_objs.append(
                    {
                        "mmcif": mmcif_str,
                        "queryIndices": q_indices,
                        "templateIndices": t_indices,
                    }
                )

        # ---- Write per-entity template JSON file --------------------------
        templates_json_path = None
        if template_objs:
            templates_json_path = (
                tpl_out_dir / f"{data_stem}_entity{entity_idx}_templates.json"
            )
            with open(templates_json_path, "w") as f:
                json.dump(template_objs, f, indent=2)
            print(
                f"[INFO] entity{entity_idx} ({base_chain_id}): wrote "
                f"{len(template_objs)} templates -> {templates_json_path}"
            )

        entity_info.append(
            {
                "type": entity_type,
                "sequence": query_seq,
                "paired_path": item.get("paired_path"),
                "unpaired_path": unpaired_path,
                "templates_json_path": (
                    str(templates_json_path.absolute()) if templates_json_path else None
                ),
                "copy": item.get("copy", 1),
            }
        )

    # ---- Emit JSON entries (count=1 per entry) ----------------------------
    # Protenix assigns chain ids sequentially over the entries below, so emit
    # one entry per chain following the order in chains_per_entity.
    chain_order = []
    for entity_idx, chain_idxs in enumerate(chains_per_entity):
        for chain_idx in chain_idxs:
            chain_order.append((chain_idx, entity_idx))
    chain_order.sort()

    json_sequences = []
    for _, entity_idx in chain_order:
        e = entity_info[entity_idx]
        json_sequences.append(
            {
                e["type"]: {
                    "sequence": e["sequence"],
                    "count": 1,
                    "pairedMsaPath": e["paired_path"],
                    "unpairedMsaPath": e["unpaired_path"],
                    "templatesPath": e["templates_json_path"],
                }
            }
        )

    # ---- Ligand entries ---------------------------------------------------
    for lig_idx, lig_entry in enumerate(ligand_list):
        lig_obj = _build_ligand_entry(lig_entry)
        if lig_obj is None:
            print(
                f"[WARN] ligand entry {lig_idx} has none of 'ccd' / 'smiles' / "
                f"'file'; skipping (entry={lig_entry})"
            )
            continue
        json_sequences.append(lig_obj)
        val = lig_obj["ligand"]["ligand"]
        cnt = lig_obj["ligand"]["count"]
        if val.startswith("CCD_"):
            print(f"[ligand] entry {lig_idx}: ccd={val} count={cnt}")
        elif val.startswith("FILE_"):
            print(f"[ligand] entry {lig_idx}: file={val[5:]} count={cnt}")
        else:
            print(f"[ligand] entry {lig_idx}: smiles=<{len(val)} chars> count={cnt}")

    # ---- Final JSON -------------------------------------------------------
    final_output = [
        {
            "name": final_name,
            "modelSeeds": data_cfg.get("seed"),
            "sequences": json_sequences,
        }
    ]

    out_path = save_dir / "input.json"
    with open(out_path, "w") as f:
        json.dump(final_output, f, indent=2)
    print(f"[INFO] input.json written -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Protenix input JSON using AF3-style JSON templates."
    )
    parser.add_argument("--data", required=True, help="Path to target YAML file")
    parser.add_argument(
        "--protenix", required=False, help="(unused, kept for CLI compat)"
    )
    parser.add_argument("--save_path", required=True, help="Output directory")
    parser.add_argument(
        "--name",
        required=False,
        help="Value for the 'name' field in output JSON "
        "(defaults to the output directory name)",
    )
    main(parser.parse_args())
