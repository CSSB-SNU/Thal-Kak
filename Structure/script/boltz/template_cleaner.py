"""Pre-process template CIFs for boltz inference.

boltz's CIF parser walks the entire structure (including ligands, water,
and chains other than the requested template) and raises ValueError on
any CCD whose pkl is missing from its bundled moldir. Templates from a
typical search hit can include modifications or cofactors that boltz
cannot resolve.

This module strips the input to the target polymer chain only and
relabels any residue still outside boltz's canonical_tokens and missing
from moldir to the polymer's UNK token, so boltz handles them via its
standard unknown-residue path.
"""

import os
import sys
from pathlib import Path

import gemmi

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTENIX_DIR = os.path.join(os.path.dirname(_HERE), "protenix")
if _PROTENIX_DIR not in sys.path:
    sys.path.insert(0, _PROTENIX_DIR)
from process_msa_to_json import extract_template_chain  # noqa: E402


CANONICAL_PROTEIN = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "UNK",
}
CANONICAL_RNA = {"A", "G", "C", "U", "N"}
CANONICAL_DNA = {"DA", "DG", "DC", "DT", "DN"}
DEFAULT_MOLDIR = Path(os.path.expanduser("~/.boltz/mols"))


def _classify(residue_names):
    """Return (canonical_set, unk_token) for a polymer based on residue names."""
    if any(n in CANONICAL_DNA for n in residue_names):
        return CANONICAL_DNA, "DN"
    if any(n in CANONICAL_RNA for n in residue_names):
        return CANONICAL_RNA, "N"
    return CANONICAL_PROTEIN, "UNK"


def _substitute_in_cif_text(cif_text, substitutions):
    """Replace residue names in entity_poly_seq.mon_id and atom_site.label_comp_id
    columns of the CIF text. Returns the modified text."""
    if not substitutions:
        return cif_text

    lines = cif_text.splitlines(keepends=True)
    out = []
    headers = []
    mon_id_col = None
    comp_id_col = None
    in_loop_header = False

    for line in lines:
        stripped = line.strip()

        if stripped == "loop_":
            headers = []
            mon_id_col = None
            comp_id_col = None
            in_loop_header = True
            out.append(line)
            continue

        if in_loop_header and stripped.startswith("_"):
            headers.append(stripped)
            if stripped == "_entity_poly_seq.mon_id":
                mon_id_col = len(headers) - 1
            elif stripped == "_atom_site.label_comp_id":
                comp_id_col = len(headers) - 1
            out.append(line)
            continue

        # transition from header to data
        if in_loop_header:
            in_loop_header = False

        if stripped == "" or stripped.startswith("#"):
            # End of loop body
            headers = []
            mon_id_col = None
            comp_id_col = None
            out.append(line)
            continue

        if mon_id_col is None and comp_id_col is None:
            out.append(line)
            continue

        cols = stripped.split()
        target_col = mon_id_col if mon_id_col is not None else comp_id_col
        if target_col is None or target_col >= len(cols):
            out.append(line)
            continue
        old = cols[target_col]
        if old in substitutions:
            cols[target_col] = substitutions[old]
            # Re-emit preserving leading indent
            indent = line[: len(line) - len(line.lstrip())]
            newline_char = "\n" if line.endswith("\n") else ""
            out.append(indent + " ".join(cols) + newline_char)
        else:
            out.append(line)

    return "".join(out)


def clean_template_for_boltz(input_path, output_cif, chain_id, moldir=None):
    """Strip + relabel a template chain for boltz consumption.

    Steps:
    1. extract_template_chain strips the input to a single polymer chain
       CIF (drops ligands, water, other chains). The output chain label
       becomes 'A'.
    2. Any residue in entity_poly_seq / atom_site whose CCD is missing from
       moldir and not a canonical token is renamed to the polymer's UNK
       token via text-level substitution on that output.

    Returns (label_chain, n_renamed). The output CIF's polymer is always
    addressed by label_chain='A' downstream.
    """
    moldir = Path(moldir) if moldir else DEFAULT_MOLDIR

    extract_template_chain(str(input_path), str(output_cif), chain_id)

    # Collect residue names produced by AF3's strip pass
    structure = gemmi.read_structure(str(output_cif))
    residue_names = []
    for model in structure:
        for chain in model:
            for residue in chain:
                residue_names.append(residue.name)
    canonical, unk = _classify(residue_names)

    substitutions = {}
    for name in set(residue_names):
        if name in canonical:
            continue
        if (moldir / f"{name}.pkl").exists():
            continue
        substitutions[name] = unk

    if substitutions:
        with open(output_cif) as f:
            text = f.read()
        text = _substitute_in_cif_text(text, substitutions)
        with open(output_cif, "w") as f:
            f.write(text)

    return "A", sum(1 for n in residue_names if n in substitutions)
