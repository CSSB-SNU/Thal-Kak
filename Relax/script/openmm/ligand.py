import os
import urllib.request


RCSB_CCD_URL = "https://files.rcsb.org/ligands/download/{code}.cif"
_BOND_ORDER = {"SING": 1, "DOUB": 2, "TRIP": 3, "QUAD": 4}


def deps_available():
    """True if the ligand-parametrization stack is usable in this env. Importable
    is not enough: GAFF needs AM1-BCC charges, and the toolkit providing them
    (ambertools) only registers when antechamber/sqm are on PATH -- without it a
    ligand resolves and then aborts the decoy inside createSystem."""
    try:
        import rdkit  # noqa: F401
        import openmmforcefields  # noqa: F401
        from openff.toolkit.utils.toolkits import GLOBAL_TOOLKIT_REGISTRY
    except Exception:
        return False
    return any(
        "am1bcc" in (getattr(tk, "SUPPORTED_CHARGE_METHODS", None) or {})
        for tk in GLOBAL_TOOLKIT_REGISTRY.registered_toolkits
    )


def _apply_atom_names(off_mol, heavy_names):
    """Name the atoms (heavy in order, then H1, H2, ...) so they survive
    to_openmm(); otherwise openff emits generic names and the ligand's atom
    identity is lost in the output PDB."""
    heavy_iter = iter(heavy_names)
    h_count = 0
    for atom in off_mol.atoms:
        if atom.atomic_number > 1:
            atom.name = next(heavy_iter, atom.name)
        else:
            h_count += 1
            atom.name = f"H{h_count}"
    return off_mol


def _heavy_view(atom_names, elements, coords_ang):
    """The residue's heavy atoms as (names, element symbols, coords). Symbols
    keep their original case -- RDKit's Atom() rejects e.g. 'CL'."""
    idx = [i for i, el in enumerate(elements) if el.upper() != "H"]
    return (
        [atom_names[i] for i in idx],
        [elements[i] for i in idx],
        [coords_ang[i] for i in idx],
    )


def _fetch_ccd_cif(code, cache_dir):
    """Return the wwPDB CCD component CIF text for ``code`` (cache, else RCSB).
    Returns None on any failure (offline node, unknown code)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{code}.cif")
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    url = RCSB_CCD_URL.format(code=code)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "thalkak-relax"})
        txt = urllib.request.urlopen(req, timeout=15).read().decode()
    except Exception as e:
        print(f"  [ligand] CCD fetch failed for {code!r}: {str(e)[:60]}")
        return None
    with open(path, "w") as f:
        f.write(txt)
    return txt


def _rdkit_from_ccd(code, cache_dir):
    """Build an RDKit mol (with bond orders + formal charges, heavy atoms only)
    from the wwPDB CCD. Returns (mol, [atom_names]) or None."""
    txt = _fetch_ccd_cif(code, cache_dir)
    if txt is None:
        return None
    import gemmi
    from rdkit import Chem

    try:
        block = gemmi.cif.read_string(txt).sole_block()
        names = list(block.find_values("_chem_comp_atom.atom_id"))
        syms = list(block.find_values("_chem_comp_atom.type_symbol"))
        charges = list(block.find_values("_chem_comp_atom.charge")) or ["0"] * len(names)
        b1 = list(block.find_values("_chem_comp_bond.atom_id_1"))
        b2 = list(block.find_values("_chem_comp_bond.atom_id_2"))
        orders = list(block.find_values("_chem_comp_bond.value_order"))
        aroms = list(block.find_values("_chem_comp_bond.pdbx_aromatic_flag")) or ["N"] * len(b1)
    except Exception as e:
        print(f"  [ligand] CCD parse failed for {code!r}: {str(e)[:60]}")
        return None

    def _clean(v):
        return gemmi.cif.as_string(v).strip()

    # One guard for the whole build: an exotic type_symbol (Chem.Atom) or a
    # structure sanitize can't resolve must degrade to None, not raise.
    try:
        names = [_clean(n) for n in names]
        syms = [_clean(s) for s in syms]
        charges = [_clean(c) for c in charges]

        rw = Chem.RWMol()
        idx_of = {}
        for i, (nm, sym) in enumerate(zip(names, syms)):
            el = sym.capitalize() if len(sym) == 2 else sym.upper()
            atom = Chem.Atom(el)
            try:
                q = int(round(float(charges[i]))) if charges[i] not in (".", "?", "") else 0
            except ValueError:
                q = 0
            atom.SetFormalCharge(q)
            rw.AddAtom(atom)
            idx_of[nm] = i

        for a1, a2, order, arom in zip(b1, b2, orders, aroms):
            a1, a2, order, arom = _clean(a1), _clean(a2), _clean(order), _clean(arom)
            if a1 not in idx_of or a2 not in idx_of:
                continue
            bt = Chem.BondType.AROMATIC if arom == "Y" else {
                1: Chem.BondType.SINGLE,
                2: Chem.BondType.DOUBLE,
                3: Chem.BondType.TRIPLE,
                4: Chem.BondType.QUADRUPLE,
            }.get(_BOND_ORDER.get(order, 1), Chem.BondType.SINGLE)
            rw.AddBond(idx_of[a1], idx_of[a2], bt)
            if arom == "Y":
                rw.GetAtomWithIdx(idx_of[a1]).SetIsAromatic(True)
                rw.GetAtomWithIdx(idx_of[a2]).SetIsAromatic(True)

        mol = rw.GetMol()
        Chem.SanitizeMol(mol)
    except Exception as e:
        print(f"  [ligand] CCD build failed for {code!r}: {str(e)[:60]}")
        return None
    return mol, names


def _heavy_mol_from_smiles(smiles):
    """RDKit heavy-atom mol (bond orders + formal charges, H implicit) parsed
    from a SMILES string, or None if it can't be parsed."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"  [ligand] could not parse provided SMILES: {smiles[:60]}")
        return None
    return Chem.RemoveHs(mol)  # heavy atoms only (H stay implicit here)


def _heavy_mol_from_ccd(code, cache_dir):
    """RDKit heavy-atom mol (bond orders + formal charges) and its CCD atom names
    for an explicit CCD ``code``. Returns (mol, [names]) or None."""
    from rdkit import Chem

    res = _rdkit_from_ccd(code, cache_dir)
    if res is None:
        return None
    full, names = res
    keep_names = [
        names[i] for i, a in enumerate(full.GetAtoms()) if a.GetAtomicNum() > 1
    ]
    ed = Chem.RWMol(full)
    for i in sorted(
        (i for i, a in enumerate(full.GetAtoms()) if a.GetAtomicNum() == 1),
        reverse=True,
    ):
        # Carry each H over as a count on its neighbour: dropping it outright
        # leaves e.g. an aromatic N-H unkekulizable, and AddHs would not put it
        # back later.
        for nb in ed.GetAtomWithIdx(i).GetNeighbors():
            nb.SetNumExplicitHs(nb.GetNumExplicitHs() + 1)
        ed.RemoveAtom(i)
    heavy = ed.GetMol()
    try:
        Chem.SanitizeMol(heavy)
    except Exception as e:
        print(f"  [ligand] CCD {code}: heavy-atom sanitize failed: {str(e)[:60]}")
        return None
    return heavy, keep_names


def _assign_coords_to_template(
    heavy_mol, template_names, atom_names, elements, coords_ang, require_names=False
):
    """Map a heavy-atom template mol onto the PDB residue's coordinates, add H,
    return an openff Molecule (or None if this template is not that molecule).

    Gate: heavy-atom count and element composition must match. Correspondence is
    by atom name, else by index+element sequence -- the latter suppressed by
    ``require_names``, since for a resname guess the name match is the only
    evidence that the resname is not a placeholder colliding with a real CCD.
    """
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    from openff.toolkit.topology import Molecule

    pdb_names, pdb_elems, pdb_coords = _heavy_view(atom_names, elements, coords_ang)
    n = heavy_mol.GetNumAtoms()
    if n != len(pdb_names):
        return None
    tmpl_elems = [a.GetSymbol().upper() for a in heavy_mol.GetAtoms()]
    pdb_elems = [el.upper() for el in pdb_elems]
    if sorted(tmpl_elems) != sorted(pdb_elems):
        return None

    coord_by_name = dict(zip(pdb_names, pdb_coords))
    if template_names and all(nm in coord_by_name for nm in template_names):
        ordered_coords = [coord_by_name[nm] for nm in template_names]
        heavy_names = list(template_names)
    elif not require_names and tmpl_elems == pdb_elems:
        ordered_coords = list(pdb_coords)
        heavy_names = list(pdb_names)
    else:
        return None

    mol = Chem.Mol(heavy_mol)
    conf = Chem.Conformer(n)
    for i, (x, y, z) in enumerate(ordered_coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.RemoveAllConformers()
    mol.AddConformer(conf, assignId=True)

    mol = Chem.AddHs(mol, addCoords=True)
    try:
        off = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
    except Exception as e:
        print(f"  [ligand] openff from provided template failed: {str(e)[:60]}")
        return None
    return _apply_atom_names(off, heavy_names)


def _mol_from_ccd_and_coords(
    code, cache_dir, atom_names, elements, coords_ang, require_names=False
):
    """Fetch CCD ``code`` and map it onto the residue's coordinates."""
    built = _heavy_mol_from_ccd(code, cache_dir)
    if built is None:
        return None
    heavy, names = built
    return _assign_coords_to_template(
        heavy, names, atom_names, elements, coords_ang, require_names
    )


def _openff_from_smiles_and_coords(smiles, atom_names, elements, coords_ang):
    """Map a reference SMILES onto the residue's heavy-atom coordinates,
    independent of atom order/names: perceive connectivity from coordinates,
    then take bond orders / aromaticity / charges from the SMILES via
    AssignBondOrdersFromTemplate. Returns an openff Molecule (heavy + H) or None."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDetermineBonds
    from rdkit.Geometry import Point3D
    from openff.toolkit.topology import Molecule

    template = _heavy_mol_from_smiles(smiles)
    if template is None:
        return None

    names, elems, coords = _heavy_view(atom_names, elements, coords_ang)
    if template.GetNumAtoms() != len(names):
        return None
    if sorted(a.GetSymbol().upper() for a in template.GetAtoms()) != sorted(
        el.upper() for el in elems
    ):
        return None

    # Target: the PDB heavy atoms at their coordinates; perceive CONNECTIVITY
    # only (single bonds from distances). Bond ORDERS come from the template.
    rw = Chem.RWMol()
    for el in elems:
        rw.AddAtom(Chem.Atom(el))
    tgt = rw.GetMol()
    conf = Chem.Conformer(len(names))
    for i, (x, y, z) in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    tgt.AddConformer(conf, assignId=True)
    try:
        rdDetermineBonds.DetermineConnectivity(tgt)
        tgt = AllChem.AssignBondOrdersFromTemplate(template, tgt)  # SMILES bond orders
        # AssignBondOrdersFromTemplate leaves atoms as radicals; clear them so
        # sanitize refills implicit H from the (now correct) bond orders.
        for a in tgt.GetAtoms():
            a.SetNoImplicit(False)
            a.SetNumRadicalElectrons(0)
        Chem.SanitizeMol(tgt)
        tgt = Chem.AddHs(tgt, addCoords=True)
        mol = Molecule.from_rdkit(tgt, allow_undefined_stereo=True)
    except Exception as e:
        print(f"  [ligand] SMILES->coordinate mapping failed: {str(e)[:70]}")
        return None
    return _apply_atom_names(mol, names)


def resolve_residue(resname, atom_names, elements, coords_nm, cache_dir, specs=None):
    """Resolve one candidate residue into an openff Molecule (with a conformer),
    or None if it cannot be parametrized (caller freezes it). coords_nm is in
    nanometers.

    Order: explicit ``specs`` ({ccd|smiles} dicts, ``resname`` ignored) -> the
    resname as a CCD code. A spec only "takes" if its template maps onto this
    residue's atoms, so non-matching ones fall through. Chemistry is never
    guessed from coordinates alone: a predicted ligand carries no hydrogens, so
    nothing pins its bond orders and a perceived graph would be fiction.
    """
    coords_ang = [(x * 10.0, y * 10.0, z * 10.0) for (x, y, z) in coords_nm]
    n_heavy = sum(1 for el in elements if el.upper() != "H")

    for spec in specs or []:
        mol = src = None
        if spec.get("ccd"):
            mol = _mol_from_ccd_and_coords(
                spec["ccd"].strip(), cache_dir, atom_names, elements, coords_ang
            )
            src = f"provided CCD {spec['ccd']}"
        if mol is None and spec.get("smiles"):
            mol = _openff_from_smiles_and_coords(
                spec["smiles"], atom_names, elements, coords_ang
            )
            src = "provided SMILES"
        if mol is not None:
            print(
                f"  [ligand] {resname}: resolved via {src} "
                f"({mol.n_atoms} atoms incl. H)"
            )
            return mol
    if specs:
        print(
            f"  [ligand] {resname}: no provided spec matched this "
            f"{n_heavy}-heavy-atom residue; trying the resname as a CCD code."
        )

    mol = _mol_from_ccd_and_coords(
        resname.strip(), cache_dir, atom_names, elements, coords_ang, require_names=True
    )
    if mol is not None:
        print(f"  [ligand] {resname}: resolved via CCD ({mol.n_atoms} atoms incl. H)")
        return mol

    print(
        f"  [ligand] {resname}: no chemistry for this {n_heavy}-heavy-atom residue "
        f"-> exclusion-frozen (pass --data_yaml / --ligand_ccd / --ligand_smiles "
        f"to relax it)"
    )
    return None


def make_template_generator(molecules, cache=None):
    """Return a GAFF-2.11 template generator for the resolved molecules, or None.
    ``cache`` (JSON path) memoizes each molecule's GAFF typing + AM1-BCC charges
    -- the bulk of wall time -- by canonical SMILES, so decoys sharing a ligand
    reuse it. Safe across relaxation.py's serial per-decoy runs."""
    if not molecules:
        return None
    from openmmforcefields.generators import GAFFTemplateGenerator

    gen = GAFFTemplateGenerator(molecules=molecules, forcefield="gaff-2.11", cache=cache)
    return gen.generator
