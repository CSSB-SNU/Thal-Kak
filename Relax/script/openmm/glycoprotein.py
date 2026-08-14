"""Carbohydrate (glycan) retyping to GLYCAM_06j for all-atom OpenMM relaxation.

Adapted from OpenMMDL (``openmmdl/openmmdl_setup/glycoprotein.py``,
https://github.com/wolberlab/OpenMMDL): the GLYCAM residue / linkage / atom-name
tables (sugar codes, the substitution-position letter map, the HexNAc acetyl
renames, Asn->NLN handling) follow that module, which verifies them against
GLYCAM-Web/gmml. The retyping is reimplemented here on OpenMM ``Topology``
objects (geometry-based linkage perception + Modeller hydrogens) rather than
OpenMMDL's PDB-text / tleap path.

Used under the MIT License:

    Copyright (c) 2024 Valerij Talagayev, Yu Chen, Niklas Piet Doering &
    Leon Obendorf (Wolber lab)

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import os
import numpy as np

import openmm
from openmm import unit
from openmm import app as openmm_app


# CCD 3-letter code -> (GLYCAM 2-char stem, {CCD atom name -> GLYCAM atom name}).
# The stem fixes monosaccharide + anomeric config + D/L. Heavy-atom names of the
# pyranose ring (C1-C6, O2-O6, O5) already coincide with GLYCAM; only decorating
# groups (the N-acetyl of HexNAc) need renaming.
_HEXNAC_RENAME = {"C7": "C2N", "O7": "O2N", "C8": "CME"}
SUGAR_CCD = {
    # beta-/alpha-D-GlcpNAc (the two core GlcNAc of N-glycans)
    "NAG": ("YB", _HEXNAC_RENAME),
    "NDG": ("YA", _HEXNAC_RENAME),
    # beta-/alpha-D-GalpNAc
    "NGA": ("VB", _HEXNAC_RENAME),
    "A2G": ("VA", _HEXNAC_RENAME),
    # beta-D-Manp (core) / alpha-D-Manp (branches)
    "BMA": ("MB", {}),
    "MAN": ("MA", {}),
    # beta-/alpha-D-Galp
    "GAL": ("LB", {}),
    "GLA": ("LA", {}),
    # beta-/alpha-D-Glcp
    "BGC": ("GB", {}),
    "GLC": ("GA", {}),
}

# GLYCAM substitution-position letter for the set of linked (child-bearing) O's.
PREFIX = {
    frozenset(): "0",
    frozenset({2}): "2", frozenset({3}): "3", frozenset({4}): "4", frozenset({6}): "6",
    frozenset({2, 3}): "Z", frozenset({2, 4}): "Y", frozenset({2, 6}): "X",
    frozenset({3, 4}): "W", frozenset({3, 6}): "V", frozenset({4, 6}): "U",
    frozenset({2, 3, 4}): "T", frozenset({2, 3, 6}): "S",
    frozenset({2, 4, 6}): "R", frozenset({3, 4, 6}): "Q",
    frozenset({2, 3, 4, 6}): "P",
}

# heavy-heavy covalent cutoff (Angstrom) for bond / linkage perception
_BOND_CUT = 1.85
# C1 <-> protein acceptor (Asn ND2 / Ser OG / Thr OG1) cutoff (Angstrom)
_LINK_CUT = 2.0


def _data_dir():
    return os.path.join(os.path.dirname(openmm_app.__file__), "data")


def deps_available():
    """True if the GLYCAM force-field + hydrogen-definition data ship with the
    installed OpenMM (they do since 8.x)."""
    d = _data_dir()
    return os.path.exists(os.path.join(d, "amber14", "GLYCAM_06j-1.xml")) and \
        os.path.exists(os.path.join(d, "glycam-hydrogens.xml"))


def _centroid(coords):
    return np.mean(np.stack(coords), axis=0)


# --------------------------------------------------------------------------- #
# tree perception
# --------------------------------------------------------------------------- #
def _find_glycan_trees(topology, pos_ang):
    """Group sugar residues into covalently connected trees.

    Returns (trees, parent), where ``trees`` is a list of residue lists and
    ``parent[res] = (parent_res, position:int)`` records the glycosidic bond
    child.C1 -> parent.O<position>. A sugar with no parent is a tree root
    (reducing end).
    """
    sugars = [r for r in topology.residues() if r.name in SUGAR_CCD]
    if not sugars:
        return [], {}

    parent = {}
    for child in sugars:
        c1 = None
        for a in child.atoms():
            if a.name == "C1":
                c1 = pos_ang[a.index]
                break
        if c1 is None:
            continue
        best = None
        for cand in sugars:
            if cand is child:
                continue
            for a in cand.atoms():
                if a.name.startswith("O") and a.name[1:].isdigit():
                    d = float(np.linalg.norm(c1 - pos_ang[a.index]))
                    if d < _BOND_CUT and (best is None or d < best[2]):
                        best = (cand, int(a.name[1:]), d)
        if best is not None:
            parent[child] = (best[0], best[1])

    # union-find over the parent/child adjacency to form trees
    idx = {r: i for i, r in enumerate(sugars)}
    dsu = list(range(len(sugars)))

    def find(i):
        while dsu[i] != i:
            dsu[i] = dsu[dsu[i]]
            i = dsu[i]
        return i

    for child, (par, _pos) in parent.items():
        a, b = find(idx[child]), find(idx[par])
        dsu[a] = b
    groups = {}
    for r in sugars:
        groups.setdefault(find(idx[r]), []).append(r)
    return list(groups.values()), parent


def _glycam_name(residue, children_positions):
    """GLYCAM residue code for a sugar given the set of its child-bearing O
    positions. Returns None if the sugar or the substitution pattern is
    unsupported."""
    stem = SUGAR_CCD[residue.name][0]
    key = frozenset(children_positions)
    pref = PREFIX.get(key)
    if pref is None:
        return None
    # GlcNAc/GalNAc cannot be substituted at O2 (occupied by the N-acetyl); such
    # a pattern means our perception is wrong -> bail on this tree.
    if stem[0] in ("Y", "V") and 2 in key:
        return None
    return pref + stem


# --------------------------------------------------------------------------- #
# protein linkage
# --------------------------------------------------------------------------- #
def _find_protein_link(root, topology, pos_ang):
    """Find the protein acceptor a tree's reducing C1 is glycosidically bonded
    to. Returns (residue, atom, kind) or None. Only N-linked Asn is templated
    here; Ser/Thr O-glycans fall back to a free (ROH) cap."""
    c1 = None
    for a in root.atoms():
        if a.name == "C1":
            c1 = pos_ang[a.index]
            break
    if c1 is None:
        return None
    for res in topology.residues():
        if res.name not in ("ASN", "NLN"):
            continue
        for a in res.atoms():
            if a.name == "ND2":
                if float(np.linalg.norm(c1 - pos_ang[a.index])) < _LINK_CUT:
                    return (res, a, "N")
    return None


# --------------------------------------------------------------------------- #
# main entry
# --------------------------------------------------------------------------- #
def retype(topology, positions, verbose=True):
    """Rewrite recognized carbohydrate residues into GLYCAM form.

    Returns ``(topology, positions, restore)``: fully-recognized glycan trees are
    GLYCAM-named, protonated, glycosidically bonded, and (for N-glycans) linked to
    their Asn (->``NLN``); free reducing ends get an ``ROH`` cap. ``restore`` lets
    ``apply_restore`` put wwPDB nomenclature back. Unrecognized sugars are left
    untouched; on any error the original inputs are returned with empty restore.
    """
    try:
        return _retype_impl(topology, positions, verbose)
    except Exception as e:  # never destabilize the pipeline
        if verbose:
            print(f"  [glycam] retyping skipped (fell back to ligand path): "
                  f"{type(e).__name__}: {str(e)[:80]}")
        return topology, positions, {}


def apply_restore(topology, restore):
    """Rename GLYCAM residues/atoms back to their original wwPDB CCD identity in
    ``topology`` (in place), using the map produced by ``retype``. Applied just
    before writing the output so the relaxed structure uses standard
    nomenclature (``NAG``/``BMA``/``MAN``/``ASN`` rather than ``4YB``/``NLN``)."""
    for res in topology.residues():
        entry = restore.get((res.chain.id, res.id, res.insertionCode))
        if entry is None:
            continue
        res.name = entry["resname"]
        amap = entry["atoms"]
        if amap:
            for atom in res.atoms():
                if atom.name in amap:
                    atom.name = amap[atom.name]


def _retype_impl(topology, positions, verbose):
    pos_nm = np.asarray(positions.value_in_unit(unit.nanometer), dtype=np.float64)
    pos_ang = pos_nm * 10.0

    trees, parent = _find_glycan_trees(topology, pos_ang)
    if not trees:
        return topology, positions, {}

    # Decide GLYCAM identity for every sugar in every fully-supported tree.
    # children_of[res] = {positions on res that bear a child}
    children_of = {}
    for child, (par, position) in parent.items():
        children_of.setdefault(par, set()).add(position)

    handled_trees = []        # list of (tree, {res: glycam_name}, root, link)
    for tree in trees:
        if any(r.name not in SUGAR_CCD for r in tree):
            continue
        names, ok = {}, True
        for r in tree:
            gname = _glycam_name(r, children_of.get(r, set()))
            if gname is None:
                ok = False
                break
            names[r] = gname
        if not ok:
            continue
        roots = [r for r in tree if r not in parent]
        if len(roots) != 1:
            continue  # cyclic / malformed -> leave to fallback
        link = _find_protein_link(roots[0], topology, pos_ang)
        handled_trees.append((tree, names, roots[0], link))

    if not handled_trees:
        return topology, positions, {}

    handled_sugars = {r for t, _, _, _ in handled_trees for r in t}

    # restore[(chain_id, res_id, icode)] = {resname: <wwPDB CCD>, atoms: {glycam
    # atom name -> original name}}; consumed by apply_restore to put standard
    # nomenclature back in the output PDB.
    restore = {}

    # ---- build the glycan-only sub-topology (GLYCAM heavy atoms + ROH caps) --
    gtop = openmm_app.Topology()
    gchain = gtop.addChain("G")
    gpos_ang = []
    new_atom = {}          # (orig_atom) -> new atom  (for sugar heavy atoms)

    for tree, names, root, link in handled_trees:
        for r in tree:
            rename = SUGAR_CCD[r.name][1]
            nr = gtop.addResidue(names[r], gchain, r.id, r.insertionCode)
            restore[("G", r.id, r.insertionCode)] = {
                "resname": r.name,
                "atoms": {v: k for k, v in rename.items()},
            }
            for a in r.atoms():
                if a.element is not None and a.element.symbol == "H":
                    continue  # drop any input H; GLYCAM H added below
                gn = rename.get(a.name, a.name)
                na = gtop.addAtom(gn, a.element, nr)
                new_atom[a] = na
                gpos_ang.append(pos_ang[a.index])
        # free reducing end -> ROH cap (skip if N-linked to protein)
        if link is None:
            c1_atom = None
            for a in root.atoms():
                if a.name == "C1":
                    c1_atom = a
                    break
            ring = [pos_ang[a.index] for a in root.atoms()
                    if a.name in ("C1", "C2", "O5")]
            c1xyz = pos_ang[c1_atom.index]
            direction = c1xyz - _centroid(ring) if len(ring) >= 2 else np.array([1.4, 0, 0])
            n = np.linalg.norm(direction)
            direction = direction / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
            # share the reducing sugar's id so the anomeric O1/HO1 merge into
            # that sugar residue (its restore entry then covers them) in output.
            roh = gtop.addResidue("ROH", gchain, root.id, root.insertionCode)
            o1 = gtop.addAtom("O1", openmm_app.Element.getBySymbol("O"), roh)
            gpos_ang.append(c1xyz + 1.4 * direction)
            new_atom[("ROH", root)] = (o1, new_atom[c1_atom])

    # intra-residue bonds (geometry) + glycosidic bonds + ROH bonds
    gp = np.stack(gpos_ang)
    for r in gtop.residues():
        ats = list(r.atoms())
        for i in range(len(ats)):
            for j in range(i + 1, len(ats)):
                if np.linalg.norm(gp[ats[i].index] - gp[ats[j].index]) < _BOND_CUT:
                    gtop.addBond(ats[i], ats[j])
    for child, (par, position) in parent.items():
        if child not in handled_sugars or par not in handled_sugars:
            continue
        c1 = next((new_atom[a] for a in child.atoms() if a.name == "C1"), None)
        opar = next((new_atom[a] for a in par.atoms()
                     if a.name == f"O{position}"), None)
        if c1 is not None and opar is not None:
            gtop.addBond(c1, opar)
    for (_tag, root), (o1, c1) in [(k, v) for k, v in new_atom.items()
                                   if isinstance(k, tuple) and k[0] == "ROH"]:
        gtop.addBond(o1, c1)

    # ---- add GLYCAM-named hydrogens to the glycan sub-topology --------------
    d = _data_dir()
    openmm_app.Modeller.loadHydrogenDefinitions(os.path.join(d, "hydrogens.xml"))
    openmm_app.Modeller.loadHydrogenDefinitions(os.path.join(d, "glycam-hydrogens.xml"))
    gmod = openmm_app.Modeller(gtop, (gp / 10.0) * unit.nanometer)
    gmod.addHydrogens()
    gtop_h, gpos_h = gmod.topology, gmod.positions

    # ---- assemble final topology: protein (Asn->NLN) + protonated glycan ----
    keep = openmm_app.Modeller(topology, positions)

    # For each N-linked tree, pick the Asn ND2 hydrogen to remove (the one
    # nearest the reducing C1) using the ORIGINAL topology/coords -- before any
    # delete rebuilds the topology and invalidates these atom references.
    drop_h = []
    asn_convert = set()   # (chain.id, res.id) of Asn residues to rename -> NLN
    for tree, names, root, link in handled_trees:
        if link is None or link[2] != "N":
            continue
        asn_res = link[0]
        asn_convert.add((asn_res.chain.id, asn_res.id))
        c1 = next((pos_ang[a.index] for a in root.atoms() if a.name == "C1"), None)
        hds = [a for a in asn_res.atoms() if a.name in ("HD21", "HD22")]
        if c1 is not None and len(hds) == 2:
            drop_h.append(min(hds, key=lambda h: np.linalg.norm(pos_ang[h.index] - c1)))

    # remove the original sugars and the linked ND2 hydrogens in one pass
    keep.delete(list(handled_sugars) + drop_h)

    # rename converted Asn -> NLN, normalize the surviving ND2 H to HD21, and
    # index each NLN's ND2 for the protein-glycan bond.
    nd2_link_atom = {}
    for res in keep.topology.residues():
        if res.name == "ASN" and (res.chain.id, res.id) in asn_convert:
            res.name = "NLN"
            restore[(res.chain.id, res.id, res.insertionCode)] = {
                "resname": "ASN", "atoms": {}
            }
            surviving = [a for a in res.atoms() if a.name in ("HD21", "HD22")]
            if surviving:
                surviving[0].name = "HD21"
            nd2 = next((a for a in res.atoms() if a.name == "ND2"), None)
            if nd2 is not None:
                nd2_link_atom[(res.chain.id, res.id)] = nd2

    n_res_before = keep.topology.getNumResidues()
    keep.add(gtop_h, gpos_h)
    added = list(keep.topology.residues())[n_res_before:]

    # protein-glycan bonds: NLN.ND2 -> reducing sugar C1
    for tree, names, root, link in handled_trees:
        if link is None or link[2] != "N":
            continue
        nd2 = nd2_link_atom.get((link[0].chain.id, link[0].id))
        c1 = None
        for r in added:
            if r.id == root.id and r.name == names[root]:
                c1 = next((a for a in r.atoms() if a.name == "C1"), None)
                break
        if nd2 is not None and c1 is not None:
            keep.topology.addBond(nd2, c1)

    if verbose:
        summ = {}
        for _t, names, _r, link in handled_trees:
            for r, gn in names.items():
                summ[gn] = summ.get(gn, 0) + 1
        nlink = sum(1 for _t, _n, _r, l in handled_trees if l is not None)
        print(f"  [glycam] retyped {len(handled_sugars)} sugar residue(s) in "
              f"{len(handled_trees)} tree(s): {dict(sorted(summ.items()))}; "
              f"{nlink} N-linked to Asn(->NLN)")

    return keep.topology, keep.positions, restore
