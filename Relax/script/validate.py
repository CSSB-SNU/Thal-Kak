"""Pre-relaxation structure validation / normalization.

Runs BEFORE openmm relaxation (invoked by ``Relax/relaxation.py`` as a
preprocessing pass) so relaxation starts from a consistent, chemically sane
structure instead of the relaxer re-deriving its own fixes. Two checks are
performed here:

1. ``canonicalize_prochiral_methyls`` — relabel Val ``CG1``/``CG2`` and Leu
   ``CD1``/``CD2`` (and their attached H) to Rosetta ``fa_standard`` handedness.
   The methyls are prochiral, so a predictor may emit either labeling; Rosetta's
   ``cart_bonded`` improper term penalizes the non-canonical one. Name
   unification only — no coordinates move.

2. ``fix_terminal_carboxylate`` — validate the whole protein C-terminal
   ``-COO(-)`` group (both C-O bonds, all three ~120deg angles, planarity), not
   just ``OXT``, and rebuild whatever is off to ideal sp2: a misplaced backbone
   ``O`` is fixed just like a misplaced/absent ``OXT``. When one oxygen is sound
   it is used as the reference for the other; when neither is, both are rebuilt
   from the backbone (N-CA-C plane). A distorted terminus is localized strain the
   minimizer cannot reliably reopen, so it is fixed up front.

This is a pure text-level PDB rewrite: only coordinates and atom names/lines
change. B-factors, occupancies and residue numbering are preserved, so the
downstream per-residue pLDDT keying (CA/C1' B-factor) is unaffected.

The only heavy dependency is PyRosetta, imported lazily and used solely to read
the canonical methyl handedness from ``fa_standard``; if it is unavailable the
methyl step is skipped with a warning (the C-terminal fix is dependency-free).

CLI::

    python validate.py -pdb_fn in.pdb -out out.pdb          # single file
    python validate.py -in_dir decoys/ -out_dir validated/  # whole directory
"""

import argparse
import glob
import math
import os

# --- C-terminal carboxylate geometry ------------------------------------------
CARBOXYLATE_BOND = 1.25  # A, ideal carboxylate C-O bond length
# An oxygen is "sound" (usable to rebuild its partner) when its C-O bond and
# CA-C-O angle fall in these bands; the angle band also gates the O-C-OXT angle.
OXY_BOND_MIN, OXY_BOND_MAX = 1.15, 1.40
OXY_ANGLE_MIN, OXY_ANGLE_MAX = 105.0, 140.0
COO_PLANAR_TOL = 0.35  # A, max distance of C off the plane of its 3 substituents
# Backbone atoms that mark a residue as a protein C-terminus candidate (the last
# such residue per chain). O/OXT are validated separately, so O is not required.
TERMINAL_BACKBONE = {"N", "CA", "C"}

# --- Prochiral methyl canonicalization ----------------------------------------
# (branch centre c, backbone ref r, methyl a, methyl b) per prochiral residue.
METHYLS = {
    "VAL": ("CB", "CA", "CG1", "CG2"),
    "LEU": ("CG", "CB", "CD1", "CD2"),
}


# --- small pure-python vector helpers (no numpy dependency) -------------------
def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a):
    return math.sqrt(_dot(a, a))


def _unit(a):
    n = _norm(a)
    return (a[0] / n, a[1] / n, a[2] / n) if n > 1e-8 else a


def _angle_deg(a, b, c):
    """Angle a-b-c at vertex b, in degrees."""
    u, v = _unit(_sub(a, b)), _unit(_sub(c, b))
    return math.degrees(math.acos(max(-1.0, min(1.0, _dot(u, v)))))


def signed_volume(c, r, a, b):
    """det[r-c, a-c, b-c]; sign is the prochiral handedness of the branch centre
    c (backbone ref r, methyls a/b). Chi-invariant, flips only on a label swap."""
    va, vb, vc = _sub(r, c), _sub(a, c), _sub(b, c)
    return (
        va[0] * (vb[1] * vc[2] - vb[2] * vc[1])
        - va[1] * (vb[0] * vc[2] - vb[2] * vc[0])
        + va[2] * (vb[0] * vc[1] - vb[1] * vc[0])
    )


def _sp2_third_vertex(C, p1, p2, bond):
    """Third bond of an sp2 centre C whose other two bonds point at p1 and p2:
    the unit vector opposite the bisector of C->p1 and C->p2, scaled by ``bond``
    (~120deg from each). ``bond`` is in the same length unit as the inputs."""
    u1 = _unit(_sub(p1, C))
    u2 = _unit(_sub(p2, C))
    bisector = _add(u1, u2)
    if _norm(bisector) < 1e-6:
        # p1 and p2 nearly antiparallel -> bisector undefined; use any in-plane
        # perpendicular instead.
        perp = _cross(u1, u2)
        direction = _unit(perp) if _norm(perp) > 1e-6 else _unit(_cross(u1, (0.0, 0.0, 1.0)))
    else:
        direction = _unit(bisector)
        direction = (-direction[0], -direction[1], -direction[2])
    return (
        C[0] + bond * direction[0],
        C[1] + bond * direction[1],
        C[2] + bond * direction[2],
    )


def sp2_oxt_position(C, O, CA, bond=CARBOXYLATE_BOND):
    """Ideal sp2 position of the terminal OXT given the backbone O and CA (both
    kept where they are): the third carboxyl-carbon bond, ~120deg from C-O and
    C-CA. Public because the OpenMM relaxer's post-PDBFixer OXT safety net reuses
    it (``bond`` in nm there, A here)."""
    return _sp2_third_vertex(C, O, CA, bond)


def _rebuild_coo_from_backbone(C, CA, N, bond=CARBOXYLATE_BOND):
    """Both carboxylate oxygens as an ideal planar sp2 -COO-, used when neither
    input oxygen can be trusted as a reference. They are placed symmetric about
    the C->CA reverse direction (each ~120deg from C-CA and from each other) and
    coplanar with N-CA-C. Returns (O, OXT)."""
    a = _unit(_sub(CA, C))
    nref = _sub(N, C)
    # in-plane component of C->N, perpendicular to the C-CA axis, sets the plane
    p = _sub(nref, (a[0] * _dot(nref, a), a[1] * _dot(nref, a), a[2] * _dot(nref, a)))
    if _norm(p) < 1e-6:  # N collinear with C-CA -> pick any perpendicular
        p = _cross(a, (0.0, 0.0, 1.0))
        if _norm(p) < 1e-6:
            p = _cross(a, (0.0, 1.0, 0.0))
    p = _unit(p)
    ca = math.cos(math.radians(120.0))
    sa = math.sin(math.radians(120.0))
    dO = (ca * a[0] + sa * p[0], ca * a[1] + sa * p[1], ca * a[2] + sa * p[2])
    dX = (ca * a[0] - sa * p[0], ca * a[1] - sa * p[1], ca * a[2] - sa * p[2])
    O = (C[0] + bond * dO[0], C[1] + bond * dO[1], C[2] + bond * dO[2])
    OXT = (C[0] + bond * dX[0], C[1] + bond * dX[1], C[2] + bond * dX[2])
    return O, OXT


def _oxygen_ok(C, X, CA):
    """True if oxygen X sits at a sane C-O bond length and CA-C-O angle."""
    if not (OXY_BOND_MIN <= _norm(_sub(X, C)) <= OXY_BOND_MAX):
        return False
    return OXY_ANGLE_MIN <= _angle_deg(CA, C, X) <= OXY_ANGLE_MAX


def _coo_ok(C, CA, O, OXT):
    """True if the whole carboxylate is well-formed: both oxygens sane, O-C-OXT in
    band, and C coplanar with its three substituents (CA, O, OXT)."""
    if not (_oxygen_ok(C, O, CA) and _oxygen_ok(C, OXT, CA)):
        return False
    if not (OXY_ANGLE_MIN <= _angle_deg(O, C, OXT) <= OXY_ANGLE_MAX):
        return False
    normal = _cross(_sub(O, CA), _sub(OXT, CA))
    if _norm(normal) < 1e-6:
        return False
    return abs(_dot(_sub(C, CA), _unit(normal))) <= COO_PLANAR_TOL


# --- PDB line helpers ---------------------------------------------------------
def _is_atom(line):
    return line.startswith("ATOM") or line.startswith("HETATM")


def _xyz(line):
    return (float(line[30:38]), float(line[38:46]), float(line[46:54]))


def _set_xyz(line, xyz):
    return line[:30] + "%8.3f%8.3f%8.3f" % (xyz[0], xyz[1], xyz[2]) + line[54:]


def _res_key(line):
    # chain + resSeq + iCode; matches the keying the relaxers use for pLDDT.
    return (line[21], line[22:26], line[26])


# --- 1. prochiral methyl name canonicalization --------------------------------
_CANON_SIGNS = None  # {resn: +1/-1}, computed once from fa_standard (lazy)


def _canonical_signs():
    """Canonical signed-volume sign per prochiral residue, read once from
    PyRosetta's ``fa_standard`` ideal residues. Returns {} (methyl step becomes a
    no-op) if PyRosetta cannot be imported/initialised."""
    global _CANON_SIGNS
    if _CANON_SIGNS is not None:
        return _CANON_SIGNS
    try:
        import pyrosetta
    except Exception as e:  # noqa: BLE001 - any import failure -> skip gracefully
        print(f"[validate] PyRosetta unavailable ({e}); skipping methyl canonicalization")
        _CANON_SIGNS = {}
        return _CANON_SIGNS

    opts = "-mute all -load_PDB_components false"
    try:
        pyrosetta.init(opts, silent=True)
    except TypeError:
        pyrosetta.init(opts)

    rosetta = pyrosetta.rosetta
    rts = rosetta.core.chemical.ChemicalManager.get_instance().residue_type_set(
        "fa_standard"
    )
    signs = {}
    for resn, (c, r, a, b) in METHYLS.items():
        res = rosetta.core.conformation.ResidueFactory.create_residue(
            rts.name_map(resn)
        )
        xyz = {nm: (res.xyz(nm).x, res.xyz(nm).y, res.xyz(nm).z) for nm in (c, r, a, b)}
        signs[resn] = 1 if signed_volume(xyz[c], xyz[r], xyz[a], xyz[b]) > 0 else -1
    _CANON_SIGNS = signs
    return signs


def _swap_methyl_name(resn, name):
    """Geminal partner name: CG1<->CG2 / CD1<->CD2 and the attached methyl
    hydrogens HG1x<->HG2x / HD1x<->HD2x; identity otherwise. Partner names are the
    same length, so PDB columns 13-16 stay aligned."""
    if resn == "VAL":
        if name == "CG1":
            return "CG2"
        if name == "CG2":
            return "CG1"
        if name.startswith("HG1"):
            return "HG2" + name[3:]
        if name.startswith("HG2"):
            return "HG1" + name[3:]
    elif resn == "LEU":
        if name == "CD1":
            return "CD2"
        if name == "CD2":
            return "CD1"
        if name.startswith("HD1"):
            return "HD2" + name[3:]
        if name.startswith("HD2"):
            return "HD1" + name[3:]
    return name


def canonicalize_prochiral_methyls(lines):
    """Relabel non-canonical Val/Leu methyl pairs to fa_standard handedness.
    Returns (new_lines, n_residues_relabeled)."""
    # Gather the four reference atoms of every prochiral residue first, so a
    # structure with no Val/Leu never pays the PyRosetta init cost.
    residues = {}
    for ln in lines:
        if not _is_atom(ln):
            continue
        resn = ln[17:20].strip()
        if resn not in METHYLS:
            continue
        name = ln[12:16].strip()
        key = (ln[21], ln[22:27])  # chain + resSeq + iCode
        residues.setdefault(key, {"_resn": resn})[name] = _xyz(ln)

    if not residues:
        return lines, 0

    signs = _canonical_signs()
    if not signs:
        return lines, 0

    # Flag residues whose observed handedness disagrees with the canonical sign.
    to_swap = set()
    for key, atoms in residues.items():
        c, r, a, b = METHYLS[atoms["_resn"]]
        if not all(nm in atoms for nm in (c, r, a, b)):
            continue  # incomplete side chain -- the relaxer rebuilds it anyway
        v = signed_volume(atoms[c], atoms[r], atoms[a], atoms[b])
        if (1 if v > 0 else -1) != signs[atoms["_resn"]]:
            to_swap.add(key)

    if not to_swap:
        return lines, 0

    out = []
    for ln in lines:
        if _is_atom(ln):
            resn = ln[17:20].strip()
            key = (ln[21], ln[22:27])
            if resn in METHYLS and key in to_swap:
                name = ln[12:16].strip()
                new = _swap_methyl_name(resn, name)
                if new != name:
                    ln = ln[:12] + ln[12:16].replace(name, new, 1) + ln[16:]
        out.append(ln)
    return out, len(to_swap)


# --- 2. C-terminal carboxylate reconstruction ---------------------------------
def _renumber_serials(lines):
    """Re-sequence ATOM/HETATM/TER serial numbers (columns 7-11). Called only
    after new atom lines are inserted, so the file stays a valid PDB."""
    serial = 0
    out = []
    for ln in lines:
        if _is_atom(ln):
            serial += 1
            ln = ln[:6] + "%5d" % (serial % 100000) + ln[11:]
        elif ln.startswith("TER"):
            serial += 1
            body = ln.rstrip("\n")
            if len(body) < 11:
                ln = "TER   %5d\n" % (serial % 100000)
            else:
                ln = ln[:6] + "%5d" % (serial % 100000) + ln[11:]
        out.append(ln)
    return out


def _oxygen_line(template_line, name, xyz, force_oxygen):
    """Build a carboxylate-oxygen ATOM line by cloning ``template_line`` (so it
    inherits occupancy/B-factor/chain/resSeq): set the atom name (cols 13-16) and
    coordinates, and force the element to O when cloning from a non-oxygen (C)."""
    ln = template_line[:12] + (" %-3s" % name) + template_line[16:]
    ln = _set_xyz(ln, xyz)
    if force_oxygen and len(ln.rstrip("\n")) >= 78:
        ln = ln[:76] + " O" + ln[78:]
    return ln


def fix_terminal_carboxylate(lines):
    """Validate and, where needed, rebuild each protein C-terminal carboxylate.

    Targets are, per chain, the last residue with an N/CA/C backbone, plus any
    residue already bearing an OXT. A well-formed COO group is left untouched;
    otherwise the sound oxygen (if any) rebuilds its partner, else both are
    rebuilt from the N-CA-C plane (see the module docstring; the branch comments
    below track each case).

    Returns (new_lines, n_created, n_repaired): oxygen atoms added, and termini
    whose existing oxygens were moved."""
    res_atoms = {}  # key -> {atom_name: line_index}
    chain_last_terminal = {}  # chain -> key of last N/CA/C-complete residue

    for i, ln in enumerate(lines):
        if not _is_atom(ln):
            continue
        key = _res_key(ln)
        res_atoms.setdefault(key, {})[ln[12:16].strip()] = i

    for key, atoms in res_atoms.items():  # insertion order == document order
        if TERMINAL_BACKBONE <= atoms.keys():
            chain_last_terminal[key[0]] = key

    targets = set(chain_last_terminal.values())
    targets |= {key for key, atoms in res_atoms.items() if "OXT" in atoms}

    created = repaired = 0
    inserts = {}  # anchor line_index -> [new_line, ...], applied after that line
    for key in sorted(targets, key=lambda k: (k[0], k[1])):
        atoms = res_atoms[key]
        if not ({"C", "CA"} <= atoms.keys()):
            continue
        C = _xyz(lines[atoms["C"]])
        CA = _xyz(lines[atoms["CA"]])
        haveO, haveX = "O" in atoms, "OXT" in atoms
        O = _xyz(lines[atoms["O"]]) if haveO else None
        X = _xyz(lines[atoms["OXT"]]) if haveX else None

        if haveO and haveX and _coo_ok(C, CA, O, X):
            continue  # well-formed carboxylate -- leave it

        okO = haveO and _oxygen_ok(C, O, CA)
        okX = haveX and _oxygen_ok(C, X, CA)

        new_o = new_x = None
        if okO:  # trust O (backbone carbonyl); rebuild/create OXT from it
            new_x = _sp2_third_vertex(C, O, CA, CARBOXYLATE_BOND)
        elif okX:  # O is the problem; rebuild it from the sound OXT
            new_o = _sp2_third_vertex(C, X, CA, CARBOXYLATE_BOND)
        elif "N" in atoms:  # neither oxygen usable -> rebuild both from backbone
            new_o, new_x = _rebuild_coo_from_backbone(C, CA, _xyz(lines[atoms["N"]]))
        elif haveO:  # last resort with no backbone N: use the (imperfect) O
            new_x = _sp2_third_vertex(C, O, CA, CARBOXYLATE_BOND)
        elif haveX:
            new_o = _sp2_third_vertex(C, X, CA, CARBOXYLATE_BOND)
        else:
            continue  # nothing to build a carboxylate from

        moved = False
        if new_o is not None:
            if haveO:
                lines[atoms["O"]] = _set_xyz(lines[atoms["O"]], new_o)
                moved = True
            else:
                tmpl = atoms["OXT"] if haveX else atoms["C"]
                inserts.setdefault(atoms["C"], []).append(
                    _oxygen_line(lines[tmpl], "O", new_o, force_oxygen=not haveX)
                )
                created += 1
        if new_x is not None:
            if haveX:
                lines[atoms["OXT"]] = _set_xyz(lines[atoms["OXT"]], new_x)
                moved = True
            else:
                anchor = tmpl = atoms["O"] if haveO else atoms["C"]
                inserts.setdefault(anchor, []).append(
                    _oxygen_line(lines[tmpl], "OXT", new_x, force_oxygen=not haveO)
                )
                created += 1
        if moved:
            repaired += 1

    if inserts:
        rebuilt = []
        for i, ln in enumerate(lines):
            rebuilt.append(ln)
            if i in inserts:
                rebuilt.extend(inserts[i])
        lines = _renumber_serials(rebuilt)

    return lines, created, repaired


# --- driver -------------------------------------------------------------------
def validate_lines(lines):
    """Apply both fixes to a list of PDB lines. Returns (new_lines, report)."""
    lines, n_methyl = canonicalize_prochiral_methyls(lines)
    lines, n_created, n_repaired = fix_terminal_carboxylate(lines)
    return lines, {
        "methyl": n_methyl,
        "coo_created": n_created,
        "coo_repaired": n_repaired,
    }


def validate_file(in_path, out_path):
    with open(in_path) as fp:
        lines = fp.readlines()
    lines, report = validate_lines(lines)
    with open(out_path, "w") as fp:
        fp.writelines(lines)
    return report


def _fmt(report):
    return (
        "methyl=%(methyl)d coo_created=%(coo_created)d coo_repaired=%(coo_repaired)d"
        % report
    )


def main():
    p = argparse.ArgumentParser(
        description="Pre-relaxation PDB validation (prochiral methyl names + "
        "C-terminal sp2 carboxylate).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-pdb_fn", type=str, help="input PDB (single-file mode)")
    p.add_argument("-out", type=str, help="output PDB (single-file mode)")
    p.add_argument("-in_dir", type=str, help="input directory of *.pdb (batch mode)")
    p.add_argument("-out_dir", type=str, help="output directory (batch mode)")
    p.add_argument(
        "-overwrite",
        action="store_true",
        help="reprocess even if the output already exists (default: skip existing)",
    )
    args = p.parse_args()

    if args.in_dir:
        if not args.out_dir:
            p.error("-out_dir is required with -in_dir")
        os.makedirs(args.out_dir, exist_ok=True)
        pdbs = sorted(glob.glob(os.path.join(args.in_dir, "*.pdb")))
        total = {"methyl": 0, "coo_created": 0, "coo_repaired": 0}
        for pdb in pdbs:
            out = os.path.join(args.out_dir, os.path.basename(pdb))
            if os.path.exists(out) and not args.overwrite:
                print(f"[validate] exists, skip: {os.path.basename(out)}")
                continue
            report = validate_file(pdb, out)
            print(f"[validate] {os.path.basename(pdb)}: {_fmt(report)}")
            for k in total:
                total[k] += report[k]
        print(f"[validate] done: {len(pdbs)} file(s), {_fmt(total)}")
    elif args.pdb_fn:
        if not args.out:
            p.error("-out is required with -pdb_fn")
        report = validate_file(args.pdb_fn, args.out)
        print(f"[validate] {os.path.basename(args.pdb_fn)}: {_fmt(report)}")
    else:
        p.error("provide either -pdb_fn/-out or -in_dir/-out_dir")


if __name__ == "__main__":
    main()
