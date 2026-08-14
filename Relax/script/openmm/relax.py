import argparse
import io
import json
import os
import sys
import time
import yaml
import numpy as np

import openmm
from openmm import unit
from openmm import app as openmm_app
from pdbfixer import PDBFixer

from ff_stack import make_forcefield
from violations_torch import find_violations_torch

# Shared pre-relaxation validators live one level up (Relax/script/validate.py);
# reuse the sp2 carboxylate geometry so it has a single source of truth.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from validate import sp2_oxt_position


ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms

# Explicit waters are dropped before building the system (solvent is implicit).
WATER_RESNAMES = {"HOH", "WAT", "H2O", "TIP", "TIP3", "TIP4", "SPC", "SOL", "DOD"}


class _Struct:
    """(topology, positions) holder used in place of PDBFile.

    Carries the live topology through minimization so GAFF-ligand bonds survive;
    a PDB round-trip would drop them and break template matching.
    """

    __slots__ = ("topology", "positions")

    def __init__(self, topology, positions):
        self.topology = topology
        self.positions = positions


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-pdb_fn", type=str, required=True)
    p.add_argument("-out_prefix", type=str, required=True)

    # restraint
    p.add_argument(
        "-lddt_cut",
        type=float,
        default=60.0,
        help="lDDT cutoff. Above -> hires params, below -> lowres params.",
    )
    p.add_argument(
        "-k_hires",
        type=float,
        default=10.0,
        help="spring constant for high-lDDT region (kcal/mol/A^2).",
    )
    p.add_argument(
        "-tol_hires",
        type=float,
        default=0.5,
        help="flat-bottom tolerance for high-lDDT region (A).",
    )
    p.add_argument(
        "-k_lowres",
        type=float,
        default=5.0,
        help="spring constant for low-lDDT region (kcal/mol/A^2).",
    )
    p.add_argument(
        "-tol_lowres",
        type=float,
        default=2.0,
        help="flat-bottom tolerance for low-lDDT region (A).",
    )
    p.add_argument(
        "-rst_set",
        type=str,
        default="non_hydrogen",
        choices=["non_hydrogen", "backbone", "c_alpha"],
    )

    # force field / solvent
    p.add_argument(
        "-implicit_solvent", type=str, default="obc2", choices=["obc2", "gbn2", "none"]
    )

    # minimizer
    p.add_argument("-max_iterations", type=int, default=2000)
    p.add_argument(
        "-tolerance",
        type=float,
        default=0.239,
        help="minimizer force tolerance in kcal/mol/A "
        "(0.239 = 10 kJ/mol/nm, OpenMM's default).",
    )
    p.add_argument(
        "-max_outer_iterations",
        type=int,
        default=3,
        help="violation-informed iterations (AF2 default 20, ColabFold 3).",
    )
    p.add_argument(
        "-platform",
        type=str,
        default="CUDA",
        choices=["CUDA", "CPU", "OpenCL", "Reference"],
    )

    p.add_argument(
        "-ccd_cache_dir",
        type=str,
        default=os.path.expanduser("~/.cache/thalkak/ccd"),
        help="cache dir for fetched wwPDB CCD component CIFs.",
    )
    # Explicit ligand chemistry, tried before the resname->CCD guess; overrides
    # the resname (often a placeholder like 'LIG' for SMILES-predicted ligands).
    p.add_argument(
        "-ligand_ccd",
        type=str,
        nargs="+",
        default=None,
        metavar="CCD",
        help="explicit wwPDB CCD code(s) for ligand(s) (e.g. STI), overriding "
        "resname; accepts multiple.",
    )
    p.add_argument(
        "-ligand_smiles",
        type=str,
        nargs="+",
        default=None,
        metavar="SMILES",
        help="explicit SMILES for ligand(s); bond graph is built from each and "
        "mapped onto the matching HETATM residue's coordinates; accepts multiple.",
    )
    p.add_argument(
        "-ligand_specs",
        type=str,
        default=None,
        help="JSON list of ligand specs [{ccd|smiles}, ...], each tried in order "
        "against every ligand residue (first that maps wins). Passed by the "
        "orchestrator from the data config's 'ligand' list (`thalkak relax "
        "--data_yaml`); -ligand_ccd/-ligand_smiles are a shorthand for direct "
        "use and are appended to this list.",
    )
    p.add_argument(
        "-relax_config",
        type=str,
        default=None,
        help="YAML of relax parameters; each value fills in the matching flag "
        "unless that flag was given on the command line.",
    )

    args = p.parse_args()

    if args.relax_config:
        with open(args.relax_config) as fp:
            cfg = yaml.safe_load(fp) or {}
        valid = {a.dest for a in p._actions}
        overrides = {}
        for key, value in cfg.items():
            if value is None:
                continue
            if key not in valid:
                print(f"  [relax] warning: unknown config key {key!r} ignored")
                continue
            overrides[key] = value
        p.set_defaults(**overrides)
        args = p.parse_args()

    # Normalize to one list -- -ligand_specs is JSON (already a list when it came
    # from the config yaml) and the shorthand flags fold into it.
    if isinstance(args.ligand_specs, str) and args.ligand_specs.strip():
        args.ligand_specs = json.loads(args.ligand_specs)
    elif not isinstance(args.ligand_specs, list):
        args.ligand_specs = []
    args.ligand_specs = (
        list(args.ligand_specs)
        + [{"ccd": c} for c in (args.ligand_ccd or [])]
        + [{"smiles": s} for s in (args.ligand_smiles or [])]
    )

    return args


def _res_key(residue):
    """Stable identity key for a residue: (chain_id, resSeq, insertionCode)."""
    return (
        (residue.chain.id or "").strip(),
        str(residue.id).strip(),
        (residue.insertionCode or "").strip(),
    )


def read_lddt_by_key(pdb_fn):
    """Read per-residue pLDDT from the B-factor column of CA / C1' atoms, keyed
    by residue identity (heterogens have no CA/C1' -> no entry -> lowres)."""
    lddt = {}
    with open(pdb_fn) as fp:
        for line in fp:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in ("CA", "C1'"):
                continue
            key = (line[21:22].strip(), line[22:26].strip(), line[26:27].strip())
            lddt[key] = float(line[60:66])
    return lddt


def read_bfactor_by_key(pdb_fn):
    """Per-residue input B-factor (first atom's), keyed by residue identity.
    Keeps the column for residues pLDDT does not cover -- heterogens have no
    CA/C1', and openff rebuilds a resolved ligand from scratch, so without this
    their B-factors would be written as 0.00."""
    bfac = {}
    with open(pdb_fn) as fp:
        for line in fp:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            key = (line[21:22].strip(), line[22:26].strip(), line[26:27].strip())
            if key not in bfac:
                bfac[key] = float(line[60:66])
    return bfac


def build_lddt_map(pdb_fn, fixed_topology):
    """Per-residue pLDDT keyed to the FIXED topology's residue identity.

    fix_pdb canonicalizes chain IDs (A, B, ...) and resSeq (1..N), so input-keyed
    pLDDT would no longer match. Instead read input pLDDT in document order and
    assign it positionally to the fixed topology's polymer residues (those with a
    CA/C1'). Predicted inputs have no gaps so the correspondence is exact; on a
    count mismatch fall back to the input-key map and warn.
    """
    ordered = []
    with open(pdb_fn) as fp:
        for line in fp:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[12:16].strip() not in ("CA", "C1'"):
                continue
            ordered.append(float(line[60:66]))

    poly = [
        r
        for r in fixed_topology.residues()
        if any(a.name in ("CA", "C1'") for a in r.atoms())
    ]
    if len(poly) != len(ordered):
        print(
            f"  WARNING: pLDDT residue count mismatch (input {len(ordered)} "
            f"CA/C1' vs topology {len(poly)} polymer residues); falling back to "
            f"input-key mapping -- some residues may default to lowres/0.00"
        )
        return read_lddt_by_key(pdb_fn)

    return {_res_key(r): v for r, v in zip(poly, ordered)}


def _strip_5prime_phosphate(topology, positions):
    """amber templates expect a 5'-OH terminus for RNA/DNA. If the input has a
    5'-phosphate on the first residue of an NA chain, remove the whole phosphate
    group so the chain can be templated. Returns new (topology, positions)."""
    # "N"/"DN"/"DX" are the ambiguous bases written for an 'N' in the input
    # sequence (protenix writes DN, chai-1 writes DX).
    NA_RES = {"A", "U", "G", "C", "T", "N", "DA", "DT", "DG", "DC", "DN", "DX"}
    REMOVE_ATOM_NAMES = {
        "P",
        "OP1",
        "OP2",
        "OP3",
        "O1P",
        "O2P",
        "O3P",
        "HOP2",
        "HOP3",
        "H5T",
    }

    drop_ids = set()
    for chain in topology.chains():
        first_residue = None
        for res in chain.residues():
            first_residue = res
            break
        if first_residue is None:
            continue
        if first_residue.name not in NA_RES:
            continue
        for atom in first_residue.atoms():
            if atom.name in REMOVE_ATOM_NAMES:
                drop_ids.add(atom.index)

    if not drop_ids:
        return topology, positions

    new_top = openmm_app.Topology()
    new_pos_nm = []
    atom_map = {}
    for chain in topology.chains():
        new_chain = new_top.addChain(chain.id)
        for res in chain.residues():
            new_res = new_top.addResidue(res.name, new_chain, res.id, res.insertionCode)
            for atom in res.atoms():
                if atom.index in drop_ids:
                    continue
                new_atom = new_top.addAtom(atom.name, atom.element, new_res)
                atom_map[atom] = new_atom
                vec = positions[atom.index].value_in_unit(unit.nanometer)
                new_pos_nm.append([vec[0], vec[1], vec[2]])
    for bond in topology.bonds():
        a1, a2 = bond[0], bond[1]
        if a1 in atom_map and a2 in atom_map:
            new_top.addBond(atom_map[a1], atom_map[a2])

    new_pos = np.array(new_pos_nm, dtype=np.float64) * unit.nanometer
    print(f"  stripped {len(drop_ids)} 5'-phosphate atoms (OP3/HOP3/H5T)")
    return new_top, new_pos


def _delete_water(fixer):
    """Remove explicit water residues (implicit solvent replaces them)."""
    waters = [r for r in fixer.topology.residues() if r.name in WATER_RESNAMES]
    if not waters:
        return
    modeller = openmm_app.Modeller(fixer.topology, fixer.positions)
    modeller.delete(waters)
    fixer.topology = modeller.topology
    fixer.positions = modeller.positions
    print(f"  dropped {len(waters)} explicit water residue(s)")


def _place_terminal_oxt(topology, positions, strain_cut=110.0):
    """Reconstruct collapsed C-terminal OXT atoms at ideal sp2 geometry.

    PDBFixer.addMissingAtoms() sometimes builds OXT on top of the backbone
    carbonyl O for distorted termini, and the minimizer can't reliably reopen
    that localized strain. For each strained OXT (O-C-OXT < ``strain_cut``) place
    it as the sp2 third vertex (opposite the CA/O bisector, ~120deg). Well-placed
    OXT atoms are left untouched.
    """
    C_OXT_BOND_NM = 0.125  # 1.25 A carboxylate C-O bond
    coords = np.array(positions.value_in_unit(unit.nanometer), dtype=np.float64)
    per_res = {}
    for atom in topology.atoms():
        if atom.name in ("CA", "C", "O", "OXT"):
            per_res.setdefault(atom.residue.index, {})[atom.name] = atom.index

    def _unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else v

    fixed = 0
    for a in per_res.values():
        if not {"CA", "C", "O", "OXT"} <= a.keys():
            continue
        C, O, CA, OXT = (coords[a[k]] for k in ("C", "O", "CA", "OXT"))
        u_o = _unit(O - C)
        u_oxt = _unit(OXT - C)
        ang = np.degrees(np.arccos(np.clip(np.dot(u_o, u_oxt), -1.0, 1.0)))
        if ang >= strain_cut:
            continue  # already a sensible carboxylate; leave it
        # sp2 third vertex, opposite the CA/O bisector -- shared geometry with the
        # pre-relaxation validator (validate.sp2_oxt_position); bond in nm here.
        coords[a["OXT"]] = sp2_oxt_position(C, O, CA, bond=C_OXT_BOND_NM)
        fixed += 1

    if fixed:
        print(f"[oxt] reconstructed {fixed} collapsed C-terminal OXT atom(s) "
              f"(O-C-OXT < {strain_cut:.0f}deg) to sp2 geometry")
    new_positions = unit.Quantity(
        [openmm.Vec3(float(x), float(y), float(z)) for x, y, z in coords],
        unit.nanometer,
    )
    return new_positions


def fix_pdb(pdb_fn):
    """PDBFixer clean-up that KEEPS heterogens (ligands/ions/glycans).

    Only water is dropped (implicit solvent). Missing atoms/hydrogens are added
    for residues PDBFixer recognizes; unknown residues are left as-is and get
    handled downstream (templated by a generator, or exclusion-frozen).
    """
    fixer = PDBFixer(filename=pdb_fn)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()

    # Keep heterogens (ligands/ions/glycans); drop only water (implicit solvent).
    _delete_water(fixer)

    new_top, new_pos = _strip_5prime_phosphate(fixer.topology, fixer.positions)
    fixer.topology = new_top
    fixer.positions = new_pos

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.0)

    # Repair any C-terminal OXT that PDBFixer built collapsed onto the backbone
    # O (see _place_terminal_oxt); the minimizer cannot reliably fix this later.
    fixer.positions = _place_terminal_oxt(fixer.topology, fixer.positions)

    # keepIds=True preserves the input chain/resSeq through the round-trip (else
    # OpenMM renumbers to A../1..N), so pLDDT keys stay stable under residue
    # reordering and the output keeps the prediction's numbering.
    buf = io.StringIO()
    openmm_app.PDBFile.writeFile(fixer.topology, fixer.positions, buf, keepIds=True)
    buf.seek(0)
    return openmm_app.PDBFile(buf)


def select_restrained(atom, rst_set):
    """Return True if this atom should be restrained under the given set."""
    if rst_set == "non_hydrogen":
        return atom.element is not None and atom.element.symbol != "H"
    elif rst_set == "backbone":
        return atom.name in (
            "N",
            "CA",
            "C",
            "O",
            "P",
            "OP1",
            "OP2",
            "O5'",
            "C5'",
            "C4'",
            "C3'",
            "O3'",
        )
    elif rst_set == "c_alpha":
        return atom.name in ("CA", "C1'")
    return False


def add_flat_bottom_restraint(
    system,
    ref_pdb,
    lddt_map,
    lddt_cut,
    k_hires,
    tol_hires,
    k_lowres,
    tol_lowres,
    rst_set,
    exclude_residues,
):
    """Flat-bottom harmonic restraint keyed to per-residue lDDT.

    ``lddt_map`` maps residue-identity keys to pLDDT; residues without an entry
    (ligands/ions/glycans, which have no CA/C1') fall to the lowres params.
    Returns the number of restrained particles.
    """
    # +1e-8 nm^2 under the sqrt regularizes the d=0 gradient (0/0 -> NaN would
    # otherwise crash minimizeEnergy at iteration 0, where d==0 for every atom).
    energy_expr = (
        "0.5 * k * step(d - tol) * (d - tol)^2;"
        "d = sqrt((x-x0)^2 + (y-y0)^2 + (z-z0)^2 + 1e-8)"
    )
    force = openmm.CustomExternalForce(energy_expr)
    force.addPerParticleParameter("k")
    force.addPerParticleParameter("tol")
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")

    k_unit = unit.kilojoules_per_mole / (unit.nanometer**2)
    len_unit = unit.nanometer

    k_hires_u = (k_hires * ENERGY / (LENGTH**2)).value_in_unit(k_unit)
    k_lowres_u = (k_lowres * ENERGY / (LENGTH**2)).value_in_unit(k_unit)
    tol_hires_u = (tol_hires * LENGTH).value_in_unit(len_unit)
    tol_lowres_u = (tol_lowres * LENGTH).value_in_unit(len_unit)

    n_added = 0
    positions = ref_pdb.positions
    for atom in ref_pdb.topology.atoms():
        if atom.residue.index in exclude_residues:
            continue
        if not select_restrained(atom, rst_set):
            continue

        plddt = lddt_map.get(_res_key(atom.residue), -1.0)
        if plddt > lddt_cut:
            k_val, tol_val = k_hires_u, tol_hires_u
        else:
            k_val, tol_val = k_lowres_u, tol_lowres_u

        x0, y0, z0 = positions[atom.index].value_in_unit(len_unit)
        force.addParticle(atom.index, [k_val, tol_val, x0, y0, z0])
        n_added += 1

    system.addForce(force)
    return n_added


def build_system(topology, implicit_solvent, template_generators=None):
    """Build an OpenMM System with the all-atom FF stack (callers must
    exclusion-freeze untemplatable residues first). A ValueError from createSystem
    means the GB model lacks radii for some atom type -> fall back to vacuum.
    Returns (system, effective_solvent).
    """
    if implicit_solvent == "none":
        nonbonded, kwargs = openmm_app.NoCutoff, {}
    else:
        nonbonded = openmm_app.CutoffNonPeriodic
        kwargs = {"nonbondedCutoff": 1.0 * unit.nanometer}

    ff = make_forcefield(implicit_solvent, template_generators)
    try:
        system = ff.createSystem(
            topology,
            nonbondedMethod=nonbonded,
            constraints=openmm_app.HBonds,
            rigidWater=True,
            **kwargs,
        )
        return system, implicit_solvent
    except ValueError as e:
        if implicit_solvent == "none":
            raise
        print(
            f"  WARNING: implicit solvent '{implicit_solvent}' has no parameters "
            f"for some atom type ({str(e).splitlines()[0][:80]}); "
            f"falling back to vacuum"
        )
        ff2 = make_forcefield("none", template_generators)
        system = ff2.createSystem(
            topology,
            nonbondedMethod=openmm_app.NoCutoff,
            constraints=openmm_app.HBonds,
            rigidWater=True,
        )
        return system, "none"


def _strip_hydrogens(topology, positions):
    """Drop every hydrogen -> heavy-atom-only (topology, positions).

    The relaxed structure is written heavy-atom-only, consistent with the
    pyrosetta method's output.
    """
    modeller = openmm_app.Modeller(topology, positions)
    hydrogens = [
        a for a in topology.atoms()
        if a.element is not None and a.element.symbol == "H"
    ]
    if hydrogens:
        modeller.delete(hydrogens)
    return modeller.topology, modeller.positions


def overwrite_bfactor_with_lddt(pdb_str, lddt_map, input_bfactor=None):
    """Rewrite the B-factor column with per-residue pLDDT (keyed by identity),
    falling back to the input B-factor for residues pLDDT does not cover
    (ligands/ions/glycans), then 0.00."""
    input_bfactor = input_bfactor or {}
    lines_out = []
    for line in pdb_str.splitlines(keepends=True):
        if not line.startswith(("ATOM", "HETATM")):
            lines_out.append(line)
            continue
        key = (line[21:22].strip(), line[22:26].strip(), line[26:27].strip())
        bf = lddt_map.get(key, input_bfactor.get(key, 0.0))
        lines_out.append(line[:60] + f"{bf:6.2f}" + line[66:])
    return "".join(lines_out)


def run_minimize(
    pdb, ref_pdb, lddt_map, args, exclude_residues, template_generators, solvent
):
    """Single minimization pass. Returns (einit, efinal, next_pdb, effective_solvent).

    ``solvent`` is the implicit-solvent model to request this pass. The caller
    threads the previous pass's *effective* solvent through, so once build_system
    has fallen back to vacuum (GB radii missing) we stop re-attempting the failing
    GB model on every outer iteration.
    """
    system, effective_solvent = build_system(
        pdb.topology, solvent, template_generators
    )

    n_restrained = 0
    if args.k_hires > 0 or args.k_lowres > 0:
        n_restrained = add_flat_bottom_restraint(
            system,
            ref_pdb,
            lddt_map,
            args.lddt_cut,
            args.k_hires,
            args.tol_hires,
            args.k_lowres,
            args.tol_lowres,
            args.rst_set,
            exclude_residues,
        )

    integrator = openmm.LangevinIntegrator(
        0 * unit.kelvin, 1.0 / unit.picosecond, 1.0 * unit.femtosecond
    )
    try:
        platform = openmm.Platform.getPlatformByName(args.platform)
    except Exception as e:
        print(
            f"  WARNING: platform '{args.platform}' unavailable ({e}), "
            f"falling back to fastest available"
        )
        platform = None
    if platform is not None:
        sim = openmm_app.Simulation(pdb.topology, system, integrator, platform)
    else:
        sim = openmm_app.Simulation(pdb.topology, system, integrator)
    actual_platform = sim.context.getPlatform().getName()
    print(f"  platform: {actual_platform}")
    sim.context.setPositions(pdb.positions)

    state0 = sim.context.getState(getEnergy=True)
    einit = state0.getPotentialEnergy().value_in_unit(ENERGY)

    force_tol = args.tolerance * ENERGY / LENGTH
    t_min0 = time.time()
    sim.minimizeEnergy(maxIterations=args.max_iterations, tolerance=force_tol)
    print(f"  minimize: {time.time() - t_min0:.1f}s")

    state1 = sim.context.getState(getEnergy=True, getPositions=True)
    efinal = state1.getPotentialEnergy().value_in_unit(ENERGY)
    positions = state1.getPositions()

    print(
        f"  restrained {n_restrained} atoms | "
        f"E_init={einit:.2f}  E_final={efinal:.2f} kcal/mol  "
        f"excluded_residues={len(exclude_residues)}"
    )

    # Reuse the live topology (bonds intact) for the next iteration; a PDB
    # round-trip would drop GAFF-ligand bonds.
    next_pdb = _Struct(sim.topology, positions)
    return einit, efinal, next_pdb, effective_solvent


def _relabel_added_residues(topology, start_res_index, src_res):
    """Rename residues added at/after ``start_res_index`` to match ``src_res``.

    ``Modeller.add`` appends the openff ligand as "UNK" on a new chain; copy the
    original name/id/insertion-code/chain back so it keeps its wwPDB identity.
    """
    src_chain_id = (src_res.chain.id or "").strip()
    for i, res in enumerate(topology.residues()):
        if i < start_res_index:
            continue
        res.name = src_res.name
        res.id = src_res.id
        res.insertionCode = src_res.insertionCode
        res.chain.id = src_chain_id


def _reassert_interchain_bonds(ref_topology, modeller):
    """Re-add inter-chain bonds that ``Modeller.delete`` silently drops (e.g. the
    GLYCAM protein-ND2 <-> glycan-C1 bond). Any inter-chain bond in
    ``ref_topology`` whose endpoints both survive in ``modeller.topology`` is
    restored (matched by chain/residue/icode/atom name). Returns the count.
    """
    index = {
        (a.residue.chain.id, a.residue.id, a.residue.insertionCode, a.name): a
        for a in modeller.topology.atoms()
    }
    existing = {frozenset((b[0].index, b[1].index)) for b in modeller.topology.bonds()}
    n = 0
    for b in ref_topology.bonds():
        a1, a2 = b[0], b[1]
        if a1.residue.chain.id == a2.residue.chain.id:
            continue
        na1 = index.get((a1.residue.chain.id, a1.residue.id, a1.residue.insertionCode, a1.name))
        na2 = index.get((a2.residue.chain.id, a2.residue.id, a2.residue.insertionCode, a2.name))
        if na1 is None or na2 is None:
            continue
        if frozenset((na1.index, na2.index)) in existing:
            continue
        modeller.topology.addBond(na1, na2)
        n += 1
    return n


def _build_relax_system(fixed, args):
    """Partition the fixed structure into the minimizable system + the frozen
    remainder.

    Every component is relaxed with its native force field where possible:
    protein/NA via amber19-all, carbohydrates via GLYCAM_06j (bonded glycans,
    N-linked to Asn), ligands via GAFF-2.11. Only residues that cannot be typed
    (sugars GLYCAM can't map, ligands GAFF can't parametrize) are exclusion-frozen
    -- kept at their input coords in the output but excluded from minimization --
    rather than cross-typed or dropped.

    Returns (sys_pdb, template_generators, frozen_top, frozen_pos, glycam_restore).
    """
    import glycoprotein

    freeze = []

    # Carbohydrates: retype recognized CCD sugars to GLYCAM_06j so they relax as
    # bonded glycan chains. Sugars not retyped (GLYCAM data/typing unavailable)
    # fall into the candidates below and are frozen.
    glycam_restore = {}
    if glycoprotein.deps_available():
        top, pos, glycam_restore = glycoprotein.retype(fixed.topology, fixed.positions)
        fixed = _Struct(top, pos)
    else:
        print("  [carbohydrate] GLYCAM data unavailable -> sugars frozen")

    base_ff = make_forcefield(args.implicit_solvent, None)
    candidates = list(base_ff.getUnmatchedResidues(fixed.topology))

    # Leftover sugars are carbohydrates whose GLYCAM FF wasn't applied -> frozen;
    # the rest are small-molecule ligands.
    freeze += [r for r in candidates if r.name in glycoprotein.SUGAR_CCD]
    lig_candidates = [r for r in candidates if r.name not in glycoprotein.SUGAR_CCD]

    # Ligands: GAFF-parametrize, else freeze.
    resolved, generators = [], []
    if lig_candidates:
        import ligand  # lazy: heavy deps live behind this module's functions

        if ligand.deps_available():
            print(
                f"[ligand] {len(lig_candidates)} candidate residue(s): "
                f"{sorted({r.name for r in lig_candidates})}"
            )
            pos = fixed.positions
            for res in lig_candidates:
                atoms = list(res.atoms())
                mol = ligand.resolve_residue(
                    res.name,
                    [a.name for a in atoms],
                    [a.element.symbol for a in atoms],
                    [pos[a.index].value_in_unit(unit.nanometer) for a in atoms],
                    args.ccd_cache_dir,
                    specs=args.ligand_specs or None,
                )
                if mol is not None:
                    resolved.append((res, mol))
                else:
                    freeze.append(res)
        else:
            print(
                "  [ligand] openff/GAFF stack or its AM1-BCC toolkit "
                "(ambertools on PATH) unavailable -> frozen"
            )
            freeze += lig_candidates

    # frozen sub-structure = every untypable residue (kept at original coords)
    freeze = list({id(r): r for r in freeze}.values())
    frozen_top = frozen_pos = None
    if freeze:
        print(
            f"[freeze] {len(freeze)} residue(s) excluded from minimization, "
            f"coords preserved: {sorted({r.name for r in freeze})}"
        )
        freeze_ids = {id(r) for r in freeze}
        fm = openmm_app.Modeller(fixed.topology, fixed.positions)
        fm.delete([r for r in fixed.topology.residues() if id(r) not in freeze_ids])
        frozen_top, frozen_pos = fm.topology, fm.positions

    # system = fixed minus every candidate, plus the openff-resolved ligands
    # (with H). Bonds are preserved (no PDB round-trip) so generators match.
    sm = openmm_app.Modeller(fixed.topology, fixed.positions)
    sm.delete([res for res, _ in resolved] + list(freeze))
    if resolved:
        from openff.units.openmm import to_openmm as _q_to_openmm

        for _res, mol in resolved:
            n_before = sm.topology.getNumResidues()
            sm.add(mol.to_topology().to_openmm(), _q_to_openmm(mol.conformers[0]))
            # openff names the re-added residue "UNK" on a fresh chain; restore
            # its original CCD identity so it stays identifiable in the output.
            _relabel_added_residues(sm.topology, n_before, _res)
        # Cache the GAFF/AM1-BCC parametrization (keyed by canonical SMILES) so
        # decoys sharing a ligand reuse it -- N antechamber runs become 1.
        cache = os.path.join(os.path.dirname(args.out_prefix), ".gaff_cache.json")
        generators = [
            ligand.make_template_generator([m for _, m in resolved], cache=cache)
        ]
        print(
            f"[ligand] parametrized {len(resolved)} ligand(s) with GAFF-2.11 "
            f"(template cache: {cache})"
        )

    # Modeller.delete above drops inter-chain bonds (e.g. the GLYCAM protein<->
    # glycan N-glycosidic bond); restore them from the pre-delete topology.
    _reassert_interchain_bonds(fixed.topology, sm)

    sys_pdb = _Struct(sm.topology, sm.positions)
    return sys_pdb, generators, frozen_top, frozen_pos, glycam_restore


def _detect_violation_device():
    """Probe torch CUDA independently of OpenMM's CUDA; fall back to CPU."""
    import torch as _torch

    if _torch.cuda.is_available():
        try:
            _ = _torch.zeros(1, device="cuda")
            return "cuda"
        except Exception:
            return "cpu"
    return "cpu"


def main():
    args = get_args()

    out_pdb = f"{args.out_prefix}.pdb"
    if os.path.exists(out_pdb):
        print(f"Output exists: {out_pdb} -- skipping")
        return

    print(f"Input: {args.pdb_fn}")
    print(
        f"FF: amber19-all + GLYCAM_06j-1 + ions | implicit={args.implicit_solvent}"
    )
    print(f"Restraint: flat-bottom harmonic, set={args.rst_set}")
    print(f"  hires (lDDT>{args.lddt_cut}): k={args.k_hires}, tol={args.tol_hires}")
    print(f"  lowres (lDDT<={args.lddt_cut}): k={args.k_lowres}, tol={args.tol_lowres}")

    t0 = time.time()

    # 1. fix & hydrogenate, keeping heterogens (only water dropped, 5'-P stripped)
    fixed = fix_pdb(args.pdb_fn)
    print(
        f"[fix_pdb] {time.time() - t0:.1f}s, "
        f"{fixed.topology.getNumAtoms()} atoms, "
        f"{fixed.topology.getNumResidues()} residues"
    )

    # 2. per-residue pLDDT keyed to the FIXED (canonicalized) topology (see
    #    build_lddt_map); restraint lookup and output B-factor share one keyspace.
    lddt_map = build_lddt_map(args.pdb_fn, fixed.topology)
    if lddt_map:
        vals = np.array(list(lddt_map.values()))
        print(
            f"[lddt] {len(vals)} polymer residues, mean={vals.mean():.1f}, "
            f"frac>{args.lddt_cut}={np.mean(vals > args.lddt_cut):.2f}"
        )

    # 3. build the relaxable system: resolve ligands to generators, freeze the
    #    unparametrizable remainder (kept at original coords in the output).
    sys_pdb, template_generators, frozen_top, frozen_pos, glycam_restore = (
        _build_relax_system(fixed, args)
    )
    if sys_pdb.topology.getNumAtoms() == 0:
        raise RuntimeError("no force-field-templatable atoms to relax")

    # immutable restraint reference: same topology, initial positions snapshot
    ref_pdb = _Struct(sys_pdb.topology, sys_pdb.positions)

    # 4. violation-informed iterative minimization (on the templatable system)
    exclude_residues = set()
    einit_first = None
    effective_solvent = args.implicit_solvent
    pdb = sys_pdb
    for it in range(args.max_outer_iterations):
        print(f"\n[iter {it}]")
        # pass the previous pass's *effective* solvent: after a vacuum fallback
        # (GB radii missing) keep requesting vacuum instead of re-failing on the
        # GB model every iteration.
        einit, efinal, pdb, effective_solvent = run_minimize(
            pdb, ref_pdb, lddt_map, args, exclude_residues, template_generators,
            effective_solvent,
        )
        if einit_first is None:
            einit_first = einit

        t_v = time.time()
        device = _detect_violation_device()
        new_violations, vinfo = find_violations_torch(
            pdb,
            pdb.positions,
            tolerance_factor=12.0,
            clash_overlap_tolerance=1.5,
            device=device,
        )
        new_violations -= exclude_residues
        print(
            f"  violations: {len(new_violations)} res "
            f"(bond={vinfo['bond']}, between_clash={vinfo['between_clash']}, "
            f"within_clash={vinfo['within_clash']}) "
            f"[{time.time() - t_v:.1f}s]"
        )

        if not new_violations:
            print("  converged (no new violations)")
            break
        exclude_residues |= new_violations

    # 5. restore wwPDB nomenclature for GLYCAM-retyped residues, then merge any
    #    frozen residues back into the output topology at their original coords.
    if glycam_restore:
        import glycoprotein
        glycoprotein.apply_restore(pdb.topology, glycam_restore)

    if frozen_top is not None:
        out_modeller = openmm_app.Modeller(pdb.topology, pdb.positions)
        out_modeller.add(frozen_top, frozen_pos)
        out_top, out_pos = out_modeller.topology, out_modeller.positions
    else:
        out_top, out_pos = pdb.topology, pdb.positions

    # Heavy-atom-only output (drop hydrogens), consistent with the pyrosetta method.
    out_top, out_pos = _strip_hydrogens(out_top, out_pos)

    buf = io.StringIO()
    openmm_app.PDBFile.writeFile(out_top, out_pos, buf, keepIds=True)  # match fix_pdb keyspace
    min_pdb_str = buf.getvalue()

    # 6. overwrite B-factor with pLDDT, then write output
    min_pdb_str = overwrite_bfactor_with_lddt(
        min_pdb_str, lddt_map, read_bfactor_by_key(args.pdb_fn)
    )
    with open(out_pdb, "w") as f:
        f.write(min_pdb_str)

    # Energy reporting. A single minimization pass (no separate FastRelax step),
    # so just initial vs final potential energy.
    E_init = float(einit_first)
    E_final = float(efinal)
    tool_tag = f"openmm_{effective_solvent}"

    print(
        "ENERGY of INITIAL / FINAL = %.3f / %.3f (%s)"
        % (E_init, E_final, tool_tag)
    )

    e_file = f"{args.out_prefix}.energy.yaml"
    with open(e_file, "w") as fp:
        yaml.dump(
            {
                os.path.basename(args.out_prefix): {
                    "tool": tool_tag,
                    "E_init": float("%.6f" % E_init),
                    "E_final": float("%.6f" % E_final),
                }
            },
            fp,
            sort_keys=False,
        )

    print(f"[done] {time.time() - t0:.1f}s")
    print(f"  output:   {out_pdb}")
    print(f"  energies: {e_file}")
    print(f"  excluded {len(exclude_residues)} residues from restraint")


if __name__ == "__main__":
    main()
