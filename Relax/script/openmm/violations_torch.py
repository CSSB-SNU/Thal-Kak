"""
AF2 find_structural_violations, ported to PyTorch.

Original sources (Apache 2.0):
  alphafold.model.folding.find_structural_violations
  alphafold.model.all_atom.between_residue_bond_loss
  alphafold.model.all_atom.between_residue_clash_loss
  alphafold.model.all_atom.within_residue_violations
Copyright 2021 DeepMind Technologies Limited.

This module reuses residue_constants (Apache 2.0) for atom14 indexing,
peptide-bond reference geometry, vdW radii, and within-residue distance bounds.
Numerical criteria match AF2 defaults:
  violation_tolerance_factor   = 12.0   (bond/angle z-score)
  clash_overlap_tolerance      = 1.5 A
"""

import numpy as np
import torch

import residue_constants as rc


# ---------------------------------------------------------------------------
# One-time precompute of atom14 lookup tables and bound matrices (numpy).
# ---------------------------------------------------------------------------

def _build_atom14_tables():
    """
    Build:
      restype_atom14_mask:  (21, 14) bool   -- which atom14 slots exist per AA
      restype_atom14_radius:(21, 14) float  -- vdW radius per slot
      resname_to_idx14:     dict[resname]   -> dict[atom_name, slot_idx]
    """
    restypes_with_unk = rc.restypes + ["X"]
    mask = np.zeros((21, 14), dtype=np.float32)
    radius = np.zeros((21, 14), dtype=np.float32)
    name_to_idx = {}
    for ri, rletter in enumerate(restypes_with_unk):
        if rletter == "X":
            resname = "UNK"
            atom_names = rc.restype_name_to_atom14_names["UNK"]
        else:
            resname = rc.restype_1to3[rletter]
            atom_names = rc.restype_name_to_atom14_names[resname]
        name_to_idx[resname] = {}
        for j, an in enumerate(atom_names):
            if not an:
                continue
            mask[ri, j] = 1.0
            radius[ri, j] = rc.van_der_waals_radius[an[0]]
            name_to_idx[resname][an] = j
    return mask, radius, name_to_idx


_ATOM14_MASK, _ATOM14_RADIUS, _RESNAME_ATOM_TO_IDX14 = _build_atom14_tables()

# amber/GLYCAM protonation & linkage variants share their parent residue's
# heavy-atom names; normalize to the standard 3-letter code so they get real
# atom14 slots. Otherwise they fall through to UNK (empty slot map): the peptide
# bonds flanking a glycosylated Asn (renamed NLN) go unchecked, and a CYX-CYX
# disulfide would even be miscounted as an inter-residue clash.
_RESNAME_ALIASES = {
    "NLN": "ASN",                               # N-linked Asn (GLYCAM)
    "HIE": "HIS", "HID": "HIS", "HIP": "HIS",   # Amber His protonation states
    "CYX": "CYS", "CYM": "CYS",                 # disulfide / deprotonated Cys
    "ASH": "ASP", "GLH": "GLU", "LYN": "LYS",   # protonated / neutral variants
}

# Within-residue lower/upper bounds (21, 14, 14)
_BOUNDS = rc.make_atom14_dists_bounds(
    overlap_tolerance=1.5,
    bond_length_tolerance_factor=15,  # AF2 default
)
_LOWER = _BOUNDS["lower_bound"]   # (21, 14, 14)
_UPPER = _BOUNDS["upper_bound"]   # (21, 14, 14)

# Peptide bond reference constants (general, PRO-as-i+1)
_CN_LEN  = np.array(rc.between_res_bond_length_c_n,        dtype=np.float32)
_CN_STD  = np.array(rc.between_res_bond_length_stddev_c_n, dtype=np.float32)
_CCN_COS = np.array(rc.between_res_cos_angles_ca_c_n,      dtype=np.float32)  # [mean, stddev]
_CNC_COS = np.array(rc.between_res_cos_angles_c_n_ca,      dtype=np.float32)  # [mean, stddev]


# ---------------------------------------------------------------------------
# Convert OpenMM topology + positions to atom14 tensors
# ---------------------------------------------------------------------------

def _topology_to_atom14(pdb, positions, device):
    """
    Build atom14 representation from OpenMM topology + positions.

    Returns:
      aatype:        (N,)        int64, restype index (UNK -> 20)
      atom14_pos:    (N, 14, 3)  float32
      atom14_mask:   (N, 14)     float32
      atom14_radius: (N, 14)     float32
      residue_index: (N,)        int64
      res_names:     list[str]   length N
    """
    from openmm import unit as _u

    if hasattr(positions, "value_in_unit"):
        pos_np = np.asarray(positions.value_in_unit(_u.angstroms),
                            dtype=np.float32)
    else:
        pos_np = np.asarray(positions, dtype=np.float32)

    residues = list(pdb.topology.residues())
    N = len(residues)

    aatype = np.full((N,), rc.unk_restype_index, dtype=np.int64)
    atom14_pos = np.zeros((N, 14, 3), dtype=np.float32)
    atom14_mask = np.zeros((N, 14), dtype=np.float32)
    atom14_radius = np.zeros((N, 14), dtype=np.float32)
    res_names = []

    for i, res in enumerate(residues):
        # normalize amber/GLYCAM variants (NLN, HIE, CYX, ...) to their parent so
        # the PRO/CYS checks and atom14 slotting below see the standard name.
        resname = _RESNAME_ALIASES.get(res.name, res.name)
        res_names.append(resname)
        if resname in rc.restype_3to1:
            rt = rc.restype_order[rc.restype_3to1[resname]]
            slot_map = _RESNAME_ATOM_TO_IDX14[resname]
        else:
            # nonstandard residue -> UNK (no within-residue bounds applied)
            rt = rc.unk_restype_index
            slot_map = {}
        aatype[i] = rt

        for atom in res.atoms():
            if atom.name not in slot_map:
                continue
            j = slot_map[atom.name]
            atom14_pos[i, j] = pos_np[atom.index]
            atom14_mask[i, j] = 1.0
            atom14_radius[i, j] = rc.van_der_waals_radius[atom.name[0]]

    return (
        torch.from_numpy(aatype).to(device),
        torch.from_numpy(atom14_pos).to(device),
        torch.from_numpy(atom14_mask).to(device),
        torch.from_numpy(atom14_radius).to(device),
        torch.arange(N, dtype=torch.long, device=device),
        res_names,
    )


# ---------------------------------------------------------------------------
# Violation components (torch, vectorized)
# ---------------------------------------------------------------------------

def _between_residue_bond_violations(
    atom14_pos, atom14_mask, res_names,
    tolerance_factor=12.0,
):
    """
    Returns (N,) bool: True if residue i or i+1 has peptide-bond violation
    (C-N length, Ca-C-N cos-angle, C-N-Ca cos-angle).
    """
    N = atom14_pos.shape[0]
    device = atom14_pos.device
    viol = torch.zeros(N, dtype=torch.bool, device=device)
    if N < 2:
        return viol

    N_IDX, CA_IDX, C_IDX = 0, 1, 2  # atom14 ordering: N, CA, C, O, ...

    has_C  = atom14_mask[:-1, C_IDX]  > 0
    has_N  = atom14_mask[1:,  N_IDX]  > 0
    has_CA_i = atom14_mask[:-1, CA_IDX] > 0
    has_CA_j = atom14_mask[1:,  CA_IDX] > 0
    pair_ok = has_C & has_N & has_CA_i & has_CA_j

    is_pro_next = torch.tensor(
        [(rn == "PRO") for rn in res_names[1:]],
        dtype=torch.bool, device=device,
    )

    c_pos  = atom14_pos[:-1, C_IDX]
    n_pos  = atom14_pos[1:,  N_IDX]
    ca_i   = atom14_pos[:-1, CA_IDX]
    ca_j   = atom14_pos[1:,  CA_IDX]

    eps = 1e-6

    # (1) C-N bond length
    d_cn = torch.norm(c_pos - n_pos, dim=-1)
    cn_len_mean = torch.where(
        is_pro_next,
        torch.full_like(d_cn, float(_CN_LEN[1])),
        torch.full_like(d_cn, float(_CN_LEN[0])),
    )
    cn_len_std = torch.where(
        is_pro_next,
        torch.full_like(d_cn, float(_CN_STD[1])),
        torch.full_like(d_cn, float(_CN_STD[0])),
    )
    z_len = torch.abs(d_cn - cn_len_mean) / (cn_len_std + eps)

    # (2) Ca(i)-C(i)-N(i+1)  -- compare as cosine
    v1 = ca_i - c_pos
    v2 = n_pos - c_pos
    cos_ccn = (v1 * v2).sum(-1) / (
        torch.norm(v1, dim=-1) * torch.norm(v2, dim=-1) + eps
    )
    ccn_mean = torch.full_like(cos_ccn, float(_CCN_COS[0]))
    ccn_std  = torch.full_like(cos_ccn, float(_CCN_COS[1]))
    z_ccn = torch.abs(cos_ccn - ccn_mean) / (ccn_std + eps)

    # (3) C(i)-N(i+1)-Ca(i+1)
    v3 = c_pos - n_pos
    v4 = ca_j - n_pos
    cos_cnc = (v3 * v4).sum(-1) / (
        torch.norm(v3, dim=-1) * torch.norm(v4, dim=-1) + eps
    )
    cnc_mean = torch.full_like(cos_cnc, float(_CNC_COS[0]))
    cnc_std  = torch.full_like(cos_cnc, float(_CNC_COS[1]))
    z_cnc = torch.abs(cos_cnc - cnc_mean) / (cnc_std + eps)

    pair_violation = (
        (z_len > tolerance_factor)
        | (z_ccn > tolerance_factor)
        | (z_cnc > tolerance_factor)
    ) & pair_ok

    viol[:-1] |= pair_violation
    viol[1:]  |= pair_violation
    return viol


def _between_residue_clash_violations(
    atom14_pos, atom14_mask, atom14_radius, residue_index, res_names,
    overlap_tolerance=1.5,
):
    """
    Returns (N,) bool: True if residue i has any inter-residue heavy-atom clash.
    Exceptions: adjacent peptide bond (C-N), disulfide (CYS SG - CYS SG).
    """
    N, A, _ = atom14_pos.shape
    device = atom14_pos.device

    flat_pos    = atom14_pos.reshape(N * A, 3)
    flat_mask   = atom14_mask.reshape(N * A)
    flat_radius = atom14_radius.reshape(N * A)
    flat_resid  = residue_index.repeat_interleave(A)
    flat_slot   = torch.arange(A, device=device).repeat(N)

    valid = flat_mask > 0
    idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    pos = flat_pos[idx]
    rad = flat_radius[idx]
    rid = flat_resid[idx]
    slot = flat_slot[idx]

    n_atoms = pos.shape[0]
    per_res_violating = torch.zeros(N, dtype=torch.bool, device=device)

    is_cys = torch.tensor(
        [rn == "CYS" for rn in res_names], dtype=torch.bool, device=device
    )

    CYS_SG_SLOT = rc.restype_name_to_atom14_names["CYS"].index("SG")  # =5
    C_SLOT = 2
    N_SLOT = 0

    chunk = 2048
    for s in range(0, n_atoms, chunk):
        e = min(s + chunk, n_atoms)
        diff = pos[s:e, None, :] - pos[None, :, :]
        dist = torch.norm(diff, dim=-1)

        lb = rad[s:e, None] + rad[None, :] - overlap_tolerance
        diff_res = rid[s:e, None] != rid[None, :]
        global_i = torch.arange(s, e, device=device)[:, None]
        global_j = torch.arange(n_atoms, device=device)[None, :]
        upper = global_j > global_i

        clash = (dist < lb) & diff_res & upper

        if clash.any():
            ii, jj = torch.nonzero(clash, as_tuple=True)
            ri = rid[s + ii]
            rj = rid[jj]
            si = slot[s + ii]
            sj = slot[jj]

            # peptide bond exception
            adj = (ri - rj).abs() == 1
            forward  = (rj == ri + 1) & (si == C_SLOT) & (sj == N_SLOT)
            backward = (ri == rj + 1) & (sj == C_SLOT) & (si == N_SLOT)
            pep_skip = adj & (forward | backward)

            # disulfide exception
            cys_i = is_cys[ri] & (si == CYS_SG_SLOT)
            cys_j = is_cys[rj] & (sj == CYS_SG_SLOT)
            ss_skip = cys_i & cys_j

            keep = ~(pep_skip | ss_skip)
            ri_k = ri[keep]
            rj_k = rj[keep]
            if ri_k.numel() > 0:
                per_res_violating.scatter_(
                    0, ri_k, torch.ones_like(ri_k, dtype=torch.bool)
                )
                per_res_violating.scatter_(
                    0, rj_k, torch.ones_like(rj_k, dtype=torch.bool)
                )
    return per_res_violating


def _within_residue_violations(aatype, atom14_pos, atom14_mask):
    """
    Returns (N,) bool: True if residue i has any intra-residue atom pair
    outside [lower_bound, upper_bound] for its restype.
    Uses precomputed _LOWER / _UPPER tables from residue_constants.
    """
    N, A, _ = atom14_pos.shape
    device = atom14_pos.device

    lower = torch.from_numpy(_LOWER).to(device)   # (21, 14, 14)
    upper = torch.from_numpy(_UPPER).to(device)   # (21, 14, 14)

    aatype_clamped = aatype.clamp(max=20)
    res_lower = lower[aatype_clamped]
    res_upper = upper[aatype_clamped]

    d = torch.cdist(atom14_pos, atom14_pos)  # (N, 14, 14)

    pair_mask = atom14_mask[:, :, None] * atom14_mask[:, None, :]
    eye = torch.eye(A, device=device).bool()
    pair_mask = pair_mask.bool() & (~eye)[None]
    has_bound = res_lower > 0  # zero entries = no constraint set

    too_close = (d < res_lower) & pair_mask & has_bound
    too_far   = (d > res_upper) & pair_mask & has_bound

    per_atom = (too_close | too_far).any(dim=-1)
    per_res  = per_atom.any(dim=-1)
    return per_res


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def find_violations_torch(
    pdb, positions,
    tolerance_factor=12.0,
    clash_overlap_tolerance=1.5,
    device="cpu",
):
    """
    AF2-style structural violation detector (torch).

    Args:
        pdb: openmm.app.PDBFile (provides topology + residue order)
        positions: openmm Quantity or np.ndarray (n_atoms, 3) in Angstrom
        tolerance_factor: z-score threshold for bond/angle (AF2: 12)
        clash_overlap_tolerance: vdW overlap allowance in A (AF2: 1.5)
        device: 'cpu' or 'cuda'

    Returns:
        violating_residues: set[int]   -- zero-indexed by topology order
        info: dict with per-component counts
    """
    aatype, atom14_pos, atom14_mask, atom14_radius, residue_index, res_names = \
        _topology_to_atom14(pdb, positions, device)

    bond_viol  = _between_residue_bond_violations(
        atom14_pos, atom14_mask, res_names,
        tolerance_factor=tolerance_factor,
    )
    clash_viol = _between_residue_clash_violations(
        atom14_pos, atom14_mask, atom14_radius, residue_index, res_names,
        overlap_tolerance=clash_overlap_tolerance,
    )
    within_viol = _within_residue_violations(
        aatype, atom14_pos, atom14_mask,
    )

    total = bond_viol | clash_viol | within_viol
    viol_set = set(torch.nonzero(total, as_tuple=False).squeeze(-1).tolist())
    info = {
        "bond":          int(bond_viol.sum().item()),
        "between_clash": int(clash_viol.sum().item()),
        "within_clash":  int(within_viol.sum().item()),
        "total":         int(total.sum().item()),
    }
    return viol_set, info
