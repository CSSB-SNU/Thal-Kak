import string

PDB_CHAIN_CHARS = string.ascii_uppercase + string.ascii_lowercase + "0123456789"

# Chain labels for CIF / fasta entity names: A-Z, then two-letter ids that cycle
# the first character fastest (AA, BA, ..., ZA, AB, BB, ..., ZB, ..., ZZ), so
# complexes with more than 26 chains still get unique labels. Indexing
# string.ascii_uppercase avoids chr(ord("A") + n), which yields '[', ']', etc.
# once the offset passes 'Z'.
CIF_CHAIN_CHARS = list(string.ascii_uppercase) + [
    a + b for b in string.ascii_uppercase for a in string.ascii_uppercase
]

def assign_chain_indices(copies, types):
    """Assign chain indices to entity copies in interleaved (round-robin) order.

    Entities sharing the same ``types`` value are round-robined together, and
    groups are emitted in ascending order of that value. Convention: ``0`` for
    protein, ``1`` for nucleic acid. This keeps protein chains indexed first
    so ``chain_query`` fields written at MSA time (which use protein-only
    round-robin) stay aligned with the structure runner's chain ids.

    Args:
        copies: list of int, copy count for each entity.
        types: list of int (same length as ``copies``). 0 = protein, 1 = NA.

    Returns:
        list of list, where result[i] = sorted chain indices for entity i.

    Example:
        >>> assign_chain_indices([2, 2, 1], types=[0, 0, 1])
        [[0, 2], [1, 3], [4]]   # proteins interleaved, NA last
    """
    if len(types) != len(copies):
        raise ValueError(
            f"len(types)={len(types)} != len(copies)={len(copies)}"
        )
    groups = [
        [i for i, t in enumerate(types) if t == g]
        for g in sorted(set(types))
    ]

    chains_per_entity = [[] for _ in copies]
    k = 0
    for group in groups:
        if not group:
            continue
        group_copies = [copies[i] for i in group]
        max_copies = max(group_copies)
        for r in range(max_copies):
            for orig_i, c in zip(group, group_copies):
                if r < c:
                    chains_per_entity[orig_i].append(k)
                    k += 1
    return chains_per_entity


def parse_bonds(data_cfg):
    """Read the data yaml's optional `bonds` into [(c1, r1, a1, c2, r2, a2), ...].

    A bond names its two atoms as {chain, residue, atom}, the residue 1-based
    within its chain. af3 and boltz take that addressing directly; protenix maps
    it onto its own entity/copy scheme. Covalent bonds are how glycans and
    covalent ligands attach, since a sugar has no polymer backbone and so cannot
    be given as a residue modification.
    """
    out = []
    for i, bond in enumerate(data_cfg.get("bonds") or []):
        try:
            a1, a2 = bond["atom1"], bond["atom2"]
            c1, r1, n1 = a1["chain"], int(a1["residue"]), a1["atom"]
            c2, r2, n2 = a2["chain"], int(a2["residue"]), a2["atom"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"bonds[{i}]: expected atom1/atom2 as {{chain, residue, atom}}, "
                f"e.g. {{chain: A, residue: 125, atom: ND2}}; got {bond!r} ({exc})"
            ) from exc
        out.append((str(c1), r1, str(n1), str(c2), r2, str(n2)))
    return out
