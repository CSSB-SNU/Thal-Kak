import numpy as np
from collections import defaultdict

NA_RESIDUES = {"A", "C", "G", "U", "DA", "DC", "DG", "DT"}

def check_bond(lines, na=False):
    if na:
        atom_list = ["O3'", "P"]
        lower_bound, upper_bound = 0.0, 1.8 # P-O single bond length maximum
    else:
        atom_list = ["C", "N"]
        lower_bound, upper_bound = 0.0, 1.5 # C-N single bond length maximum
    
    chain = lines[0][21]
    former_res, latter_res = [], []
    init_res_num = None
    prev_res_num = 0
    hits = 0
    for line in lines:
        atom_name = line[12:16].strip()
        if atom_name not in atom_list:
            continue

        res_num = line[22:26].strip()
        coord = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        
        if init_res_num is None:
            init_res_num = int(res_num)
        
        if int(res_num) - prev_res_num == 1:
            if atom_name == atom_list[0]:
                former_res.append(coord)
                hits += 1
            elif atom_name == atom_list[1]:
                latter_res.append(coord)
                hits += 1
        else:
            print(f"Non-consecutive residue numbers found in chain {chain}; between {prev_res_num} and {res_num}")
            return False

        if hits == 2:
            hits = 0
            prev_res_num = int(res_num)

    former_res.pop()
    latter_res.pop(0)
    former_res, latter_res = np.array(former_res), np.array(latter_res)
    dists = np.linalg.norm(former_res - latter_res, axis=1)
    ok = np.all((dists >= lower_bound) & (dists <= upper_bound))
    bad_idx = np.where((dists < lower_bound) | (dists > upper_bound))[0]

    if ok:
        return True
    else:
        print(f"Invalid bond length between residues in chain {chain}; distances={list(dists[bad_idx])}, residue indices={list(init_res_num + bad_idx)}")
        return False

def validation(pdb):
    print('Checking bond length between residues of %s'%pdb)

    with open(pdb, 'r') as f:
        lines = f.readlines()

    chain_residues = defaultdict(set)
    chain_lines = defaultdict(list)

    for line in lines:
        if not line.startswith("ATOM"):
            continue
        chain = line[21]
        resname = line[17:20].strip()
        chain_residues[chain].add(resname)
        chain_lines[chain].append(line.strip())
    
    count_valid_chain = 0
    invalid_chain = []
    for chain, lines in chain_lines.items():
        if chain_residues[chain].issubset(NA_RESIDUES):
            valid = check_bond(lines, na=True)
        else:
            valid = check_bond(lines, na=False)
        
        if valid:
            count_valid_chain += 1
        else:
            invalid_chain.append(chain)

    if count_valid_chain == len(chain_lines.keys()):
        print('All chains are valid.')
        return True
    else:
        return False
