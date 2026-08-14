# Thal-Kak
# Copyright 2026 CSSB, Seoul National University
#
# Licensed under the Apache License, Version 2.0 (see LICENSE).
#
# ---------------------------------------------------------------------------
# Third-party attribution (see also NOTICE)
#
# build_hhsearch_db() in this file is derived from ColabFold's
# mk_hhsearch_db() (colabfold/batch.py,
# https://github.com/sokrypton/ColabFold):
#
#   MIT License
#
#   Copyright (c) 2021 Sergey Ovchinnikov
#
#   Permission is hereby granted, free of charge, to any person obtaining a
#   copy of this software and associated documentation files (the "Software"),
#   to deal in the Software without restriction, including without limitation
#   the rights to use, copy, modify, merge, publish, distribute, sublicense,
#   and/or sell copies of the Software, and to permit persons to whom the
#   Software is furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.
#
# MODIFIED by CSSB Thal-Kak (2026): per-chain selection via chain_filter;
# .pdb files parsed directly with PDBParser instead of being converted to
# mmCIF; mmCIF validation/repair dropped; insertion-coded residues retained
# rather than raising; multi-model files use model 0 rather than raising;
# hetero/solvent residues skipped; residue_constants.restype_3to1 replaced
# with Bio.SeqUtils.seq1; per-entry sequences returned to the caller.
#
# The remainder of this file — parse_hhr_to_m8_rows() and
# generate_m8_from_hhsearch() — is original to this work. The latter drives
# the hhsearch binary using the same documented -i/-d/-o/-maxseq invocation
# as AlphaFold's HHSearch.query() (Apache License 2.0, Copyright 2021
# DeepMind Technologies Limited); no AlphaFold source is reproduced.
# ---------------------------------------------------------------------------

import subprocess
import shutil
from pathlib import Path
import tempfile
import re
import os, sys, glob
from Bio.PDB import MMCIFParser, PDBIO, Select, PDBParser
from Bio.SeqUtils import seq1
from collections import defaultdict


def build_hhsearch_db(template_dir, chain_filter=None):
    """Build pdb70 HH-suite database from CIF/PDB files in template_dir.

    chain_filter: optional {file_stem: iterable_of_chain_ids}. When given,
    structures whose stem is missing from the filter are skipped entirely,
    and only the listed chains of the remaining files are indexed.

    Returns a dict mapping entry_name (e.g. "6blh_G") -> one-letter sequence,
    so callers can emit fallback m8 rows when hhsearch misses a desired template.
    """
    template_path = Path(template_dir)

    for f in template_path.glob("pdb70*"):
        os.remove(f)

    entry_seqs = {}
    with open(template_path / "pdb70_a3m.ffdata", "w") as a3m_f, \
         open(template_path / "pdb70_a3m.ffindex", "w") as a3m_idx, \
         open(template_path / "pdb70_cs219.ffdata", "w") as cs219_f, \
         open(template_path / "pdb70_cs219.ffindex", "w") as cs219_idx:

        n = 1000000
        offset = 0
        struct_files = list(template_path.glob("*.cif")) + list(template_path.glob("*.pdb"))
        for struct_file in struct_files:
            allowed = None
            if chain_filter is not None:
                if struct_file.stem not in chain_filter:
                    continue
                allowed = set(chain_filter[struct_file.stem])
            ext = struct_file.suffix.lower()
            parser = MMCIFParser(QUIET=True) if ext == ".cif" else PDBParser(QUIET=True)
            try:
                structure = parser.get_structure("none", str(struct_file))
            except Exception as e:
                print(f"[Warning] Could not parse {struct_file}: {e}")
                continue

            models = list(structure.get_models())
            if not models:
                print(f"[Warning] No models in {struct_file}; skipping")
                continue
            if len(models) > 1:
                print(
                    f"[Process] {struct_file.name} has {len(models)} models; using model 0"
                )
            model = models[0]
            for chain in model:
                if allowed is not None and chain.id not in allowed:
                    continue
                residues = []
                for res in chain:
                    # res.id = (hetflag, resseq, icode). Skip waters / ligands
                    # / hetero (hetflag != " "). Icoded polymer residues
                    # (102, 102A, 102B, ...) are kept in author order: hh-suite's
                    # cif2fasta.py indexes by label_seq_id and includes them in
                    # the FASTA, and gemmi's get_polymer().make_one_letter_sequence
                    # also keeps them, so dropping here would shift the offsets
                    # hhsearch reports into the m8.
                    if res.id[0] == " ":
                        aa = seq1(res.get_resname())
                        residues.append(aa if aa else "X")
                protein_str = "".join(residues)
                if not protein_str:
                    continue

                entry_name = f"{struct_file.stem}_{chain.id}"
                entry_seqs[entry_name] = protein_str
                a3m_str = f">{entry_name}\n{protein_str}\n\0"
                a3m_f.write(a3m_str)
                a3m_idx.write(f"{n}\t{offset}\t{len(a3m_str)}\n")
                cs219_idx.write(f"{n}\t{offset}\t{len(protein_str)}\n")
                cs219_f.write("\n\0")
                offset += len(a3m_str)
                n += 1

    return entry_seqs


def parse_hhr_to_m8_rows(hhr_text):
    """Parse HHR output and return list of dicts with m8-compatible fields.

    Each dict contains: name, pident, length, mismatch, gapopen,
    query_start, query_end, hit_start, hit_end, evalue, bitscore.
    Positions are 1-indexed (matching m8/BLAST convention).
    """
    results = []
    blocks = re.split(r'\nNo \d+\n', hhr_text)

    for block in blocks[1:]:
        lines = block.strip().split('\n')
        if not lines or not lines[0].startswith('>'):
            continue

        name = lines[0][1:].split()[0]

        # Parse stats: Probab=... E-value=... Score=... Identities=...
        stats = {}
        for line in lines[1:6]:
            if 'Probab=' in line:
                for m in re.finditer(r'([\w-]+)=([\d.eE+-]+%?)', line):
                    stats[m.group(1)] = m.group(2).rstrip('%')
                break

        if not stats:
            continue

        evalue = float(stats.get('E-value', '999'))
        score = float(stats.get('Score', '0'))
        pident = float(stats.get('Identities', '0'))
        aligned_cols = int(stats.get('Aligned_cols', '0'))

        # Parse alignment blocks (may span multiple line groups)
        q_starts, q_ends, q_seqs = [], [], []
        t_starts, t_ends, t_seqs = [], [], []

        for line in lines:
            if line.startswith('Q ') and not line.startswith('Q Consensus'):
                m = re.match(r'^Q\s+\S+\s+(\d+)\s+(\S+)\s+(\d+)', line)
                if m:
                    q_starts.append(int(m.group(1)))
                    q_seqs.append(m.group(2))
                    q_ends.append(int(m.group(3)))

            if line.startswith('T ') and not line.startswith('T Consensus') \
               and not line.startswith('T ss_'):
                m = re.match(r'^T\s+\S+\s+(\d+)\s+(\S+)\s+(\d+)', line)
                if m:
                    t_starts.append(int(m.group(1)))
                    t_seqs.append(m.group(2))
                    t_ends.append(int(m.group(3)))

        if not q_starts or not t_starts:
            continue

        q_full = ''.join(q_seqs)
        t_full = ''.join(t_seqs)
        mismatch = 0
        gapopen = 0
        in_gap = False
        cigar_ops = []
        for q, t in zip(q_full, t_full):
            if q == '-' and t == '-':
                continue
            if q == '-':
                op = 'D'
            elif t == '-':
                op = 'I'
            else:
                op = 'M'
                if q.upper() != t.upper():
                    mismatch += 1
            if op in ('I', 'D'):
                if not in_gap:
                    gapopen += 1
                    in_gap = True
            else:
                in_gap = False
            if cigar_ops and cigar_ops[-1][0] == op:
                cigar_ops[-1][1] += 1
            else:
                cigar_ops.append([op, 1])
        cigar = ''.join(f"{n}{o}" for o, n in cigar_ops)

        results.append({
            'name': name,
            'pident': pident,
            'length': aligned_cols,
            'mismatch': mismatch,
            'gapopen': gapopen,
            'query_start': q_starts[0],
            'query_end': q_ends[-1],
            'hit_start': t_starts[0],
            'hit_end': t_ends[-1],
            'evalue': evalue,
            'bitscore': score,
            'comment': cigar,
        })

    return results


def generate_m8_from_hhsearch(num_map, a3m_list, template_list, output_dir):
    """
    Generate an m8 file using hhsearch with pre-generated MSA (a3m) profiles.
    Replaces generate_m8 which required MMseqs2.
    num_map: dict mapping num keys from raw m8 file to list of labels you want to put in m8 file's first field (e.g. {"101": ["H1140_A"], "102": ["H1140_B"]})
    Then, the m8 file will have lines like:
    H1140_A  template1_A  99.5  150  0  0  1  150  1  150  1e-50  200.0
    H1140_B  template1_B  98.0  150  3  0  1  150  1  150  1e-45  180.0
    """
    hits_fp = output_dir / "hits.m8"
    # Reset any prior content so re-runs start fresh.
    if hits_fp.exists():
        hits_fp.unlink()

    # Read query sequence lengths from the unpaired a3m (first record is query).
    def _query_seq_len(a3m_path):
        with open(a3m_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        for i, l in enumerate(lines):
            if l.startswith(">") and i + 1 < len(lines):
                return len(lines[i + 1])
        return 0

    # chain mapping
    chain_map = defaultdict(int)
    for i, a3m in enumerate(a3m_list):
        unp = a3m["unpaired_path"]
        chains = unp[:-4].split("_unpaired_msa_chains_")[-1].split("_")
        for c in chains:
            chain_map[c.upper()] = i

    # Group templates by query chain index
    templates_by_chain = defaultdict(list)
    for info in template_list:
        chain_idx = chain_map[info["chain_query"][0]]
        cif_path = info["path"]
        chain_template = info["chain_template"][0]
        subject_id = f'{os.path.basename(cif_path)[:-4]}_{chain_template}'
        templates_by_chain[chain_idx].append((cif_path, subject_id))

    for chain_idx, chain_templates in templates_by_chain.items():
        unpaired = a3m_list[chain_idx]["unpaired_path"]
        current_chain = unpaired[:-4].split("_unpaired_msa_chains_")[-1].split("_")[0].upper()
        num_key = str(101 + chain_idx)
        labels = num_map[num_key]
        q_len = _query_seq_len(unpaired)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Copy template structures to temp directory and build a
            # {stem: {chain_ids}} filter so build_hhsearch_db only indexes
            # the chains actually referenced as subjects for this query.
            chain_filter = defaultdict(set)
            for cif_path, subject_id in chain_templates:
                shutil.copy(cif_path, tmpdir_path / os.path.basename(cif_path))
                stem_cif_path = os.path.basename(cif_path)[:-4]
                chain_filter[stem_cif_path].add(subject_id[len(stem_cif_path) + 1:])

            # Build hhsearch database from template structures
            entry_seqs = build_hhsearch_db(tmpdir_path, chain_filter=dict(chain_filter))
            print(f"[Process] Built hhsearch DB for chain {current_chain} "
                  f"({len(entry_seqs)} chains indexed)")

            hhr_path = tmpdir_path / "result.hhr"
            cmd = [
                "hhsearch",
                "-i", str(unpaired),
                "-d", str(tmpdir_path / "pdb70"),
                "-o", str(hhr_path),
                "-maxseq", "1_000_000",
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"[Error] hhsearch failed for chain {current_chain}: "
                      f"{e.stderr if hasattr(e, 'stderr') else e}")
                hits = []
            else:
                with open(hhr_path) as f:
                    hhr_text = f.read()
                hits = parse_hhr_to_m8_rows(hhr_text)

            desired = {sid for _, sid in chain_templates}
            found = set()

            # m8 columns (tab-delimited): query_id, subject_id, pident, length,
            # mismatch, gapopen, query_start, query_end, subject_start,
            # subject_end, evalue, bitscore, comment. pident is written as a
            # 0-1 fraction (matching the reference leave_hits output); comment
            # holds a CIGAR-like string (e.g. "100M2D23M2I104M").
            with open(hits_fp, "a") as f:
                for hit in hits:
                    if hit['name'] in desired and hit['name'] not in found:
                        found.add(hit['name'])
                        for label in labels:
                            row = [
                                label,
                                hit['name'],
                                f"{hit['pident'] / 100:.3f}",
                                str(hit['length']),
                                str(hit['mismatch']),
                                str(hit['gapopen']),
                                str(hit['query_start']),
                                str(hit['query_end']),
                                str(hit['hit_start']),
                                str(hit['hit_end']),
                                f"{hit['evalue']:.3E}",
                                f"{hit['bitscore']:.0f}",
                                hit['comment'],
                            ]
                            f.write("\t".join(row) + "\n")
                        print(f"[Process] hhsearch hit: {hit['name']} "
                              f"(identity={hit['pident']:.1f}%, "
                              f"e-value={hit['evalue']:.1e}, "
                              f"cigar={hit['comment']})")


    if hits_fp.is_file():
        print(f"[Process] Created template hits m8 at {hits_fp}")
    else:
        print("[Warning] hhsearch produced no hits.m8 rows")

    return hits_fp
