"""Make a query fasta from a CASP fasta file, honouring its stoichiometry."""

import os, re
import argparse

# `N` = ambiguous nucleotide.
NUCLEIC_ACIDS = set("ACGTUN")


def is_nucleic_acid(seq):
    """True iff every residue is in {A,C,G,T,U,N} (and non-empty).

    A/C/G/T/U/N are also valid amino-acid codes, so a peptide drawn only from
    those letters would classify as NA. No real CASP chain is that short.
    """
    s = seq.upper()
    return len(s) > 0 and set(s).issubset(NUCLEIC_ACIDS)


def get_na_type(seq):
    """Classify an NA sequence:
    - U present, or T absent  -> rna  (covers ambiguous 'ACGN')
    - T present and U absent   -> dna
    - both T and U present     -> ValueError (fail-fast)
    """
    s = seq.upper()
    has_t, has_u = "T" in s, "U" in s
    if has_t and has_u:
        raise ValueError(
            f"sequence contains both T and U (cannot classify DNA/RNA): {seq[:40]}..."
        )
    return "rna" if (has_u or not has_t) else "dna"


def read_fasta_records(fasta):
    """Parse FASTA into [(header, sequence), ...]; robust to multi-line
    sequences, blank lines, and whitespace. Raises ValueError on data before
    any header or an empty file. Shared FASTA reader — also imported by
    MSA/local_msa/common/input.py (parse_inputs) so both MSA paths use one parser.
    """
    records, header, buf = [], None, []
    with open(fasta, "r") as f:
        for raw in f.read().splitlines():
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(buf)))
                header, buf = line[1:].strip(), []
            else:
                if header is None:
                    raise ValueError(f"{fasta}: sequence data before any '>' header")
                buf.append("".join(line.split()))
    if header is not None:
        records.append((header, "".join(buf)))
    if not records:
        raise ValueError(f"{fasta}: empty FASTA (no '>' records)")
    return records


def parse_fasta(fasta, stoi, out_dir):

    k = os.path.basename(fasta).split(".")[0]
    records = read_fasta_records(fasta)
    stoi_parsed = re.findall(r"([A-Z])(\d+|n)", stoi)
    if not stoi_parsed:
        raise ValueError(f"unrecognized stoi string {stoi!r}")
    if "".join(f"{c}{n}" for c, n in stoi_parsed) != stoi:
        raise ValueError(
            f"stoi {stoi!r} contains content not matched by [A-Z](\\d+|n) "
            f"tokens (got tokens {stoi_parsed})"
        )

    # Reject silent truncation: every record must have a stoi token & vice-versa.
    if len(records) != len(stoi_parsed):
        raise ValueError(
            f"{fasta}: {len(records)} FASTA record(s) but stoi {stoi!r} has "
            f"{len(stoi_parsed)} token(s); these must match (tokens: {stoi_parsed})"
        )

    protein_entities = []  # [{"sequence","copy","orig_chain"}]
    na_chains = []
    seen_stoi_keys = set()
    for (_hdr, seq), (chain_chr, n) in zip(records, stoi_parsed):
        if chain_chr in seen_stoi_keys:
            raise ValueError(f"stoi {stoi!r}: chain {chain_chr} appears more than once")
        seen_stoi_keys.add(chain_chr)
        n = 1 if n == "n" else int(n)
        if n < 1:
            raise ValueError(f"stoi {stoi!r}: count for {chain_chr} must be >= 1")
        if is_nucleic_acid(seq):
            na_type = get_na_type(seq)
            na_chains.append({"sequence": seq, "type": na_type, "copy": n})
            print(f"  Found {na_type.upper()} chain {chain_chr} (copy: {n})")
            continue
        protein_entities.append({"sequence": seq, "copy": n, "orig_chain": chain_chr})

    # Copy-major round robin: one copy of every entity, then the next copy, so
    # A2B2 lays out as entity0, entity1, entity0, entity1. Same order as
    # `colab_a3m_to_yaml.split_colab_a3m_write_yaml` assigns chain letters.
    final = []
    max_copies = max((e["copy"] for e in protein_entities), default=0)
    for r in range(max_copies):
        for e in protein_entities:
            if r < e["copy"]:
                final.append(e["sequence"])
    parsed_path = f"{out_dir}/{k}_parsed.fa"
    with open(parsed_path, "w") as fasta_file:
        fasta_file.write(f">{k}\n")
        fasta_file.write(":".join(final))

    return parsed_path, na_chains, protein_entities

def main(args):
    fasta = args.fasta
    stoi = args.stoi
    out_dir = args.out_dir
    
    out_fa = parse_fasta(fasta, stoi, out_dir)
    print(out_fa)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument("--fasta", default=None, help="Path to input fasta")
    argparser.add_argument("--stoi", default="A1", help="stoichiometry information")
    argparser.add_argument("--out_dir", default=None, help="Path to output directory")
    args = argparser.parse_args()    
    main(args)
