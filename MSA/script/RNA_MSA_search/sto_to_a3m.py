import os
import glob
import argparse
from Bio import AlignIO


def sto_to_a3m(sto_path, output_path):
    """Convert a Stockholm alignment to a3m. Match/insert states are defined by
    the query — the alignment's FIRST record — not by the Stockholm markup."""
    try:
        alignment = AlignIO.read(sto_path, "stockholm")
        num_seqs = len(alignment)
        if num_seqs == 0:
            return False, 0

        query_record = alignment[0]
        query_seq_str = str(query_record.seq)

        # Columns where the query has a base = the match states.
        match_cols = [
            i for i, char in enumerate(query_seq_str) if char not in ("-", ".")
        ]
        match_cols_set = set(match_cols)

        with open(output_path, "w") as f_out:
            for record in alignment:
                header = f">{record.id}"
                if record.description and record.description != "<unknown description>":
                    header += f" {record.description}"
                f_out.write(f"{header}\n")

                # a3m: match column -> uppercase base, or '-' for a gap;
                # insert column -> lowercase base, gaps dropped entirely.
                raw_seq = str(record.seq)
                new_seq = []

                for i, char in enumerate(raw_seq):
                    if i in match_cols_set:
                        if char in ("-", "."):
                            new_seq.append("-")
                        else:
                            new_seq.append(char.upper())
                    else:
                        if char not in ("-", "."):
                            new_seq.append(char.lower())

                f_out.write("".join(new_seq) + "\n")

        print(f"[Converted] {sto_path} -> {output_path} ({num_seqs} seqs)")
        return True, num_seqs

    except Exception as e:
        print(f"Error converting {sto_path}: {e}")
        return False, 0


def convert(input_path, output_path):
    """Convert .sto to .a3m. Input can be a single file."""
    if os.path.isfile(input_path):
        sto_to_a3m(input_path, output_path)
    else:
        print(f"Input {input_path} is not a file. Please provide a valid .sto file.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input .sto file")
    parser.add_argument("--output", required=True, help="output .a3m file")
    args = parser.parse_args()
    convert(args.input, args.output)
