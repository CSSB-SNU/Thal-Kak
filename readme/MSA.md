# MSA

MSA and template generation is stage 1 of the Thal-Kak pipeline. It takes a CASP-style FASTA and stoichiometry string and produces per-chain multiple sequence alignments, AF2 template hits, and the **data yaml** that every downstream stage consumes. RNA and DNA chains are handled automatically.

## Where this stage sits

```
FASTA + stoi  ──►  [MSA]  ──►  data yaml  ──►  Structure  ──►  Relax
```

## Methods (`--msa`)

| Option | Backend | What it does |
|--------|---------|--------------|
| `colab` | ColabFold (`colabfold_search`) via the configured env | MMseqs2 search against ColabFold DBs, then AF2 template lookup; the combined a3m is split per chain into paired / unpaired files |
| `custom` | none — your own alignment | No search. Splits the a3m given by `--a3m_path` per chain, exactly as the `colab` output is split |
| `mmseqs_cssb` | local MMseqs2 | Local MMseqs2 search against the databases under `db/` (no remote server) |
| `hhblits_cssb` | local HHblits | Local HHblits search against the databases under `db/` |
| `mmseqs_hhblits_cssb` | local MMseqs2 + HHblits | Runs both engines and merges the alignments |

`custom` needs no database either: pass `--a3m_path <file.a3m>` (`Method.a3m_path`
in a full-mode input) and that alignment is used for the protein chains instead of
searching for one. The file must be in ColabFold-complex format — the first line
is `#<comma-separated chain lengths>\t<comma-separated copy counts>` — because it
goes through the same per-chain paired / unpaired split as a `colab` a3m. A header
that disagrees with the declared entities is rejected rather than split into
alignments for the wrong sequences. RNA and DNA chains are unaffected and still
take the routes below.

`colab` needs no local database. The `*_cssb` modes run entirely locally against
the databases under `db/` and take an optional `--msa_config` (default:
`examples/msa_config.{mmseqs,hhblits,combined}.yaml`). Install those databases
with `./install_db.sh` (see [../install/README.md](../install/README.md)); where
they live is set by `db_paths.yaml` at the repository root.
See [../MSA/cssb_msa/README.md](../MSA/cssb_msa/README.md) and
[../MSA/cssb_template/README.md](../MSA/cssb_template/README.md).

RNA / DNA handling is automatic, regardless of `--msa`:
- Each FASTA record is checked against the nucleic-acid alphabet (`{A, C, G, T, U}`); records that match are routed out of the protein path.
- **RNA chains** → NHMMER-based MSA search against the local RNA MSA database (under `db/rna`), output as a3m.
- **DNA chains** → no MSA; the FASTA is referenced directly in the data yaml.

## CLI

```
thalkak msa --msa colab \
            --seq <target.fa> \
            --stoi A1 \
            [--output_dir DIR]
```

## Inputs

- `--seq`: CASP FASTA. One record per distinct sequence, in chain order.
- `--stoi`: stoichiometry string, e.g. `A1`, `A2B1`, `A1B1C2`. `An` (literal `n`) marks an unknown copy count for chain `A` and is treated as `1`.
- `--a3m_path`: required by `--msa custom`, rejected by the other modes. The ColabFold-format a3m to use for the protein chains.
- `--output_dir`: optional override; defaults to the directory containing the FASTA.

## Outputs

Under `<output_dir>/msa/<msa_method>/`:

| File | Produced for |
|------|--------------|
| `<target>_paired_msa_chains_<x>.a3m` | each protein chain |
| `<target>_unpaired_msa_chains_<x>.a3m` | each protein chain |
| `<target>_na_<i>.a3m` | each RNA chain |
| `<target>_na_<i>.fa` | each DNA chain |
| `<pdb>_<chain>.cif` | AF2 template hits, when found |
| `method_log.yaml` | `{msa, seq, stoi, template_config, templates}` (the `*_cssb` engines also record `dbs`, `merge`, `dedup`, `caps`); inherited by Structure |

And one **data yaml** at `<output_dir>/<target>.yaml` matching the [data yaml schema](Structure.md#data-yaml-schema). The header reminds you to fill in `job_name`, `output_dir`, and `seed` before running structure prediction; in `thalkak full` mode these are filled in automatically.

## Skip-if-already-done

Each call writes / checks `method_log.yaml`. `(msa, seq, stoi)` must match a
previous run and `*.a3m` files must already exist; then `colab` reuses the
cached files as is.

For the `*_cssb` modes the match is stricter, because a `--msa_config` swap can
change the alignment: the recorded `dbs`, merge mode key, `caps` and template
settings must also match, and for `dedup`-reading merge modes the `dedup` mode
too. Only settings the active merge mode actually reads are compared, so
editing a knob that mode ignores still reuses the existing MSA. Any mismatch
rebuilds rather than silently reusing.

RNA chains are skipped independently, per chain: an existing non-empty
`<target>_na_<i>.a3m` from a run with the same `(msa, seq, stoi)` is reused.

## Caveats

- Stoichiometry letters do not have to start at `A`; they are paired positionally with FASTA records.
- Chain IDs in downstream structure outputs are assigned in the order chains appear in the data yaml (see the templates-field section of [Structure.md](Structure.md)).
- The ColabFold raw a3m contains a header line like `#L1,L2  C1,C2` that must be removed when hand-preparing a3m. The pipeline handles this for you automatically.