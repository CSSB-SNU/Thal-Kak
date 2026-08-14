# Local Template Search

Local replacement for ColabFold's template artifacts. The default CSSB
dispatch is engine-specific: `mmseqs_cssb` searches BioMolDB with mmseqs
using the persisted UniRef30 profile, while `hhblits_cssb` uses the local
hmmbuild+hmmsearch engine (`engine_hmmer`, no external container) seeded
with the hhblits UniRef100 a3m.

## What it does

Given a protein query plus either a UniRef-derived a3m (hmmer path) or
the persisted UniRef30 profile from `mmseqs_cssb` (mmseqs path), produces
ColabFold-format outputs that Protenix / Boltz-2 / Chai-1 already know how
to consume:

```
out_dir/
├── pdb70.m8                       # 13-column TSV
└── templates_<query_id>/
    └── <pdb_id>.cif × N           # gunzipped, plain mmCIF
```

The downstream `Structure/script/protenix/process_msa_to_json.py`
finds the m8 via `cif_path.parent.parent / "pdb70.m8"`, so this layout
plugs in without any reader changes.

## Prereqs

Everything lives under one root, `db/template/<snapshot>/` by default. Both
halves come from `db_paths.yaml` at the repo root: `template_root` is the
directory holding snapshots, and `template_snapshot` picks the one in use, so
several can sit side by side and a rebuild is a one-line change.
`mmseqs` and HMMER come from the active env's PATH.

```
db/template/<snapshot>/
├── cif/raw/<pdb[1:3]>/<pdb>.cif.gz     # BioMolDB structures, both engines
├── mmseqs/<snapshot lowercased>_pdb    # BioMolDB mmseqs DB, mmseqs engine
├── fasta/pdb_seqres_protein.fasta      # protein-only seqres, hmmer engine
├── metadata/cif_metadata.tsv           # deposition dates (fallback)
└── release_dates.tsv                   # initial release dates (date cutoff)
```

`scripts/build_seqres_from_biomoldb.py` regenerates the seqres FASTA from a
BioMolDB snapshot; it is only needed when the snapshot itself is replaced.

## Usage (auto via `--msa mmseqs_cssb` / `--msa hhblits_cssb`)

Template search runs automatically inside `thalkak --msa mmseqs_cssb`
(and `hhblits_cssb`) after `build_a3m`, so
`<msa_dir>/<target>_env/pdb70.m8` and `templates_<query_id>/<pdb>.cif`
exist by the time downstream models start.

```bash
# Templates default-on (any structure model consumes them)
thalkak full --msa mmseqs_cssb --structure boltz2 ...
```

Knobs live in each per-mode config yaml's `template:` block
(`examples/msa_config.{mmseqs,hhblits,combined}.yaml`; loaded by default;
pass `--msa_config <path>` to swap the file for one invocation):

```yaml
template:
  enable:    true                 # false → skip template search
  engine:    auto                 # mmseqs_cssb→mmseqs, hhblits_cssb→hmmer.
                                  # Absent from the combined yaml: that mode
                                  # always runs both engines and ignores it.
  query_a3m: uniref30             # hmmer seed (uniref100 in the hhblits/combined
                                  # yaml); ignored by the mmseqs engine
  max_date:  "3000-01-01"         # release-date cutoff (both engines); 3000-01-01 = none
  max_hits:  20                   # per chain, both engines
```

Edit the default yaml in place to change behavior.

## Standalone (Python API)

For ad-hoc template search on a single chain without re-running MSA, call the
engine directly:

```python
from MSA.cssb_template.engine_hmmer import run_local_hmmer_template_search

result = run_local_hmmer_template_search(
    query_sequence="MSV...",
    query_a3m_path="/path/to/uniref30.a3m",
    out_dir="/path/to/out_dir",
    query_id="101",
    query_chain_id="A",
)
# result.template_entries → list[TemplateEntry] for the data yaml
# result.m8_path → out_dir/pdb70.m8
```

`TemplateEntry.to_dict()` produces the `{path, chain_template,
chain_query}` dict that goes into `data.yaml:templates`.
