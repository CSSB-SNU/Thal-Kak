# MSA/cssb_msa

Local CSSB MSA pipeline. `build_a3m()` (`mmseqs/build.py`) takes a
query fasta + stoi string (e.g. `A1`, `A1B1`) and emits one ColabFold-complex master a3m
using local mmseqs2 DBs (UniRef30 + UniRef100 + bfd_reduced +
mgnify_clusters + logan_human + envhog_std as defaults). Output is
ColabFold-complex format
identical to `--msa colab`, so downstream Structure / Relax code is
unchanged.

The package is split into 5 engine/responsibility sub-packages:
`mmseqs/` (mmseqs builder + runner + search_*), `hhblits/` (hhblits
builder + runner + search), `combined/` (mmseqs+hhblits merge),
`plot/` (coverage + Neff), `common/` (input parse + db_registry
+ dedup + assemble). Top-level `__init__`
re-exports the public API (`build_a3m`, `build_a3m_hhblits`, `DBSpec`,
`DEFAULT_REGISTRY`, `parse_inputs`, `ParsedInputs`, `parse_stoi`,
`BuildResult`, `PlotResult`, `plot_cssb_msa`) so external callers see
no change.

## Prereqs

- **`db_paths.yaml` at the repo root says where the databases are** — by
  default `db/msa_mmseqs/` (mmseqs), `db/msa_hhblits/` (HHblits),
  `db/template/<snapshot>/` and `db/rna/`. Each database sits in its own
  directory named after its registry key, files inside stemmed `db`
  (`common/db_registry.py`). To keep a format, or one database, on another
  disk, name it in that file before installing — `install_db.sh` reads it and
  installs where it says.
- **Multimers need the mmseqs UniRef30 database in every mode**, `hhblits_cssb`
  included — chain pairing always runs through mmseqs `search_pair` against
  `db/msa_mmseqs/uniref30_2302/`, plus its `db_mapping` / `db_taxonomy`
  sidecars (`./install_db.sh --family mmseqs uniref30_2302`). Single-chain targets skip that
  step entirely.
- mmseqs, hhblits and HMMER come from the active env's PATH; a missing binary
  fails loud.

## Usage

```bash
# Full pipeline (templates + plots default-on). Set `msa: mmseqs_cssb` in the
# input yaml's Method section; see examples/input.yaml.
thalkak full --input examples/input.yaml

# MSA only — output at <seq_parent>/msa/mmseqs_cssb/
thalkak msa --msa mmseqs_cssb --seq <target.fa> --stoi A1

# HHblits engine, or the mmseqs+hhblits combined engine (same flags, swap --msa)
thalkak msa --msa hhblits_cssb        --seq <target.fa> --stoi A1
thalkak msa --msa mmseqs_hhblits_cssb --seq <target.fa> --stoi A1

```

All MSA settings (DB list, search params, template/plot toggles) live in a
per-mode config yaml — [`msa_config.mmseqs.yaml`](../../examples/msa_config.mmseqs.yaml),
[`msa_config.hhblits.yaml`](../../examples/msa_config.hhblits.yaml), or
[`msa_config.combined.yaml`](../../examples/msa_config.combined.yaml) — auto-selected by
`--msa <mode>` when no `--msa_config` is given. Pass `--msa_config <path>`
(on `thalkak msa|full`) to swap the file for one invocation; otherwise edit
the default in place. Each file is self-contained: the engine block(s) (`dbs`
+ search knobs; the combined yaml has both `mmseqs:` and `hhblits:`) plus its
own `merge:` (`mode: leveled | cap_walk`), `caps:` (per-DB and total depth
limits), `template:` and `plot:` blocks.

## Output layout

```
{msa_dir}/
├── {target}.a3m                                 # master a3m (inference input)
├── {target}_unpaired_msa_chains_{a,b,...}.a3m   # per-chain split
├── {target}_paired_msa_chains_{a,b,...}.a3m
├── {target}.yaml                                # data yaml (with templates:)
├── method_log.yaml
├── {target}_env/                                # template search hits
│   ├── pdb70.m8
│   └── templates_<query_id>/<pdb>.cif × N
└── plots/                                       # MSA depth diagnostics
    ├── <chain>_coverage.png                     # merged (split a3m, post-dedup)
    ├── <chain>_neff.png                         # merged + per-DB curve overlay
    ├── multimer_coverage.png                    # complex only
    ├── per_db/<db>/<chain>_{coverage,neff}.png  # raw a3m per (chain × DB)
    └── neff_summary.json                        # incl. per_db scalars per chain
```

Per-DB plots cover all DBs whose raw a3ms exist under `_workdir/raw/`:
the default mmseqs inference DBs (`uniref30_2302`,
`uniref100_2026_01`, `bfd_reduced`, `mgnify_clusters`,
`logan_human`, `envhog_std`).
The `pair/` subdir is intentionally skipped (paired multimer rows are
chain-aligned and need a different visualization). Disable per-DB
plots by editing the default yaml: set `plot.per_db_enable: false`
(or `plot.enable: false` for all plots). Neff cost on large env DBs
is bounded by `plot.subsample_cap` (default 10000, a few GB peak per
chain × DB).

Output layout: `<base_dir>/msa/mmseqs_cssb/`, where `<base_dir>` =
`--output_dir` (msa subcommand) / `--base_dir` (full subcommand), or
the FASTA's parent dir if neither is set.

## Caveats

- Default DB load mode is mmap (mode 2) and requires `.idx` sidecars.
  Missing `.idx` auto-degrades to mode 0 (slower wall, much higher RAM).
  Run `mmseqs createindex` per DB to bring mmap back.
- Results differ numerically from ColabFold API: cssb searches each
  environmental DB separately, whereas the API uses the single merged
  `colabfold_envdb_202108`. Expected, not a bug.
- The two engines number their `dbs` list differently. `mmseqs_cssb`
  requires `uniref30_2302` first (it is the Stage A profile *and* the
  pairing DB, and the run is refused otherwise); `hhblits_cssb` takes
  whichever UniRef DB you put first as its primary pass — UniRef100 by
  default — and pairs through uniref30 regardless. Copying one `dbs`
  list into the other config is therefore not safe.
