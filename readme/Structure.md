# Structure

Structure prediction is stage 2 of the Thal-Kak pipeline. It consumes the data yaml emitted by [MSA](MSA.md) plus a model-specific config yaml, runs the chosen predictor across the requested seeds, computes per-seed/per-sample confidence, and stages everything into a flattened `common/` directory consumed by the downstream relaxation stage.

## Where this stage sits

```
MSA  ──►  data yaml + model yaml  ──►  [Structure]  ──►  common/*.pdb + confidence.csv  ──►  Relax
```

## Methods (`--model` / `--structure`)

| Option | Backend | Weights cache (default) |
|--------|---------|-------------------------|
| `boltz2` | Boltz-2 | `~/.boltz` |
| `chai1` | Chai-1 | `Structure/submodules/chai-lab/downloads/` |
| `protenix_v1` | Protenix v1 generation (`protenix_base_default_v1.0.0`, 368M base) | `Structure/submodules/protenix/checkpoint/` |
| `protenix_v2` | Protenix `protenix-v2` (464M scaled-up) | `Structure/submodules/protenix/checkpoint/` |
| `esmfold2` | ESMFold2 (MSA-free or MSA-augmented) | `~/.cache/huggingface/` |

All backends share the same input / output contract and run in the one unified environment (pixi or conda). Each downloads its own weights to the default cache location above on first run.

## CLI

```
thalkak structure --model boltz2 \
                  --data_config <output_dir>/<target>.yaml \
                  --model_config examples/model_config.yaml
```

## Inputs

- `--data_config`: data yaml, produced by the [MSA](MSA.md) stage (`thalkak msa` writes it to `<output_dir>/<target>.yaml`). The fields `job_name`, `output_dir`, and `seed` must be filled in. `seed` may be a single int or a list of ints. See [Data yaml schema](#data-yaml-schema).
- `--model_config`: model config yaml, keyed by model name — one section per model (`boltz2` / `chai1` / `protenix_v1` / `protenix_v2` / `esmfold2`); the requested model's section is extracted automatically. Default `examples/model_config.yaml`. See [Model yaml schemas](#model-yaml-schemas).

## Outputs

Under `<output_dir>/<model>_results_<target>_<job_name>[_timestamp]/`:

- `common/`: a flattened directory holding all decoy PDBs, the per-method confidence CSV (`<target>_results_summary.csv`), PAE / pLDDT PNGs, and `method_log.yaml`. The confidence CSV holds each backend's own scores, such as `ranking_score`. The `method_log.yaml` here is inherited from MSA, with `structure: <model>` appended. **This is the directory that top-5 selection and `relax` are pointed at.**
- Model-native sub-directories are preserved in each upstream tool's original layout (e.g. `seed_*/predictions/` for Protenix).

Chain IDs in the output PDBs are assigned by cycling copies before entities: copy 1 of every entity in `a3m` order, then copy 2 of every entity, and so on (e.g. with entities `[E1: copy=3, E2: copy=2]`, chain order is `E1, E2, E1, E2, E1`).

If the result root already exists, a timestamp suffix (`_YYYY_MM_DD_HH_MM_SS`) is appended so reruns don't clobber prior runs.

## Data yaml schema

The data yaml is produced by the [MSA](MSA.md) stage and consumed by every Structure runner. Full schema:

```yaml
a3m:
- paired_path: str(AlphaFold3-style a3m Path) | null
  unpaired_path: str(AlphaFold3-style a3m Path) | null
  copy: int
  type: str(protein|dna|rna)
- ...

ligand (Optional):
- smiles: str(smiles) # exclusive with ccd
  copy: int
- ccd: str(CCD ID)    # exclusive with smiles
  copy: int
- ...

templates (Optional):
- path: str(cif Path)
  chain_template: str(Chain)
  chain_query:
  - str(Chain)
  - str(Chain)
  - ...

job_name: str
output_dir: str(Path)
seed:
- int
- int
- ...
```

### `a3m` field

| Field | Meaning |
|-------|---------|
| `paired_path` | Path to the paired MSA for one sequence |
| `unpaired_path` | Path to the unpaired MSA for one sequence |
| `copy` | Number of copies of this sequence in the complex (e.g. an A2 complex → `2`) |
| `type` | Entity type: `protein`, `dna`, or `rna` |

When the query has 2+ distinct sequences, paired and unpaired MSAs must be prepared **separately for each sequence**. The MSA stage handles this automatically; for hand-prepared MSAs, see "Hand-preparing a3m" below.

### `templates` field

| Field | Meaning |
|-------|---------|
| `chain_template` | Which chain of the template to use |
| `chain_query` | Which predicted-sequence chains the template applies to |

<details>
<summary><b>Hand-preparing a3m (when bypassing the MSA stage)</b></summary>

Suppose you have an A1B1-stoichiometry protein and obtained the following ColabFold MSA:

```
Colabfold_MSA.a3m

#21,21  1,1   <-- The first line of a ColabFold MSA contains the length of each sequence
              <-- and the copy count. This must be removed before use.
>101    102
AAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBB
|<-----A chain----->||<-----B chain----->|
>Uniref_something
AAAAAAAAAAAAAAAAAAAAABBBBBBBBBBBBBBBBBBBBB
|<-----A chain----->||<-----B chain----->|

>101
AAAAAAAAAAAAAAAAAAAAA---------------------
|<-----A chain----->||<-----B chain----->|
>Uniref_something
AAAAAAAAAAAAAAAAAAAAA---------------------
|<-----A chain----->||<-----B chain----->|

>102
---------------------BBBBBBBBBBBBBBBBBBBBB
|<-----A chain----->||<-----B chain----->|
>Uniref_something
---------------------BBBBBBBBBBBBBBBBBBBBB
|<-----A chain----->||<-----B chain----->|
```

The **paired MSA for sequence A** keeps only A-chain information, with the aligned B-chain region stripped:

```
A_paired_MSA.a3m

>101    102
AAAAAAAAAAAAAAAAAAAAA
|<-----A chain----->|
>Uniref_something
AAAAAAAAAAAAAAAAAAAAA
|<-----A chain----->|
```

The **unpaired MSA for sequence A** similarly removes B-chain aligned-gap regions:

```
A_unpaired_MSA.a3m

>101
AAAAAAAAAAAAAAAAAAAAA
|<-----A chain----->|
>Uniref_something
AAAAAAAAAAAAAAAAAAAAA
|<-----A chain----->|
```

The **paired MSA for sequence B** keeps only B-chain information:

```
B_paired_MSA.a3m

>101    102
BBBBBBBBBBBBBBBBBBBBB
|<-----B chain----->|
>Uniref_something
BBBBBBBBBBBBBBBBBBBBB
|<-----B chain----->|
```

The **unpaired MSA for sequence B** similarly removes A-chain aligned-gap regions:

```
B_unpaired_MSA.a3m

>102
BBBBBBBBBBBBBBBBBBBBB
|<-----B chain----->|
>Uniref_something
BBBBBBBBBBBBBBBBBBBBB
|<-----B chain----->|
```

</details>

## Model yaml schemas

The pipeline ships defaults in `examples/model_config.yaml` — one section per model (`boltz2` / `chai1` / `protenix_v1` / `protenix_v2` / `esmfold2`), keyed by model name. Edit the section for the model you're invoking; the schemas below describe each section's keys.

<details>
<summary><b>Boltz-2</b></summary>

```yaml
n_samples: int
no_kernels: bool          # default: True. True disables Boltz's optimized (trifast) kernels; False uses them (faster / less memory, needs trifast installed)
output_format: str        # pdb | mmcif
recycling_steps: int      # default: 3
sampling_steps: int       # default: 200
subsample_msa: bool       # default: False. True enables Boltz's native MSA subsampler (random 1024 rows per recycle); omit/False = full MSA
constraints (Optional):
  - bond:
      atom1: [CHAIN_ID, RES_IDX, ATOM_NAME]
      atom2: [CHAIN_ID, RES_IDX, ATOM_NAME]
  - pocket:
      binder: CHAIN_ID
      contacts: [[CHAIN_ID, RES_IDX/ATOM_NAME], [CHAIN_ID, RES_IDX/ATOM_NAME]]
      max_distance: DIST_ANGSTROM    # default: 6.0
      force: false                   # default: false. If true, a potential is used to enforce the pocket constraint.
  - contact:
      token1: [CHAIN_ID, RES_IDX/ATOM_NAME]
      token2: [CHAIN_ID, RES_IDX/ATOM_NAME]
      max_distance: DIST_ANGSTROM    # default: 6.0
      force: false                   # default: false. If true, a potential is used to enforce the contact constraint.
```

Example:

```yaml
n_samples: 5
no_kernels: True
output_format: pdb
recycling_steps: 3
sampling_steps: 200
subsample_msa: True
constraints:
  - pocket:
      binder: C
      contacts: [[A, 24], [A, 112]]
  - pocket:
      binder: D
      contacts: [[B, 24], [B, 112]]
```

See the [Boltz-2 official docs](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md) for full details.

</details>

<details>
<summary><b>Chai-1</b></summary>

```yaml
num-trunk-samples:                  # default: 1
num-diffn-samples:                  # default: 5
num-diffn-timesteps:                # default: 200
recycle-msa-subsample:              # default: 0
num-trunk-recycles:                 # default: 3
constraint-path: path to constraint # default: null. If set, the constraints are used to enforce structure prediction.
use-esm-embeddings:                 # default: True
fasta-names-as-cif-chains:          # default: False
```

Example:

```yaml
num-trunk-samples: 1
num-diffn-samples: 5
num-diffn-timesteps: 200
recycle-msa-subsample: 0
num-trunk-recycles: 3
constraint-path:
use-esm-embeddings: True
fasta-names-as-cif-chains: False
```

See the [Chai-1 official README](https://github.com/chaidiscovery/chai-lab/blob/main/README.md) for full details.

</details>

<details>
<summary><b>Protenix (<code>protenix_v1</code> / <code>protenix_v2</code>)</b></summary>

The two methods run different Protenix generations. `model_name` is required and names the checkpoint: the shipped config sets `protenix_base_default_v1.0.0` (368M base) for `protenix_v1` and `protenix-v2` (464M scaled-up) for `protenix_v2`. Change it to select another checkpoint of the same generation, e.g. `protenix_base_20250630_v1.0.0` for `protenix_v1`. Both support MSA, RNA MSA, and templates.

```yaml
# protenix_v1
model_name: str           # Protenix checkpoint name, e.g. protenix_base_default_v1.0.0
N_cycle: int              # number of recycling iterations
N_sample: int             # number of diffusion samples
N_step: int               # diffusion steps per sample
data.msa.min_size.test: int | null   # Optional. MSA subsampling: null/omit = Protenix native per-recycle subsampling; set 16384 (featurization cap) to force full MSA (raw MSAs deeper than 16384 are truncated at featurization)
```

```yaml
# protenix_v2 — same keys (model_name is e.g. protenix-v2), plus:
use_tfg_guidance: bool    # enable Training-Free Guidance (TFG) sampling
```

**`use_tfg_guidance`** (`protenix_v2` only): when `True`, the runner turns on Protenix's Training-Free Guidance pass, which refines diffusion sampling without retraining the model, at the cost of extra inference time per sample. Leave `False` for vanilla sampling. It is skipped automatically for single-chain inputs, which trip a Protenix bug in the steric potential. Putting this key in a `protenix_v1` section is an error rather than a silent no-op.

`protenix_v2` weights are no longer downloadable from Protenix's official endpoint; see [the note in the top-level README](../README.md#install) for how they are fetched and verified. `protenix_v1` weights download normally on first run.

</details>

<details>
<summary><b>ESMFold2</b></summary>

```yaml
model_variant: str          # "biohub/ESMFold2" (full, MSA-capable) or "biohub/ESMFold2-Fast"
use_msa: bool               # only the full variant consumes MSA; false (or -Fast) skips a3m construction
num_loops: int              # trunk refinement iterations
num_sampling_steps: int     # diffusion sampling steps per sample
num_diffusion_samples: int  # samples produced per seed (each becomes one PDB)
msa_max_depth: int | null   # per-loop MSA row subsampling; null disables it
msa_column_mask_rate: float # MSA column masking; 0.0 disables it
```

These names match `esm.ESMFold2InputBuilder.fold()` / `ESMFold2Model.forward()`. The keys below do not: they are levers this runner applies to the loaded model, and each is off when absent.

```yaml
tf32: bool                  # TF32 on the fp32 matmul path (speed; changes numerics)
kernel_backend: str | null  # "cuequivariance" | "fused" -- trunk kernel path (speed; changes numerics)
cueq_msa: bool              # route the MSA encoder's trimul through cueq too (needs kernel_backend: cuequivariance)
chunk_size: int | null      # tile L^2 transients in the trunk (memory; can cost speed)
offload_lm: bool            # park the ESM-C backbone on CPU during folding (memory; no result change)
confidence_chunk: int | null # run the confidence head N diffusion samples at a time (memory)
```

`kernel_backend` and `chunk_size` call the model's own `set_kernel_backend()` / `set_chunk_size()`. The rest are this runner's: `tf32` is a PyTorch global, `cueq_msa` reaches the MSA encoder that `set_kernel_backend()` does not, and `offload_lm` / `confidence_chunk` install and remove their own hooks around the fold.

**`confidence_chunk`**: the confidence head, not the trunk, sets ESMFold2's memory peak — it replicates the pair representation to `(num_diffusion_samples, L, L, d_pair)` and runs a folding trunk over it, so the peak grows with the sample count. Everything after that replication is independent per sample, so the head can run in chunks and have its per-sample outputs concatenated. Set it to 1 (or another divisor of `num_diffusion_samples`) when a large complex runs out of memory. It applies to single-protein inference only; other call shapes pass through unchanged. Chunking is mathematically equivalent, but running the head on a smaller batch reorders float reductions, so scores shift within noise (measured on T1201 at `num_diffusion_samples: 4` with tf32 + cuequivariance: ranking_score by <= 1e-4, mean_plddt by <= 0.06, ranking order unchanged).

**Note:** ESMFold2 is language-model based. The `-Fast` variant (or `use_msa: false`) runs MSA-free. Kernel backends need compute capability >= 8.0 (bf16); on an older GPU the runner reports this and falls back to the pure-PyTorch path.

</details>

## Caveats

- The timestamp-suffix on rerun is intentional. A Relax stage pointed at a previous result root keeps working while a new run is in flight.
