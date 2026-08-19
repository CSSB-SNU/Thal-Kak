# Relax

Relaxation is the final stage of the Thal-Kak pipeline. Given a directory of
decoy PDBs, it writes relaxed copies into `<decoy_dir>/relaxed/<method>/`. In
`full` pipeline mode this stage relaxes the per-job top-5 (selected by the
model's own confidence) in place.

## Where this stage sits

```
Structure  ──►  top-5 (top5/<job>/model_{1..5}.pdb)  ──►  [Relax]  ──►  top5/<job>/relaxed/<relax>/
```

## Methods (`--relax`)

| Option | Behavior |
|--------|----------|
| `none` | Pass-through. Copies each validated `*.pdb` to `<decoy_dir>/relaxed/none/<name>_unrelaxed.pdb` (no minimization). |
| `openmm` | All-atom OpenMM minimization with the Amber-family force-field stack (`amber19-all` for protein/NA, GLYCAM_06j for glycans, GAFF-2.11 for ligands, monatomic ions) + implicit solvent (OBC2 default; GBn2 or vacuum selectable via the relax config, with automatic vacuum fallback when GB lacks radii). Every component is relaxed with its native force field; anything that can't be typed is frozen at its input coordinates rather than dropped. Writes `<name>_relaxed_openmm.pdb` plus per-decoy `<out_prefix>.energy.yaml` (`E_init` / `E_final`), merged into `energies.yaml`. |

Both methods skip decoys whose output already exists, so re-running over a
partially-relaxed directory is cheap.

## Pre-relaxation validation

Before either method runs, the orchestrator runs
[`Relax/script/validate.py`](../Relax/script/validate.py) over every decoy and
writes normalized copies into a scratch directory; both methods then read those.
This centralizes structural fixes so relaxation starts from chemically sane
input, and `none` hands the same normalized geometry downstream:

- **Prochiral methyl naming** — Val `CG1`/`CG2` and Leu `CD1`/`CD2` (and their
  attached H) are relabeled to the standard PDB/IUPAC handedness. Names only, no
  atoms move.
- **C-terminal carboxylate** — the whole `-COO(-)` group (both C–O bonds, the
  ~120° angles, planarity) is validated and rebuilt to ideal sp2 geometry when
  distorted; missing `O`/`OXT` are created.

The rewrite is text-level: B-factors, occupancies and residue numbering are
preserved, so per-residue pLDDT keying is unaffected.

## Configuration (`--relax_config`)

`openmm` reads a relax config yaml (default `examples/openmm.yaml`) for solvent
choice and restraint parameters. Override with `--relax_config <yaml>`.

## CLI

```
thalkak relax --decoy_dir <dir of decoy PDBs> --relax openmm
```

## Inputs / outputs

- **Input**: `<decoy_dir>/*.pdb`
- **Output**: `<decoy_dir>/relaxed/<method>/`
  - `<name>_relaxed_openmm.pdb` (openmm) or `<name>_unrelaxed.pdb` (none)
  - `energies.yaml`: `{ <decoy_basename>: { E_init, E_final, tool } }` (each worker
    drops a per-decoy `*.energy.yaml`, merged into `energies.yaml` by the orchestrator)

## What `openmm` does

Adapted from AlphaFold2's relaxation (`alphafold.relax`), with implicit solvent
(instead of AF2's vacuum), pLDDT-dependent flat-bottom coordinate restraints
(high-confidence regions held tight, low-confidence regions freer), and a
broadened force-field stack that keeps and parametrizes heterogens (glycans via
GLYCAM_06j, small-molecule ligands via GAFF-2.11 with AM1-BCC charges, ions).
Violation-informed iterative minimization follows the AF2 scheme. Runs in the
unified `thalkak` conda env.

## Caveats

- Point at the flattened `common/` directory (or a top-5 directory), **not**
  the raw `seed_*/` subdirectories produced by some predictors.
- `relax=none` still writes files (with an `_unrelaxed` suffix) so downstream
  consumers see a consistent input contract.
