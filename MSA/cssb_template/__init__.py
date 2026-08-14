"""Local template search against the BioMolDB seqres/cif snapshot.

Two engines, both emitting the same `<msa_dir>/<target>_env/`
contract (`pdb70.m8` + per-chain cifs):

  - `engine_mmseqs.run_mmseqs_for_msa_dir`  — mmseqs vs BioMolDB (mmseqs_cssb)
  - `engine_hmmer.run_hmmer_for_msa_dir`    — local hmmbuild+hmmsearch (hhblits_cssb)

The shared engine-agnostic core (dataclasses, multi-chain orchestrator,
query-a3m resolution, cif gunzip) lives in `_common.py`.
"""
