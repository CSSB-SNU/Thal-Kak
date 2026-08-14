"""Combined mmseqs+hhblits MSA mode (`mmseqs_hhblits_cssb`).

Runs BOTH the mmseqs and hhblits per-source builders into separate workdirs,
then performs a Stage-2 cross-source merge whose shape depends on `merge.mode`:
under `leveled` (the default) each chain's per-DB a3ms from both sources are
grouped by DB kind and filtered once per group; under `cap_walk` the two
sources' already-merged a3ms are concatenated mmseqs-first and re-capped.
Paired rows are merged mmseqs-first in both modes, and template hits from both
engines are merged mmseqs-first. See `build.py` / `merge.py` /
`merge_templates.py`.
"""
