# Installing the local databases

`install_db.sh` at the repository root installs the databases the local MSA
modes read — `--msa mmseqs_local`, `--msa hhblits_local`, `--msa
mmseqs_hhblits_local` — plus the local template search snapshot and the RNA
databases the RNA/RNP path needs.

This file is what you need before running it: what it installs, how much disk
that takes, and the commands.

## Contents

- [What gets installed](#what-gets-installed)
- [Installed size on disk](#installed-size-on-disk)
- [Quick start](#quick-start)

## What gets installed

Four families. They differ by three orders of magnitude in size — see
[Installed size on disk](#installed-size-on-disk) — and come from different
sources, so `--family` is required and has no default.

| Family | What | Needed by |
|---|---|---|
| `mmseqs` | the MMseqs2 search databases | `mmseqs_local`, `mmseqs_hhblits_local` |
| `template` | the local template search snapshot | template search on any local mode |
| `rna` | Rfam + RNAcentral, clustered | any RNA or RNP target |
| `hhblits` | the HH-suite (FFindex) builds | `hhblits_local`, `mmseqs_hhblits_local` |

**The RNA databases are not optional.** The public ColabFold API returns protein
alignments only, so without them `msa_generation` refuses rather than align an
RNA chain against nothing.

**Multimers on `hhblits_local` also need the mmseqs `uniref30_2302`** with its
`db_mapping` / `db_taxonomy` sidecars, because the HH-suite UniRef100 carries no
taxonomy and pairing is delegated to mmseqs.

**`install_db.sh` repairs the hhblits `uniref100_2026_01` index as it installs.**
Nine of its cs219 prefilter records came out of the original build empty — zero
column states. hhblits scans every cs219 record on every search, so those nine put
an error line in every log and make it die with SIGSEGV on short queries: a crash,
not a worse alignment, and non-monotonic in query length, so no guard on the
caller's side avoids it. The installer removes those nine entries. They held no
usable profile in the first place, so nothing that worked is given up — what goes
away is the crash. The other 39,312,371 clusters and all of the a3m and hhm data
are untouched. A copy that is already on disk is repaired the same way on a
re-run, and one that is already repaired is left alone.

## Installed size on disk

Every database `install_db.sh` can install. Figures are `du -sb` measurements of a
completed install (2026-08-19), not manifest estimates. Archives are deleted after a
successful install and are not counted here.

### mmseqs db

| Database | Installed |
|---|---:|
| `mgnify_clusters` | 692.1 GiB |
| `uniref100_2026_01` | 489.3 GiB |
| `uniref30_2302` | 371.5 GiB |
| `bfd_reduced` | 142.3 GiB |
| `logan_human` | 102.0 GiB |
| `envhog_std` | 90.3 GiB |
| **Total (6)** | **1,887.5 GiB — 1.84 TiB** |

### hhsuite db

| Database | Installed |
|---|---:|
| `bfd` | 1,772.4 GiB |
| `uniref100_2026_01` | 1,301.5 GiB |
| `uniref30_2302` | 261.2 GiB |
| `mgnify_clusters` | 150.3 GiB |
| `logan_nonhuman` | 55.9 GiB |
| `envhog` | 11.5 GiB |
| `logan_human` | 2.0 GiB |
| **Total (7)** | **3,554.8 GiB — 3.47 TiB** |

### rna db

| Database | Installed |
|---|---:|
| `rnacentral` | 26.9 GiB |
| `rfam` | 1.2 GiB |
| **Total (2)** | **28.1 GiB** |

### template db

| Database | Installed |
|---|---:|
| `BioMolDB_20260224` | 81.2 GiB |
| **Total (1)** | **81.2 GiB** |

These figures come from one host, and which of them reproduce depends on the host.

- **The six mmseqs databases** are indexed with `mmseqs createindex --split 0`,
  which splits the index into `db.idx.0 .. db.idx.N` when it does not fit in the
  memory free at build time, so any of their figures can move by several GiB with
  the split count.
- **The two rna databases** are clustered from EBI's CURRENT releases, so their
  size follows what EBI publishes on the day you build.

### All families

| Family | Installed |
|---|---:|
| hhblits (7) | 3.47 TiB |
| mmseqs (6) | 1.84 TiB |
| template (1) | 0.08 TiB |
| rna (2) | 0.03 TiB |
| **Total (16)** | **5.42 TiB** |

Peak usage during installation is higher than the final figure, because an archive sits
on disk next to the tree it is being unpacked into. Installing one family at a time
keeps that overhead to a single archive and needs about **5.7 TiB**; fetching everything
up front first needs 6.4 TiB.

## Quick start

Everything the installer runs — `zstd`, `aria2c`, `mmseqs`, `makehmmerdb`,
`esl-sfetch`, and the Python that reads `db_paths.yaml` — comes from the project
environment, so create and activate that first.

```bash
# create and activate the project environment first, then run these inside it

./install_db.sh --family mmseqs                             # install all mmseqs db
./install_db.sh --family mmseqs uniref30_2302 bfd_reduced   # only the named ones
./install_db.sh --family rna                                # install the RNA databases
./install_db.sh --family all                                # all four
./install_db.sh --family mmseqs --status                    # what is already there
```

