<!-- Draft. Lives at aws_open_data/README.md in CSSB-SNU/Thal-Kak and is also
     uploaded to the bucket root as README.md. Adapted from the Hugging Face
     dataset card; only what differs between Hugging Face and AWS was changed.
     Before launch: point the nine matching rows of
     install/manifest/databases.tsv at this bucket, and remove the status note. -->

# Thal-Kak Multiple Sequence Alignment Databases

> **Status: in preparation.** The bucket `s3://thal-kak-msa` and its Registry
> of Open Data page are not live yet, so the links and commands below do not
> work yet.

The sequence databases that the local MSA modes of
**[Thal-Kak](https://github.com/CSSB-SNU/Thal-Kak)** search —
`--msa mmseqs_local`, `--msa hhblits_local` and `--msa mmseqs_hhblits_local` —
in a public Amazon S3 bucket listed on the
[Registry of Open Data on AWS](https://registry.opendata.aws/thal-kak-msa/).

> **With Thal-Kak, install these with
> [`install_db.sh`](#with-install_dbsh-from-thal-kak).** Each database is one
> `.tar.zst` archive, the same file as on our Hugging Face repository; the
> installer downloads it, checks its SHA-256, extracts it and sets it up in the
> layout the pipeline expects.

## At a glance
| | |
|---|---|
| Bucket | `s3://thal-kak-msa/`, region `us-east-1`; no AWS account needed |
| Archives | 9 (`mmseqs/` 4, `hhblits/` 5), one per database |
| Download | **398.9 GiB** total |
| After extraction | 1.77 TiB, plus 0.70 TiB of MMseqs2 search indexes built on your machine |
| Compression | zstd (`zstd` required to unpack) |
| Licenses | mixed, per database — CC0 1.0 / CC BY 4.0 / CC BY-SA 4.0 / **no grant** (Logan). See [License and attribution](#license-and-attribution). |

## What is in this bucket

> **These are derived databases, not upstream mirrors.** The sequences of
> UniRef100, MGnify and the reduced BFD were re-clustered, EnVhogDB was
> re-clustered at its upstream `standard` level, and Logan was built into an
> MMseqs2 expandable-profile database; `hhblits/envhog` is upstream's own
> HH-suite build. So the counts below are ours and will not match numbers
> computed on the upstream distribution. What was done to each source is recorded in
> [`LICENSE.txt`](https://thal-kak-msa.s3.amazonaws.com/LICENSE.txt), under `Changes`.

```
s3://thal-kak-msa/
├── README.md                this document
├── LICENSE.txt              license, attribution, citation and changes per database
├── mmseqs/<key>.tar.zst     one MMseqs2 database per archive
└── hhblits/<key>.tar.zst    one HH-suite database per archive
```

Each archive holds one complete database built from one upstream source. It is
the same file as the corresponding archive on our Hugging Face repository, and
its SHA-256 is listed in Thal-Kak's `install/manifest/databases.tsv`. The files
inside are named after the key, e.g. `mgnify_clusters_seq` or
`envhog_a3m.ffdata`. Download only the archives you need.

### `mmseqs/` — MMseqs2 expandable profile databases

Searched by `--msa mmseqs_local`.
The `.idx` search index is **not** shipped — `install_db.sh` builds it locally,
which is where the Index column goes. Index sizes depend on the host and can
differ from these by several GiB, because `mmseqs createindex` splits the index
to fit the memory free at build time.

| Archive | Download | Extracted | Index (local) | Representatives | Members |
|---|--:|--:|--:|--:|--:|
| `mmseqs/mgnify_clusters.tar.zst` | 126.9 GiB | 202.3 GiB | 470.8 GiB | 313,067,917 | 717,738,164 |
| `mmseqs/envhog_std.tar.zst` | 20.7 GiB | 36.7 GiB | 53.6 GiB | 11,742,979 | 129,896,064 |
| `mmseqs/bfd_reduced.tar.zst` | 18.6 GiB | 27.6 GiB | 114.7 GiB | 51,652,611 | 65,984,053 |
| `mmseqs/logan_human.tar.zst` | 12.0 GiB | 28.1 GiB | 73.9 GiB | 61,657,544 | 71,364,503 |

Each archive holds four MMseqs2 databases. Each is a data file plus its
`.index` and `.dbtype`, so 12 files per archive; the archive also carries the
`<key>_h` links to `<key>_seq_h`, through which MMseqs2 reads headers.

| Database | Holds |
|---|---|
| `<key>` | a consensus sequence for each cluster; the prefilter searches these |
| `<key>_seq` | every member sequence |
| `<key>_aln` | the alignment of each cluster's members to its representative |
| `<key>_seq_h` | the headers of the member sequences |

`mmseqs expandaln` uses `_aln` and `_seq` to add the members of every hit
cluster to the alignment.

### `hhblits/` — HH-suite (FFindex) databases

Searched by `--msa hhblits_local`.

| Archive | Download | Extracted | Clusters (cs219) | Member sequences |
|---|--:|--:|--:|--:|
| `hhblits/uniref100_2026_01.tar.zst` | 157.2 GiB | 1,301.5 GiB | 39,312,371 ¹ | 475,217,228 |
| `hhblits/mgnify_clusters.tar.zst` | 33.7 GiB | 150.3 GiB | 31,779,642 | 278,185,288 |
| `hhblits/logan_nonhuman.tar.zst` | 24.2 GiB | 55.9 GiB | 28,702,814 | 130,348,747 |
| `hhblits/envhog.tar.zst` | 4.7 GiB | 11.5 GiB | 2,203,457 | 25,550,069 |
| `hhblits/logan_human.tar.zst` | 0.8 GiB | 2.0 GiB | 1,645,263 | 7,385,713 |

¹ After setup. The `cs219` index in the archive lists 39,312,380 records, 9 of
them empty; `install_db.sh` removes those 9.

Each archive holds three FFindex databases. Each is an `.ffdata` file plus its
`.ffindex`, so 6 files per archive.

| Database | Holds |
|---|---|
| `<key>_a3m` | the multiple sequence alignment of each cluster |
| `<key>_hhm` | profile HMMs; not every cluster has one |
| `<key>_cs219` | each cluster in the column-state alphabet that the HHblits prefilter scans |

## What is *not* here

`install_db.sh` installs seven more databases that this bucket does not hold.
Three are in our [Hugging Face repository](https://huggingface.co/datasets/cssbsnu/Thal-Kak_local_db);
the other four come from their upstream providers.

| Database | Installer command | `install_db.sh` fetches it from |
|---|---|---|
| MMseqs2 UniRef100 2026_01 | `--family mmseqs uniref100_2026_01` | Hugging Face |
| HH-suite UniRef30 2023_02 | `--family hhblits uniref30_2302` | Hugging Face |
| Template snapshot `BioMolDB_20260224` | `--family template` | Hugging Face |
| MMseqs2 UniRef30 2023_02 | `--family mmseqs uniref30_2302` | `opendata.mmseqs.org` (the ColabFold build) |
| HH-suite BFD (full) | `--family hhblits bfd` | `storage.googleapis.com/alphafold-databases` |
| Rfam | `--family rna rfam` | EMBL-EBI |
| RNAcentral | `--family rna rnacentral` | EMBL-EBI |

Four consequences worth knowing before you plan a download:

- **Thal-Kak's default database lists reach beyond this bucket.**
  `mmseqs_local` also searches `uniref30_2302` and `uniref100_2026_01`;
  `hhblits_local` also searches `uniref30_2302` and `bfd`.
- **Heteromers on `hhblits_local` also need the MMseqs2 `uniref30_2302`.** The
  HH-suite UniRef100 carries no taxonomy, so multimer pairing is delegated to
  mmseqs, which needs that database's `db_mapping` / `db_taxonomy` sidecars.
  `install_db.sh` refreshes them by default from
  `uniref30_2302_newtaxonomy.tar.gz`.
- **`logan_nonhuman` exists in HH-suite format only.** Its MMseqs2 build
  extracts to 1.4 TB and is not published anywhere. This asymmetry is
  deliberate.
- **RNA and RNP targets need the `rna` family**, which is not here at all — it
  is built locally from Rfam and RNAcentral.

## Getting the databases

Extract the archives onto local disk and search there, not from S3: the MMseqs2
prefilter reads its whole index on every search, and HHblits makes many small
random reads.

### With `install_db.sh` from Thal-Kak

The installer downloads each archive, checks its SHA-256, extracts it into
place atomically, normalizes the file stems to the layout `db_paths.yaml`
expects, and builds the MMseqs2 `.idx` where one is needed.

```bash
git clone https://github.com/CSSB-SNU/Thal-Kak.git
cd Thal-Kak
# create and activate the project environment first — it provides
# zstd, aria2c and mmseqs, which the installer calls.

./install_db.sh --family mmseqs                        # every mmseqs database
./install_db.sh --family mmseqs mgnify_clusters        # just this one
./install_db.sh --family hhblits envhog logan_human    # several by name
./install_db.sh --family template                      # template snapshot for the default search
./install_db.sh --family mmseqs --status               # what is already installed
```

Databases that are not in this bucket come from the sources listed in
[What is not here](#what-is-not-here). To install somewhere other than
`<repo>/db`, edit `db_paths.yaml` **before** running the installer — it reads
that file and never writes it.

## License and attribution

**The Apache-2.0 license on the Thal-Kak source code does not apply to anything
in this bucket.** Each database carries the license of the source it derives
from; no database mixes sources. The authoritative record — upstream license,
the license we place on our database, attribution, citation, and exactly what
we changed — is [`LICENSE.txt`](https://thal-kak-msa.s3.amazonaws.com/LICENSE.txt). The summary:

| Archives | License we grant | Source |
|---|---|---|
| `hhblits/uniref100_2026_01` | CC BY 4.0 | UniProt UniRef100, release 2026_01 |
| `mmseqs/bfd_reduced` | CC BY-SA 4.0 | BFD reduced subset, via the AlphaFold databases |
| `mmseqs/mgnify_clusters`, `hhblits/mgnify_clusters` | CC0 1.0 | MGnify protein database, release 2024_04 |
| `mmseqs/envhog_std`, `hhblits/envhog` | CC BY 4.0 | EnVhogDB |
| `mmseqs/logan_human`, `hhblits/logan_human`, `hhblits/logan_nonhuman` | **none** | Logan v1 (SRA cutoff 2023-12-10) |

Three things that are easy to miss:

- **The Logan databases carry no license grant.** No license instrument has
  been applied to that data by its distributor, so we grant no rights in it and
  make no representation about your right to use or redistribute it. It is
  passed through subject to the NCBI notice reproduced in `LICENSE.txt`,
  entry 5.
- **We do not license the output of the pipeline.** An alignment produced by a
  run contains material from whichever databases you searched. We make no
  claim over it either.
- **One license claim is disputed upstream** (`bfd_reduced`). The dispute is
  written out in `LICENSE.txt`; we follow the more restrictive claim for
  anything we redistribute.

Everything here is provided **AS IS, WITHOUT WARRANTY OF ANY KIND**, either by
the upstream provider or by us, and without any representation that it is fit
for a particular purpose.

## Citing

Cite the upstream database for whichever archive you used — each entry in
[`LICENSE.txt`](https://thal-kak-msa.s3.amazonaws.com/LICENSE.txt)
carries its `Cite` line, and for the CC BY / CC BY-SA databases that attribution
is a license condition. For the pipeline that searches them, cite: Bae J. et al.
Thal-Kak: unifying biomolecular structure predictors reveals a sampling–selection
gap. *bioRxiv* 2026.08.19.745680 (2026).
doi:[10.64898/2026.08.19.745680](https://doi.org/10.64898/2026.08.19.745680)

```bibtex
@article{bae2026thal,
  title={Thal-Kak: unifying biomolecular structure predictors reveals a sampling-selection gap},
  author={Bae, Junhyeok and Jo, Soohyun and Kim, Yeajin and Kim, Dongyun and Kim, Kwanwook and Park, Sanghyun and Park, Sanggeun and Myung, Sojung and Shin, Hyunho and Kim, Min Hyeok and others},
  journal={bioRxiv},
  pages={2026--08},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Links

- Registry of Open Data entry: <https://registry.opendata.aws/thal-kak-msa/>
- Tutorial notebook: [Get To Know A Dataset](https://github.com/CSSB-SNU/Thal-Kak/blob/main/aws_open_data/get-to-know-a-dataset.ipynb)
- Pipeline: <https://github.com/CSSB-SNU/Thal-Kak>
- Installer documentation: `install/README.md` in that repository
- Questions and problems: <https://github.com/CSSB-SNU/Thal-Kak/issues>
- The same archives on Hugging Face, plus the three marked Hugging Face above:
  <https://huggingface.co/datasets/cssbsnu/Thal-Kak_local_db>
- Per-database provenance and licensing: [`LICENSE.txt`](https://thal-kak-msa.s3.amazonaws.com/LICENSE.txt)
