"""The MSA database registry — every database this code knows how to search.

A key being here means "the pipeline can search this if you have it", not
"you have it". `install/manifest/databases.tsv` is the list of what can
actually be installed, and it states each case; one of the ten keys has
no row there at all: `envhog_prof`, an alternative clustering level of
the same source data as the default `envhog_std`, kept for the
comparison below and for anyone rebuilding at that level.

Two others have a row that names an upstream rather than an archive of
ours: `uniref30_2302` (the ColabFold build, plus the pairing-taxonomy
overlay `./install_db.sh --family mmseqs uniref30_2302`) and `bfd` in HHblits format
(hosted by AlphaFold). `logan_nonhuman` has a row for the HHblits format
only; its mmseqs archive is unpublished at 1.4 TB extracted.

Paths follow one rule, so no entry states one: each DB lives in its own
directory named after its registry key, under the per-format root, and the
files inside are always stemmed `db`:

    <mmseqs_root>/<key>/db{,_h,_seq,_seq_h,_aln}{,.index,.dbtype}
    <hhblits_root>/<key>/db_{a3m,cs219,hhm}.ff{data,index}

The key therefore appears exactly once in a path, and there is no DB
whose filenames deviate from the convention. The roots, and any per-DB
exception to them, come from `db_paths.yaml` at the repo root; this module
asks `MSA/db_paths.py:db_dir()` rather than composing paths itself.

A missing installation is not detected here. For mmseqs it surfaces in
the search steps, which check for `<basename>.dbtype`
(`mmseqs/search_{uniref,envdb,pair}.py`); for HHblits it surfaces in
`has_hhblits_db()`, which stats the `_cs219.ffindex` prefilter table.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from MSA.db_paths import db_dir, validate_overrides

DBKind = Literal["uniref", "env"]

# Filename stem inside every DB directory, both formats. Uniform so the
# directory name carries the identity and the path never repeats it.
DB_STEM = "db"


@dataclass(frozen=True)
class DBSpec:
    """One MSA database. May expose either or both of an mmseqs sibling
    (consumed by `--msa mmseqs_local`) and an hhblits sibling (consumed by
    `--msa hhblits_local`). The two flags declare which builds EXIST for this
    key; neither one stats the disk (see the module docstring).

    `mmseqs`: an mmseqs-format build exists, i.e. `dbbase` would hold
    `db`, `db_h`, `db_aln`, `db_seq`, `db_seq_h` (each with matching
    `.dbtype`/`.index`), plus `db_mapping`/`db_taxonomy` when `pairable`.
    `db.idx` is optional (performance only). False for an HHblits-only
    entry, e.g. `bfd`, whose full-member cluster MSAs have no mmseqs build.

    `hhblits`: an hhblits-format build exists. HHblits takes
    `hhblits_db_stem` as a prefix and discovers `<stem>_a3m.ff{data,index}`,
    `<stem>_hhm.ff{data,index}`, `<stem>_cs219.ff{data,index}` itself.
    False means `hhblits_local` cannot use this DB.

    `expandable` (env-kind mmseqs only): True iff the DB carries cluster
    member alignments (`db_aln`, or `_aln` packed into `db.idx`) so Stage C
    can run `mmseqs expandaln`. Set False for a FLAT (non-clustered) env DB
    built by plain `createdb`+`createindex`: expandaln there dies with
    "getData: local id (...) >= db size", so `search_envdb` skips it and
    aligns the raw search hits instead. No entry below needs this today —
    every mmseqs env build here is clustered.
    """

    key: str
    kind: DBKind
    pairable: bool  # True iff colabfold_search mmseqs_search_pair is valid
    mmseqs: bool = True
    hhblits: bool = True
    expandable: bool = True

    @property
    def dbbase(self) -> Path | None:
        return db_dir(self.key, "mmseqs") if self.mmseqs else None

    @property
    def basename(self) -> str | None:
        return DB_STEM if self.mmseqs else None

    @property
    def hhblits_db_stem(self) -> Path | None:
        return db_dir(self.key, "hhblits") / DB_STEM if self.hhblits else None

    def formats(self) -> set[str]:
        """The per-database formats this entry has an installed build for."""
        return {
            fmt
            for fmt, installed in (("mmseqs", self.mmseqs), ("hhblits", self.hhblits))
            if installed
        }

    def has_mmseqs_db(self) -> bool:
        return self.mmseqs

    def dbtype_path(self) -> Path:
        if not self.mmseqs:
            raise RuntimeError(
                f"{self.key}: no mmseqs sibling DB — this entry is hhblits-only"
            )
        return self.dbbase / f"{DB_STEM}.dbtype"

    def has_idx(self) -> bool:
        if not self.mmseqs:
            return False
        return (self.dbbase / f"{DB_STEM}.idx").is_file() or (
            self.dbbase / f"{DB_STEM}.idx.index"
        ).is_file()

    def has_hhblits_db(self) -> bool:
        if not self.hhblits:
            return False
        # HHblits requires the cs219 prefilter table at minimum.
        return Path(f"{self.hhblits_db_stem}_cs219.ffindex").is_file()


DEFAULT_REGISTRY: dict[str, DBSpec] = {
    spec.key: spec
    for spec in [
        DBSpec(key="uniref30_2302", kind="uniref", pairable=True),
        DBSpec(key="uniref100_2026_01", kind="uniref", pairable=True),
        # `bfd_reduced` and `bfd` reference the same underlying source
        # data (BFD clustered at id30/c90, the standard ColabFold/AF2
        # "reduced BFD") but in two formats with different content depth:
        #
        #   - bfd_reduced (mmseqs)  : 65.98M cluster CENTROID sequences
        #     only (~22 GB across main + _seq + _aln). Its _aln is too
        #     small to hold full member-to-centroid alignments, so
        #     `expandaln` against this DB returns ~centroids only.
        #
        #   - bfd (hhblits)         : 65.98M cluster a3ms each containing
        #     the FULL cluster member MSA (1.5 TB a3m.ffdata). HHblits
        #     hits expand to cluster members.
        #
        # → distinct keys so the mmseqs_local vs hhblits_local builders
        #   pick the correct one without sharing a misleading name.
        DBSpec(key="bfd_reduced", kind="env", pairable=False, hhblits=False),
        DBSpec(key="bfd", kind="env", pairable=False, mmseqs=False),
        DBSpec(key="mgnify_clusters", kind="env", pairable=False),
        DBSpec(key="logan_human", kind="env", pairable=False),
        DBSpec(key="logan_nonhuman", kind="env", pairable=False),
        # hhblits-only, like `bfd` above: the per-family a3m/hhm that
        # hhblits_local reads comes from EnVhogDB's own release. The mmseqs
        # builds of the same source are the clustered `envhog_*` keys below.
        DBSpec(key="envhog", kind="env", pairable=False, mmseqs=False),
        # Two exprofiledb rebuilds of the same upstream proteins, both driven by
        # a clustering enVhogDB publishes itself (as a flat TSV of base62 IDs —
        # no mmseqs cluster DB is distributed for any level, so the TSV is
        # decoded and fed to `tsv2db`). Both carry _seq/_aln, so Stage C takes
        # the normal expandaln path and the prefilter stops scanning the raw
        # 129.9M sequences. They differ only in WHICH clustering level is used.
        #
        # mmseqs-only, both: the HHblits sibling (per-family a3m/hhm) is
        # installed under the `envhog` key, so hhblits_local keeps using that.
        DBSpec(
            key="envhog_std",
            kind="env",
            pairable=False,
            # enVhogDB's own `standard` level: `mmseqs cluster -s 4
            # --min-seq-id 0.3 --cov-mode 0 -c 0.7 -e 0.001`, the parameter
            # analogue of the 30%/80% linclust behind every other DB here.
            # The mmseqs default since it keeps 75% of pooled Neff for a 4.6x
            # smaller index and an 11x smaller prefilter.
            hhblits=False,
        ),
        DBSpec(
            key="envhog_prof",
            kind="env",
            pairable=False,
            # enVhogDB's own `enVhog` level: 2,203,457 curated viral families
            # grouped by profile-vs-consensus search with NO sequence-identity
            # threshold (E<=1e-3, 60% mutual coverage) — nearer a superfamily
            # than a sequence cluster. Kept for the minority of chains where
            # that remoteness pays; too shallow on the median chain to be the
            # default.
            hhblits=False,
        ),
    ]
}


# `db_paths.yaml` may point individual databases somewhere else, but it is read
# before this registry exists and so cannot check the keys it is given. This is
# the first moment both halves are available, so a typo'd or wrongly-formatted
# override fails here — at startup, naming the key — instead of resolving to a
# path nobody installed.
validate_overrides({key: spec.formats() for key, spec in DEFAULT_REGISTRY.items()})


# The one DB the mmseqs builder supports as `dbs[0]`. That slot is three jobs at
# once — Stage A profile source, multimer pairing DB, and the hardcoded name of
# the raw output dir (`mmseqs/build.py`) — and only this key is supported in all
# three. Enforced by `__main__._validate_mmseqs_block` and `mmseqs/build.py`.
# The hhblits builder has no such constraint: its `dbs[0]` is the Phase 2 primary
# only, it pairs through this DB regardless (see `msa_config.hhblits.yaml:pair_db`),
# and it names raw dirs by key.
PRIMARY_UNIREF_DB = "uniref30_2302"


# Merge-time DB priority (UniRef first → env; more specific → broader).
# Both `bfd_reduced` and `bfd` occupy the same priority slot — they're
# alternative BFD-format DBs consumed by different builders.
MERGE_ORDER: tuple[str, ...] = (
    "uniref30_2302",
    "uniref100_2026_01",
    "bfd_reduced",
    "bfd",
    "mgnify_clusters",
    "logan_human",
    "logan_nonhuman",
    "envhog",
    "envhog_std",
    "envhog_prof",
)


# Cap-walk DB priority (high -> low) for the per-DB hit caps feature
# (`common/cap_merge.py`). Distinct from MERGE_ORDER (which stays
# unchanged): the envhog builds outrank the logan pair here, and both
# BFD-format slots (`bfd_reduced` = mmseqs, `bfd` = hhblits) sit at the
# same rank between the uniref DBs and the broad env DBs. The cap-walk
# intersects this with each engine's effective db set, so unused keys are
# skipped.
CAP_PRIORITY: tuple[str, ...] = (
    "uniref30_2302",
    "uniref100_2026_01",
    "bfd_reduced",
    "bfd",
    "mgnify_clusters",
    "envhog",
    "envhog_std",
    "envhog_prof",
    "logan_human",
    "logan_nonhuman",
)


# Both order lists must cover the whole registry, and for opposite reasons:
# `cap_walk` walks CAP_PRIORITY and `common/caps.py:validate_caps` fails loud on an
# effective db missing from it, but `leveled` walks `MERGE_ORDER ∩ dbs` with no such
# check — a key absent from MERGE_ORDER would be dropped from every merged a3m
# SILENTLY, so assert the invariant at import.
for _name, _order in (("MERGE_ORDER", MERGE_ORDER), ("CAP_PRIORITY", CAP_PRIORITY)):
    _missing = set(DEFAULT_REGISTRY) - set(_order)
    _extra = set(_order) - set(DEFAULT_REGISTRY)
    if _missing or _extra:
        raise AssertionError(
            f"{_name} must list every DEFAULT_REGISTRY key exactly once: "
            f"missing={sorted(_missing)} unknown={sorted(_extra)}"
        )
del _name, _order, _missing, _extra


# The DB set a run searches is not defined here: it is each per-mode config
# yaml's `dbs` list (examples/msa_config.{mmseqs,hhblits,combined}.yaml), which
# is the single source of truth for that. This module only says which keys
# exist and where each one lives.


def select(keys: list[str] | None = None) -> list[DBSpec]:
    if keys is None:
        return list(DEFAULT_REGISTRY.values())
    missing = [k for k in keys if k not in DEFAULT_REGISTRY]
    if missing:
        raise KeyError(f"Unknown DB keys: {missing}. Known: {list(DEFAULT_REGISTRY)}")
    return [DEFAULT_REGISTRY[k] for k in keys]
