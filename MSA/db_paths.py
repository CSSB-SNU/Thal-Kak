"""Where the MSA stage's databases are — resolved from `db_paths.yaml`.

The repo root holds `db_paths.yaml`; this module reads it and exposes the
answers as module constants. A fresh clone works with no editing, because the
shipped file says `db_root: db` and every other location is derived from it.
Users who install onto another disk name that disk in the yaml instead of
symlinking `<repo>/db` at it.

Three layers, most specific wins:

    db_root                     everything under one directory
    <family>_root               one format on its own disk
    databases.<key>.<format>    one database on its own disk

Relative paths in the yaml resolve against the yaml's own directory, never the
current working directory, so the same config means the same thing from any cwd.

Layout under `db_root`, created by install_db.sh (see install/manifest/databases.tsv):

    <db_root>/
    |-- msa_mmseqs/<db_key>/db{,_h,_seq,_seq_h,_aln}{,.index,.dbtype}
    |-- msa_hhblits/<db_key>/db_{a3m,cs219,hhm}.ff{data,index}
    |-- template/<template_snapshot>/
    |   |-- fasta/pdb_seqres_protein.fasta
    |   |-- mmseqs/<snapshot-lowercased>_pdb{,_h}{,.index,.dbtype,.lookup}
    |   |-- cif/raw/<pdb[1:3]>/<pdb>.cif.gz
    |   |-- metadata/cif_metadata.tsv
    |   `-- release_dates.tsv
    `-- rna/{rfam,rnacentral}/            (nhmmer .mdf + .fasta + .ssi)

Database directories are named by their registry key (db_registry.py) and the
files inside are always stemmed `db`, so a `databases:` override only has to
name a directory.

Note the two meanings of "template root": the yaml key `template_root` is the
directory that CONTAINS snapshots, while `TEMPLATE_ROOT` below is the snapshot
itself (`template_root / template_snapshot`), which is what the template code
reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# MSA/db_paths.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]

DB_PATHS_YAML = _REPO_ROOT / "db_paths.yaml"

# The per-database formats a `databases:` override can name.
FAMILIES = ("mmseqs", "hhblits")

# Subdirectory of `db_root` used when the matching `*_root` key is absent. The
# default layout is stated here once and nowhere else — the installer reads
# this rather than composing `db_root/msa_mmseqs` in shell, so the installer and
# the search cannot drift apart.
DEFAULT_SUBDIR = {
    "mmseqs": "msa_mmseqs",
    "hhblits": "msa_hhblits",
    "template": "template",
    "rna": "rna",
}

_REQUIRED_KEYS = ("db_root", "template_snapshot")
_OPTIONAL_KEYS = (
    "mmseqs_root",
    "hhblits_root",
    "template_root",
    "rna_root",
    "databases",
)


class DbPathsError(RuntimeError):
    """Anything wrong with db_paths.yaml. Always names the file and the fix."""


@dataclass(frozen=True)
class DbPaths:
    """One parsed `db_paths.yaml`. Built by `load()`; never mutated."""

    source: Path
    db_root: Path
    mmseqs_root: Path
    hhblits_root: Path
    template_root: Path  # includes the snapshot
    rna_root: Path
    template_snapshot: str
    overrides: dict[str, dict[str, Path]]
    given: frozenset[str] = frozenset()

    def dir_source(self, key: str, family: str) -> str:
        """Which layer answered — for messages that would otherwise surprise.

        A more specific key silently beats a less specific one, which is the one
        thing about this file that is hard to see. Printing the layer next to the
        directory makes "why is it looking there" answerable without reading the
        yaml, so the installer shows it in `--dry-run` and `--status`.
        """
        if family in FAMILIES and self.overrides.get(key, {}).get(family) is not None:
            return f"databases.{key}.{family}"
        return f"{family}_root" if f"{family}_root" in self.given else "db_root"

    def db_dir(self, key: str, family: str) -> Path:
        """The directory holding one database's files — the whole fallback."""
        if family not in FAMILIES:
            raise ValueError(f"unknown database format {family!r}; expected one of {list(FAMILIES)}")
        override = self.overrides.get(key, {}).get(family)
        if override is not None:
            return override
        root = self.mmseqs_root if family == "mmseqs" else self.hhblits_root
        return root / key

    def validate_overrides(self, known: dict[str, set[str]]) -> None:
        """Check every `databases:` entry against the database registry.

        `known` maps each registry key to the formats that database has a build
        for. It is passed in rather than imported because db_registry imports
        THIS module, so importing it back would be circular. Every caller that
        resolves a database directory goes through db_registry and therefore
        runs this; a caller that only reads the roots does not, and cannot be
        affected by a bad override either.
        """
        for key, per_format in sorted(self.overrides.items()):
            if key not in known:
                raise DbPathsError(
                    f"{self.source}: `databases.{key}` is not a database key. "
                    f"Known keys: {sorted(known)}"
                )
            unbuilt = sorted(set(per_format) - known[key])
            if unbuilt:
                has = sorted(known[key])
                raise DbPathsError(
                    f"{self.source}: `databases.{key}` gives a path for "
                    f"{unbuilt}, but {key} has no build in that format. "
                    f"{key} is available as: {has if has else 'neither format'}."
                )


def load(path: Path | str = DB_PATHS_YAML) -> DbPaths:
    """Parse a db_paths.yaml. Pure: no globals read or written."""
    path = Path(path)
    if not path.is_file():
        raise DbPathsError(
            f"{path} is missing. It ships with the repo and says where the MSA "
            "databases are installed; nothing regenerates it. Restore it with "
            f"`git checkout -- {path.name}`."
        )
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as exc:
        raise DbPathsError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise DbPathsError(
            f"{path}: the top level must be `key: value` entries, "
            f"got {type(raw).__name__}"
        )

    unknown = sorted(set(raw) - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS))
    if unknown:
        raise DbPathsError(
            f"{path}: unknown key(s) {unknown}. Valid keys: "
            f"{sorted(_REQUIRED_KEYS + _OPTIONAL_KEYS)}"
        )
    for key in _REQUIRED_KEYS:
        if raw.get(key) is None:
            raise DbPathsError(f"{path}: `{key}` is required and must not be empty.")

    base = path.parent

    def as_path(value: object, where: str) -> Path:
        if not isinstance(value, str) or not value.strip():
            raise DbPathsError(
                f"{path}: `{where}` must be a non-empty path, got {value!r}"
            )
        candidate = Path(value.strip()).expanduser()
        # Relative to the yaml, not to the cwd — the same file has to mean the
        # same thing whether the pipeline runs from the repo or from a job dir.
        return candidate if candidate.is_absolute() else base / candidate

    db_root = as_path(raw["db_root"], "db_root")

    def family_root(key: str, family: str) -> Path:
        if raw.get(key) is None:
            return db_root / DEFAULT_SUBDIR[family]
        return as_path(raw[key], key)

    snapshot = raw["template_snapshot"]
    if not isinstance(snapshot, str) or not snapshot.strip():
        raise DbPathsError(
            f"{path}: `template_snapshot` must be a directory name, got {snapshot!r}"
        )
    snapshot = snapshot.strip()

    overrides: dict[str, dict[str, Path]] = {}
    per_db = raw.get("databases") or {}
    if not isinstance(per_db, dict):
        raise DbPathsError(
            f"{path}: `databases` must map a database key to `format: path` "
            f"entries, got {type(per_db).__name__}"
        )
    for key, per_format in per_db.items():
        if not isinstance(per_format, dict):
            raise DbPathsError(
                f"{path}: `databases.{key}` must map a format to a path, "
                f"e.g. `mmseqs: /disk/{key}`"
            )
        bad = sorted(set(per_format) - set(FAMILIES))
        if bad:
            raise DbPathsError(
                f"{path}: `databases.{key}` names unknown format(s) {bad}. "
                f"Valid formats: {list(FAMILIES)}"
            )
        overrides[key] = {
            fmt: as_path(value, f"databases.{key}.{fmt}")
            for fmt, value in per_format.items()
        }

    return DbPaths(
        source=path,
        db_root=db_root,
        mmseqs_root=family_root("mmseqs_root", "mmseqs"),
        hhblits_root=family_root("hhblits_root", "hhblits"),
        template_root=family_root("template_root", "template") / snapshot,
        rna_root=family_root("rna_root", "rna"),
        template_snapshot=snapshot,
        overrides=overrides,
        given=frozenset(k for k in _OPTIONAL_KEYS if raw.get(k) is not None),
    )


PATHS = load()

LOCALDB_ROOT = PATHS.db_root
MMSEQS_DB_ROOT = PATHS.mmseqs_root
HHBLITS_DB_ROOT = PATHS.hhblits_root
RNA_DB_ROOT = PATHS.rna_root
TEMPLATE_SNAPSHOT = PATHS.template_snapshot
TEMPLATE_ROOT = PATHS.template_root

db_dir = PATHS.db_dir
dir_source = PATHS.dir_source
validate_overrides = PATHS.validate_overrides
