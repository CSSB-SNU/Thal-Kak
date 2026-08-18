#!/bin/bash

resolve_targets() {
    (
        cd "${REPO_ROOT:?REPO_ROOT unset}" || exit 1
        python - "$@" <<'PY'
import sys

from MSA.local_msa.common.db_registry import DEFAULT_REGISTRY
from MSA.db_paths import DEFAULT_SUBDIR, PATHS

fam, keys = sys.argv[1], sys.argv[2:]
snap = PATHS.template_snapshot

if fam not in DEFAULT_SUBDIR:
    sys.exit(f"unknown database family {fam!r}; expected one of {sorted(DEFAULT_SUBDIR)}")


def where(key):
    if fam == "template":
        return PATHS.template_root
    if fam == "rna":
        return PATHS.rna_root / key
    return PATHS.db_dir(key, fam)


out = []
for k in keys:
    if fam == "template":
        if k != snap:
            sys.exit(
                f"manifest row {k!r} does not match `template_snapshot: {snap}` "
                f"in {PATHS.source}.\n"
                "  Set template_snapshot to the snapshot you are installing, or "
                "install the one the config asks for."
            )
        out.append((k, where(k), "-", "-"))
    elif fam == "rna":
        out.append((k, where(k), "-", "-"))
    else:
        spec = DEFAULT_REGISTRY.get(k)
        if spec is None:
            sys.exit(
                f"{k!r} is not a database this pipeline knows. Known keys: "
                f"{sorted(DEFAULT_REGISTRY)}"
            )
        if not getattr(spec, fam):
            has = [f for f in ("mmseqs", "hhblits") if getattr(spec, f)]
            sys.exit(
                f"{k!r} has no {fam} build (db_registry.py says: "
                f"{', '.join(has) if has else 'none'}). Installing it under the "
                f"{fam} root would put files where nothing reads them."
            )
        out.append((k, where(k),
                    "yes" if spec.pairable else "no",
                    "yes" if spec.expandable else "no"))

for k, d, pairable, expandable in out:
    print(f"{k}\t{d}\t{pairable}\t{expandable}\t{PATHS.dir_source(k, fam)}")
PY
    ) || fatal "could not work out where the databases go (the error above says why).
  If that error names a missing module, the project environment is not active:
      pixi shell                  # or: conda activate thalkak"
}

family_root() {
    (
        cd "${REPO_ROOT:?REPO_ROOT unset}" || exit 1
        python - "$1" <<'PY'
import sys

from MSA.db_paths import PATHS

fam = sys.argv[1]
roots = {
    "mmseqs": PATHS.mmseqs_root,
    "hhblits": PATHS.hhblits_root,
    "template": PATHS.template_root,
    "rna": PATHS.rna_root,
}
if fam not in roots:
    sys.exit(f"unknown database family {fam!r}; expected one of {sorted(roots)}")
print(roots[fam])
PY
    ) || fatal "could not read db_paths.yaml (the error above says why).
  If that error names a missing module, the project environment is not active:
      pixi shell                  # or: conda activate thalkak"
}
