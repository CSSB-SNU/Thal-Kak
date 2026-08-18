#!/bin/bash
set -eo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
LIB="$REPO_ROOT/install/lib"
MANIFEST="$REPO_ROOT/install/manifest/databases.tsv"

log()   { printf '%s\n' "$*"; }
info()  { printf '  %s\n' "$*"; }
warn()  { printf 'WARNING: %s\n' "$*" >&2; }
fatal() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

human() {
    awk -v b="${1:-0}" 'BEGIN{
        split("B KiB MiB GiB TiB PiB", u, " ")
        i = 1
        while (b >= 1024 && i < 6) { b /= 1024; i++ }
        if (i == 1) printf "%d %s", b, u[i]; else printf "%.1f %s", b, u[i]
    }'
}

usage() {
    cat <<'EOF'
install_db.sh — install the local MSA, template and RNA databases that
`--msa mmseqs_local`, `--msa hhblits_local` and `--msa mmseqs_hhblits_local` read.

Usage
    ./install_db.sh --family LIST [KEY ...]

Everything here comes from the project environment — zstd, aria2, mmseqs,
makehmmerdb, esl-sfetch, and the python that reads db_paths.yaml:

    pixi install && pixi shell                       # pixi (recommended), or:
    ./install.sh && conda activate thalkak
    ./install_db.sh --family all

With no KEY the whole family is installed; with KEYs, only those, from one
family at a time.

Every database is fetched from the url in install/manifest/databases.tsv, and
downloads are cached under --archive-dir so a re-run resumes instead of
starting over. The one archive that is not a manifest row is the uniref30_2302
pairing taxonomy overlay, a second download onto a database already installed —
see --no-taxonomy below.

To install somewhere other than <repo>/db, edit db_paths.yaml first — that file
is the only thing that decides, and the pipeline reads the same answer, so an
install cannot land where the search will not look. `--dry-run` prints each
destination and which line of the yaml chose it.

Options
    --family LIST       REQUIRED. Comma-separated: mmseqs, hhblits, template,
                        rna, or all. There is no default: the families run from
                        ~11 GB to ~3.6 TB and come from different sources.
    --status            report per-database state and exit
    --dry-run           print the plan and the disk maths, change nothing
    --archive-dir DIR   where downloads are kept (default <root>/_archives)
    --keep-archives     do not delete an archive after a successful install
    --threads N         threads for index building and RNA clustering
                        (default: SLURM alloc or nproc)
    --mmseqs PATH       mmseqs binary (default: from PATH)
    --createindex-extra "ARGS"   passed through to mmseqs createindex
    --no-taxonomy       install uniref30_2302 without the 2025-08 pairing
                        taxonomy overlay, which is otherwise applied whenever
                        that database is installed

The RNA databases are NOT optional. The ColabFold API returns protein
alignments only, so an RNA or RNP target has no remote fallback: without them
it runs on the query sequence alone.

install/README.md is the reference for the rest — the yaml's keys, what is free
and what is fixed, and how each database was built.
EOF
    exit "${1:-0}"
}

# shellcheck source=install/lib/paths.sh
. "$LIB/paths.sh"
# shellcheck source=install/lib/manifest.sh
. "$LIB/manifest.sh"
# shellcheck source=install/lib/fetch.sh
. "$LIB/fetch.sh"
# shellcheck source=install/lib/extract.sh
. "$LIB/extract.sh"
# shellcheck source=install/lib/preflight.sh
. "$LIB/preflight.sh"
# shellcheck source=install/lib/build.sh
. "$LIB/build.sh"

FAMILY_ARG=""
SEL=()
ARCHIVE_DIR=""
KEEP_ARCHIVES=0
DRY_RUN=0
STATUS=0
MMSEQS_BIN=""
THREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
CREATEINDEX_EXTRA=()
DO_TAXONOMY=1

need_value() {
    [[ $# -ge 2 && -n "$2" && "$2" != -* ]] || fatal "$1 needs a value"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --family)            need_value "$@"; FAMILY_ARG="$2"; shift 2 ;;
        --archive-dir)       need_value "$@"; ARCHIVE_DIR="$2"; shift 2 ;;
        --keep-archives)     KEEP_ARCHIVES=1; shift ;;
        --dry-run)           DRY_RUN=1; shift ;;
        --status)            STATUS=1; shift ;;
        --threads)           need_value "$@"; THREADS="$2"; shift 2 ;;
        --mmseqs)            need_value "$@"; MMSEQS_BIN="$2"; shift 2 ;;
        --createindex-extra) need_value "$@"
                             IFS=' ' read -r -a CREATEINDEX_EXTRA <<<"$2"; shift 2 ;;
        --no-taxonomy)       DO_TAXONOMY=0; shift ;;
        -h|--help)           usage 0 ;;
        --*) printf 'unknown option: %s\n\n' "$1" >&2; usage 1 ;;
        *)   SEL+=("$1"); shift ;;
    esac
done

ALL_FAMILIES=(mmseqs hhblits template rna)

[[ -n "$FAMILY_ARG" ]] || fatal "--family is required: mmseqs, hhblits, template, rna, or all.
  They run from ~11 GB to ~3.6 TB and come from different sources, so there is
  no sensible default. See --help and install/README.md.
      e.g.  ./install_db.sh --family all"

FAMILIES=()
if [[ "$FAMILY_ARG" == "all" ]]; then
    FAMILIES=("${ALL_FAMILIES[@]}")
else
    IFS=',' read -r -a FAMILIES <<<"$FAMILY_ARG"
    for fam in "${FAMILIES[@]}"; do
        case " ${ALL_FAMILIES[*]} " in
            *" $fam "*) ;;
            *) fatal "unknown family '$fam'; expected one of: ${ALL_FAMILIES[*]}, all" ;;
        esac
    done
fi

if [[ ${#SEL[@]} -gt 0 && ${#FAMILIES[@]} -gt 1 ]]; then
    fatal "naming databases works within one family at a time (--family ${FAMILIES[0]}),
  because a key such as uniref30_2302 exists in more than one of them."
fi

wanted() {
    [[ ${#SEL[@]} -eq 0 ]] && return 0
    local k
    for k in "${SEL[@]}"; do [[ "$k" == "$1" ]] && return 0; done
    return 1
}

[[ -f "$MANIFEST" ]] || fatal "manifest not found: $MANIFEST"
validate_manifest

if [[ $STATUS -eq 0 ]]; then
    require_tools tar sha256sum md5sum find awk df
    for fam in "${FAMILIES[@]}"; do
        case "$fam" in
            mmseqs|hhblits|template) require_tools zstd ;;
        esac
        case "$fam" in
            rna) require_tools makehmmerdb esl-sfetch gzip ;;
        esac
    done
    for fam in "${FAMILIES[@]}"; do
        [[ "$fam" == "mmseqs" || "$fam" == "rna" ]] || continue
        [[ -n "$MMSEQS_BIN" ]] || MMSEQS_BIN=$(command -v mmseqs || true)
        [[ -n "$MMSEQS_BIN" ]] || fatal "mmseqs not found on PATH (createindex for the
  mmseqs family, clustering for the rna one). Activate the project environment,
  or pass --mmseqs PATH."
    done
fi

INSTALL_ROOT=$(family_root "${FAMILIES[0]}")
[[ -n "$INSTALL_ROOT" ]] || fatal "cannot work out where to install"
if [[ $STATUS -eq 0 ]]; then
    mkdir -p "$INSTALL_ROOT" || fatal "cannot create $INSTALL_ROOT"
fi
[[ -n "$ARCHIVE_DIR" ]] || ARCHIVE_DIR="$INSTALL_ROOT/_archives"

declare -A TARGET PAIRABLE EXPANDABLE SOURCE FAMILY_OF
declare -a PLAN=()

for fam in "${FAMILIES[@]}"; do
    keys=()
    while IFS= read -r row; do
        key=$(manifest_field "$row" $F_KEY)
        wanted "$key" || continue
        keys+=("$key")
        PLAN+=("$fam"$'\t'"$row")
    done < <(read_rows "$fam")
    [[ ${#keys[@]} -gt 0 ]] || continue

    resolved=$(resolve_targets "$fam" "${keys[@]}") || exit 1
    while IFS=$'\t' read -r k d p e s; do
        [[ -n "$k" ]] || continue
        TARGET["$fam/$k"]=$d
        PAIRABLE["$fam/$k"]=$p
        EXPANDABLE["$fam/$k"]=$e
        SOURCE["$fam/$k"]=$s
        FAMILY_OF["$k"]=$fam
    done <<<"$resolved"

    for key in "${keys[@]}"; do
        [[ -n "${TARGET["$fam/$key"]}" ]] || fatal "no destination came back for $fam/$key"
    done
done

for name in "${SEL[@]}"; do
    [[ -n "${FAMILY_OF["$name"]}" ]] && continue
    resolve_targets "${FAMILIES[0]}" "$name" >/dev/null
    fatal "'$name' is a ${FAMILIES[0]} database but has no row in
      $MANIFEST
  so there is nothing to install for it."
done

if [[ ${#PLAN[@]} -eq 0 && $STATUS -eq 0 ]]; then
    fatal "nothing to install: the manifest has no ${FAMILIES[*]} rows"
fi

target_of() { printf '%s' "${TARGET["$1/$2"]}"; }
source_of() { printf '%s' "${SOURCE["$1/$2"]}"; }

if [[ $STATUS -eq 1 ]]; then
    report_status
    exit 0
fi

plan_and_preflight

if [[ $DRY_RUN -eq 1 ]]; then
    log "--dry-run: stopping here. Would install:"
    for entry in "${INSTALL_PLAN[@]}"; do
        fam=${entry%%$'\t'*}; row=${entry#*$'\t'}
        key=$(manifest_field "$row" $F_KEY)
        stem=$(manifest_field "$row" $F_STEM)
        printf '  %-9s %-22s -> %s  [%s]  %s\n' "$fam" "$key" \
            "$(target_of "$fam" "$key")" "$(source_of "$fam" "$key")" \
            "$([[ "$stem" == "-" ]] && echo "(no rename)" || echo "(stem $stem -> db)")"
    done
    exit 0
fi

acquire_lock "$INSTALL_ROOT"
pick_downloader
log "downloader  : $DOWNLOADER"
log

INSTALLED=()
for entry in "${INSTALL_PLAN[@]}"; do
    fam=${entry%%$'\t'*}; row=${entry#*$'\t'}
    key=$(manifest_field "$row" $F_KEY)
    stem=$(manifest_field "$row" $F_STEM)
    archive=$(manifest_field "$row" $F_ARCHIVE)
    url=$(manifest_field "$row" $F_URL)
    sha=$(manifest_field "$row" $F_SHA)
    idxcol=$(manifest_field "$row" $F_INDEX)

    target=$(target_of "$fam" "$key")
    log "[$fam/$key]"

    if is_installed "$fam" "$target" "$key"; then
        info "already installed at $target"
    else
        ARCHIVE_PATH=""
        fetch_archive "$archive" "$ARCHIVE_DIR/$fam" "$sha" "$url"

        if [[ "$fam" == "mmseqs" && "$key" == "uniref30_2302" ]]; then
            install_uniref30 "$target" "$ARCHIVE_PATH"
        elif [[ "$fam" == "rna" ]]; then
            build_rna_from_upstream "$target" "$ARCHIVE_PATH" "$key"
        else
            extract_archive "$ARCHIVE_PATH" "$target"
            normalize_stem "$STAGE_DIR" "$stem"
        fi

        case "$fam" in
            mmseqs)   verify_mmseqs_db "$STAGE_DIR" \
                          "${PAIRABLE["$fam/$key"]}" "${EXPANDABLE["$fam/$key"]}" ;;
            hhblits)  verify_hhblits_db "$STAGE_DIR" ;;
            template) verify_template_db "$STAGE_DIR" "$key" ;;
            rna)      verify_rna_db "$STAGE_DIR" "$key" ;;
        esac

        mkdir -p "$(dirname "$target")" || fatal "cannot create $(dirname "$target")"
        superseded=""
        if [[ -e "$target" ]]; then
            superseded="$target.superseded.$$"
            rm -rf "$superseded"
            mv "$target" "$superseded" || fatal "cannot move the existing $target aside"
        fi
        if ! mv "$STAGE_DIR" "$target"; then
            [[ -n "$superseded" ]] && mv "$superseded" "$target" 2>/dev/null
            fatal "cannot move $STAGE_DIR -> $target (the previous copy was put back)"
        fi
        STAGE_DIR=""
        [[ -n "$superseded" ]] && rm -rf "$superseded"
        info "installed $target"

        if [[ $KEEP_ARCHIVES -eq 0 && $FETCHED -eq 1 ]]; then
            rm -f "$ARCHIVE_PATH"
            info "archive removed (--keep-archives to keep it)"
        fi
    fi

    if [[ "$idxcol" == "yes" ]]; then
        case "$fam" in
            mmseqs) build_index "$target" "$MMSEQS_BIN" "$THREADS" "${CREATEINDEX_EXTRA[@]}" ;;
            rna)    build_index_rna "$target" "$key" ;;
        esac
    fi

    if [[ "$fam" == "mmseqs" && "$key" == "uniref30_2302" && $DO_TAXONOMY -eq 1 ]]; then
        install_taxonomy_overlay "$target"
    fi

    if [[ "$fam" == "hhblits" && "$key" == "uniref100_2026_01" ]]; then
        prune_empty_cs219 "$target"
    fi

    INSTALLED+=("$fam/$key")
    log
done

rmdir "$ARCHIVE_DIR"/* 2>/dev/null || true
rmdir "$ARCHIVE_DIR" 2>/dev/null || true

log "=== installed ${#INSTALLED[@]} database(s), where db_paths.yaml says ==="
log
log "Verified. Put exactly this list in the config yaml's \`dbs:\` — it is what is"
log "actually on disk, in manifest order:"
log
for k in "${INSTALLED[@]}"; do
    printf '  - %s\n' "${k##*/}"
done
log
warn_if_rna_missing
log "Then run:  thalkak msa --msa mmseqs_local --seq <fasta> --stoi <e.g. A1>"
