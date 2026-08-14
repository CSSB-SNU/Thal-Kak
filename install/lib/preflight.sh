#!/bin/bash

LOCK_DIR=""
STAGE_DIR=""
INSTALL_PLAN=()

require_tools() {
    local missing=() t
    for t in "$@"; do
        command -v "$t" >/dev/null 2>&1 || missing+=("$t")
    done
    [[ ${#missing[@]} -eq 0 ]] || fatal "missing required tool(s): ${missing[*]}
  These ship in the project environment. Activate it first:
      pixi shell                  # or: conda activate thalkak"
}

acquire_lock() {
    local root=$1
    LOCK_DIR="$root/.install_db.lock"
    if ! mkdir "$LOCK_DIR" 2>/dev/null; then
        local owner="unknown"
        [[ -f "$LOCK_DIR/owner" ]] && owner=$(cat "$LOCK_DIR/owner")
        fatal "another install is already running on $root (lock held by $owner).
  If that process is gone, remove the stale lock:
      rmdir $LOCK_DIR"
    fi
    printf 'pid=%s host=%s started=%s\n' "$$" "$(hostname)" "$(date '+%F %T')" \
        >"$LOCK_DIR/owner"
    trap cleanup_on_exit EXIT INT TERM
}

cleanup_on_exit() {
    if [[ -n "$STAGE_DIR" && -d "$STAGE_DIR" ]]; then
        warn "removing the unfinished staging directory $STAGE_DIR"
        rm -rf "$STAGE_DIR"
    fi
    [[ -n "$LOCK_DIR" && -d "$LOCK_DIR" ]] || return 0
    rm -f "$LOCK_DIR/owner"
    rmdir "$LOCK_DIR" 2>/dev/null || true
}

avail_bytes() {
    local p=$1
    while [[ ! -d "$p" && "$p" != "/" ]]; do p=$(dirname "$p"); done
    df -PB1 "$p" 2>/dev/null | awk 'NR==2 {print $4}'
}

warn_if_quota_short() {
    local need=$1 dest=${2:-$INSTALL_ROOT} p fs headroom
    command -v quota >/dev/null 2>&1 || return 0

    p=${dest:-/}
    while [[ ! -d "$p" && "$p" != "/" ]]; do p=$(dirname "$p"); done
    fs=$(df -P "$p" 2>/dev/null | awk 'NR==2 {print $1}')
    [[ -n "$fs" ]] || return 0

    headroom=$(quota -w 2>/dev/null | awk -v fs="$fs" '
        $1 == fs && $2 ~ /^[0-9]+$/ && $3+0 > 0 {
            h = ($3 - $2) * 1024
            print (h > 0 ? h : 0)
            exit
        }' || true)
    [[ "$headroom" =~ ^[0-9]+$ ]] || return 0
    if [[ "$headroom" -lt "$need" ]]; then
        warn "your disk QUOTA on $fs has less headroom than this install needs:
         quota headroom $(human "$headroom") vs needed $(human "$need")
         df sees more free space than your quota will let you use, so the
         install would fail partway. Point db_root, or one family root, at a
         partition without a quota in db_paths.yaml."
    fi
}

warn_if_rna_missing() {
    local missing=() k d resolved
    resolved=$(resolve_targets rna rfam rnacentral) || {
        warn "cannot tell whether the RNA databases are installed (see above)"
        return 0
    }
    while IFS=$'\t' read -r k d _; do
        [[ -n "$k" ]] || continue
        is_installed rna "$d" "$k" || missing+=("$k")
    done <<<"$resolved"
    [[ ${#missing[@]} -gt 0 ]] || return 0
    warn "the RNA databases are not installed (${missing[*]}).
         The ColabFold API does not return RNA alignments, so an RNA or RNP
         target has no fallback and would run on the query sequence alone.
             ./install_db.sh --family rna"
}

report_status() {
    local fam key dir row src state
    log "=== install_db.sh --status ==="
    log "config      : $REPO_ROOT/db_paths.yaml"
    log
    for fam in "${FAMILIES[@]}"; do
        log "[$fam]"
        while IFS= read -r row; do
            key=$(manifest_field "$row" $F_KEY)
            wanted "$key" || continue
            dir=$(target_of "$fam" "$key")
            if [[ ! -d "$dir" ]]; then
                state="not installed"
            elif ! is_installed "$fam" "$dir" "$key"; then
                state="INCOMPLETE (directory exists, no database files)"
            elif [[ "$(manifest_field "$row" $F_INDEX)" == "yes" ]] \
                 && ! has_index "$fam" "$dir" "$key"; then
                state="installed, no index"
            else
                state="installed"
            fi
            src=$(source_of "$fam" "$key")
            if [[ "$src" == "db_root" ]]; then
                printf '  %-22s %s\n' "$key" "$state"
            else
                printf '  %-22s %-48s [%s -> %s]\n' "$key" "$state" "$src" "$dir"
            fi
        done < <(read_rows "$fam")
    done
    log
    warn_if_rna_missing
}

check_space() {
    local label=$1 dir=$2 need=$3 have
    have=$(avail_bytes "$dir")
    printf '    %-9s %-11s -> %s  (%s available)%s\n' \
        "$label" "$(human "$need")" "$dir" "$(human "$have")" \
        "$([[ "$have" -lt "$need" ]] && printf '  << SHORT')"
    warn_if_quota_short "$need" "$dir"
    [[ "$have" -ge "$need" ]]
}

plan_and_preflight() {
    local sum_archive=0 sum_raw=0 sum_idx=0 n_rows=0
    local unknown_idx=() already=()
    local entry fam row key archive ab rb ib idxcol need
    declare -A need_by_fam=()

    INSTALL_PLAN=()
    for entry in "${PLAN[@]}"; do
        fam=${entry%%$'\t'*}; row=${entry#*$'\t'}
        key=$(manifest_field "$row" $F_KEY)
        archive=$(manifest_field "$row" $F_ARCHIVE)

        if is_installed "$fam" "$(target_of "$fam" "$key")" "$key"; then
            already+=("$key")
            n_rows=$(( n_rows + 1 ))
            INSTALL_PLAN+=("$entry")
            continue
        fi

        ab=$(manifest_field "$row" $F_ABYTES); rb=$(manifest_field "$row" $F_RBYTES)
        ib=$(manifest_field "$row" $F_IBYTES); idxcol=$(manifest_field "$row" $F_INDEX)
        [[ "$ab" == "-" ]] && ab=0
        [[ "$rb" == "-" ]] && rb=0
        sum_raw=$(( sum_raw + rb ))
        need_by_fam["$fam"]=$(( ${need_by_fam["$fam"]:-0} + rb ))
        if [[ ! -f "$ARCHIVE_DIR/$fam/$archive" ]]; then
            sum_archive=$(( sum_archive + ab ))
        fi
        if [[ "$idxcol" == "yes" ]]; then
            if [[ "$ib" == "-" ]]; then
                unknown_idx+=("$key")
            else
                sum_idx=$(( sum_idx + ib ))
                need_by_fam["$fam"]=$(( ${need_by_fam["$fam"]} + ib ))
            fi
        fi
        n_rows=$(( n_rows + 1 ))
        INSTALL_PLAN+=("$entry")
    done

    log "=== install_db.sh ==="
    log "repo        : $REPO_ROOT"
    log "config      : $REPO_ROOT/db_paths.yaml"
    log "install root: $INSTALL_ROOT"
    log "families    : ${FAMILIES[*]}"
    [[ ${#SEL[@]} -gt 0 ]] && log "databases   : ${SEL[*]}"
    log "archive dir : $ARCHIVE_DIR"
    log

    [[ ${#already[@]} -eq 0 ]] || log "already installed (not counted below): ${already[*]}"

    need=$(( sum_raw + sum_idx + sum_archive ))

    log "plan: $n_rows database(s)"
    log "  download   $(human "$sum_archive")"
    log "  extracted  $(human "$sum_raw")"
    log "  index      $(human "$sum_idx")$([[ ${#unknown_idx[@]} -gt 0 ]] && echo "  + unmeasured: ${unknown_idx[*]}")"
    log "  peak need  $(human "$need")"

    local short=() dest fam_need
    [[ $sum_archive -eq 0 ]] || check_space archives "$ARCHIVE_DIR" "$sum_archive" || short+=("archives")
    for fam in "${FAMILIES[@]}"; do
        fam_need=${need_by_fam["$fam"]:-0}
        [[ $fam_need -gt 0 ]] || continue
        dest=$(family_root "$fam")
        check_space "$fam" "$dest" "$fam_need" || short+=("$fam")
    done
    log

    if [[ ${#unknown_idx[@]} -gt 0 ]]; then
        warn "no measured index size for: ${unknown_idx[*]}
         The peak figure above is therefore a LOWER bound."
    fi

    for entry in "${INSTALL_PLAN[@]}"; do
        row=${entry#*$'\t'}
        [[ "${entry%%$'\t'*}" == "rna" && "$(manifest_field "$row" $F_URL)" != "-" ]] || continue
        warn "the RNA databases are CLUSTERED here, not unpacked, and the figures above
         do not include the working set mmseqs needs for that -- roughly the
         decompressed input again, about 33 GB for RNAcentral. It is written
         under $ARCHIVE_DIR/rna and removed afterwards.
         The sequences come from EBI's CURRENT release, so what you build is
         what they publish today; that is not the same database anyone who
         built it on another day has."
        break
    done

    if [[ ${#short[@]} -gt 0 ]]; then
        fatal "not enough space for: ${short[*]}   (marked SHORT above)
  Each line above is its own destination and its own filesystem. Point db_root,
  one family root, or one database at a bigger partition in db_paths.yaml, or
  name a subset of the databases -- but a partial set is recorded in
  method_log.yaml and becomes part of the MSA skip key, so a shallow MSA can be
  cached as if it were complete. See install/README.md."
    fi
}
