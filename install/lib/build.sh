#!/bin/bash

TAX_URL=https://opendata.mmseqs.org/colabfold/uniref30_2302_newtaxonomy.tar.gz
TAX_ARCHIVE=uniref30_2302_newtaxonomy.tar.gz
TAX_SHA=b94f03c15a9e4f3bfb4cab88ea7c5bef1f6b51c98bf9fbf3434d15a7e620b289
TAX_ARCHIVE_BYTES=1975608472
TAX_RAW_BYTES=6505843214
TAX_MEMBER_MAPPING=uniref30_2302_db_mapping
TAX_MEMBER_TAXONOMY=uniref30_2302_db_taxonomy
TAX_BACKUP_SUFFIX=.pre2508
TAX_INSTALLED_BYTES=5820218761
TAX_INSTALLED_TAXONOMY_BYTES=685624453

install_taxonomy_overlay() {
    local dir=$1 f

    for f in db.dbtype db_seq.index; do
        [[ -e "$dir/$f" ]] || fatal "no uniref30_2302 database at $dir (missing $f).
  Install the database first, then apply the overlay:
      ./install_db.sh --family mmseqs uniref30_2302"
    done

    if [[ -e "$dir/db_mapping" ]]; then
        local have have_tax
        have=$(stat -Lc %s "$dir/db_mapping" 2>/dev/null || echo 0)
        have_tax=$(stat -Lc %s "$dir/db_taxonomy" 2>/dev/null || echo 0)
        if [[ "$have" == "$TAX_INSTALLED_BYTES" \
           && "$have_tax" == "$TAX_INSTALLED_TAXONOMY_BYTES" ]]; then
            info "pairing taxonomy already at the 2025-08 files — nothing to do"
            return 0
        fi
        warn "db_mapping ($(human "$have")) and db_taxonomy ($(human "$have_tax")) are not
         the 2025-08 pair. Both are moved aside to *$TAX_BACKUP_SUFFIX. Pairing that
         reads one of the two against the other's vintage is wrong for every hit."
    fi

    ARCHIVE_PATH=""
    fetch_archive "$TAX_ARCHIVE" "$ARCHIVE_DIR/mmseqs" "$TAX_SHA" "$TAX_URL"

    local tmp="$dir/.newtaxonomy.tmp.$$"
    rm -rf "$tmp"; mkdir -p "$tmp" || fatal "cannot create $tmp"
    info "extracting $TAX_ARCHIVE"
    tar xzf "$ARCHIVE_PATH" -C "$tmp" || { rm -rf "$tmp"; fatal "cannot extract $ARCHIVE_PATH"; }
    for f in "$TAX_MEMBER_MAPPING" "$TAX_MEMBER_TAXONOMY"; do
        [[ -f "$tmp/$f" ]] || { rm -rf "$tmp"; fatal "$TAX_ARCHIVE did not contain $f — wrong archive?"; }
    done

    local n_map n_seq
    n_map=$(wc -l < "$tmp/$TAX_MEMBER_MAPPING")
    n_seq=$(wc -l < "$dir/db_seq.index")
    info "key space: mapping $n_map lines vs db_seq $n_seq entries"
    if [[ "$n_map" != "$n_seq" ]]; then
        rm -rf "$tmp"
        fatal "mapping/_seq key-space mismatch ($n_map vs $n_seq).
  This overlay does not belong to the database in $dir. Nothing was changed."
    fi

    local src dst pair moved=() backed=()
    for pair in "$TAX_MEMBER_MAPPING:db_mapping" "$TAX_MEMBER_TAXONOMY:db_taxonomy"; do
        dst=${pair##*:}
        [[ -e "$dir/$dst" || -L "$dir/$dst" ]] || continue
        rm -rf "${dir:?}/$dst$TAX_BACKUP_SUFFIX"
        mv "$dir/$dst" "$dir/$dst$TAX_BACKUP_SUFFIX" \
            || { rm -rf "$tmp"; fatal "cannot move aside $dst — nothing was changed"; }
        backed+=("$dst")
    done

    restore_taxonomy() {
        local d
        for d in "${moved[@]}";  do rm -f "${dir:?}/$d"; done
        for d in "${backed[@]}"; do mv -f "$dir/$d$TAX_BACKUP_SUFFIX" "$dir/$d"; done
    }

    for pair in "$TAX_MEMBER_MAPPING:db_mapping" "$TAX_MEMBER_TAXONOMY:db_taxonomy"; do
        src=${pair%%:*}; dst=${pair##*:}
        if ! mv "$tmp/$src" "$dir/$dst"; then
            restore_taxonomy
            rm -rf "$tmp"
            fatal "cannot install $dst — the previous taxonomy was put back"
        fi
        moved+=("$dst")
        info "installed $dst  ($(human "$(stat -Lc %s "$dir/$dst")"))"
    done
    rm -rf "$tmp"
    info "multimer pairing now uses the 2025-08-04 taxonomy; the .idx needs no rebuild"
}

CS219_MAX_EMPTY_LEN=1
CS219_KNOWN_EMPTY=9

prune_empty_cs219() {
    local dir=$1
    local idx="$dir/db_cs219.ffindex"
    local n_bad before after ids tmp

    [[ -f "$idx" ]] || fatal "no db_cs219.ffindex at $dir"

    IFS=$'\t' read -r n_bad before ids < <(awk -F'\t' -v m="$CS219_MAX_EMPTY_LEN" '
        $3+0 <= m { n++; ids = ids " " $1 }
        END { printf "%d\t%d\t%s\n", n+0, NR, ids }' "$idx")

    if [[ "$n_bad" -eq 0 ]]; then
        info "cs219: no empty records"
        return 0
    fi

    [[ "$n_bad" -eq "$CS219_KNOWN_EMPTY" ]] \
        || warn "cs219 carries $n_bad empty records, not the $CS219_KNOWN_EMPTY this build is
         known for. All of them are dropped, but the extra ones are unaccounted for."

    tmp="$idx.prune.$$"
    awk -F'\t' -v m="$CS219_MAX_EMPTY_LEN" '$3+0 > m' "$idx" >"$tmp" \
        || { rm -f "$tmp"; fatal "cannot rewrite db_cs219.ffindex — nothing was changed"; }

    after=$(wc -l <"$tmp")
    if [[ $(( before - after )) -ne "$n_bad" ]]; then
        rm -f "$tmp"
        fatal "the rewrite drops $(( before - after )) records, not $n_bad. Nothing was changed."
    fi

    mv "$tmp" "$idx" || { rm -f "$tmp"; fatal "cannot replace db_cs219.ffindex"; }
    info "cs219: $before -> $after records, dropped$ids"
    info "those ids are unsearchable now; their a3m and hhm records are untouched"
}

cluster_memory_limit() {
    local mb=${SLURM_MEM_PER_NODE:-}
    if [[ -z "$mb" && -n "${SLURM_MEM_PER_CPU:-}" && -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
        mb=$(( SLURM_MEM_PER_CPU * SLURM_CPUS_PER_TASK ))
    fi
    [[ -n "$mb" ]] || mb=$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo 2>/dev/null)
    [[ "$mb" =~ ^[0-9]+$ && "$mb" -gt 512 ]] || return 0
    printf '%dM\n' $(( mb * 80 / 100 ))
}

build_rna_from_upstream() {
    local target=$1 archive=$2 key=$3
    local work="$ARCHIVE_DIR/rna/${key}_work" fasta module mem
    local -a args=()

    rm -rf "$work"; mkdir -p "$work" || fatal "cannot create $work"
    fasta="$work/$(basename "${archive%.gz}")"
    info "decompressing $(basename "$archive")"
    gzip -dc "$archive" >"$fasta" || { rm -rf "$work"; fatal "cannot decompress $archive"; }

    case "$key" in
        rfam)       module=easy-cluster  ;;
        rnacentral) module=easy-linclust ;;
        *) rm -rf "$work"; fatal "no clustering recipe for the rna database '$key'" ;;
    esac

    STAGE_DIR="$(dirname "$target")/.stage.$(basename "$target").$$"
    rm -rf "$STAGE_DIR"; mkdir -p "$STAGE_DIR" || fatal "cannot create $STAGE_DIR"

    mem=$(cluster_memory_limit)
    args=("$module" "$fasta" "$work/${key}_clust" "$work/tmp"
          --min-seq-id 0.9 -c 0.8 --cov-mode 1 --threads "$THREADS")
    [[ -n "$mem" ]] && args+=(--split-memory-limit "$mem")
    info "mmseqs $module${mem:+, memory ceiling $mem} (this is the slow step)"
    "$MMSEQS_BIN" "${args[@]}" \
        || { rm -rf "$work" "$STAGE_DIR"; fatal "mmseqs $module failed for $key"; }

    mv "$work/${key}_clust_rep_seq.fasta" "$STAGE_DIR/${key}_clust_rep_seq.fasta" \
        || { rm -rf "$work" "$STAGE_DIR"
             fatal "mmseqs $module wrote no representative sequences for $key"; }
    rm -rf "$work"
}

install_uniref30() {
    local target=$1 archive=$2
    local work="$ARCHIVE_DIR/mmseqs/uniref30_work" kind src

    rm -rf "$work"; mkdir -p "$work" || fatal "cannot create $work"
    info "extracting TSVs into $work"
    tar -xzf "$archive" -C "$work" || { rm -rf "$work"; fatal "extraction failed: $archive"; }

    STAGE_DIR="$(dirname "$target")/.stage.$(basename "$target").$$"
    rm -rf "$STAGE_DIR"; mkdir -p "$STAGE_DIR" || fatal "cannot create $STAGE_DIR"
    info "mmseqs tsv2exprofiledb (this is the slow step)"
    "$MMSEQS_BIN" tsv2exprofiledb "$work/uniref30_2302" "$STAGE_DIR/db" \
        || { rm -rf "$work" "$STAGE_DIR"; fatal "tsv2exprofiledb failed"; }

    for kind in mapping taxonomy; do
        [[ -e "$STAGE_DIR/db_$kind" ]] && continue
        src=$(find "$work" -maxdepth 1 -type f -name "*_$kind" -print -quit)
        [[ -n "$src" ]] && { mv "$src" "$STAGE_DIR/db_$kind"
                             info "pairing file: $(basename "$src") -> db_$kind"; }
    done
    rm -rf "$work"
}
