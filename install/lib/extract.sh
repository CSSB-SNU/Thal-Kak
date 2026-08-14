#!/bin/bash

extract_archive() {
    local archive=$1 target=$2
    local parent stage
    parent=$(dirname "$target")
    stage="$parent/.stage.$(basename "$target").$$"

    mkdir -p "$parent"
    rm -rf "$stage"
    mkdir -p "$stage" || fatal "cannot create staging dir $stage"

    info "extracting into $stage"
    case "$archive" in
        *.tar.zst) zstd -dc "$archive" | tar -x -p -C "$stage" ;;
        *.tar.gz)  tar -xzp -f "$archive" -C "$stage" ;;
        *) rm -rf "$stage"
           fatal "unknown archive format: $(basename "$archive")
  Expected .tar.zst or .tar.gz." ;;
    esac || { rm -rf "$stage"; fatal "extraction failed: $archive"; }

    STAGE_DIR=$stage
}

_stem_belongs() { [[ "$1" == "$2" || "$1" == "$2"[._]* ]]; }

normalize_stem() {
    local dir=$1 src=$2
    [[ "$src" == "-" ]] && return 0

    local n_file=0 n_link=0 f base rest tgt dst leftover=()

    for f in "$dir"/*; do
        [[ -L "$f" ]] || continue
        base=${f##*/}
        _stem_belongs "$base" "$src" || continue
        tgt=$(readlink "$f")
        case "$tgt" in
            */*) fatal "unexpected non-sibling symlink in archive: $base -> $tgt" ;;
        esac
        _stem_belongs "$tgt" "$src" || fatal "symlink $base points at $tgt, which is not
  part of the manifest stem '$src'. Refusing to guess; check the manifest."
        tgt="db${tgt#"$src"}"
        rest=${base#"$src"}
        dst="$dir/db$rest"
        if [[ "$dst" != "$f" && ( -e "$dst" || -L "$dst" ) ]]; then
            fatal "$base would become db$rest, which the archive already contains.
  Refusing rather than overwrite it."
        fi
        ln -sfn "$tgt" "$dst" || fatal "cannot rewrite symlink $base"
        [[ "$dst" == "$f" ]] || rm -f "$f"
        n_link=$((n_link + 1))
    done

    for f in "$dir"/*; do
        [[ -f "$f" && ! -L "$f" ]] || continue
        base=${f##*/}
        _stem_belongs "$base" "$src" || continue
        rest=${base#"$src"}
        dst="$dir/db$rest"
        [[ "$dst" == "$f" ]] && continue
        if [[ -e "$dst" || -L "$dst" ]]; then
            fatal "$base would become db$rest, which already exists.
  Refusing rather than lose one of them."
        fi
        mv "$f" "$dst" || fatal "cannot rename $base -> db$rest"
        n_file=$((n_file + 1))
    done

    for f in "$dir"/*; do
        base=${f##*/}
        _stem_belongs "$base" "$src" && leftover+=("$base")
    done
    [[ ${#leftover[@]} -eq 0 ]] || fatal "files still named after the old stem '$src':
$(printf '      %s\n' "${leftover[@]}")"

    info "stem normalised: $src* -> db* ($n_file files, $n_link symlinks)"
}

verify_mmseqs_db() {
    local dir=$1 pairable=$2 expandable=$3
    local f missing=()

    for f in db db.dbtype db.index db_h db_h.dbtype db_h.index; do
        [[ -e "$dir/$f" ]] || missing+=("$f")
    done
    if [[ "$expandable" == "yes" ]]; then
        for f in db_seq db_seq.dbtype db_seq.index db_aln db_aln.dbtype db_aln.index; do
            [[ -e "$dir/$f" ]] || missing+=("$f")
        done
    fi
    if [[ "$pairable" == "yes" ]]; then
        for f in db_mapping db_taxonomy; do
            [[ -e "$dir/$f" ]] || missing+=("$f")
        done
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        fatal "$(basename "$dir"): incomplete mmseqs DB, missing: ${missing[*]}
  Searching would still 'work' while producing broken headers or silently
  skipping multimer pairing, so this is refused."
    fi
}

verify_hhblits_db() {
    local dir=$1 f missing=()
    for f in db_a3m.ffdata db_a3m.ffindex db_cs219.ffdata db_cs219.ffindex; do
        [[ -e "$dir/$f" ]] || missing+=("$f")
    done
    [[ ${#missing[@]} -eq 0 ]] || fatal "$(basename "$dir"): incomplete HHblits DB, missing: ${missing[*]}"
    [[ -e "$dir/db_hhm.ffdata" ]] || warn "$(basename "$dir"): no db_hhm.ff* (HHblits will build profiles on the fly)"
}

is_installed() {
    local fam=$1 dir=$2 key=$3
    [[ -d "$dir" ]] || return 1
    case "$fam" in
        mmseqs)   [[ -e "$dir/db.dbtype" ]] ;;
        hhblits)  [[ -e "$dir/db_cs219.ffindex" ]] ;;
        template) [[ -e "$dir/release_dates.tsv" && -d "$dir/cif/raw" ]] ;;
        rna)      [[ -s "$dir/${key}_clust_rep_seq.fasta" ]] ;;
        *)        return 1 ;;
    esac
}

has_index() {
    local fam=$1 dir=$2 key=$3
    case "$fam" in
        mmseqs) [[ -f "$dir/db.idx.ok" ]] ;;
        rna)    [[ -s "$dir/${key}_v_latest.mdf" \
                   && -s "$dir/${key}_clust_rep_seq.fasta.ssi" ]] ;;
        *)      return 0 ;;
    esac
}

verify_rna_db() {
    local dir=$1 key=$2
    [[ -s "$dir/${key}_clust_rep_seq.fasta" ]] || fatal \
"$key: incomplete RNA DB, missing ${key}_clust_rep_seq.fasta
  nhmmer would find nothing and msa_gen.py would report no hits rather than
  failing, so this is refused."
}

build_index_rna() {
    local dir=$1 key=$2
    local fa="$dir/${key}_clust_rep_seq.fasta" mdf="$dir/${key}_v_latest.mdf"

    if [[ ! -s "$mdf" ]]; then
        info "makehmmerdb (this is the slow step)"
        makehmmerdb "$fa" "$mdf" || { rm -f "$mdf"; fatal "makehmmerdb failed for $key"; }
    else
        info "nhmmer database present"
    fi
    if [[ ! -s "$fa.ssi" ]]; then
        esl-sfetch --index "$fa" >/dev/null || fatal "esl-sfetch --index failed for $key"
    fi
}

verify_template_db() {
    local dir=$1 snapshot=$2 missing=()

    [[ -e "$dir/fasta/pdb_seqres_protein.fasta" ]] || missing+=("fasta/pdb_seqres_protein.fasta (DEFAULT_SEQRES_FASTA)")
    [[ -e "$dir/metadata/cif_metadata.tsv"      ]] || missing+=("metadata/cif_metadata.tsv (DEFAULT_CIF_METADATA)")
    [[ -e "$dir/release_dates.tsv"              ]] || missing+=("release_dates.tsv (DEFAULT_RELEASE_DATES)")
    [[ -d "$dir/cif/raw"                        ]] || missing+=("cif/raw/ (DEFAULT_MMCIF_DIR)")

    # Named after the SNAPSHOT, not after `$dir` -- `$dir` is the staging
    # directory (`.stage.<snapshot>.<pid>`), while engine_mmseqs.py reads
    # `<snapshot lowercased>_pdb`.
    local pdb_db="$dir/mmseqs/$(printf '%s' "$snapshot" | tr '[:upper:]' '[:lower:]')_pdb"
    [[ -e "$pdb_db.dbtype" ]] || missing+=("mmseqs/$(basename "$pdb_db").dbtype (DEFAULT_PDB_DB)")

    if [[ ${#missing[@]} -gt 0 ]]; then
        fatal "$(basename "$dir"): incomplete template DB, missing:
      $(printf '%s\n      ' "${missing[@]}")
  Template search would still run and then fail per target, or drop the date
  cutoff without a word, so this is refused."
    fi

    [[ -e "$pdb_db.idx" ]] ||
        info "no mmseqs index for the template DB (searches run at --db-load-mode 1)"
}

build_index() {
    local dir=$1 mmseqs=$2 threads=$3; shift 3
    local marker="$dir/db.idx.ok"

    # `createindex --split 0` writes ONE `db.idx` when the index fits in memory
    # and numbered `db.idx.0..N` when it does not -- 506 GB of mgnify index on a
    # 755 GB host splits into four. `db.idx.index` is written either way, and it
    # is what db_registry.has_idx() reads, so it is the presence test here too.
    # Checking `db.idx` alone rejects a split index that the pipeline would
    # happily mmap, and then deletes it and rebuilds it to the same shape.
    if [[ -f "$marker" && ( -f "$dir/db.idx" || -f "$dir/db.idx.index" ) ]]; then
        info "index present (db.idx.ok)"
        return 0
    fi
    if [[ -e "$dir/db.idx" || -e "$dir/db.idx.index" ]]; then
        warn "$(basename "$dir"): db.idx exists without a completion marker — it may be
         truncated. Removing and rebuilding."
        rm -f "$dir"/db.idx*
    fi

    info "mmseqs createindex (this is the slow step)"
    "$mmseqs" createindex "$dir/db" "$dir/tmp_createindex" \
        --split 0 --threads "$threads" "$@" \
        || { rm -f "$dir"/db.idx*; fatal "createindex failed for $(basename "$dir")"; }
    rm -rf "$dir/tmp_createindex"

    [[ -e "$dir/db.idx" || -e "$dir/db.idx.index" ]] \
        || fatal "createindex reported success but neither db.idx nor db.idx.index is there"
    printf 'mmseqs=%s\nbuilt=%s\nhost=%s\n' \
        "$("$mmseqs" version 2>/dev/null)" "$(date '+%F %T')" "$(hostname)" >"$marker"
}
