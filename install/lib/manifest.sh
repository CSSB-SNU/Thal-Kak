#!/bin/bash

F_FAMILY=1 F_KEY=2 F_STEM=3 F_ARCHIVE=4 F_URL=5 F_SHA=6
F_ABYTES=7 F_RBYTES=8 F_IBYTES=9 F_WBYTES=10 F_INDEX=11

MANIFEST_FIELDS=11

read_rows() {
    local family=$1
    awk -F'\t' -v f="$family" '
        /^[[:space:]]*(#|$)/ { next }
        $1 == "family"       { next }
        $1 == f              { print }
    ' "$MANIFEST"
}

validate_manifest() {
    local bad
    bad=$(awk -F'\t' -v want="$MANIFEST_FIELDS" '
        /^[[:space:]]*(#|$)/ { next }
        NF != want { printf "  line %d: %d fields, expected %d  (%s)\n", NR, NF, want, $2; next }
        ($4 == "-") && ($5 == "-") {
            printf "  line %d: %s names neither an archive nor a url\n", NR, $2; next
        }
        ($4 == "-") != ($5 == "-") {
            printf "  line %d: %s has an archive but no url, or a url but no archive\n", NR, $2
        }
    ' "$MANIFEST")
    [[ -z "$bad" ]] || fatal "$MANIFEST has malformed row(s):
$bad
  Columns are tab-separated and read by position:
    family key stem archive url sha256 archive_bytes raw_bytes idx_bytes work_bytes index
  The url is the only place an archive is fetched from, so every row names both.
  A database we do not publish has no row at all."
}

manifest_field() {
    local row=$1 index=$2
    printf '%s' "$row" | cut -d$'\t' -f"$index"
}

verify_checksum() {
    local file=$1 want=$2 algo=sha256 got
    if [[ "$want" == "-" ]]; then
        warn "no checksum in the manifest for $(basename "$file") — not verified.
         A truncated or corrupt archive will surface as a tar error rather than
         silently, but it will not be caught here."
        return 0
    fi
    case "$want" in md5:*) algo=md5; want=${want#md5:} ;; esac
    got=$("${algo}sum" "$file" | awk '{print $1}')
    [[ "$got" == "$want" ]] || fatal "$algo mismatch for $file
      expected $want
      got      $got
  The download is corrupt. Delete the file and re-run."
}
