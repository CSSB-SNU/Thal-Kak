#!/bin/bash

DOWNLOADER=""
FETCHED=0

pick_downloader() {
    if command -v aria2c >/dev/null 2>&1; then
        DOWNLOADER=aria2c
    elif command -v curl >/dev/null 2>&1; then
        DOWNLOADER=curl
    elif command -v wget >/dev/null 2>&1; then
        DOWNLOADER=wget
    else
        fatal "no downloader found (need aria2c, curl or wget).
  aria2c ships in the project environment: activate it first
  (pixi shell, or conda activate thalkak)"
    fi
}

fetch_archive() {
    local name=$1 dir=$2 sha=$3 url=$4
    local final="$dir/$name" part="$dir/$name.part"

    FETCHED=0
    mkdir -p "$dir"

    if [[ -f "$final" ]]; then
        info "archive present, verifying: $name"
        verify_checksum "$final" "$sha"
        ARCHIVE_PATH=$final
        return 0
    fi

    info "downloading $name"
    rm -f "$part"
    case "$DOWNLOADER" in
        aria2c) aria2c -x8 -s8 -c --console-log-level=warn --summary-interval=0 \
                       -d "$dir" -o "$name.part" "$url" ;;
        curl)   curl -fL --retry 3 --retry-delay 5 -o "$part" "$url" ;;
        wget)   wget -c -O "$part" "$url" ;;
        *)      fatal "no downloader selected (pick_downloader was not called)" ;;
    esac || fatal "download failed: $url"

    verify_checksum "$part" "$sha"
    mv -f "$part" "$final" || fatal "cannot rename $part -> $final"
    ARCHIVE_PATH=$final
    FETCHED=1
}
