#!/usr/bin/env bash
set -euo pipefail
ROOT=/results/dflash-tilelang-draft-verify
for file in partial.cu combine.cu; do
    test -s "$ROOT/$file"
    printf 'BEGIN_%s\n' "${file//./_}"
    base64 -w 0 "$ROOT/$file"
    printf '\nEND_%s\n' "${file//./_}"
done
