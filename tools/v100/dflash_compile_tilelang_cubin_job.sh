#!/usr/bin/env bash
set -euo pipefail
SOURCE=/results/dflash-tilelang-draft-verify
OUT=/results/dflash-tilelang-native-cubin
rm -rf "$OUT"
mkdir -p "$OUT"
INCLUDE=$(python3 - <<'PY'
from pathlib import Path
import tilelang
root = Path(tilelang.__file__).resolve().parent
for candidate in (root / 'include', root.parent / 'include', root / '3rdparty' / 'tilelang' / 'include'):
    if (candidate / 'tl_templates').is_dir():
        print(candidate)
        break
else:
    matches = list(root.rglob('tl_templates'))
    if not matches:
        raise SystemExit(f'tl_templates not found below {root}')
    print(matches[0].parent)
PY
)
echo "TileLang include: $INCLUDE"
nvcc -std=c++17 -arch=sm_70 -cubin -I"$INCLUDE" "$SOURCE/partial.cu" -o "$OUT/partial.cubin"
nvcc -std=c++17 -arch=sm_70 -cubin -I"$INCLUDE" "$SOURCE/combine.cu" -o "$OUT/combine.cubin"
sha256sum "$OUT"/*.cubin
ls -lh "$OUT"/*.cubin
