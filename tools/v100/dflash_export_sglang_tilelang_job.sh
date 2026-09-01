#!/usr/bin/env bash
set -euo pipefail
OUT=/results/dflash-sglang-tilelang-native
rm -rf "$OUT"
mkdir -p "$OUT"
python3 - "$OUT" <<'PY'
from pathlib import Path
import sys
from sglang.srt.layers.attention.tilelang_fa_v100._kernels_paged_verify import get_paged_verify_kernels
out = Path(sys.argv[1])
partial, combine, splits = get_paged_verify_kernels(
    batch=1,
    heads=8,
    heads_kv=2,
    dim=128,
    block_size=16,
    num_pages=1024,
    max_blocks=1024,
    causal=False,
    fp8_kv=False,
    min_tokens_per_split=128,
)
for name, kernel in (("partial", partial), ("combine", combine)):
    kernel.export_sources(kernel_path=str(out / f"{name}.cu"), host_path=str(out / f"{name}_host.cc"))
    kernel.export_ptx(str(out / f"{name}.ptx"))
    kernel.export_sass(str(out / f"{name}.sass"))
print(f"SGLANG_TILELANG_EXPORT splits={splits}")
PY
TILELANG_INCLUDE=$(
    python3 - <<'PY'
from pathlib import Path
import tilelang
root = Path(tilelang.__file__).resolve().parent
matches = list(root.rglob('tl_templates'))
if not matches:
    raise SystemExit(f'tl_templates not found below {root}')
print(matches[0].parent)
PY
)
CUTLASS_HEADER=$(find /opt /usr/local -path '*/cutlass/numeric_types.h' -print -quit 2>/dev/null)
test -n "$CUTLASS_HEADER"
CUTLASS_INCLUDE=$(dirname "$(dirname "$CUTLASS_HEADER")")
nvcc -std=c++17 -arch=sm_70 -cubin -I"$TILELANG_INCLUDE" -I"$CUTLASS_INCLUDE" "$OUT/partial.cu" -o "$OUT/partial.cubin"
nvcc -std=c++17 -arch=sm_70 -cubin -I"$TILELANG_INCLUDE" -I"$CUTLASS_INCLUDE" "$OUT/combine.cu" -o "$OUT/combine.cubin"
sha256sum "$OUT"/*
ls -lh "$OUT"/*
sleep 300
