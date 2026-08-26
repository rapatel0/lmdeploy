#!/usr/bin/env bash
# Test the economic assumption behind speculation: is a K+1 prefill-shaped
# forward cheaper than K+1 decode forwards? Everything else assumes it.
set -uo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-bench-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1

finish() {
    rc=$?
    echo "$rc" > "${RESULTS}/exit_code"
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT

echo "=== build ==="
cd /src
bash /src/tools/v100/build_v100.sh || { echo "FAIL: build" >&2; exit 2; }

WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' 2>/dev/null |
    sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ] || { echo "FAIL: no wheel" >&2; exit 2; }
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1

MODEL="${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
cd /
echo
echo "=== prefill vs decode per-token cost ==="
stdbuf -oL -eL python3 /src/tools/v100/bench_prefill_vs_decode.py \
    --model-dir "${MODEL}" --tp 4 --emit-json "${RESULTS}/bench.json" 2>&1
