#!/usr/bin/env bash
# Match SGLang EAGLE/NEXTN prefill rotation and draft-zero position semantics.
set -uo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-mtp-eagle-rotation-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
finish() {
    rc=$?
    echo "${rc}" >"${RESULTS}/exit_code"
    [ -f "${RESULTS}/completed" ] || echo KILLED >"${RESULTS}/incomplete"
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT

cat /src/SOURCE_STAMP
if ! bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1; then
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -30
    exit 2
fi
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1

run_arm() {
    local label=$1
    local k=$2
    local rotation=$3
    echo "=== ${label}: K=${k} eagle_rotation=${rotation} ==="
    TM_MTP_LOCAL_TOP1=1 TM_MTP_FROZEN_KV=0 TM_MTP_EAGLE_ROTATION="${rotation}" \
        python3 /job/bench_decode.py \
        --model "${MODEL_DIR}" --tp "${TP}" --num-draft-tokens "${k}" \
        --input-tokens 1024 --output-tokens 256 --trials 2 \
        --require-mtp --json-out "${RESULTS}/${label}.json"
}

run_arm k1_shifted 1 0 || exit $?
run_arm k1_rotated 1 1 || exit $?
run_arm k4_shifted 4 0 || exit $?
run_arm k4_rotated 4 1 || exit $?

python3 - "${RESULTS}" <<'PY'
import json
import os
import sys
root = sys.argv[1]
print("=== EAGLE rotation A/B ===")
for k in (1, 4):
    old = json.load(open(os.path.join(root, f"k{k}_shifted.json")))["mean_decode_tok_s"]
    new = json.load(open(os.path.join(root, f"k{k}_rotated.json")))["mean_decode_tok_s"]
    print(f"K={k}: shifted={old:.2f} rotated={new:.2f} ratio={new / old:.3f}x")
PY

touch "${RESULTS}/completed"
echo MTP_EAGLE_ROTATION_COMPLETE
