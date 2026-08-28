#!/usr/bin/env bash
# Compare sequentially appended draft KV with SGLang-style frozen committed KV.
set -uo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-mtp-frozen-kv-${SRC_COMMIT}
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
    local frozen=$2
    echo "=== ${label}: K=4 frozen_kv=${frozen} ==="
    TM_MTP_LOCAL_TOP1=1 TM_MTP_FROZEN_KV="${frozen}" python3 /job/bench_decode.py \
        --model "${MODEL_DIR}" --tp "${TP}" --num-draft-tokens 4 \
        --input-tokens 1024 --output-tokens 256 --trials 2 \
        --require-mtp --json-out "${RESULTS}/${label}.json"
}

run_arm appended 0 || exit $?
run_arm frozen 1 || exit $?

python3 - "${RESULTS}" <<'PY'
import json
import os
import sys
root = sys.argv[1]
a = json.load(open(os.path.join(root, "appended.json")))["mean_decode_tok_s"]
f = json.load(open(os.path.join(root, "frozen.json")))["mean_decode_tok_s"]
print("=== frozen KV A/B ===")
print(f"K=4: appended={a:.2f} frozen={f:.2f} ratio={f / a:.3f}x")
PY

touch "${RESULTS}/completed"
echo MTP_FROZEN_KV_COMPLETE
