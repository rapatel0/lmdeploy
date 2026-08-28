#!/usr/bin/env bash
# A/B full-vocabulary TP all-gather against one-candidate-per-rank draft top-1.
set -uo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-mtp-local-top1-${SRC_COMMIT}
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
    echo "FAIL: build" >&2
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -30
    exit 2
fi
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1

run_arm() {
    local label=$1
    local k=$2
    local local_top1=$3
    echo "=== ${label}: K=${k} local_top1=${local_top1} ==="
    TM_MTP_LOCAL_TOP1="${local_top1}" python3 /job/bench_decode.py \
        --model "${MODEL_DIR}" --tp "${TP}" --num-draft-tokens "${k}" \
        --input-tokens 1024 --output-tokens 256 --trials 2 \
        --require-mtp --json-out "${RESULTS}/${label}.json"
}

run_arm k1_full 1 0 || exit $?
run_arm k1_top1 1 1 || exit $?
run_arm k4_full 4 0 || exit $?
run_arm k4_top1 4 1 || exit $?

python3 - "${RESULTS}" <<'PY'
import json
import os
import sys

root = sys.argv[1]
rows = {}
for label in ("k1_full", "k1_top1", "k4_full", "k4_top1"):
    rows[label] = json.load(open(os.path.join(root, f"{label}.json")))["mean_decode_tok_s"]
print("=== local TP top-1 A/B ===")
for k in (1, 4):
    old = rows[f"k{k}_full"]
    new = rows[f"k{k}_top1"]
    print(f"K={k}: full={old:.2f} top1={new:.2f} ratio={new / old:.3f}x")
PY

touch "${RESULTS}/completed"
echo MTP_LOCAL_TOP1_COMPLETE
