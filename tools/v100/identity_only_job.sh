#!/usr/bin/env bash
# Build and run only the exactness gates. Use this for correctness iterations;
# the benchmark has no meaning until identity passes.
set -uo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-identity-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
finish() {
    rc=$?
    echo "$rc" >"${RESULTS}/exit_code"
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT

cat /src/SOURCE_STAMP
echo "=== build ==="
bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || {
    tail -60 "${RESULTS}/build.log"
    exit 2
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1

MODEL="${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
TP="${TP:-4}"
K="${NUM_DRAFT_TOKENS:-4}"

echo "=== normal identity ==="
python3 /src/tools/v100/verify_spec_identity.py --model-dir "${MODEL}" --tp "${TP}" \
    --num-draft-tokens "${K}" --json-out "${RESULTS}/identity.json"
NORMAL=$?

echo "=== forced-reject identity ==="
TM_MTP_FORCE_REJECT=1 python3 /src/tools/v100/verify_spec_identity.py \
    --model-dir "${MODEL}" --tp "${TP}" --num-draft-tokens "${K}" \
    --json-out "${RESULTS}/identity_force_reject.json"
FORCED=$?

echo "IDENTITY_RESULT normal=${NORMAL} forced=${FORCED}"
[ "${NORMAL}" -eq 0 ] && [ "${FORCED}" -eq 0 ]
