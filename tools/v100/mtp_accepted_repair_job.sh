#!/usr/bin/env bash
# Rebuild accepted draft KV from verifier hidden states before the next chain.
set -uo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-mtp-accepted-repair-${SRC_COMMIT}
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
    local label=$1 k=$2 repair=$3
    echo "=== ${label}: K=${k} accepted_repair=${repair} ==="
    TM_MTP_LOCAL_TOP1=1 TM_MTP_TP_REDUCE=1 TM_MTP_EAGLE_ROTATION=1 \
        TM_MTP_FROZEN_KV=0 TM_MTP_ACCEPTED_REPAIR="${repair}" \
        python3 /job/bench_decode.py --model "${MODEL_DIR}" --tp "${TP}" \
        --num-draft-tokens "${k}" --input-tokens 1024 --output-tokens 256 --trials 2 \
        --require-mtp --json-out "${RESULTS}/${label}.json"
}
run_arm k1_control 1 0 || exit $?
run_arm k1_repair 1 1 || exit $?
run_arm k4_control 4 0 || exit $?
run_arm k4_repair 4 1 || exit $?

python3 - "${RESULTS}" <<'PY'
import json, os, sys
r=sys.argv[1]
print('=== Accepted KV repair A/B ===')
for k in (1,4):
 a=json.load(open(os.path.join(r,f'k{k}_control.json')))['mean_decode_tok_s']
 b=json.load(open(os.path.join(r,f'k{k}_repair.json')))['mean_decode_tok_s']
 print(f'K={k}: control={a:.2f} repair={b:.2f} ratio={b/a:.3f}x')
PY
touch "${RESULTS}/completed"
echo MTP_ACCEPTED_REPAIR_COMPLETE
