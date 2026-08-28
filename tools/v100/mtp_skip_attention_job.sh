#!/usr/bin/env bash
# Test whether the MTP attention sublayer earns its cost after history proved invariant.
set -uo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-mtp-skip-attn-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
finish() { rc=$?; echo "${rc}" >"${RESULTS}/exit_code"; echo "artifacts in ${RESULTS} (exit ${rc})"; }
trap finish EXIT
cat /src/SOURCE_STAMP
if ! bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1; then
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -30
    exit 2
fi
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
run_arm() {
    local label=$1 k=$2 skip=$3
    echo "=== ${label}: K=${k} skip_attn=${skip} ==="
    TM_MTP_LOCAL_TOP1=1 TM_MTP_TP_REDUCE=1 TM_MTP_EAGLE_ROTATION=1 \
        TM_MTP_ACCEPTED_REPAIR=0 TM_MTP_SKIP_ATTN="${skip}" \
        python3 /job/bench_decode.py --model "${MODEL_DIR}" --tp "${TP}" \
        --num-draft-tokens "${k}" --input-tokens 1024 --output-tokens 256 --trials 2 \
        --require-mtp --json-out "${RESULTS}/${label}.json"
}
for k in 3 4; do
    run_arm k${k}_attention "${k}" 0 || exit $?
    run_arm k${k}_skip "${k}" 1 || exit $?
done
python3 - "${RESULTS}" <<'PY'
import json,os,sys
r=sys.argv[1]
for k in (3,4):
 a=json.load(open(os.path.join(r,f'k{k}_attention.json')))['mean_decode_tok_s']
 b=json.load(open(os.path.join(r,f'k{k}_skip.json')))['mean_decode_tok_s']
 print(f'K={k}: attention={a:.2f} skip={b:.2f} ratio={b/a:.3f}x')
PY
echo MTP_SKIP_ATTN_COMPLETE
