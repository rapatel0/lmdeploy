#!/usr/bin/env bash
# Restore the TP all-reduces after MTP row-parallel wo and w2 projections.
set -uo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-mtp-tp-reduce-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"${RESULTS}/exit_code"; echo "artifacts in ${RESULTS} (exit ${rc})"' EXIT
cat /src/SOURCE_STAMP

if ! bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1; then
   grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -30
   exit 2
fi
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1

run_arm() {
   local label=$1 k=$2 reduce=$3
   echo "=== ${label}: K=${k} tp_reduce=${reduce} ==="
   TM_MTP_LOCAL_TOP1=1 TM_MTP_EAGLE_ROTATION=0 TM_MTP_FROZEN_KV=0 TM_MTP_TP_REDUCE="${reduce}" \
      python3 /job/bench_decode.py --model "${MODEL_DIR}" --tp "${TP}" \
      --num-draft-tokens "${k}" --input-tokens 1024 --output-tokens 256 --trials 2 \
      --require-mtp --json-out "${RESULTS}/${label}.json"
}
run_arm k1_local 1 0 || exit $?
run_arm k1_reduced 1 1 || exit $?
run_arm k4_local 4 0 || exit $?
run_arm k4_reduced 4 1 || exit $?

python3 - "${RESULTS}" <<'PY'
import json, os, sys
r=sys.argv[1]
print('=== MTP TP reduction A/B ===')
for k in (1,4):
 a=json.load(open(os.path.join(r,f'k{k}_local.json')))['mean_decode_tok_s']
 b=json.load(open(os.path.join(r,f'k{k}_reduced.json')))['mean_decode_tok_s']
 print(f'K={k}: local={a:.2f} reduced={b:.2f} ratio={b/a:.3f}x')
PY
echo MTP_TP_REDUCE_COMPLETE
