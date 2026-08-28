#!/usr/bin/env bash
# Screen speculative depths after restoring TP reductions.
set -uo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-mtp-depth-sweep-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
finish() {
    rc=$?
    echo "${rc}" >"${RESULTS}/exit_code"
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT
cat /src/SOURCE_STAMP
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ] || exit 2
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1

for k in 0 1 2 3 4 5 6 7; do
    echo "=== depth K=${k} ==="
    args=()
    [ "${k}" -eq 0 ] || args+=(--require-mtp)
    repair=0
    [ "${k}" -lt 2 ] || repair=1
    TM_MTP_LOCAL_TOP1=1 TM_MTP_TP_REDUCE=1 TM_MTP_EAGLE_ROTATION=1 \
        TM_MTP_FROZEN_KV=0 TM_MTP_ACCEPTED_REPAIR="${repair}" \
        python3 /job/bench_decode.py --model "${MODEL_DIR}" --tp "${TP}" \
        --num-draft-tokens "${k}" --input-tokens 1024 --output-tokens 256 --trials 1 \
        "${args[@]}" --json-out "${RESULTS}/k${k}.json" || exit $?
done
python3 - "${RESULTS}" <<'PY'
import json, os, sys
r=sys.argv[1]
base=json.load(open(os.path.join(r,'k0.json')))['mean_decode_tok_s']
print('=== MTP depth sweep ===')
for k in range(8):
 v=json.load(open(os.path.join(r,f'k{k}.json')))['mean_decode_tok_s']
 print(f'K={k}: {v:.2f} tok/s ({v/base:.3f}x K=0)')
PY
echo MTP_DEPTH_SWEEP_COMPLETE
