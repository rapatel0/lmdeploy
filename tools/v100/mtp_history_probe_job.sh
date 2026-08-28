#!/usr/bin/env bash
# Falsify whether the MTP attention output depends on its cached prompt history.
set -uo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-mtp-history-probe-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"${RESULTS}/exit_code"; echo "artifacts in ${RESULTS} (exit ${rc})"' EXIT

cat /src/SOURCE_STAMP
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ] || { echo 'FAIL: no staged wheel' >&2; exit 2; }
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1

run_arm() {
    local label=$1
    local shift=$2
    echo "=== ${label}: probe_shift=${shift} ==="
    TM_MTP_LOCAL_TOP1=1 TM_MTP_EAGLE_ROTATION=0 TM_MTP_FROZEN_KV=0 TM_MTP_PROBE_SHIFT="${shift}" \
        python3 /job/bench_decode.py --model "${MODEL_DIR}" --tp "${TP}" \
        --num-draft-tokens 1 --input-tokens 1024 --output-tokens 256 --trials 1 \
        --require-mtp --json-out "${RESULTS}/${label}.json"
}
run_arm full_history 0 || exit $?
run_arm one_token_history -1000 || exit $?
python3 - "${RESULTS}" <<'PY'
import json, os, sys
r=sys.argv[1]
a=json.load(open(os.path.join(r,'full_history.json')))['mean_decode_tok_s']
b=json.load(open(os.path.join(r,'one_token_history.json')))['mean_decode_tok_s']
print(f'HISTORY_PROBE full={a:.2f} one_token={b:.2f} ratio={b/a:.3f}x')
PY
echo MTP_HISTORY_PROBE_COMPLETE
