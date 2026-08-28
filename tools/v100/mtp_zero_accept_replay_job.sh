#!/usr/bin/env bash
# Measure the exactness-preserving fallback for zero-accept verification steps.
set -uo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-mtp-zero-replay-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
finish() {
    rc=$?
    echo "${rc}" >"${RESULTS}/exit_code"
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
    local label=$1 k=$2 replay=$3 ambiguous=$4 margin=$5
    echo "=== ${label}: K=${k} zero_replay=${replay} ambiguous_replay=${ambiguous} margin=${margin} ==="
    local extra=()
    [ "${k}" = 0 ] || extra+=(--require-mtp)
    TM_MTP_LOCAL_TOP1=1 TM_MTP_TP_REDUCE=1 TM_MTP_EAGLE_ROTATION=1 \
        TM_MTP_ACCEPTED_REPAIR=1 TM_MTP_ZERO_ACCEPT_REPLAY="${replay}" \
        TM_MTP_AMBIGUOUS_REPLAY="${ambiguous}" TM_MTP_AMBIGUITY_MARGIN="${margin}" python3 /src/tools/v100/bench_decode.py --model "${MODEL_DIR}" --tp "${TP}" \
        --num-draft-tokens "${k}" --input-tokens 1024 --output-tokens 256 --trials 2 \
        --json-out "${RESULTS}/${label}.json" "${extra[@]}"
}
run_arm k0 0 0 0 0 || exit $?
run_arm k4_control 4 0 0 0 || exit $?
run_arm k4_margin_003 4 0 1 0.03125 || exit $?
run_arm k4_margin_006 4 0 1 0.0625 || exit $?
run_arm k4_margin_012 4 0 1 0.125 || exit $?
run_arm k4_replay 4 1 0 0 || exit $?
python3 - "${RESULTS}" <<'PY'
import json,os,sys
r=sys.argv[1]
names=('k0','k4_control','k4_margin_003','k4_margin_006','k4_margin_012','k4_replay')
vals={n:json.load(open(os.path.join(r,n+'.json')))['mean_decode_tok_s'] for n in names}
for n,v in vals.items(): print(f'{n}: {v:.2f} tok/s ({v/vals["k0"]:.3f}x K=0)')
PY
echo MTP_ZERO_ACCEPT_REPLAY_COMPLETE
