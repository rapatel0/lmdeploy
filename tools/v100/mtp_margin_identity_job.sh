#!/usr/bin/env bash
# Find the smallest verifier top-two margin that restores forced-reject identity.
set -uo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-mtp-margin-identity-${SRC_COMMIT}
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
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
export TM_MTP_TP_REDUCE=1 TM_MTP_EAGLE_ROTATION=1 TM_MTP_LOCAL_TOP1=1
export TM_MTP_FROZEN_KV=0 TM_MTP_ACCEPTED_REPAIR=1 TM_MTP_SKIP_ATTN=0
export TM_MTP_AMBIGUOUS_REPLAY=1 TM_MTP_FORCE_REJECT=1
for item in 003:0.03125 006:0.0625 012:0.125; do
    label=${item%%:*}
    margin=${item#*:}
    echo "=== forced identity margin=${margin} ==="
    TM_MTP_AMBIGUITY_MARGIN="${margin}" python3 /src/tools/v100/verify_spec_identity.py \
        --model-dir "${MODEL_DIR}" --tp "${TP}" --num-draft-tokens 4 \
        --baseline-replicas 1 --skip-narrowing \
        --json-out "${RESULTS}/identity_${label}.json"
    rc=$?
    if [ "${rc}" = 0 ]; then
        echo "MTP_MARGIN_IDENTITY_PASS margin=${margin}"
        exit 0
    fi
    echo "MTP_MARGIN_IDENTITY_FAIL margin=${margin} rc=${rc}"
done
exit 6
