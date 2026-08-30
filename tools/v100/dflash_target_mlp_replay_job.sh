#!/usr/bin/env bash
# Isolate the first material target MLP mismatch with exact SGLang input replay.
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-target-mlp-replay-${SRC_COMMIT:-unknown}
BASE_REF="$(find /results -maxdepth 3 -type d -path '*-dflash-target-trajectory-*/parity/lmdeploy' -print | sort | tail -1)"
SG_REF="$(find /results -maxdepth 4 -type d -path '*-sglang-dflash-parity-*/trace/sglang' -print | sort | tail -1)"
[ -n "${BASE_REF}" ] && [ -n "${SG_REF}" ] || { echo 'FAIL: missing same-input target traces' >&2; exit 4; }
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"${RESULTS}/exit_code"; echo "artifacts in ${RESULTS} (exit ${rc})"' EXIT
cat /src/SOURCE_STAMP
python3 /job/prepare_dflash_target_mlp_replay.py --sglang "${SG_REF}" --output "${RESULTS}/replay"
rm -f /wheels/lmdeploy-*.whl
bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || {
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -100
    exit 2
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ]
sha256sum "${WHEEL}" | tee "${RESULTS}/wheel.sha256"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
export TM_LOG_LEVEL=INFO
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}"
    --num-draft-tokens 7 --speculative-algorithm dflash2
    --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
    --speculative-dflash-block-size 8 --speculative-draft-window 2048
    --input-tokens 1000 --sglang-corpus /sglang-corpus
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
    --cache-max-entry-count 0.05)

TM_DFLASH_TARGET_TRAJECTORY=1 \
TM_DFLASH_TARGET_PARITY_DIR="${RESULTS}/parity" \
TM_DFLASH_TARGET_MLP_REPLAY_DIR="${RESULTS}/replay" \
    python3 /job/bench_decode.py "${common[@]}" --output-tokens 64 --trials 1 \
    --json-out "${RESULTS}/trace.json" 2>&1 | tee "${RESULTS}/trace.log"
grep -q 'target layer-0 MLP input replay active' "${RESULTS}/trace.log"

for arm in baseline replay; do
    if [ "${arm}" = replay ]; then
        replay_env=(env TM_DFLASH_TARGET_MLP_REPLAY_DIR="${RESULTS}/replay")
    else
        replay_env=(env -u TM_DFLASH_TARGET_MLP_REPLAY_DIR)
    fi
    "${replay_env[@]}" python3 /job/bench_decode.py "${common[@]}" --output-tokens 256 --trials 5 \
        --json-out "${RESULTS}/${arm}.json" 2>&1 | tee "${RESULTS}/${arm}.log"
done

python3 /job/analyze_dflash_target_mlp_replay.py \
    --baseline-trace "${BASE_REF}" --replay-trace "${RESULTS}/parity/lmdeploy" \
    --sglang "${SG_REF}" --benchmark-root "${RESULTS}" --output "${RESULTS}/analysis.json" \
    | tee "${RESULTS}/analysis.log"

env -u TM_DFLASH_TARGET_MLP_REPLAY_DIR \
    python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 128 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    2>&1 | tee "${RESULTS}/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "${RESULTS}/identity.log"
touch "${RESULTS}/completed"
echo DFLASH_TARGET_MLP_REPLAY_COMPLETE
