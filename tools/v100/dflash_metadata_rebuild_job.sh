#!/usr/bin/env bash
# Test post-rollback DFlash draft-attention metadata rebuild on TP4 V100.
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-metadata-rebuild-${SRC_COMMIT:-unknown}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"${RESULTS}/exit_code"; echo "artifacts in ${RESULTS} (exit ${rc})"' EXIT
cat /src/SOURCE_STAMP
if [ "${REUSE_WHEEL:-0}" != 1 ]; then
    rm -f /wheels/lmdeploy-*.whl
    bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || {
        grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -80
        exit 2
    }
fi
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ]
sha256sum "${WHEEL}" | tee "${RESULTS}/wheel.sha256"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
export TM_LOG_LEVEL=INFO
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" --num-draft-tokens 7
    --speculative-algorithm dflash2 --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
    --speculative-dflash-block-size 8 --speculative-draft-window 2048 --input-tokens 1000
    --sglang-corpus /sglang-corpus --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
    --cache-max-entry-count 0.05)
for arm in legacy rebuild; do
    if [ "${arm}" = rebuild ]; then rebuild=1; else rebuild=0; fi
    echo "=== ${arm}: rebuild=${rebuild} ==="
    TM_DFLASH_REBUILD_METADATA_AFTER_ROLLBACK="${rebuild}" \
        TM_DFLASH_ASSERT_DRAFT_METADATA=0 \
        TM_DFLASH_METADATA_TRACE=1 \
        python3 /job/bench_decode.py "${common[@]}" --output-tokens 256 --trials 5 --json-out "${RESULTS}/${arm}.json" \
        2>&1 | tee "${RESULTS}/${arm}.log"
done

echo "=== assertion and live cu_k_len smoke ==="
TM_DFLASH_REBUILD_METADATA_AFTER_ROLLBACK=1 \
    TM_DFLASH_ASSERT_DRAFT_METADATA=1 \
    python3 /job/bench_decode.py "${common[@]}" --output-tokens 64 --trials 1 \
    --json-out "${RESULTS}/assert_smoke.json" 2>&1 | tee "${RESULTS}/assert_smoke.log"
python3 /job/analyze_dflash_metadata_rebuild.py "${RESULTS}" | tee "${RESULTS}/analysis.log"
TM_DFLASH_REBUILD_METADATA_AFTER_ROLLBACK=1 \
    TM_DFLASH_ASSERT_DRAFT_METADATA=1 \
    python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 128 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    2>&1 | tee "${RESULTS}/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "${RESULTS}/identity.log"
touch "${RESULTS}/completed"
echo DFLASH_METADATA_REBUILD_COMPLETE
