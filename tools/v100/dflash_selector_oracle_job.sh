#!/usr/bin/env bash
# Measure which per-slot transition scale would select each observed verifier target.
set -euo pipefail
SRC_COMMIT=$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP)
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-selector-oracle-${SRC_COMMIT:-unknown}
mkdir -p "$RESULTS"
exec > >(tee -a "$RESULTS/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"$RESULTS/exit_code"; echo "artifacts in $RESULTS (exit $rc)"' EXIT
rm -f /wheels/lmdeploy-*.whl
bash /src/tools/v100/build_v100_fast.sh >"$RESULTS/build.log" 2>&1 || {
    grep -aE 'error:|Error [0-9]+' "$RESULTS/build.log" | head -80
    exit 2
}
WHEEL=$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
pip install --no-deps --force-reinstall "$WHEEL" 2>&1 | tail -1
TM_DFLASH_SELECTOR_TRANSITION_SCALE=1 TM_DFLASH_TARGET_CANDIDATE_RANK=1 \
    python3 /job/bench_decode.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
    --num-draft-tokens 7 --speculative-algorithm dflash2 \
    --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
    --input-tokens 1000 --output-tokens 1536 --trials 1 --sglang-corpus /sglang-corpus \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    --cache-max-entry-count 0.05 --json-out "$RESULTS/bench.json" 2>&1 | tee "$RESULTS/bench.log"
[ "$(grep -c 'DFLASH_SELECTOR_SCALE_ORACLE slot=' "$RESULTS/bench.log")" -eq 28 ]
grep 'DFLASH_TARGET_CANDIDATE_RANK\|DFLASH_SELECTOR_SCALE_ORACLE' "$RESULTS/bench.log" | tail -32
touch "$RESULTS/completed"
echo DFLASH_SELECTOR_ORACLE_COMPLETE
