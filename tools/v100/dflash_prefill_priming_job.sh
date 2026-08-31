#!/usr/bin/env bash
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-prefill-priming-${SRC_COMMIT:-unknown}
mkdir -p "$RESULTS"
exec > >(tee -a "$RESULTS/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"$RESULTS/exit_code"; echo "artifacts in $RESULTS (exit $rc)"' EXIT
rm -f /wheels/lmdeploy-*.whl
bash /src/tools/v100/build_v100_fast.sh >"$RESULTS/build.log" 2>&1 || {
 grep -aE 'error:|Error [0-9]+' "$RESULTS/build.log" | head -80
 exit 2
}
W="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "$W" | tail -1
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" --num-draft-tokens 7 --speculative-algorithm dflash2 --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" --speculative-dflash-block-size 8 --speculative-draft-window 2048 --input-tokens 1000 --output-tokens 256 --trials 5 --sglang-corpus /sglang-corpus --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 --cache-max-entry-count .05)
for mode in control priming; do
 flag=0
 [ "$mode" = priming ] && flag=1
 TM_DFLASH_PREFILL_PRIMING=$flag TM_SPEC_TRACE=$flag python3 /job/bench_decode.py "${common[@]}" --json-out "$RESULTS/$mode.json" 2>&1 | tee "$RESULTS/$mode.log"
done
grep -q '\[trace\] fwd uid=.*in_len=8 drafts=7 in=\[1596,' "$RESULTS/priming.log"
TM_DFLASH_PREFILL_PRIMING=1 python3 /job/verify_dflash_audited.py --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 128 --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 2>&1 | tee "$RESULTS/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "$RESULTS/identity.log"
touch "$RESULTS/completed"
echo DFLASH_PREFILL_PRIMING_COMPLETE
