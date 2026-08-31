#!/usr/bin/env bash
# Test per-slot transition scales derived from target-feedback attribution.
set -euo pipefail
SRC_COMMIT=$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP)
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-selector-slot-scale-${SRC_COMMIT:-unknown}
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
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}"
    --num-draft-tokens 7 --speculative-algorithm dflash2
    --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
    --speculative-dflash-block-size 8 --speculative-draft-window 2048
    --input-tokens 1000 --output-tokens 1536 --trials 5 --sglang-corpus /sglang-corpus
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
    --cache-max-entry-count 0.05)
TM_DFLASH_SELECTOR_TRANSITION_SCALE=1 TM_DFLASH_SELECTOR_SLOT_SCALES=0 \
    python3 /job/bench_decode.py "${common[@]}" --json-out "$RESULTS/control.json" 2>&1 | tee "$RESULTS/control.log"
TM_DFLASH_SELECTOR_TRANSITION_SCALE=2 TM_DFLASH_SELECTOR_SLOT_SCALES=0 \
    python3 /job/bench_decode.py "${common[@]}" --json-out "$RESULTS/global2.json" 2>&1 | tee "$RESULTS/global2.log"
TM_DFLASH_SELECTOR_TRANSITION_SCALE=1 TM_DFLASH_SELECTOR_SLOT_SCALES=1 \
    python3 /job/bench_decode.py "${common[@]}" --json-out "$RESULTS/slot.json" 2>&1 | tee "$RESULTS/slot.log"
python3 - "$RESULTS" <<'PY'
import json,pathlib,re,sys
root=pathlib.Path(sys.argv[1]);pat=re.compile(r'final commit length ([0-9.]+), raw ([0-9.]+) over ([0-9]+)')
for arm in ('control','global2','slot'):
 data=json.loads((root/f'{arm}.json').read_text());m=pat.findall((root/f'{arm}.log').read_text(errors='replace'));assert m
 print(f'VERDICT {arm} decode={data["mean_decode_tok_s"]:.4f} commit={m[-1][0]} raw={m[-1][1]} steps={m[-1][2]}')
PY
TM_DFLASH_SELECTOR_TRANSITION_SCALE=1 TM_DFLASH_SELECTOR_SLOT_SCALES=1 \
    python3 /job/verify_dflash_audited.py --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" --corpus /sglang-corpus \
    --tp "${TP:-4}" --input-tokens 1000 --output-tokens 128 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 2>&1 | tee "$RESULTS/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "$RESULTS/identity.log"
touch "$RESULTS/completed"
echo DFLASH_SELECTOR_SLOT_SCALE_COMPLETE
