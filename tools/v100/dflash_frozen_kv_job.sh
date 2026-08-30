#!/usr/bin/env bash
# Qualify SGLang-parity frozen-KV DFlash proposal attention on TP4 V100.
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-frozen-kv-${SRC_COMMIT:-unknown}
mkdir -p "$RESULTS"
exec > >(tee -a "$RESULTS/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"$RESULTS/exit_code"; echo "artifacts in $RESULTS (exit $rc)"' EXIT
cat /src/SOURCE_STAMP
if [ "${REUSE_WHEEL:-0}" != 1 ]; then
    rm -f /wheels/lmdeploy-*.whl
    bash /src/tools/v100/build_v100_fast.sh >"$RESULTS/build.log" 2>&1 || {
        grep -aE 'error:|Error [0-9]+' "$RESULTS/build.log" | head -80
        exit 2
    }
fi
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "$WHEEL" ]
sha256sum "$WHEEL" | tee "$RESULTS/wheel.sha256"
pip install --no-deps --force-reinstall "$WHEEL" 2>&1 | tail -1
export TM_LOG_LEVEL=INFO
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" --num-draft-tokens 7
    --speculative-algorithm dflash2 --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
    --speculative-dflash-block-size 8 --speculative-draft-window 2048 --input-tokens 1000
    --sglang-corpus /sglang-corpus --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
    --cache-max-entry-count 0.05)
for arm in legacy frozen; do
    if [ "$arm" = frozen ]; then frozen=1; else frozen=0; fi
    echo "=== $arm: anchor_inclusive_frozen_kv=$frozen ==="
    TM_DFLASH_ANCHOR_INCLUSIVE_FRONTIER="$frozen" \
        TM_DFLASH_ASSERT_DRAFT_METADATA="$frozen" \
        python3 /job/bench_decode.py "${common[@]}" --output-tokens 256 --trials 5 \
        --json-out "$RESULTS/$arm.json" 2>&1 | tee "$RESULTS/$arm.log"
done
python3 - "$RESULTS" <<'PY' | tee "$RESULTS/analysis.log"
import json, pathlib, re, statistics, sys
root = pathlib.Path(sys.argv[1])
for arm in ('legacy', 'frozen'):
    data = json.loads((root / f'{arm}.json').read_text())
    print(f'{arm.upper()}_JSON', json.dumps(data, sort_keys=True))
    text = (root / f'{arm}.log').read_text(errors='replace')
    matches = re.findall(r'\[spec\] final commit length ([0-9.]+), raw ([0-9.]+) over (\d+) verification steps', text)
    if not matches:
        raise SystemExit(f'FAIL: no DFlash acceptance summary in {arm}.log')
    committed, raw, steps = matches[-1]
    print(f'{arm.upper()}_ACCEPTANCE committed={committed} raw={raw} steps={steps}')
print('DFLASH_FROZEN_KV_ANALYSIS_COMPLETE')
PY
echo '=== exact audited identity ==='
TM_DFLASH_ANCHOR_INCLUSIVE_FRONTIER=1 \
    TM_DFLASH_ASSERT_DRAFT_METADATA=1 \
    python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    2>&1 | tee "$RESULTS/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "$RESULTS/identity.log"
touch "$RESULTS/completed"
echo DFLASH_FROZEN_KV_QUALIFICATION_COMPLETE
