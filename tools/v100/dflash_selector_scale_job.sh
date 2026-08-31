#!/usr/bin/env bash
# Sweep the relative transition-codebook contribution in the exact DFlash selector.
set -euo pipefail
SRC_COMMIT=$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP)
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-selector-scale-${SRC_COMMIT:-unknown}
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
labels=(s0 s05 s1 s15 s2)
scales=(0 0.5 1 1.5 2)
for i in "${!labels[@]}"; do
    arm=${labels[$i]}
    scale=${scales[$i]}
    TM_DFLASH_SELECTOR_TRANSITION_SCALE=$scale python3 /job/bench_decode.py "${common[@]}" \
        --json-out "$RESULTS/$arm.json" 2>&1 | tee "$RESULTS/$arm.log"
done
BEST=$(python3 - "$RESULTS" <<'PY'
import json,pathlib,re,sys
root=pathlib.Path(sys.argv[1]);pattern=re.compile(r'final commit length ([0-9.]+), raw ([0-9.]+) over ([0-9]+)')
scales={'s0':'0','s05':'0.5','s1':'1','s15':'1.5','s2':'2'};rows=[]
for arm,scale in scales.items():
 data=json.loads((root/f'{arm}.json').read_text());matches=pattern.findall((root/f'{arm}.log').read_text(errors='replace'));assert matches
 commit,raw,steps=matches[-1];rows.append((float(commit),arm,scale,float(raw),int(steps),data['mean_decode_tok_s']))
 print(f'VERDICT {arm} scale={scale} decode={data["mean_decode_tok_s"]:.4f} commit={commit} raw={raw} steps={steps}',file=sys.stderr)
print(max(rows)[2])
PY
)
echo "BEST_SCALE=$BEST"
TM_DFLASH_SELECTOR_TRANSITION_SCALE=$BEST python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 128 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    2>&1 | tee "$RESULTS/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "$RESULTS/identity.log"
touch "$RESULTS/completed"
echo DFLASH_SELECTOR_SCALE_COMPLETE
