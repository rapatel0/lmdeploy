#!/usr/bin/env bash
# Counter-ordered fresh-process confirmation plus target-in-top16 attribution.
set -euo pipefail
SRC_COMMIT=$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP)
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-selector-scale-confirm-${SRC_COMMIT:-unknown}
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
    --input-tokens 1000 --output-tokens 1536 --trials 1 --sglang-corpus /sglang-corpus
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
    --cache-max-entry-count 0.05)
orders=("1 2 2.5" "2.5 2 1" "2 1 2.5" "1 2.5 2")
round=0
for order in "${orders[@]}"; do
    round=$((round + 1))
    for scale in $order; do
        label=${scale/./p}
        arm=r${round}_s${label}
        TM_DFLASH_SELECTOR_TRANSITION_SCALE=$scale python3 /job/bench_decode.py "${common[@]}" \
            --json-out "$RESULTS/$arm.json" 2>&1 | tee "$RESULTS/$arm.log"
    done
done
BEST=$(python3 - "$RESULTS" <<'PY'
import json,pathlib,re,statistics,sys
root=pathlib.Path(sys.argv[1]);pat=re.compile(r'final commit length ([0-9.]+), raw ([0-9.]+) over ([0-9]+)');groups={}
for p in sorted(root.glob('r*_s*.log')):
 scale=p.stem.split('_s')[1].replace('p','.');m=pat.findall(p.read_text(errors='replace'));assert m,p
 data=json.loads(p.with_suffix('.json').read_text());groups.setdefault(scale,[]).append((float(m[-1][0]),float(m[-1][1]),data['mean_decode_tok_s']))
for scale,rows in sorted(groups.items(),key=lambda x:float(x[0])):
 print(f'CONFIRM scale={scale} commit_mean={statistics.mean(x[0] for x in rows):.4f} commit_values={[x[0] for x in rows]} decode_mean={statistics.mean(x[2] for x in rows):.4f}',file=sys.stderr)
print(max(groups,key=lambda s:statistics.mean(x[0] for x in groups[s])))
PY
)
echo "BEST_SCALE=$BEST"
for scale in 1 "$BEST"; do
    label=${scale/./p}
    TM_DFLASH_SELECTOR_TRANSITION_SCALE=$scale TM_DFLASH_TARGET_CANDIDATE_RANK=1 \
        python3 /job/bench_decode.py "${common[@]}" --json-out "$RESULTS/rank_s${label}.json" \
        2>&1 | tee "$RESULTS/rank_s${label}.log"
    grep -q 'DFLASH_TARGET_CANDIDATE_RANK' "$RESULTS/rank_s${label}.log"
done
TM_DFLASH_SELECTOR_TRANSITION_SCALE=$BEST python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 128 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 2>&1 | tee "$RESULTS/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "$RESULTS/identity.log"
touch "$RESULTS/completed"
echo DFLASH_SELECTOR_SCALE_CONFIRM_COMPLETE
