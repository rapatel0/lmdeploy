#!/usr/bin/env bash
set -euo pipefail
COMMIT=$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP)
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-qk-norm-warp32-${COMMIT:-unknown}
mkdir -p "$RESULTS"; exec > >(tee "$RESULTS/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"$RESULTS/exit_code"; echo "artifacts $RESULTS exit=$rc"' EXIT
rm -f /wheels/lmdeploy-*.whl
bash /src/tools/v100/build_v100_fast.sh >"$RESULTS/build.log" 2>&1 || { grep -aE 'error:|Error [0-9]+' "$RESULTS/build.log" | head -100; exit 2; }
WHEEL=$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
pip install --no-deps --force-reinstall "$WHEEL" 2>&1 | tail -1
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp 4 --num-draft-tokens 7
 --speculative-algorithm dflash2 --speculative-draft-model /models/Qwen3.8-27B-DFlash2
 --speculative-dflash-block-size 8 --speculative-draft-window 2048 --input-tokens 1000 --output-tokens 256
 --trials 5 --sglang-corpus /sglang-corpus
 --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
 --cache-max-entry-count 0.05)
for arm in legacy warp32; do
 flag=0; [ "$arm" = warp32 ] && flag=1
 echo "=== $arm TM_DFLASH_QK_NORM_WARP32=$flag ==="
 TM_DFLASH_QK_NORM_WARP32="$flag" python3 /job/bench_decode.py "${common[@]}" \
  --json-out "$RESULTS/$arm.json" 2>&1 | tee "$RESULTS/$arm.log"
done
python3 - "$RESULTS" <<'PY' | tee "$RESULTS/analysis.log"
import json,pathlib,re,sys
p=pathlib.Path(sys.argv[1])
for arm in ('legacy','warp32'):
 d=json.loads((p/f'{arm}.json').read_text()); t=(p/f'{arm}.log').read_text(errors='replace')
 m=re.findall(r'\[spec\] final commit length ([0-9.]+), raw ([0-9.]+) over (\d+) verification steps',t)
 if not m: raise SystemExit(f'FAIL no acceptance {arm}')
 print('ARM',arm,'decode',d['mean_decode_tok_s'],'commit',m[-1][0],'raw',m[-1][1],'steps',m[-1][2])
print('DFLASH_QK_NORM_WARP32_ANALYSIS_COMPLETE')
PY
echo '=== audited identity warp32 ==='
TM_DFLASH_QK_NORM_WARP32=1 python3 /job/verify_dflash_audited.py \
 --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --draft-model /models/Qwen3.8-27B-DFlash2 \
 --corpus /sglang-corpus --tp 4 --input-tokens 1000 --output-tokens 256 \
 --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
 2>&1 | tee "$RESULTS/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "$RESULTS/identity.log"
touch "$RESULTS/completed"
echo DFLASH_QK_NORM_WARP32_QUALIFICATION_COMPLETE
