#!/usr/bin/env bash
# Qualify SGLang's FP32 target residual contract, separately and with frozen proposal KV.
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-target-fp32-${SRC_COMMIT:-unknown}
mkdir -p "$RESULTS"
exec > >(tee -a "$RESULTS/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"$RESULTS/exit_code"; echo "artifacts in $RESULTS (exit $rc)"' EXIT
cat /src/SOURCE_STAMP
rm -f /wheels/lmdeploy-*.whl
bash /src/tools/v100/build_v100_fast.sh >"$RESULTS/build.log" 2>&1 || {
 grep -aE 'error:|Error [0-9]+' "$RESULTS/build.log" | head -100
 exit 2
}
WHEEL=$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
pip install --no-deps --force-reinstall "$WHEEL" 2>&1 | tail -1
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" --num-draft-tokens 7
 --speculative-algorithm dflash2 --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
 --speculative-dflash-block-size 8 --speculative-draft-window 2048 --input-tokens 1000 --output-tokens 256
 --trials 5 --sglang-corpus /sglang-corpus
 --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
 --cache-max-entry-count 0.05)
for arm in fp16_legacy fp32_legacy fp32_frozen; do
 fp32=0
 frozen=0
 [ "$arm" = fp16_legacy ] || fp32=1
 [ "$arm" = fp32_frozen ] && frozen=1
 echo "=== $arm fp32=$fp32 frozen=$frozen ==="
 TM_DFLASH_TARGET_FP32_RESIDUAL="$fp32" \
  TM_DFLASH_ANCHOR_INCLUSIVE_FRONTIER="$frozen" \
  TM_DFLASH_ASSERT_DRAFT_METADATA="$frozen" \
  python3 /job/bench_decode.py "${common[@]}" --json-out "$RESULTS/$arm.json" 2>&1 | tee "$RESULTS/$arm.log"
done
python3 - "$RESULTS" <<'PY' | tee "$RESULTS/analysis.log"
import json,pathlib,re,sys
root=pathlib.Path(sys.argv[1])
for arm in ('fp16_legacy','fp32_legacy','fp32_frozen'):
 d=json.loads((root/f'{arm}.json').read_text());text=(root/f'{arm}.log').read_text(errors='replace')
 m=re.findall(r'\[spec\] final commit length ([0-9.]+), raw ([0-9.]+) over (\d+) verification steps',text)
 if not m:raise SystemExit(f'FAIL: no acceptance summary for {arm}')
 commit,raw,steps=m[-1]
 print('ARM',arm,'decode',d['mean_decode_tok_s'],'commit',commit,'raw',raw,'steps',steps)
 if arm!='fp16_legacy' and text.count('DFLASH_TARGET_FP32_RESIDUAL_ACTIVE')<4:
  raise SystemExit(f'FAIL: FP32 activation proof missing for {arm}')
print('DFLASH_TARGET_FP32_ANALYSIS_COMPLETE')
PY
echo '=== exact audited identity: fp32 frozen ==='
TM_DFLASH_TARGET_FP32_RESIDUAL=1 TM_DFLASH_ANCHOR_INCLUSIVE_FRONTIER=1 TM_DFLASH_ASSERT_DRAFT_METADATA=1 \
 python3 /job/verify_dflash_audited.py \
 --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
 --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
 --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
 --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
 2>&1 | tee "$RESULTS/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "$RESULTS/identity.log"
touch "$RESULTS/completed"
echo DFLASH_TARGET_FP32_QUALIFICATION_COMPLETE
