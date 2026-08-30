#!/usr/bin/env bash
# No-rebuild frozen-KV follow-up: exact context replay and identity.
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-frozen-kv-followup-${SRC_COMMIT:-unknown}
mkdir -p "$RESULTS"
exec > >(tee -a "$RESULTS/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"$RESULTS/exit_code"; echo "artifacts in $RESULTS (exit $rc)"' EXIT
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "$WHEEL" ]
pip install --no-deps --force-reinstall "$WHEEL" 2>&1 | tail -1
SG=/results/20260830_223929-sglang-dflash-parity-1a86d3c5a8b6/trace/sglang
mapfile -t REPLAY < <(
   python3 - "$SG" <<'PY'
import glob,json,pathlib,sys
root=sorted(glob.glob(sys.argv[1]+'/rank-*-pid-*'))[0]
rows={r['name']:r for r in map(json.loads,open(root+'/manifest.jsonl'))}
for name in ('target.full_context','context.full_norm'):
 print(pathlib.Path(root,rows[name]['file']))
PY
)
FULL_CONTEXT=${REPLAY[0]}
FULL_NORM=${REPLAY[1]}
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" --num-draft-tokens 7
   --speculative-algorithm dflash2 --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
   --speculative-dflash-block-size 8 --speculative-draft-window 2048 --input-tokens 1000
   --sglang-corpus /sglang-corpus --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
   --cache-max-entry-count 0.05)
echo '=== exact SGLang context replay acceptance ==='
TM_DFLASH_ANCHOR_INCLUSIVE_FRONTIER=1 \
   TM_DFLASH_CONTEXT_REPLAY_FILE="$FULL_CONTEXT" \
   TM_DFLASH_CONTEXT_NORM_REPLAY_FILE="$FULL_NORM" \
   python3 /job/bench_decode.py "${common[@]}" --output-tokens 256 --trials 1 \
   --json-out "$RESULTS/replay.json" 2>&1 | tee "$RESULTS/replay.log"
grep -a '\[spec\] final commit length' "$RESULTS/replay.log" | tail -1 | tee "$RESULTS/replay_acceptance.txt"
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
echo DFLASH_FROZEN_KV_FOLLOWUP_COMPLETE
