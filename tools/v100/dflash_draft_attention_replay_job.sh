#!/usr/bin/env bash
# One-build exact-boundary replay for the first DFlash draft attention layer.
set -euo pipefail
SRC_COMMIT=$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP)
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-draft-attention-replay-${SRC_COMMIT:-unknown}
mkdir -p "$RESULTS"
exec > >(tee -a "$RESULTS/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"$RESULTS/exit_code"; echo "artifacts in $RESULTS (exit $rc)"' EXIT

rm -f /wheels/lmdeploy-*.whl
bash /src/tools/v100/build_v100_fast.sh >"$RESULTS/build.log" 2>&1 || {
    grep -aE 'error:|Error [0-9]+' "$RESULTS/build.log" | head -80
    exit 2
}
WHEEL=$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$WHEEL" ]
pip install --no-deps --force-reinstall "$WHEEL" 2>&1 | tail -1

SG_ROOT=$(find /results -maxdepth 4 -type d -path '*-sglang-dflash-parity-*/trace/sglang' | sort | tail -1)
[ -n "$SG_ROOT" ]
mapfile -t REPLAY_FILES < <(
    python3 - "$SG_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
directories = sorted(path for path in root.glob("rank-*-pid-*") if path.is_dir())
assert len(directories) == 4, directories
names = (
    "target.full_context",
    "layer0.attention.conv_side0",
    "layer0.attention.wo_reduced",
)
paths = []
for name in names:
    payloads = []
    rank_paths = []
    for directory in directories:
        records = {record["name"]: record for record in map(json.loads, (directory / "manifest.jsonl").read_text().splitlines())}
        record = records[name]
        path = directory / record["file"]
        payloads.append(hashlib.sha256(path.read_bytes()).hexdigest())
        rank_paths.append(path)
    assert len(set(payloads)) == 1, (name, payloads)
    paths.append(rank_paths[0])
print(*paths, sep="\n")
PY
)
[ "${#REPLAY_FILES[@]}" -eq 3 ]
CONTEXT=${REPLAY_FILES[0]}
ATTN_INPUT=${REPLAY_FILES[1]}
ATTN_OUTPUT=${REPLAY_FILES[2]}
printf 'context=%s\nattention_input=%s\nattention_output=%s\n' "$CONTEXT" "$ATTN_INPUT" "$ATTN_OUTPUT"

common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}"
    --num-draft-tokens 7 --speculative-algorithm dflash2
    --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
    --speculative-dflash-block-size 8 --speculative-draft-window 2048
    --input-tokens 1000 --output-tokens 64 --trials 1
    --sglang-corpus /sglang-corpus
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
    --cache-max-entry-count 0.05)

run_arm() {
    local arm=$1 input_replay=$2 output_replay=$3
    mkdir -p "$RESULTS/$arm"
    TM_DFLASH_REDUCE_BEFORE_CONV=1 \
        TM_DFLASH_CONTEXT_REPLAY_FILE="$CONTEXT" \
        TM_DFLASH_DRAFT_ATTENTION_INPUT_REPLAY_FILE="$input_replay" \
        TM_DFLASH_DRAFT_ATTENTION_OUTPUT_REPLAY_FILE="$output_replay" \
        TM_DFLASH_PARITY_DIR="$RESULTS/$arm" \
        python3 /job/bench_decode.py "${common[@]}" --json-out "$RESULTS/$arm.json" \
        2>&1 | tee "$RESULTS/$arm.log"
    python3 /job/compare_dflash_parity.py \
        --lmdeploy "$RESULTS/$arm/lmdeploy" --sglang "$SG_ROOT" \
        --output "$RESULTS/$arm-compare.json" >"$RESULTS/$arm-compare.log"
}

run_arm control "" ""
run_arm attention_input "$ATTN_INPUT" ""
run_arm attention_output "" "$ATTN_OUTPUT"
[ "$(grep -c 'TM_DFLASH_DRAFT_ATTENTION_INPUT_REPLAY_FILE' "$RESULTS/attention_input.log")" -eq 4 ]
[ "$(grep -c 'TM_DFLASH_DRAFT_ATTENTION_OUTPUT_REPLAY_FILE' "$RESULTS/attention_output.log")" -eq 4 ]

python3 - "$RESULTS" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
names = (
    "context.norm",
    "block.initial_norm",
    "layer0.attention.conv_side0",
    "layer0.attention.conv_side1",
    "layer0.attention.norm_output",
    "layer0.mlp.conv_side1",
    "layer0.mlp.norm_output",
)
for arm in ("control", "attention_input", "attention_output"):
    report = json.loads((root / f"{arm}-compare.json").read_text())
    rows = {row["lmdeploy"]: row for row in report["comparisons"]}
    print(f"ARM {arm}")
    for name in names:
        row = rows[name]
        print(name, "status", row["status"], "max", row.get("max_abs"), "rms", row.get("rms"))
PY

touch "$RESULTS/completed"
echo DFLASH_DRAFT_ATTENTION_REPLAY_COMPLETE
