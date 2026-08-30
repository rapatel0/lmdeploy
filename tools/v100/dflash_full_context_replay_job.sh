#!/usr/bin/env bash
# Replay SGLang's complete post-communication prompt context in one build.
set -euo pipefail
SRC_COMMIT=$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP)
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-full-context-replay-${SRC_COMMIT:-unknown}
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

SG_ROOT=""
while IFS= read -r candidate; do
    if grep -q 'target.full_context' "$candidate"/rank-*/manifest.jsonl 2>/dev/null; then
        SG_ROOT=$candidate
        break
    fi
done < <(find /results -maxdepth 4 -type d -path '*-sglang-dflash-parity-*/trace/sglang' | sort -r)
[ -n "$SG_ROOT" ] || {
    echo 'FAIL: no full SGLang context trace' >&2
    exit 4
}
FULL_CONTEXT=$(
    python3 - "$SG_ROOT" <<'PY'
import hashlib
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
directories = sorted(path for path in root.glob("rank-*-pid-*") if path.is_dir())
assert len(directories) == 4, directories
paths = []
hashes = []
for directory in directories:
    records = {record["name"]: record for record in map(json.loads, (directory / "manifest.jsonl").read_text().splitlines())}
    record = records["target.full_context"]
    assert record["dtype"] == "f16" and record["shape"] == [1000, 25600], record
    path = directory / record["file"]
    assert path.stat().st_size == 1000 * 25600 * 2, path
    paths.append(path)
    hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
assert len(set(hashes)) == 1, hashes
print(paths[0])
PY
)
echo "sglang=$SG_ROOT full_context=$FULL_CONTEXT"

common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}"
    --num-draft-tokens 7 --speculative-algorithm dflash2
    --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
    --speculative-dflash-block-size 8 --speculative-draft-window 2048
    --input-tokens 1000 --sglang-corpus /sglang-corpus
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
    --cache-max-entry-count 0.05)

run_perf() {
    local arm=$1 replay=$2 reduce=$3
    TM_DFLASH_CONTEXT_REPLAY_FILE="$replay" TM_DFLASH_REDUCE_BEFORE_CONV="$reduce" \
        python3 /job/bench_decode.py "${common[@]}" --output-tokens 1536 --trials 5 \
        --json-out "$RESULTS/$arm.json" 2>&1 | tee "$RESULTS/$arm.log"
}
run_perf control "" 0
run_perf full_context "$FULL_CONTEXT" 0
run_perf full_context_reduce_first "$FULL_CONTEXT" 1
[ "$(grep -c 'replaying parity target context rows=1000' "$RESULTS/full_context.log")" -eq 4 ]
[ "$(grep -c 'replaying parity target context rows=1000' "$RESULTS/full_context_reduce_first.log")" -eq 4 ]

mkdir -p "$RESULTS/parity"
TM_DFLASH_CONTEXT_REPLAY_FILE="$FULL_CONTEXT" TM_DFLASH_REDUCE_BEFORE_CONV=1 \
    TM_DFLASH_PARITY_DIR="$RESULTS/parity" \
    python3 /job/bench_decode.py "${common[@]}" --output-tokens 64 --trials 1 \
    --json-out "$RESULTS/parity.json" 2>&1 | tee "$RESULTS/parity.log"
python3 /job/compare_dflash_parity.py \
    --lmdeploy "$RESULTS/parity/lmdeploy" --sglang "$SG_ROOT" \
    --output "$RESULTS/compare.json" >"$RESULTS/compare.log"
[ "$(grep -c 'replaying parity target context rows=1000' "$RESULTS/parity.log")" -eq 4 ]
[ "$(grep -c 'replaying parity target context rows=1' "$RESULTS/parity.log")" -eq 4 ]

python3 - "$RESULTS" <<'PY'
import json
import pathlib
import re
import sys
root = pathlib.Path(sys.argv[1])
pattern = re.compile(r"final commit length ([0-9.]+), raw ([0-9.]+) over ([0-9]+)")
for arm in ("control", "full_context", "full_context_reduce_first"):
    data = json.loads((root / f"{arm}.json").read_text())
    matches = list(pattern.finditer((root / f"{arm}.log").read_text()))
    assert matches, arm
    match = matches[-1]
    print(
        "PERF",
        arm,
        "decode",
        data["mean_decode_tok_s"],
        "commit",
        match.group(1),
        "raw",
        match.group(2),
        "steps",
        match.group(3),
    )
report = json.loads((root / "compare.json").read_text())
rows = {row["lmdeploy"]: row for row in report["comparisons"]}
for name in (
    "context.fc",
    "context.norm",
    "block.initial_norm",
    "layer0.attention.conv_side0",
    "layer0.attention.conv_side1",
    "layer0.attention.norm_output",
    "layer0.mlp.conv_side1",
    "layer0.mlp.norm_output",
):
    row = rows[name]
    print("PARITY", name, "status", row["status"], "max", row.get("max_abs"), "rms", row.get("rms"))
PY

touch "$RESULTS/completed"
echo DFLASH_FULL_CONTEXT_REPLAY_COMPLETE
