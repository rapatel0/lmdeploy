#!/usr/bin/env bash
set -euo pipefail

SRC_COMMIT=$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP)
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-tilelang-graph-identity-${SRC_COMMIT:-unknown}
mkdir -p "$RESULTS"
exec > >(tee -a "$RESULTS/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"$RESULTS/exit_code"; [ -f "$RESULTS/completed" ] || echo KILLED >"$RESULTS/incomplete"; echo "artifacts in $RESULTS (exit $rc)"' EXIT

cat /src/SOURCE_STAMP
rm -f /wheels/lmdeploy-*.whl
bash /src/tools/v100/build_v100_fast.sh >"$RESULTS/build.log" 2>&1 || {
    grep -aE 'error:|Error [0-9]+' "$RESULTS/build.log" | head -100
    exit 2
}
WHEEL=$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$WHEEL" ]
sha256sum "$WHEEL" | tee "$RESULTS/wheel.sha256"
pip install --no-deps --force-reinstall "$WHEEL" 2>&1 | tail -1

export TM_LOG_LEVEL=INFO
export TM_DFLASH_TILELANG_DRAFT_ATTENTION=1
export TM_DFLASH_DRAFT_GRAPH=1
export TM_DFLASH_GRAPH_TRACE=1
export TM_DFLASH_PERSISTENT_WORKSPACE=1
export TM_DFLASH_LOCAL_TOPK=1
export TM_DFLASH_PAGED_Q8=0
export TM_MTP_FROZEN_KV=1
export TM_DFLASH_ANCHOR_INCLUSIVE_FRONTIER=1

common=(
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
    --tp "${TP:-4}"
    --num-draft-tokens 7
    --speculative-algorithm dflash2
    --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
    --speculative-dflash-block-size 8
    --speculative-draft-window 2048
    --input-tokens 1000
    --output-tokens 256
    --sglang-corpus /sglang-corpus
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
    --cache-max-entry-count 0.05
)

python3 /job/bench_decode.py "${common[@]}" --trials 1 --json-out "$RESULTS/graph.json" 2>&1 |
    tee "$RESULTS/graph.log"

captures=$(grep -c '\[DFlash2\] draft graph captured phase=' "$RESULTS/graph.log" || true)
replays=$(grep -c '\[DFlash2\] draft graph replay phase=' "$RESULTS/graph.log" || true)
selected=$(grep -c 'DFLASH_TILELANG_SELECTOR selected=true' "$RESULTS/graph.log" || true)
disabled=$(grep -c 'draft graph capture disabled' "$RESULTS/graph.log" || true)
[ "$captures" -ge 4 ] || {
    echo "FAIL: draft graph captures=$captures" >&2
    exit 3
}
[ "$replays" -ge 4 ] || {
    echo "FAIL: draft graph replays=$replays" >&2
    exit 3
}
[ "$selected" -ge 1 ] || {
    echo "FAIL: TileLang selector count=$selected" >&2
    exit 3
}
[ "$disabled" -eq 0 ] || {
    echo "FAIL: graph capture disabled on $disabled ranks/phases" >&2
    exit 3
}
echo "DFLASH_TILELANG_GRAPH_CAPTURE_REPLAY_PASS captures=$captures replays=$replays ranks=4"

python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 2>&1 |
    tee "$RESULTS/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "$RESULTS/identity.log"
[ "$(grep -c 'DFLASH_TILELANG_SELECTOR selected=true' "$RESULTS/identity.log" || true)" -ge 1 ]
echo DFLASH_TILELANG_K0_K7_IDENTITY_PASS

touch "$RESULTS/completed"
echo DFLASH_TILELANG_GRAPH_IDENTITY_COMPLETE
