#!/usr/bin/env bash
# Capture one production-order TurboMind DFlash block and compare all five
# attention layers with the audited SGLang verifier artifact.
set -euo pipefail
SRC_COMMIT=$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP)
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-native-attention-${SRC_COMMIT:-unknown}
mkdir -p "$RESULTS"
exec > >(tee -a "$RESULTS/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"$RESULTS/exit_code"; [ -f "$RESULTS/completed" ] || echo KILLED >"$RESULTS/incomplete"; echo "artifacts in $RESULTS (exit $rc)"' EXIT

rm -f /wheels/lmdeploy-*.whl
bash /src/tools/v100/build_v100_fast.sh >"$RESULTS/build.log" 2>&1 || {
    grep -aE 'error:|Error [0-9]+' "$RESULTS/build.log" | head -100
    exit 2
}
WHEEL=$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$WHEEL" ]
pip install --no-deps --force-reinstall "$WHEEL" 2>&1 | tail -1

SG=${SGLANG_DFLASH_TRACE_ROOT:-/results/20260901_082555-sglang-dflash-parity-48f21dc99772/trace/sglang}
python3 /job/validate_sglang_dflash_trace.py "$SG" --block-index 1
mapfile -t REPLAY < <(
    python3 - "$SG" <<'PY'
import glob, hashlib, json, os, sys
roots = sorted(glob.glob(sys.argv[1] + '/rank-*-pid-*'))
assert len(roots) == 4, roots
for name, shape, size in (
    ('target.full_context', [1000, 25600], 51200000),
    ('context.full_norm', [1000, 5120], 10240000),
    ('layer0.attention.norm_output', [8, 5120], 81920),
):
    paths, hashes = [], []
    for root in roots:
        records = {row['name']: row for row in map(json.loads, open(root + '/manifest.jsonl'))}
        row = records[name]
        assert row['shape'] == shape and row['dtype'] == 'f16', row
        path = root + '/' + row['file']
        assert os.path.getsize(path) == size, path
        paths.append(path)
        hashes.append(hashlib.sha256(open(path, 'rb').read()).hexdigest())
    assert len(set(hashes)) == 1, (name, hashes)
    print(paths[0])
PY
)
[ "${#REPLAY[@]}" -eq 3 ]
FULL_CONTEXT=${REPLAY[0]}
FULL_NORM=${REPLAY[1]}
INITIAL_NORM=${REPLAY[2]}
mkdir -p "$RESULTS/parity"
RUN_ENV=(env
    TM_DFLASH_CONTEXT_REPLAY_FILE="$FULL_CONTEXT"
    TM_DFLASH_CONTEXT_NORM_REPLAY_FILE="$FULL_NORM"
    TM_DFLASH_BLOCK_INITIAL_NORM_REPLAY_FILE="$INITIAL_NORM"
    TM_DFLASH_ANCHOR_INCLUSIVE_FRONTIER=1
    TM_DFLASH_ASSERT_DRAFT_METADATA=1
    TM_DFLASH_REDUCE_BEFORE_CONV=0
    TM_DFLASH_TILELANG_DRAFT_ATTENTION=1
    TM_DFLASH_SELECTOR_LOGIT_SCALE=1
    TM_DFLASH_PARITY_DIR="$RESULTS/parity")
if [ "${TM_DFLASH_NATIVE_REPLAY_FLATTENED:-0}" = 1 ]; then
    FLATTENED_KV_REPLAY="$RESULTS/flattened_kv_tp4.bin"
    python3 - "$SG" "$FLATTENED_KV_REPLAY" <<'PY'
import glob, json, numpy as np, pathlib, sys
roots = sorted(glob.glob(sys.argv[1] + '/rank-*-pid-*'))
assert len(roots) == 4, roots
with open(sys.argv[2], 'wb') as output:
    for root in roots:
        records = {row['name']: row for row in map(json.loads, open(root + '/manifest.jsonl'))}
        cache_k = np.fromfile(pathlib.Path(root, records['layer0.attention.tilelang.k']['file']), dtype='<f2').reshape(-1, 2, 128)
        cache_v = np.fromfile(pathlib.Path(root, records['layer0.attention.tilelang.v']['file']), dtype='<f2').reshape(-1, 2, 128)
        output.write(np.stack((cache_k.transpose(1, 0, 2), cache_v.transpose(1, 0, 2)), axis=1).tobytes())
PY
    RUN_ENV+=(TM_DFLASH_DRAFT_FLATTENED_KV_REPLAY_FILE="$FLATTENED_KV_REPLAY"
        TM_DFLASH_DRAFT_FLATTENED_KV_REPLAY_CONTEXT_LEN=1008)
fi

"${RUN_ENV[@]}" python3 /job/bench_decode.py \
    --model /models/Qwen3.8-27B-FP8 --tp 4 --num-draft-tokens 7 \
    --speculative-algorithm dflash2 --speculative-draft-model /models/Qwen3.8-27B-DFlash2 \
    --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
    --input-tokens 1000 --output-tokens 64 --trials 1 --sglang-corpus /sglang-corpus \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    --cache-max-entry-count 0.05 --json-out "$RESULTS/parity.json" 2>&1 | tee "$RESULTS/parity.log"

[ "$(grep -c 'replaying parity target context rows=1000 ' "$RESULTS/parity.log")" -eq 4 ]
[ "$(grep -c 'replaying normalized parity context rows=1000 ' "$RESULTS/parity.log")" -eq 4 ]
[ "$(grep -c 'TM_DFLASH_BLOCK_INITIAL_NORM_REPLAY_FILE' "$RESULTS/parity.log")" -eq 4 ]
[ "$(grep -c 'DFLASH_TILELANG_SELECTOR selected=true' "$RESULTS/parity.log")" -ge 1 ]
[ "$(grep -c 'DFLASH_METADATA_REBUILD_ACTIVE' "$RESULTS/parity.log")" -ge 4 ]
python3 - "$RESULTS/parity/lmdeploy" <<'PY'
import glob, json, sys
roots = sorted(glob.glob(sys.argv[1] + '/rank-*-pid-*'))
assert len(roots) == 4, roots
for root in roots:
    names = {row['name'] for row in map(json.loads, open(root + '/manifest.jsonl'))}
    required = {
        f'layer{layer}.attention.{boundary}'
        for layer in range(5)
        for boundary in ('qkv_projection', 'qkv_post_process', 'flattened_kv', 'core_output')
    }
    assert required <= names, (root, sorted(required - names))
print('DFLASH_NATIVE_ALL_LAYER_TRACE_PASS')
PY

python3 /job/compare_dflash_attention_layers.py \
    --lmdeploy "$RESULTS/parity/lmdeploy" \
    --sglang "$SG" \
    --output "$RESULTS/all_layer_attention.json" | tee "$RESULTS/all_layer_attention.log"
if [ -f /job/compare_dflash_parity.py ]; then
    python3 /job/compare_dflash_parity.py \
        --lmdeploy "$RESULTS/parity/lmdeploy" \
        --sglang "$SG" \
        --output "$RESULTS/all_layer_model_parity.json" | tee "$RESULTS/all_layer_model_parity.log"
fi

touch "$RESULTS/completed"
echo DFLASH_NATIVE_ATTENTION_TRACE_COMPLETE
