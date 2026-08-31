#!/usr/bin/env bash
set -euo pipefail
SRC_COMMIT=$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP)
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-context-kv-parity-${SRC_COMMIT:-unknown}
mkdir -p "$RESULTS"
exec > >(tee -a "$RESULTS/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"$RESULTS/exit_code"; echo "artifacts in $RESULTS (exit $rc)"' EXIT

rm -f /wheels/lmdeploy-*.whl
bash /src/tools/v100/build_v100_fast.sh >"$RESULTS/build.log" 2>&1 || {
    grep -aE 'error:|Error [0-9]+' "$RESULTS/build.log" | head -100
    exit 2
}
WHEEL=$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
[ -n "$WHEEL" ]
pip install --no-deps --force-reinstall "$WHEEL" 2>&1 | tail -1

SG=${SGLANG_DFLASH_REPLAY_ROOT:-/results/20260830_223929-sglang-dflash-parity-1a86d3c5a8b6/trace/sglang}
mapfile -t REPLAY < <(
    python3 - "$SG" <<'PY'
import glob, hashlib, json, os, sys
roots = sorted(glob.glob(sys.argv[1] + '/rank-*-pid-*'))
assert len(roots) == 4, roots
for name, shape, size in (
    ('target.full_context', [1000, 25600], 51200000),
    ('context.full_norm', [1000, 5120], 10240000),
    ('layer0.attention.norm_output', [8, 5120], 81920),
    ('layer0.attention.conv_side0', [8, 5120], 81920),
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
[ "${#REPLAY[@]}" -eq 4 ]
FULL_CONTEXT=${REPLAY[0]}
FULL_NORM=${REPLAY[1]}
INITIAL_NORM=${REPLAY[2]}
ATTENTION_INPUT=${REPLAY[3]}
REPLAY_EXACT_QK=${REPLAY_EXACT_QK:-1}
REPLAY_PROJECTED_QK=${REPLAY_PROJECTED_QK:-0}
REPLAY_ATTENTION_INPUT=${REPLAY_ATTENTION_INPUT:-0}
Q_REPLAY=$RESULTS/q_normalized_tp4.bin
K_REPLAY=$RESULTS/k_normalized_tp4.bin
FLATTENED_KV_REPLAY=$RESULTS/flattened_kv_tp4.bin
Q_PROJECTION_REPLAY=$RESULTS/q_projection_tp4.bin
K_PROJECTION_REPLAY=$RESULTS/k_projection_tp4.bin
read -r LIVE_TILELANG REPLAY_CONTEXT_LEN < <(python3 - "$SG" <<'PY'
import glob, json, sys
roots = sorted(glob.glob(sys.argv[1] + '/rank-*-pid-*'))
rows = {row['name']: row for row in map(json.loads, open(roots[0] + '/manifest.jsonl'))}
record = rows.get('layer0.attention.tilelang.k')
print(1 if record else 0, record['shape'][0] if record else 1000)
PY
)
if [ "$REPLAY_EXACT_QK" = 1 ]; then
    python3 - "$SG" "$Q_REPLAY" "$K_REPLAY" "$LIVE_TILELANG" <<'PY'
import glob, json, pathlib, sys

def interleave_rope(payload, rows, width):
    import numpy as np
    value = np.frombuffer(payload, dtype='<f2').reshape(rows, width // 128, 2, 64)
    return value.transpose(0, 1, 3, 2).copy().tobytes()
roots = sorted(glob.glob(sys.argv[1] + '/rank-*-pid-*'))
assert len(roots) == 4
for name, output, expected in (
    ('layer0.attention.tilelang.q' if int(sys.argv[4]) else 'layer0.attention.q_rotated', sys.argv[2], 8 * 1024 * 2),
    ('layer0.attention.k_normalized', sys.argv[3], 8 * 256 * 2),
):
    with open(output, 'wb') as stream:
        for root in roots:
            records = {row['name']: row for row in map(json.loads, open(root + '/manifest.jsonl'))}
            row = records[name]
            payload = pathlib.Path(root, row['file']).read_bytes()
            assert len(payload) == expected, (name, root, len(payload))
            stream.write(payload if name.endswith(('q_rotated', 'tilelang.q')) else interleave_rope(payload, 8, 256))
PY
    python3 - "$SG" "$FLATTENED_KV_REPLAY" "$LIVE_TILELANG" <<'PY'
import glob, json, numpy as np, pathlib, sys

roots = sorted(glob.glob(sys.argv[1] + '/rank-*-pid-*'))
assert len(roots) == 4
with open(sys.argv[2], 'wb') as output:
    for root in roots:
        records = {row['name']: row for row in map(json.loads, open(root + '/manifest.jsonl'))}
        if int(sys.argv[3]):
            cache_k = np.fromfile(pathlib.Path(root, records['layer0.attention.tilelang.k']['file']), dtype='<f2').reshape(-1, 2, 128)
            cache_v = np.fromfile(pathlib.Path(root, records['layer0.attention.tilelang.v']['file']), dtype='<f2').reshape(-1, 2, 128)
        else:
            cache_k = np.fromfile(pathlib.Path(root, records['context.prompt.layer0.cache_k']['file']), dtype='<f2').reshape(1000, 2, 128)
            cache_v = np.fromfile(pathlib.Path(root, records['context.prompt.layer0.cache_v']['file']), dtype='<f2').reshape(1000, 2, 128)
        cache_k = cache_k.transpose(1, 0, 2).copy()
        cache_v = cache_v.transpose(1, 0, 2).copy()
        output.write(np.stack((cache_k, cache_v), axis=1).tobytes())
PY
fi
if [ "$REPLAY_PROJECTED_QK" = 1 ]; then
    python3 - "$SG" "$Q_PROJECTION_REPLAY" "$K_PROJECTION_REPLAY" <<'PY'
import glob, json, numpy as np, pathlib, sys

def interleave(value):
    rows, width = value.shape
    return value.reshape(rows, width // 128, 2, 64).transpose(0, 1, 3, 2).copy()
roots = sorted(glob.glob(sys.argv[1] + '/rank-*-pid-*'))
assert len(roots) == 4
with open(sys.argv[2], 'wb') as q_out, open(sys.argv[3], 'wb') as k_out:
    for root in roots:
        records = {row['name']: row for row in map(json.loads, open(root + '/manifest.jsonl'))}
        row = records['layer0.attention.qkv_projection']
        value = np.fromfile(pathlib.Path(root, row['file']), dtype='<f2').reshape(8, 1536)
        q_out.write(interleave(value[:, :1024]).tobytes())
        k_out.write(interleave(value[:, 1024:1280]).tobytes())
PY
fi
printf 'full_context=%s\nfull_norm=%s\ninitial_norm=%s\nattention_input=%s\nflattened_kv=%s\nreplay_exact_qk=%s\nreplay_projected_qk=%s\nreplay_attention_input=%s\n' \
    "$FULL_CONTEXT" "$FULL_NORM" "$INITIAL_NORM" "$ATTENTION_INPUT" "$FLATTENED_KV_REPLAY" "$REPLAY_EXACT_QK" "$REPLAY_PROJECTED_QK" "$REPLAY_ATTENTION_INPUT"

mkdir -p "$RESULTS/parity"
RUN_ENV=(env
    TM_DFLASH_CONTEXT_REPLAY_FILE="$FULL_CONTEXT"
    TM_DFLASH_CONTEXT_NORM_REPLAY_FILE="$FULL_NORM"
    TM_DFLASH_BLOCK_INITIAL_NORM_REPLAY_FILE="$INITIAL_NORM"
    TM_DFLASH_ANCHOR_INCLUSIVE_FRONTIER=1
    TM_DFLASH_ASSERT_DRAFT_METADATA=1
    TM_DFLASH_REDUCE_BEFORE_CONV=1
    TM_DFLASH_PARITY_DIR="$RESULTS/parity")
if [ "$REPLAY_EXACT_QK" = 1 ]; then
    RUN_ENV+=(TM_DFLASH_DRAFT_Q_NORM_REPLAY_FILE="$Q_REPLAY"
        TM_DFLASH_DRAFT_K_NORM_REPLAY_FILE="$K_REPLAY"
        TM_DFLASH_DRAFT_FLATTENED_KV_REPLAY_FILE="$FLATTENED_KV_REPLAY"
        TM_DFLASH_DRAFT_FLATTENED_KV_REPLAY_CONTEXT_LEN="$REPLAY_CONTEXT_LEN")
fi
if [ "$REPLAY_PROJECTED_QK" = 1 ]; then
    RUN_ENV+=(TM_DFLASH_DRAFT_Q_PROJECTION_REPLAY_FILE="$Q_PROJECTION_REPLAY"
        TM_DFLASH_DRAFT_K_PROJECTION_REPLAY_FILE="$K_PROJECTION_REPLAY")
fi
if [ "$REPLAY_ATTENTION_INPUT" = 1 ]; then
    RUN_ENV+=(TM_DFLASH_DRAFT_ATTENTION_INPUT_REPLAY_FILE="$ATTENTION_INPUT")
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
if [ "$REPLAY_EXACT_QK" = 1 ]; then
    [ "$(grep -c 'TM_DFLASH_DRAFT_Q_NORM_REPLAY_FILE' "$RESULTS/parity.log")" -eq 4 ]
    [ "$(grep -c 'TM_DFLASH_DRAFT_K_NORM_REPLAY_FILE' "$RESULTS/parity.log")" -eq 4 ]
    [ "$(grep -c 'TM_DFLASH_DRAFT_FLATTENED_KV_REPLAY_FILE' "$RESULTS/parity.log")" -eq 4 ]
fi
if [ "$REPLAY_PROJECTED_QK" = 1 ]; then
    [ "$(grep -c 'TM_DFLASH_DRAFT_Q_PROJECTION_REPLAY_FILE' "$RESULTS/parity.log")" -eq 4 ]
    [ "$(grep -c 'TM_DFLASH_DRAFT_K_PROJECTION_REPLAY_FILE' "$RESULTS/parity.log")" -eq 4 ]
fi
if [ "$REPLAY_ATTENTION_INPUT" = 1 ]; then
    [ "$(grep -c 'TM_DFLASH_DRAFT_ATTENTION_INPUT_REPLAY_FILE' "$RESULTS/parity.log")" -eq 4 ]
fi

python3 /job/compare_dflash_parity.py \
    --lmdeploy "$RESULTS/parity/lmdeploy" --sglang "$SG" \
    --output "$RESULTS/compare.json" >"$RESULTS/compare.log" || true

python3 - "$RESULTS/parity/lmdeploy" "$SG" <<'PY'
import glob, json, numpy as np, pathlib, sys
lm_roots = sorted(glob.glob(sys.argv[1] + '/rank-*-pid-*'))
sg_roots = sorted(glob.glob(sys.argv[2] + '/rank-*-pid-*'))
assert len(lm_roots) == len(sg_roots) == 4
DT = {'f16': '<f2', 'f32': '<f4', 'i32': '<i4', 'i64': '<i8'}
def records(root):
    return {row['name']: row for row in map(json.loads, open(root + '/manifest.jsonl'))}
def load(root, rows, name):
    row = rows[name]
    return np.fromfile(root + '/' + row['file'], dtype=DT[row['dtype']]).reshape(row['shape']).astype(np.float32)
def stats(a, b):
    assert a.shape == b.shape, (a.shape, b.shape)
    delta = a - b
    return int(np.count_nonzero(a != b)), float(np.max(np.abs(delta))), float(np.sqrt(np.mean(delta * delta, dtype=np.float64)))
def reorder_rope(value):
    shape = value.shape
    assert shape[-1] % 128 == 0
    return value.reshape(-1, shape[-1] // 128, 2, 64).transpose(0, 1, 3, 2).reshape(shape)
for rank, (lm_root, sg_root) in enumerate(zip(lm_roots, sg_roots)):
    lm, sg = records(lm_root), records(sg_root)
    required_lm = {
        'context.prompt.layer0.qkv_projection',
        'context.prompt.layer0.qkv_pre_process',
        'context.prompt.layer0.qkv_post_process',
        'layer0.attention.qkv_projection',
        'layer0.attention.qkv_pre_process',
        'layer0.attention.qkv_post_process',
        'layer0.attention.flattened_kv',
        'layer0.attention.core_output',
    }
    assert required_lm <= lm.keys(), (rank, sorted(required_lm - lm.keys()))
    prompt = load(lm_root, lm, 'context.prompt.layer0.qkv_projection').reshape(1000, -1)
    assert prompt.shape[1] == 1536, prompt.shape
    draft_projection = load(lm_root, lm, 'layer0.attention.qkv_projection').reshape(8, -1)
    draft_normalized = load(lm_root, lm, 'layer0.attention.qkv_pre_process').reshape(8, -1)
    assert draft_projection.shape[1] == draft_normalized.shape[1] == 1536
    sg_projection = load(sg_root, sg, 'layer0.attention.qkv_projection')
    flattened_storage = load(lm_root, lm, 'layer0.attention.flattened_kv').reshape(-1)
    if 'layer0.attention.tilelang.k' in sg:
        sg_flat_k = load(sg_root, sg, 'layer0.attention.tilelang.k').transpose(1, 0, 2)
        sg_flat_v = load(sg_root, sg, 'layer0.attention.tilelang.v').transpose(1, 0, 2)
        sg_attention_q = load(sg_root, sg, 'layer0.attention.tilelang.q').reshape(8, 1024)
        sg_attention_output = load(sg_root, sg, 'layer0.attention.tilelang.output').reshape(8, 1024)
    else:
        sg_flat_k = load(sg_root, sg, 'context.prompt.layer0.cache_k').transpose(1, 0, 2)
        sg_flat_v = load(sg_root, sg, 'context.prompt.layer0.cache_v').transpose(1, 0, 2)
        sg_attention_q = load(sg_root, sg, 'layer0.attention.q_rotated')
        sg_attention_output = load(sg_root, sg, 'layer0.attention.core_output')
    key_count = sg_flat_k.shape[1]
    flattened = flattened_storage[:2 * 2 * key_count * 128].reshape(2, 2, key_count, 128)
    pairs = {
        'context.k_projection': (prompt[:, 1024:1280], reorder_rope(load(sg_root, sg, 'context.prompt.layer0.k_projection'))),
        'context.v_projection': (prompt[:, 1280:1536], load(sg_root, sg, 'context.prompt.layer0.v_projection')),
        'draft.q_projection': (draft_projection[:, :1024], reorder_rope(sg_projection[:, :1024])),
        'draft.k_projection': (draft_projection[:, 1024:1280], reorder_rope(sg_projection[:, 1024:1280])),
        'draft.v_projection': (draft_projection[:, 1280:1536], sg_projection[:, 1280:1536]),
        'draft.q_attention': (draft_normalized[:, :1024], sg_attention_q),
        'draft.k_normalized': (draft_normalized[:, 1024:1280], reorder_rope(load(sg_root, sg, 'layer0.attention.k_normalized'))),
        'draft.cache_k': (flattened[:, 0, :key_count, :], sg_flat_k),
        'draft.cache_v': (flattened[:, 1, :key_count, :], sg_flat_v),
        'draft.core_output': (load(lm_root, lm, 'layer0.attention.core_output'), sg_attention_output),
    }
    for name, (left, right) in pairs.items():
        print('BOUNDARY', 'rank', rank, name, 'different/max/rms', stats(left, right))
PY

touch "$RESULTS/completed"
echo DFLASH_CONTEXT_KV_PARITY_COMPLETE
