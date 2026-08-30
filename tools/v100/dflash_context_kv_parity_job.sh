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

SG=/results/20260830_223929-sglang-dflash-parity-1a86d3c5a8b6/trace/sglang
mapfile -t REPLAY < <(
    python3 - "$SG" <<'PY'
import glob, hashlib, json, os, sys
roots = sorted(glob.glob(sys.argv[1] + '/rank-*-pid-*'))
assert len(roots) == 4, roots
for name, shape, size in (
    ('target.full_context', [1000, 25600], 51200000),
    ('context.full_norm', [1000, 5120], 10240000),
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
[ "${#REPLAY[@]}" -eq 2 ]
FULL_CONTEXT=${REPLAY[0]}
FULL_NORM=${REPLAY[1]}
Q_REPLAY=$RESULTS/q_normalized_tp4.bin
K_REPLAY=$RESULTS/k_normalized_tp4.bin
python3 - "$SG" "$Q_REPLAY" "$K_REPLAY" <<'PY'
import glob, json, pathlib, sys
roots = sorted(glob.glob(sys.argv[1] + '/rank-*-pid-*'))
assert len(roots) == 4
for name, output, expected in (
    ('layer0.attention.q_normalized', sys.argv[2], 8 * 1024 * 2),
    ('layer0.attention.k_normalized', sys.argv[3], 8 * 256 * 2),
):
    with open(output, 'wb') as stream:
        for root in roots:
            records = {row['name']: row for row in map(json.loads, open(root + '/manifest.jsonl'))}
            row = records[name]
            payload = pathlib.Path(root, row['file']).read_bytes()
            assert len(payload) == expected, (name, root, len(payload))
            stream.write(payload)
PY
printf 'full_context=%s\nfull_norm=%s\nq_replay=%s\nk_replay=%s\n' \
    "$FULL_CONTEXT" "$FULL_NORM" "$Q_REPLAY" "$K_REPLAY"

mkdir -p "$RESULTS/parity"
TM_DFLASH_CONTEXT_REPLAY_FILE="$FULL_CONTEXT" \
    TM_DFLASH_CONTEXT_NORM_REPLAY_FILE="$FULL_NORM" \
    TM_DFLASH_REDUCE_BEFORE_CONV=1 \
    TM_DFLASH_PARITY_DIR="$RESULTS/parity" \
    python3 /job/bench_decode.py \
    --model /models/Qwen3.8-27B-FP8 --tp 4 --num-draft-tokens 7 \
    --speculative-algorithm dflash2 --speculative-draft-model /models/Qwen3.8-27B-DFlash2 \
    --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
    --input-tokens 1000 --output-tokens 64 --trials 1 --sglang-corpus /sglang-corpus \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    --cache-max-entry-count 0.05 --json-out "$RESULTS/parity.json" 2>&1 | tee "$RESULTS/parity.log"
[ "$(grep -c 'replaying parity target context rows=1000 ' "$RESULTS/parity.log")" -eq 4 ]
[ "$(grep -c 'replaying normalized parity context rows=1000 ' "$RESULTS/parity.log")" -eq 4 ]

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
    flattened = load(lm_root, lm, 'layer0.attention.flattened_kv')
    sg_prompt_k = load(sg_root, sg, 'context.prompt.layer0.cache_k')
    sg_prompt_v = load(sg_root, sg, 'context.prompt.layer0.cache_v')
    sg_block_k = load(sg_root, sg, 'layer0.attention.k_rotated').reshape(8, 2, 128)
    sg_block_v = sg_projection[:, 1280:1536].reshape(8, 2, 128)
    sg_flat_k = np.concatenate([sg_prompt_k, sg_block_k], axis=0).transpose(1, 0, 2)
    sg_flat_v = np.concatenate([sg_prompt_v, sg_block_v], axis=0).transpose(1, 0, 2)
    key_count = sg_flat_k.shape[1]
    pairs = {
        'context.k_projection': (prompt[:, 1024:1280], reorder_rope(load(sg_root, sg, 'context.prompt.layer0.k_projection'))),
        'context.v_projection': (prompt[:, 1280:1536], load(sg_root, sg, 'context.prompt.layer0.v_projection')),
        'draft.q_projection': (draft_projection[:, :1024], reorder_rope(sg_projection[:, :1024])),
        'draft.k_projection': (draft_projection[:, 1024:1280], reorder_rope(sg_projection[:, 1024:1280])),
        'draft.v_projection': (draft_projection[:, 1280:1536], sg_projection[:, 1280:1536]),
        'draft.q_normalized': (draft_normalized[:, :1024], reorder_rope(load(sg_root, sg, 'layer0.attention.q_normalized'))),
        'draft.k_normalized': (draft_normalized[:, 1024:1280], reorder_rope(load(sg_root, sg, 'layer0.attention.k_normalized'))),
        'draft.cache_k': (flattened[:, 0, :key_count, :], reorder_rope(sg_flat_k)),
        'draft.cache_v': (flattened[:, 1, :key_count, :], sg_flat_v),
        'draft.core_output': (
            load(lm_root, lm, 'layer0.attention.core_output'),
            load(sg_root, sg, 'layer0.attention.core_output'),
        ),
    }
    for name, (left, right) in pairs.items():
        print('BOUNDARY', 'rank', rank, name, 'different/max/rms', stats(left, right))
PY

touch "$RESULTS/completed"
echo DFLASH_CONTEXT_KV_PARITY_COMPLETE
