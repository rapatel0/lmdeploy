#!/usr/bin/env bash
# Compare legacy versus SGLang-style full-product DFlash initial RMSNorm on V100.
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-full-product-rmsnorm-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
finish() {
    rc=$?
    echo "${rc}" >"${RESULTS}/exit_code"
    [ -f "${RESULTS}/completed" ] || echo KILLED >"${RESULTS}/incomplete"
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT

cat /src/SOURCE_STAMP
bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || {
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -80
    exit 2
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
cd /
export TM_LOG_LEVEL=INFO

SGLANG_ROOT="${SGLANG_PARITY_ROOT:-/results/20260829_032508-sglang-dflash-parity-b753831db680/trace/sglang}"
REPLAY_FILE="$(
    python3 - "${SGLANG_ROOT}" <<'PY'
import hashlib
import json
import pathlib
import struct
import sys
root = pathlib.Path(sys.argv[1])
directories = sorted(path for path in root.glob("rank-*-pid-*") if path.is_dir())
assert len(directories) == 4, directories
expected_ids = [1144] + [248070] * 7
expected_positions = list(range(1000, 1008))
paths = []
hashes = []
initial_norm_hashes = []
for rank, directory in enumerate(directories):
    assert directory.name.startswith(f"rank-{rank}-"), directory
    records = {r["name"]: r for r in map(json.loads, (directory / "manifest.jsonl").read_text().splitlines())}
    target = records["target.post_layer_residual"]
    initial_norm = records["layer0.attention.norm_output"]
    path = directory / target["file"]
    payload = path.read_bytes()
    assert target["dtype"] == "f16" and target["shape"] == [1, 25600] and len(payload) == 51200
    ids = list(struct.unpack("<8q", (directory / records["block.ids"]["file"]).read_bytes()))
    positions = list(struct.unpack("<8q", (directory / records["block.positions"]["file"]).read_bytes()))
    assert ids == expected_ids and positions == expected_positions
    paths.append(path)
    hashes.append(hashlib.sha256(payload).hexdigest())
    initial_norm_payload = (directory / initial_norm["file"]).read_bytes()
    initial_norm_hashes.append(hashlib.sha256(initial_norm_payload).hexdigest())
assert len(set(hashes)) == 1, hashes
assert len(set(initial_norm_hashes)) == 1, initial_norm_hashes
print(paths[0])
PY
)"
echo "SGLang context replay file: ${REPLAY_FILE}"

bench_arm() {
    local full=$1
    TM_DFLASH_FULL_PRODUCT_RMSNORM="${full}" \
        python3 /job/bench_decode.py \
        --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
        --num-draft-tokens 7 --speculative-algorithm dflash2 \
        --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
        --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
        --input-tokens 1000 --output-tokens 256 --trials 5 \
        --sglang-corpus /sglang-corpus \
        --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
        --cache-max-entry-count 0.05 --json-out "${RESULTS}/full_${full}.json" \
        2>&1 | tee "${RESULTS}/full_${full}.log"
}

trace_arm() {
    local full=$1
    mkdir -p "${RESULTS}/trace_${full}"
    TM_DFLASH_FULL_PRODUCT_RMSNORM="${full}" \
        TM_DFLASH_CONTEXT_REPLAY_FILE="${REPLAY_FILE}" \
        TM_DFLASH_PARITY_DIR="${RESULTS}/trace_${full}" \
        python3 /job/bench_decode.py \
        --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
        --num-draft-tokens 7 --speculative-algorithm dflash2 \
        --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
        --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
        --input-tokens 1000 --output-tokens 64 --trials 1 \
        --sglang-corpus /sglang-corpus \
        --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
        --cache-max-entry-count 0.05 --json-out "${RESULTS}/trace_${full}.json" \
        2>&1 | tee "${RESULTS}/trace_${full}.log"
    python3 /job/compare_dflash_parity.py \
        --lmdeploy "${RESULTS}/trace_${full}/lmdeploy" --sglang "${SGLANG_ROOT}" \
        --output "${RESULTS}/compare_${full}.json" | tee "${RESULTS}/compare_${full}.log"
}

bench_arm 0
bench_arm 1
trace_arm 0
trace_arm 1

echo "=== full-product audited identity ==="
TM_DFLASH_FULL_PRODUCT_RMSNORM=1 \
    python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 | tee "${RESULTS}/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "${RESULTS}/identity.log"

python3 - "${RESULTS}" "${REPLAY_FILE}" <<'PY'
import hashlib
import json
import pathlib
import re
import struct
import sys
root = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2]).read_bytes()
expected_ids = [1144] + [248070] * 7
required = {
    "target.post_layer_residual", "context.norm", "block.ids", "block.embedding",
    "block.initial_norm", "selector.candidate_ids", "selector.selected_ids",
}
rows = {}
for full in (0, 1):
    report = json.loads((root / f"compare_{full}.json").read_text())
    by_name = {row["lmdeploy"]: row for row in report["comparisons"]}
    rows[full] = by_name["block.initial_norm"]
    assert by_name["target.post_layer_residual"]["status"] == "match"
    assert by_name["context.norm"]["status"] == "match"
    assert by_name["block.ids"]["status"] == "match"
    assert by_name["block.embedding"]["status"] == "match"
    directories = sorted((root / f"trace_{full}" / "lmdeploy").glob("rank-*-device-*-pid-*"))
    assert len(directories) == 4, directories
    hashes = {"block.embedding": [], "block.initial_norm": []}
    for rank, directory in enumerate(directories):
        assert directory.name.startswith(f"rank-{rank}-"), directory
        records = {r["name"]: r for r in map(json.loads, (directory / "manifest.jsonl").read_text().splitlines())}
        assert required <= records.keys(), (directory, sorted(required - records.keys()))
        target = (directory / records["target.post_layer_residual"]["file"]).read_bytes()
        assert target == source, directory
        ids = list(struct.unpack("<8i", (directory / records["block.ids"]["file"]).read_bytes()))
        assert ids == expected_ids, (directory, ids)
        for name in hashes:
            payload = (directory / records[name]["file"]).read_bytes()
            hashes[name].append(hashlib.sha256(payload).hexdigest())
    for name, values in hashes.items():
        assert len(set(values)) == 1, (full, name, values)
legacy = rows[0]
full = rows[1]
if full["status"] == "match":
    print("DFLASH_FULL_PRODUCT_RMSNORM_PARITY_PASS")
else:
    print("DFLASH_FULL_PRODUCT_RMSNORM_PARITY_FALSIFIED")
print("legacy_initial_norm", legacy)
print("full_product_initial_norm", full)
for arm in (0, 1):
    bench = json.loads((root / f"full_{arm}.json").read_text())
    matches = re.findall(r"final commit length ([0-9.]+), raw [0-9.]+ over ([0-9]+) verification steps", (root / f"full_{arm}.log").read_text())
    assert matches, arm
    commit_length = float(matches[-1][0])
    decode_tok_s = float(bench["mean_decode_tok_s"])
    normalized_cycle_ms = 1000.0 * commit_length / decode_tok_s
    print(f"full_{arm}_decode_tok_s", decode_tok_s)
    print(f"full_{arm}_commit_length", commit_length)
    print(f"full_{arm}_normalized_cycle_ms", normalized_cycle_ms)
PY

touch "${RESULTS}/completed"
echo DFLASH_FULL_PRODUCT_RMSNORM_COMPLETE
