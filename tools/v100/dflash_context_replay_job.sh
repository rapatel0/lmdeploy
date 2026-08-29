#!/usr/bin/env bash
# Isolate DFlash context-projector parity by replaying SGLang's first target residual in TurboMind.
set -euo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-context-replay-${SRC_COMMIT}
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
source_paths = []
source_hashes = []
context_hashes = {"context.fc": [], "context.norm": []}
for rank, directory in enumerate(directories):
    assert directory.name.startswith(f"rank-{rank}-"), directory
    records = {r["name"]: r for r in map(json.loads, (directory / "manifest.jsonl").read_text().splitlines())}
    for name in ("target.post_layer_residual", "context.fc", "context.norm", "block.ids", "block.positions"):
        assert name in records, (directory, name)
    target = records["target.post_layer_residual"]
    assert target["dtype"] == "f16" and target["shape"] == [1, 25600], target
    target_path = directory / target["file"]
    payload = target_path.read_bytes()
    assert len(payload) == 51200, target_path
    source_paths.append(target_path)
    source_hashes.append(hashlib.sha256(payload).hexdigest())
    for name in context_hashes:
        context_payload = (directory / records[name]["file"]).read_bytes()
        context_hashes[name].append(hashlib.sha256(context_payload).hexdigest())
    ids_record = records["block.ids"]
    pos_record = records["block.positions"]
    ids = list(struct.unpack("<8q", (directory / ids_record["file"]).read_bytes()))
    positions = list(struct.unpack("<8q", (directory / pos_record["file"]).read_bytes()))
    assert ids == expected_ids, (directory, ids)
    assert positions == expected_positions, (directory, positions)
assert len(set(source_hashes)) == 1, source_hashes
for name, hashes in context_hashes.items():
    assert len(set(hashes)) == 1, (name, hashes)
print(source_paths[0])
PY
)"
echo "SGLang context replay file: ${REPLAY_FILE}"

run_trace() {
    local arm=$1
    local replay=$2
    mkdir -p "${RESULTS}/${arm}"
    if [ -n "${replay}" ]; then
        TM_DFLASH_CONTEXT_REPLAY_FILE="${replay}" \
            TM_DFLASH_PARITY_DIR="${RESULTS}/${arm}" \
            python3 /job/bench_decode.py \
            --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
            --num-draft-tokens 7 --speculative-algorithm dflash2 \
            --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
            --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
            --input-tokens 1000 --output-tokens 64 --trials 1 \
            --sglang-corpus /sglang-corpus \
            --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
            --cache-max-entry-count 0.05 --json-out "${RESULTS}/${arm}.json" \
            2>&1 | tee "${RESULTS}/${arm}.log"
    else
        TM_DFLASH_PARITY_DIR="${RESULTS}/${arm}" \
            python3 /job/bench_decode.py \
            --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" \
            --num-draft-tokens 7 --speculative-algorithm dflash2 \
            --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
            --speculative-dflash-block-size 8 --speculative-draft-window 2048 \
            --input-tokens 1000 --output-tokens 64 --trials 1 \
            --sglang-corpus /sglang-corpus \
            --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
            --cache-max-entry-count 0.05 --json-out "${RESULTS}/${arm}.json" \
            2>&1 | tee "${RESULTS}/${arm}.log"
    fi
}

run_trace native ""
run_trace replay "${REPLAY_FILE}"
[ "$(grep -c '\[DFlash2\] replaying first eligible target context' "${RESULTS}/replay.log")" -eq 4 ]

python3 - "${RESULTS}" "${REPLAY_FILE}" <<'PY'
import hashlib
import json
import pathlib
import struct
import sys
root = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2]).read_bytes()
expected_ids = [1144] + [248070] * 7
required = {
    "target.post_layer_residual", "context.fc", "context.norm", "block.ids",
    "block.embedding", "layer0.attention.conv_side0", "layer4.mlp.norm_output",
    "selector.candidate_ids", "selector.unary_scores", "selector.selected_ids",
}
for arm in ("native", "replay"):
    directories = sorted(path for path in (root / arm / "lmdeploy").glob("rank-*-device-*-pid-*") if path.is_dir())
    assert len(directories) == 4, (arm, directories)
    context_hashes = {"context.fc": [], "context.norm": []}
    for rank, directory in enumerate(directories):
        assert directory.name.startswith(f"rank-{rank}-"), directory
        records = {r["name"]: r for r in map(json.loads, (directory / "manifest.jsonl").read_text().splitlines())}
        assert required <= records.keys(), (directory, sorted(required - records.keys()))
        ids_record = records["block.ids"]
        ids = list(struct.unpack("<8i", (directory / ids_record["file"]).read_bytes()))
        assert ids == expected_ids, (directory, ids)
        if arm == "replay":
            target = (directory / records["target.post_layer_residual"]["file"]).read_bytes()
            assert target == source, directory
        for name in context_hashes:
            payload = (directory / records[name]["file"]).read_bytes()
            context_hashes[name].append(hashlib.sha256(payload).hexdigest())
    for name, hashes in context_hashes.items():
        assert len(set(hashes)) == 1, (arm, name, hashes)
print("DFLASH_CONTEXT_REPLAY_TP4_TRACE_PASS")
PY

python3 /job/compare_dflash_parity.py \
    --lmdeploy "${RESULTS}/native/lmdeploy" --sglang "${SGLANG_ROOT}" \
    --output "${RESULTS}/native_compare.json" | tee "${RESULTS}/native_compare.log"
python3 /job/compare_dflash_parity.py \
    --lmdeploy "${RESULTS}/replay/lmdeploy" --sglang "${SGLANG_ROOT}" \
    --output "${RESULTS}/replay_compare.json" | tee "${RESULTS}/replay_compare.log"

python3 - "${RESULTS}" <<'PY'
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
native = json.loads((root / "native_compare.json").read_text())
replay = json.loads((root / "replay_compare.json").read_text())
by_name = lambda report: {row["lmdeploy"]: row for row in report["comparisons"]}
native_rows = by_name(native)
replay_rows = by_name(replay)
assert native_rows["target.post_layer_residual"]["status"] == "mismatch", native_rows["target.post_layer_residual"]
assert replay_rows["target.post_layer_residual"]["status"] == "match", replay_rows["target.post_layer_residual"]
assert replay_rows["context.fc"]["status"] == "mismatch", replay_rows["context.fc"]
assert replay_rows["context.fc"]["rms"] < 0.061, replay_rows["context.fc"]
assert replay_rows["context.norm"]["status"] == "match", replay_rows["context.norm"]
assert replay_rows["context.norm"]["max_abs"] <= 0.00390625, replay_rows["context.norm"]
assert replay_rows["context.norm"]["rms"] < 0.0003, replay_rows["context.norm"]
assert replay_rows["block.ids"]["status"] == "match", replay_rows["block.ids"]
assert replay_rows["block.embedding"]["status"] == "match", replay_rows["block.embedding"]
print("DFLASH_CONTEXT_REPLAY_PASS")
print("replay_context_fc", replay_rows["context.fc"])
print("replay_context_norm", replay_rows["context.norm"])
print("replay_earliest", replay["earliest_mismatch"])
PY

touch "${RESULTS}/completed"
echo DFLASH_CONTEXT_REPLAY_COMPLETE
