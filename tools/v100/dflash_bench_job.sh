#!/usr/bin/env bash
# Matched TP4, batch-one DFlash2 throughput benchmark at the SGLang shape.
set -uo pipefail
MODEL="${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
DRAFT_MODEL="${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-bench-${SRC_COMMIT}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
finish() {
    rc=$?
    echo "${rc}" >"${RESULTS}/exit_code"
    echo "artifacts in ${RESULTS} (exit ${rc})"
}
trap finish EXIT
cat /src/SOURCE_STAMP
if ! bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1; then
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -40
    exit 2
fi
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
cd / || exit $?
export TM_LOG_LEVEL=INFO

for K in 0 7; do
    echo "=== DFlash2 benchmark K=${K} ==="
    ARGS=(
        --model "${MODEL}" --tp "${TP:-4}"
        --num-draft-tokens "${K}"
        --input-tokens 1000 --output-tokens 256 --trials 3
        --sglang-corpus /sglang-corpus
        --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
        --cache-max-entry-count 0.05
        --json-out "${RESULTS}/bench_k${K}.json"
    )
    if [ "${K}" -gt 0 ]; then
        ARGS+=(
            --speculative-algorithm dflash2
            --speculative-draft-model "${DRAFT_MODEL}"
            --speculative-dflash-block-size 8
            --speculative-draft-window 2048
        )
    fi
    python3 /src/tools/v100/bench_decode.py "${ARGS[@]}" 2>&1 | tee "${RESULTS}/bench_k${K}.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || exit 3
done

python3 - "${RESULTS}/bench_k0.json" "${RESULTS}/bench_k7.json" <<'PY'
import json
import sys

with open(sys.argv[1]) as f:
    base = json.load(f)
with open(sys.argv[2]) as f:
    spec = json.load(f)
b = base['mean_decode_tok_s']
s = spec['mean_decode_tok_s']
print(f'DFLASH_BENCH_RESULT baseline={b:.3f} dflash={s:.3f} speedup={s / b:.3f}x')
PY

echo DFLASH_BENCH_COMPLETE
