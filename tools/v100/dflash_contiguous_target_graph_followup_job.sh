#!/usr/bin/env bash
# Finish analysis and identity after the original graph job's analyzer-only failure.
set -euo pipefail
ROOT="$(find /results -maxdepth 1 -type d -name '*-dflash-contiguous-target-graph-*' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${ROOT}" ] || { echo 'FAIL: no contiguous target graph result' >&2; exit 2; }
echo "DFLASH_CONTIGUOUS_TARGET_GRAPH_RESULT=${ROOT}"
python3 /job/analyze_dflash_contiguous_target_graph.py "${ROOT}" | tee "${ROOT}/analysis.log"
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ] || { echo 'FAIL: graph wheel missing' >&2; exit 2; }
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
TM_DFLASH_CONTIGUOUS_TARGET_GRAPH=1 \
    python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 128 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    2>&1 | tee "${ROOT}/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "${ROOT}/identity.log"
touch "${ROOT}/completed"
echo DFLASH_CONTIGUOUS_TARGET_GRAPH_COMPLETE
