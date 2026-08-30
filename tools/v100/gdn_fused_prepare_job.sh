#!/usr/bin/env bash
# One-build TP4 qualification for exact Q=8 SM70 GDN preparation fusion.
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-gdn-fused-prepare-${SRC_COMMIT:-unknown}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"${RESULTS}/exit_code"; echo "artifacts in ${RESULTS} (exit ${rc})"' EXIT
cat /src/SOURCE_STAMP
rm -f /wheels/lmdeploy-*.whl
bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || {
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -100
    exit 2
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ]
sha256sum "${WHEEL}" | tee "${RESULTS}/wheel.sha256"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
export TM_LOG_LEVEL=INFO
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}"
    --num-draft-tokens 7 --speculative-algorithm dflash2
    --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
    --speculative-dflash-block-size 8 --speculative-draft-window 2048
    --input-tokens 1000 --sglang-corpus /sglang-corpus
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
    --cache-max-entry-count 0.05)

for arm_mode in legacy:0 beta:1 full:2; do
    arm=${arm_mode%:*}
    mode=${arm_mode#*:}
    echo "=== ${arm}: fused_prepare=${mode} ==="
    TM_GDN_SM70_FUSED_PREPARE="${mode}" \
        python3 /job/bench_decode.py "${common[@]}" --output-tokens 256 --trials 5 \
        --json-out "${RESULTS}/${arm}.json" 2>&1 | tee "${RESULTS}/${arm}.log"
done

NSYS="$(command -v nsys 2>/dev/null || true)"
if [ -z "${NSYS}" ] && [ -x /opt/nsys/nsys ]; then NSYS=/opt/nsys/nsys; fi
[ -n "${NSYS}" ] || {
    echo 'FAIL: nsys unavailable' >&2
    exit 2
}
export FT_NVTX=ON
for arm_mode in legacy:0 beta:1 full:2; do
    arm=${arm_mode%:*}
    mode=${arm_mode#*:}
    TM_GDN_SM70_FUSED_PREPARE="${mode}" \
        "${NSYS}" profile --force-overwrite=true --trace=cuda,nvtx,osrt --cuda-memory-usage=true \
        --capture-range=cudaProfilerApi --capture-range-end=stop --output="${RESULTS}/profile_${arm}" \
        python3 /job/bench_decode.py "${common[@]}" --output-tokens 128 --trials 1 --cuda-profiler-range \
        --json-out "${RESULTS}/profile_${arm}.json" 2>&1 | tee "${RESULTS}/profile_${arm}.log"
    "${NSYS}" stats --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum --format csv \
        --output "${RESULTS}/profile_${arm}_stats" "${RESULTS}/profile_${arm}.nsys-rep" \
        >"${RESULTS}/profile_${arm}_stats.log" 2>&1
done

python3 /job/analyze_gdn_fused_prepare.py "${RESULTS}" | tee "${RESULTS}/analysis.log"

TM_GDN_SM70_FUSED_PREPARE=2 \
    python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 128 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    2>&1 | tee "${RESULTS}/identity.log"
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "${RESULTS}/identity.log"
touch "${RESULTS}/completed"
echo GDN_FUSED_PREPARE_COMPLETE
