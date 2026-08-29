#!/usr/bin/env bash
# Finalize grouped-scale qualification from completed A/B, NCU, and Nsight artifacts.
set -euo pipefail

SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
SOURCE_RESULTS="$(find /results -maxdepth 1 -type d -name '*-dflash-fp8-m8-reuse-followup-*' -print | sort | tail -1)"
[ -n "${SOURCE_RESULTS}" ] || { echo 'FAIL: no follow-up result' >&2; exit 2; }
for file in baseline.json baseline.log reuse.json reuse.log profile_baseline.json profile_baseline.log \
    profile_reuse.json profile_reuse.log profile_baseline_stats_cuda_gpu_kern_sum.csv \
    profile_baseline_stats_nvtx_sum.csv profile_reuse_stats_cuda_gpu_kern_sum.csv \
    profile_reuse_stats_nvtx_sum.csv wheel.sha256; do
    [ -f "${SOURCE_RESULTS}/${file}" ] || { echo "FAIL: missing ${SOURCE_RESULTS}/${file}" >&2; exit 2; }
done
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-fp8-m8-reuse-final-${SRC_COMMIT}
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
printf 'finalizing from %s\n' "${SOURCE_RESULTS}"
cp "${SOURCE_RESULTS}"/{baseline.json,baseline.log,reuse.json,reuse.log,profile_baseline.json,profile_baseline.log,profile_reuse.json,profile_reuse.log,profile_baseline_stats_cuda_gpu_kern_sum.csv,profile_baseline_stats_nvtx_sum.csv,profile_reuse_stats_cuda_gpu_kern_sum.csv,profile_reuse_stats_nvtx_sum.csv,wheel.sha256} "${RESULTS}/"

WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
expected_sha="$(cut -d' ' -f1 "${RESULTS}/wheel.sha256")"
actual_sha="$(sha256sum "${WHEEL}" | cut -d' ' -f1)"
[ "${actual_sha}" = "${expected_sha}" ] || { echo 'FAIL: wheel SHA mismatch' >&2; exit 2; }
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
python3 /src/tools/v100/analyze_dflash_fp8_m8_reuse_scale.py "${RESULTS}" | tee "${RESULTS}/analysis.log"

echo '=== reuse-scale exact audited identity ==='
set +e
TM_SM70_FP8_M8_REUSE_SCALE=1 python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    2>&1 | tee "${RESULTS}/identity.log"
identity_rc=${PIPESTATUS[0]}
set -e
identity_pass=0
grep -q '^DFLASH_AUDITED_IDENTITY_PASS$' "${RESULTS}/identity.log" && identity_pass=1

python3 - "${RESULTS}/analysis.json" "${RESULTS}/qualification.json" "${identity_pass}" "${identity_rc}" <<'PY'
import json, sys
analysis_path, output_path = sys.argv[1], sys.argv[2]
identity_pass, identity_rc = bool(int(sys.argv[3])), int(sys.argv[4])
data = json.load(open(analysis_path, encoding="utf-8"))
checks = {
    "identity": identity_pass and identity_rc == 0,
    "m8_kernel_improvement_at_least_1pct": data["profile_m8_fp8_change_pct"] <= -1.0,
    "unprofiled_cycle_improvement_at_least_0_5pct": data["unprofiled_cycle_change_pct"] <= -0.5,
    "profiled_cycle_regression_at_most_0_5pct": data["profiled_cycle_change_pct"] <= 0.5,
}
result = {
    "status": "qualified" if all(checks.values()) else "rejected",
    "checks": checks,
    "unprofiled_cycle_change_pct": data["unprofiled_cycle_change_pct"],
    "profiled_cycle_change_pct": data["profiled_cycle_change_pct"],
    "profile_m8_fp8_change_pct": data["profile_m8_fp8_change_pct"],
}
open(output_path, "w", encoding="utf-8").write(json.dumps(result, indent=2) + "\n")
print("DFLASH_FP8_M8_REUSE_SCALE_" + result["status"].upper(), json.dumps(result, sort_keys=True))
PY

touch "${RESULTS}/completed"
echo DFLASH_FP8_M8_REUSE_SCALE_FINALIZE_COMPLETE
