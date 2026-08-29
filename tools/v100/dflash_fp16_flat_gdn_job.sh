#!/usr/bin/env bash
# Qualify transposed and unpacked-native SM70 FP16 GDN projection paths.
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-fp16-flat-gdn-${SRC_COMMIT:-unknown}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"${RESULTS}/exit_code"; echo "artifacts in ${RESULTS} (exit ${rc})"' EXIT
for driver in /job/bench_decode.py /job/verify_dflash_audited.py /job/dflash_fp16_flat_gdn_microbench.cu; do
    [ -f "${driver}" ] || {
        echo "FAIL: missing ${driver}" >&2
        exit 2
    }
done
rm -f /wheels/lmdeploy-*.whl
build_started=$(date +%s)
bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || {
    grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -100
    exit 2
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ -n "${WHEEL}" ] && [ "$(stat -c %Y "${WHEEL}")" -ge "${build_started}" ]
sha256sum "${WHEEL}" | tee "${RESULTS}/wheel.sha256"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
NSYS="$(command -v nsys 2>/dev/null || true)"
[ -n "${NSYS}" ] || NSYS=/opt/nsys/nsys
NVCC="$(command -v nvcc)"
LIB="$(
    python3 - <<'PY'
import glob, lmdeploy, os
root=os.path.dirname(lmdeploy.__file__)
items=glob.glob(root+'/**/libturbomind.so', recursive=True)
print(items[0] if items else '')
PY
)"
LIBDIR="$(dirname "${LIB}")"
FMT_HEADER="$(find /src/build -path '*/fmt/format.h' -print -quit)"
FMT_INCLUDE="${FMT_HEADER%/fmt/format.h}"
PYLIB="$(find /usr /opt -name 'libpython3*.so*' -print -quit 2>/dev/null)"
[ -n "${LIB}" ] && [ -n "${FMT_HEADER}" ] && [ -n "${PYLIB}" ] || {
    echo "FAIL: microbench link inputs unavailable LIB=${LIB} FMT_HEADER=${FMT_HEADER} PYLIB=${PYLIB}" >&2
    exit 2
}
"${NVCC}" -std=c++17 -O2 -lineinfo -gencode arch=compute_70,code=sm_70 \
    -I/src -I"${FMT_INCLUDE}" /job/dflash_fp16_flat_gdn_microbench.cu "${LIB}" \
    -Xlinker -rpath -Xlinker "${LIBDIR}" -lcudart -Xlinker "${PYLIB}" \
    -o /tmp/dflash_fp16_flat_gdn_microbench >"${RESULTS}/compile_microbench.log" 2>&1 || {
    tail -100 "${RESULTS}/compile_microbench.log"
    exit 2
}

arms=(baseline transposed native)
modes=(0 1 2)
export TM_GEMM_TUNE='top_k=0,clusters=0,min_iter=2,max_iter=10,max_time=2.0'
export TM_GEMM_TUNE_VERBOSE=1
for i in "${!arms[@]}"; do
    arm=${arms[$i]} mode=${modes[$i]}
    mkdir -p "${RESULTS}/micro_${arm}"
    TM_SM70_FP16_FLAT_GDN="${mode}" TM_FP16_GDN_DUMP_DIR="${RESULTS}/micro_${arm}" \
        /tmp/dflash_fp16_flat_gdn_microbench 2>&1 | tee "${RESULTS}/micro_${arm}.log"
    grep -q "SM70_FP16_FLAT_GDN_MICRO_PASS mode=${mode}" "${RESULTS}/micro_${arm}.log"
    if [ "${mode}" -eq 2 ]; then grep -q 'SM70_FP16_FLAT_GDN_REGISTERED device=0 candidates=6' "${RESULTS}/micro_${arm}.log"; fi
done
python3 - "${RESULTS}" <<'PY'
import json, math, sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); report={}
for m in (1,7,8):
    arrays={a:np.fromfile(root/f"micro_{a}/m{m}.bin",dtype=np.float16).astype(np.float32)
            for a in ("baseline","transposed","native")}
    for lhs,rhs in (("baseline","transposed"),("transposed","native")):
        d=arrays[lhs]-arrays[rhs]; key=f"m{m}_{lhs}_vs_{rhs}"
        row={"max_abs":float(np.max(np.abs(d))),"rms":float(np.sqrt(np.mean(d*d))),
             "differing":int(np.count_nonzero(d)),
             "finite":bool(np.isfinite(arrays[lhs]).all() and np.isfinite(arrays[rhs]).all())}
        report[key]=row
        if not row["finite"] or row["max_abs"]>0.125 or row["rms"]>0.01:
            raise SystemExit(f"FP16_FLAT_GDN_MICRO_PARITY_FAIL {key} {row}")
(root/'micro_parity.json').write_text(json.dumps(report,indent=2)+'\n')
print('FP16_FLAT_GDN_MICRO_PARITY_PASS',json.dumps(report,sort_keys=True))
PY

for i in "${!arms[@]}"; do
    arm=${arms[$i]} mode=${modes[$i]}
    TM_SM70_FP16_FLAT_GDN="${mode}" "${NSYS}" profile --force-overwrite=true --trace=cuda \
        --capture-range=cudaProfilerApi --capture-range-end=stop --output="${RESULTS}/micro_profile_${arm}" \
        /tmp/dflash_fp16_flat_gdn_microbench 2>&1 | tee "${RESULTS}/micro_profile_${arm}.log"
    "${NSYS}" stats --report cuda_gpu_kern_sum --format csv --output "${RESULTS}/micro_profile_${arm}_stats" \
        "${RESULTS}/micro_profile_${arm}.nsys-rep" >/dev/null 2>&1
done
if grep -Eq 'Operand_B<[^>]*half' "${RESULTS}/micro_profile_baseline_stats_cuda_gpu_kern_sum.csv"; then exit 3; fi
if grep -Eq 'Operand_B<[^>]*half' "${RESULTS}/micro_profile_transposed_stats_cuda_gpu_kern_sum.csv"; then exit 3; fi
grep -Eq 'Operand_B<[^>]*half' "${RESULTS}/micro_profile_native_stats_cuda_gpu_kern_sum.csv"

audit_weights() {
    local log=$1 mode=$2
    python3 - "${log}" "${mode}" <<'PY'
import re,sys
text=open(sys.argv[1],errors='replace').read(); mode=int(sys.argv[2])
for device in range(4):
    ids=[int(x) for x in re.findall(rf"SM70_FP16_FLAT_GDN_WEIGHT device={device} index=(\d+) mode={mode} shape=5120x4120",text)]
    if ids != list(range(48)): raise SystemExit(f"device {device}: bad weight markers {ids}")
PY
}
validate_bench() {
    python3 - "$1" "$2" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); n=int(sys.argv[2]); rows=d.get('trials',[])
assert len(rows)==n and all(not r.get('degenerate') and r.get('output_tokens')==256 for r in rows)
PY
}
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" --num-draft-tokens 7
    --speculative-algorithm dflash2 --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
    --speculative-dflash-block-size 8 --speculative-draft-window 2048 --input-tokens 1000 --output-tokens 256
    --sglang-corpus /sglang-corpus --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
    --cache-max-entry-count 0.05)
for i in "${!arms[@]}"; do
    arm=${arms[$i]} mode=${modes[$i]}
    TM_SM70_FP16_FLAT_GDN="${mode}" python3 /job/bench_decode.py "${common[@]}" --trials 5 \
        --json-out "${RESULTS}/${arm}.json" 2>&1 | tee "${RESULTS}/${arm}.log"
    validate_bench "${RESULTS}/${arm}.json" 5
    if [ "${mode}" -ne 0 ]; then audit_weights "${RESULTS}/${arm}.log" "${mode}"; fi
    if [ "${mode}" -eq 2 ]; then
        for device in 0 1 2 3; do grep -q "SM70_FP16_FLAT_GDN_REGISTERED device=${device}" "${RESULTS}/${arm}.log"; done
    fi
done

export FT_NVTX=ON
for i in "${!arms[@]}"; do
    arm=${arms[$i]} mode=${modes[$i]}
    TM_SM70_FP16_FLAT_GDN="${mode}" "${NSYS}" profile --force-overwrite=true --trace=cuda,nvtx,osrt \
        --capture-range=cudaProfilerApi --capture-range-end=stop --output="${RESULTS}/profile_${arm}" \
        python3 /job/bench_decode.py "${common[@]}" --output-tokens 128 --trials 1 --cuda-profiler-range \
        --json-out "${RESULTS}/profile_${arm}.json" 2>&1 | tee "${RESULTS}/profile_${arm}.log"
    "${NSYS}" stats --report nvtx_sum,cuda_api_sum,cuda_gpu_kern_sum --format csv \
        --output "${RESULTS}/profile_${arm}_stats" "${RESULTS}/profile_${arm}.nsys-rep" >/dev/null 2>&1
    "${NSYS}" export --type sqlite --force-overwrite=true --output="${RESULTS}/profile_${arm}.sqlite" \
        "${RESULTS}/profile_${arm}.nsys-rep" >/dev/null 2>&1
done
unset FT_NVTX
python3 /src/tools/v100/analyze_dflash_fp16_flat_gdn.py "${RESULTS}" | tee "${RESULTS}/analysis.log"
WINNER="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"] or "")' "${RESULTS}/analysis.json")"
[ "${WINNER}" = native ] || {
    touch "${RESULTS}/completed"
    echo DFLASH_FP16_FLAT_GDN_REJECTED
    exit 4
}

set +e
TM_SM70_FP16_FLAT_GDN=2 python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 128 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    2>&1 | tee "${RESULTS}/identity_128.log"
identity128_rc=${PIPESTATUS[0]}
TM_SM70_FP16_FLAT_GDN=2 python3 /job/verify_dflash_audited.py \
    --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" \
    --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 256 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    2>&1 | tee "${RESULTS}/identity_256.log"
identity256_rc=${PIPESTATUS[0]}
set -e
python3 - "${RESULTS}" "${identity128_rc}" "${identity256_rc}" <<'PY'
import json,re,sys
root=sys.argv[1]; r128=int(sys.argv[2]); r256=int(sys.argv[3]); text=open(root+'/identity_256.log',errors='replace').read()
m=re.search(r'first_difference=(\d+)',text); pos=int(m.group(1)) if m else None
qualified=r128==0 and (r256==0 or pos in {145,220})
a=json.load(open(root+'/analysis.json')); a.update({"status":"qualified" if qualified else "rejected",
 "identity_128_rc":r128,"identity_256_rc":r256,"identity_256_failure_position":pos})
open(root+'/qualification.json','w').write(json.dumps(a,indent=2)+'\n')
print('DFLASH_FP16_FLAT_GDN_'+a['status'].upper(),json.dumps(a,sort_keys=True))
raise SystemExit(0 if qualified else 4)
PY
touch "${RESULTS}/completed"
echo DFLASH_FP16_FLAT_GDN_COMPLETE
