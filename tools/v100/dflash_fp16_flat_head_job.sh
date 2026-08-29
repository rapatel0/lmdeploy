#!/usr/bin/env bash
# One-build qualification of transposed/native SM70 FP16 vocabulary-head routing.
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-fp16-flat-head-${SRC_COMMIT:-unknown}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"${RESULTS}/exit_code"; echo "artifacts in ${RESULTS} (exit ${rc})"' EXIT
for f in bench_decode.py verify_dflash_audited.py dflash_fp16_flat_head_microbench.cu; do [ -f "/job/${f}" ] || exit 2; done
rm -f /wheels/lmdeploy-*.whl
started=$(date +%s)
bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || {
   grep -aE 'error:|Error [0-9]+' "${RESULTS}/build.log" | head -100
   exit 2
}
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ "$(stat -c %Y "${WHEEL}")" -ge "${started}" ]
sha256sum "${WHEEL}" | tee "${RESULTS}/wheel.sha256"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
NSYS="$(command -v nsys 2>/dev/null || true)"
[ -n "${NSYS}" ] || NSYS=/opt/nsys/nsys
NVCC="$(command -v nvcc)"
LIB="$(
   python3 - <<'PY'
import glob,lmdeploy,os
r=os.path.dirname(lmdeploy.__file__); x=glob.glob(r+'/**/_turbomind*.so',recursive=True)+glob.glob(r+'/**/libturbomind.so',recursive=True)
print(x[0] if x else '')
PY
)"
LIBDIR="$(dirname "${LIB}")"
FMT_HEADER="$(find /src/build -path '*/fmt/format.h' -print -quit)"
FMT_INCLUDE="${FMT_HEADER%/fmt/format.h}"
PYLIB="$(find /usr /opt -name 'libpython3*.so*' -print -quit 2>/dev/null)"
[ -n "${LIB}" ] && [ -n "${FMT_HEADER}" ] && [ -n "${PYLIB}" ]
"${NVCC}" -std=c++17 -O2 -lineinfo -gencode arch=compute_70,code=sm_70 -I/src -I"${FMT_INCLUDE}" \
   /job/dflash_fp16_flat_head_microbench.cu "${LIB}" -Xlinker -rpath -Xlinker "${LIBDIR}" \
   -lcudart -Xlinker "${PYLIB}" -o /tmp/dflash_fp16_flat_head_microbench >"${RESULTS}/compile_microbench.log" 2>&1 || {
   tail -100 "${RESULTS}/compile_microbench.log"
   exit 2
}

arms=(baseline transposed native)
modes=(0 1 2)
export TM_GEMM_TUNE='top_k=0,clusters=0,min_iter=2,max_iter=10,max_time=2.0'
export TM_GEMM_TUNE_VERBOSE=1
for i in "${!arms[@]}"; do
   arm=${arms[$i]}
   mode=${modes[$i]}
   mkdir -p "${RESULTS}/micro_${arm}"
   TM_SM70_FP16_FLAT_GDN=0 TM_SM70_FP16_FLAT_HEAD="${mode}" TM_FP16_HEAD_DUMP_DIR="${RESULTS}/micro_${arm}" \
      /tmp/dflash_fp16_flat_head_microbench 2>&1 | tee "${RESULTS}/micro_${arm}.log"
   grep -q "SM70_FP16_FLAT_HEAD_MICRO_PASS mode=${mode}" "${RESULTS}/micro_${arm}.log"
done
python3 - "${RESULTS}" <<'PY'
import json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); report={}
for m in (1,7,8):
 a={x:np.fromfile(root/f'micro_{x}/m{m}.bin',dtype=np.float16).astype(np.float32) for x in ('baseline','transposed','native')}
 for lhs,rhs in (('baseline','transposed'),('transposed','native')):
  d=a[lhs]-a[rhs]; row={'max_abs':float(np.max(np.abs(d))),'rms':float(np.sqrt(np.mean(d*d))),'differing':int(np.count_nonzero(d)),'finite':bool(np.isfinite(a[lhs]).all() and np.isfinite(a[rhs]).all())}
  report[f'm{m}_{lhs}_vs_{rhs}']=row
  if not row['finite'] or row['max_abs']>0.125 or row['rms']>0.01: raise SystemExit(f'HEAD_MICRO_PARITY_FAIL {m} {lhs} {rhs} {row}')
(root/'micro_parity.json').write_text(json.dumps(report,indent=2)+'\n'); print('FP16_FLAT_HEAD_MICRO_PARITY_PASS',json.dumps(report,sort_keys=True))
PY
for i in "${!arms[@]}"; do
    arm=${arms[$i]}; mode=${modes[$i]}
    TM_SM70_FP16_FLAT_GDN=0 TM_SM70_FP16_FLAT_HEAD="${mode}" "${NSYS}" profile --force-overwrite=true \
        --trace=cuda --output="${RESULTS}/micro_profile_${arm}" /tmp/dflash_fp16_flat_head_microbench \
        2>&1 | tee "${RESULTS}/micro_profile_${arm}.log"
    "${NSYS}" stats --report cuda_gpu_kern_sum --format csv --output "${RESULTS}/micro_profile_${arm}_stats" \
        "${RESULTS}/micro_profile_${arm}.nsys-rep" >/dev/null 2>&1
done
if grep -Eq 'Operand_B<[^>]*half' "${RESULTS}/micro_profile_baseline_stats_cuda_gpu_kern_sum.csv"; then exit 3; fi
if grep -Eq 'Operand_B<[^>]*half' "${RESULTS}/micro_profile_transposed_stats_cuda_gpu_kern_sum.csv"; then exit 3; fi
grep -Eq 'Operand_B<[^>]*half' "${RESULTS}/micro_profile_native_stats_cuda_gpu_kern_sum.csv"

common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" --num-draft-tokens 7 --speculative-algorithm dflash2
   --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" --speculative-dflash-block-size 8 --speculative-draft-window 2048
   --input-tokens 1000 --output-tokens 256 --sglang-corpus /sglang-corpus --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 --cache-max-entry-count 0.05)
validate() {
   python3 - "$1" "$2" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); n=int(sys.argv[2]); assert len(d['trials'])==n and all(not x.get('degenerate') and x.get('output_tokens')==256 for x in d['trials'])
PY
}
for i in "${!arms[@]}"; do
   arm=${arms[$i]}
   mode=${modes[$i]}
   TM_SM70_FP16_FLAT_GDN=0 TM_SM70_FP16_FLAT_HEAD="${mode}" python3 /job/bench_decode.py "${common[@]}" --trials 5 --json-out "${RESULTS}/${arm}.json" 2>&1 | tee "${RESULTS}/${arm}.log"
   validate "${RESULTS}/${arm}.json" 5
   if [ "${mode}" -ne 0 ]; then for d in 0 1 2 3; do grep -q "SM70_FP16_FLAT_HEAD_ACTIVE device=${d} mode=${mode} shape=5120x62080" "${RESULTS}/${arm}.log"; done; fi
done
export FT_NVTX=ON
for i in "${!arms[@]}"; do
   arm=${arms[$i]}
   mode=${modes[$i]}
   TM_SM70_FP16_FLAT_GDN=0 TM_SM70_FP16_FLAT_HEAD="${mode}" "${NSYS}" profile --force-overwrite=true --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop --output="${RESULTS}/profile_${arm}" \
      python3 /job/bench_decode.py "${common[@]}" --output-tokens 128 --trials 1 --cuda-profiler-range --json-out "${RESULTS}/profile_${arm}.json" 2>&1 | tee "${RESULTS}/profile_${arm}.log"
   "${NSYS}" export --type sqlite --force-overwrite=true --output="${RESULTS}/profile_${arm}.sqlite" "${RESULTS}/profile_${arm}.nsys-rep" >/dev/null 2>&1
done
unset FT_NVTX
python3 /src/tools/v100/analyze_dflash_fp16_flat_head.py "${RESULTS}" | tee "${RESULTS}/analysis.log"
[ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["winner"] or "")' "${RESULTS}/analysis.json")" = native ] || exit 4

# Current production baseline versus both qualified flat catalogs enabled.
for arm in production combined; do
   if [ "${arm}" = production ]; then head=0; else head=2; fi
   TM_SM70_FP16_FLAT_GDN=2 TM_SM70_FP16_FLAT_HEAD="${head}" python3 /job/bench_decode.py "${common[@]}" --trials 5 --json-out "${RESULTS}/${arm}.json" 2>&1 | tee "${RESULTS}/${arm}.log"
   validate "${RESULTS}/${arm}.json" 5
done
python3 - "${RESULTS}" <<'PY'
import json,re,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]); pat=re.compile(r'final commit length ([0-9.]+), raw [0-9.]+ over (\d+) verification steps')
def cycle(name):
 d=json.loads((root/f'{name}.json').read_text()); c=statistics.median(float(x[0]) for x in pat.findall((root/f'{name}.log').read_text(errors='replace')))
 return 1000*c/float(d['mean_decode_tok_s'])
a=cycle('production'); b=cycle('combined'); pct=100*(b/a-1)
(root/'production_comparison.json').write_text(json.dumps({'production_cycle_ms':a,'combined_cycle_ms':b,'combined_cycle_pct':pct},indent=2)+'\n')
print(f'FP16_FLAT_HEAD_PRODUCTION_COMPARISON combined_cycle_pct={pct:.3f}')
if pct > -1.0: raise SystemExit('combined production gain below 1%')
PY
for arm in native combined; do
   if [ "${arm}" = native ]; then gdn=0; else gdn=2; fi
   TM_SM70_FP16_FLAT_GDN="${gdn}" TM_SM70_FP16_FLAT_HEAD=2 python3 /job/verify_dflash_audited.py --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" --corpus /sglang-corpus --tp "${TP:-4}" --input-tokens 1000 --output-tokens 128 --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 2>&1 | tee "${RESULTS}/identity_${arm}.log"
   grep -q '^DFLASH_AUDITED_IDENTITY_PASS' "${RESULTS}/identity_${arm}.log"
done
touch "${RESULTS}/completed"
echo DFLASH_FP16_FLAT_HEAD_COMPLETE
