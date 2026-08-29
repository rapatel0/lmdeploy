#!/usr/bin/env bash
# Rebuilt-wheel smoke for default-on flat native SM70 FP16 GDN routing.
set -euo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-fp16-flat-gdn-default-${SRC_COMMIT:-unknown}
mkdir -p "${RESULTS}"
exec > >(tee -a "${RESULTS}/console.log") 2>&1
trap 'rc=$?; echo "$rc" >"${RESULTS}/exit_code"; echo "artifacts in ${RESULTS} (exit ${rc})"' EXIT
rm -f /wheels/lmdeploy-*.whl
started=$(date +%s)
bash /src/tools/v100/build_v100_fast.sh >"${RESULTS}/build.log" 2>&1 || { tail -100 "${RESULTS}/build.log"; exit 2; }
WHEEL="$(find /wheels -maxdepth 1 -name 'lmdeploy-*.whl' -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)"
[ "$(stat -c %Y "${WHEEL}")" -ge "${started}" ]
sha256sum "${WHEEL}" | tee "${RESULTS}/wheel.sha256"
pip install --no-deps --force-reinstall "${WHEEL}" 2>&1 | tail -1
NSYS="$(command -v nsys 2>/dev/null || true)"; [ -n "${NSYS}" ] || NSYS=/opt/nsys/nsys
common=(--model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" --tp "${TP:-4}" --num-draft-tokens 7
        --speculative-algorithm dflash2 --speculative-draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}"
        --speculative-dflash-block-size 8 --speculative-draft-window 2048 --input-tokens 1000 --output-tokens 128
        --sglang-corpus /sglang-corpus --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01
        --cache-max-entry-count 0.05)
export FT_NVTX=ON
"${NSYS}" profile --force-overwrite=true --trace=cuda,nvtx,osrt --capture-range=cudaProfilerApi --capture-range-end=stop \
    --output="${RESULTS}/default" python3 /job/bench_decode.py "${common[@]}" --trials 1 --cuda-profiler-range \
    --json-out "${RESULTS}/default.json" 2>&1 | tee "${RESULTS}/default.log"
"${NSYS}" export --type sqlite --force-overwrite=true --output="${RESULTS}/default.sqlite" \
    "${RESULTS}/default.nsys-rep" >/dev/null 2>&1
unset FT_NVTX
TM_SM70_FP16_FLAT_GDN=0 python3 /job/bench_decode.py "${common[@]}" --trials 1 \
    --json-out "${RESULTS}/legacy.json" 2>&1 | tee "${RESULTS}/legacy.log"
python3 - "${RESULTS}" <<'PY'
import json,sqlite3,sys
from pathlib import Path
root=Path(sys.argv[1]); text=(root/'default.log').read_text(errors='replace')
for device in range(4):
    marker=f'SM70_FP16_FLAT_GDN_ACTIVE device={device} mode=2 shape=5120x4120'
    if text.count(marker)!=1: raise SystemExit(f'missing default marker: {marker}')
if 'SM70_FP16_FLAT_GDN_ACTIVE' in (root/'legacy.log').read_text(errors='replace'):
    raise SystemExit('legacy arm unexpectedly activated flat GDN')
db=sqlite3.connect(root/'default.sqlite')
ranges=dict(db.execute("""SELECT k.deviceId,COUNT(DISTINCT n.rowid) FROM NVTX_EVENTS n
 JOIN StringIds nt ON nt.id=n.textId JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
 ON r.globalTid=n.globalTid AND r.start BETWEEN n.start AND n.end
 JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=r.correlationId
 AND k.globalPid=(r.globalTid & -16777216) WHERE nt.value='targetVerify' GROUP BY k.deviceId"""))
rows=db.execute("""SELECT k.deviceId,s.value,COUNT(*) FROM NVTX_EVENTS n
 JOIN StringIds nt ON nt.id=n.textId JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
 ON r.globalTid=n.globalTid AND r.start BETWEEN n.start AND n.end
 JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=r.correlationId
 AND k.globalPid=(r.globalTid & -16777216) JOIN StringIds s ON s.id=k.demangledName
 WHERE nt.value='targetVerify' AND ((lower(s.value) LIKE '%gemm_kernel%' AND lower(s.value) LIKE '%operand_b<%'
 AND lower(s.value) LIKE '%half%' AND lower(s.value) NOT LIKE '%operand_b_pack%') OR
 (lower(s.value) LIKE '%cutlass%gemm%' AND ((k.gridX=8 AND k.gridY=33) OR (k.gridX=33 AND k.gridY=8))))
 GROUP BY k.deviceId,s.value""").fetchall(); db.close()
native={}; fallback=0
for device,name,count in rows:
    if 'turbomind::gemm::gemm_kernel' in name: native[device]=native.get(device,0)+count
    else: fallback+=count
expected={device:count*48 for device,count in ranges.items()}
if native!=expected or fallback: raise SystemExit(f'route failure native={native} expected={expected} fallback={fallback}')
def metric(name):
    d=json.loads((root/f'{name}.json').read_text()); return d['mean_decode_tok_s']
result={'default_decode_tok_s':metric('default'),'legacy_decode_tok_s':metric('legacy'),
        'native_launches_by_device':native,'target_verify_ranges_by_device':ranges,'fallback_launches':fallback}
(root/'smoke.json').write_text(json.dumps(result,indent=2)+'\n')
print('DFLASH_FP16_FLAT_GDN_DEFAULT_ROUTE_PASS',json.dumps(result,sort_keys=True))
PY
python3 /job/verify_dflash_audited.py --model "${MODEL_DIR:-/models/Qwen3.8-27B-FP8}" \
    --draft-model "${DFLASH_MODEL_DIR:-/models/Qwen3.8-27B-DFlash2}" --corpus /sglang-corpus --tp "${TP:-4}" \
    --input-tokens 1000 --output-tokens 128 \
    --expected-prompt-sha256 9ac441c0409e992b270fbe9cb47ca11bf00f66dc903dcd0fd32ad00b70007a01 \
    2>&1 | tee "${RESULTS}/identity_128.log"
grep -q 'DFLASH_AUDITED_IDENTITY_PASS tokens=128' "${RESULTS}/identity_128.log"
touch "${RESULTS}/completed"
echo DFLASH_FP16_FLAT_GDN_DEFAULT_COMPLETE
