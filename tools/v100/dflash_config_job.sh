#!/usr/bin/env bash
# Build and validate the first DFlash2 configuration slice on the V100 image.
set -uo pipefail
SRC_COMMIT="$(sed -n 's/^commit=\(.\{12\}\).*/\1/p' /src/SOURCE_STAMP 2>/dev/null)"
[ -n "${SRC_COMMIT}" ] || SRC_COMMIT=unknown
RESULTS=/results/$(date +%Y%m%d_%H%M%S)-dflash-config-${SRC_COMMIT}
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
cd /src || exit $?
python3 - <<'PY' | tee "${RESULTS}/config-test.log"
from lmdeploy.messages import TurbomindEngineConfig
import _turbomind as tm

cfg = TurbomindEngineConfig(
    num_draft_tokens=7,
    speculative_algorithm='dflash2',
    speculative_draft_model='/models/Qwen3.8-27B-DFlash2',
)
assert cfg.num_draft_tokens == cfg.speculative_dflash_block_size - 1
for kwargs in (
    dict(speculative_algorithm='unknown'),
    dict(speculative_algorithm='dflash2'),
    dict(speculative_algorithm='dflash2', speculative_draft_model='/draft', num_draft_tokens=4),
):
    try:
        TurbomindEngineConfig(**kwargs)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError(f'invalid config was accepted: {kwargs}')

ec = tm.EngineConfig()
ec.speculative_algorithm = cfg.speculative_algorithm
ec.speculative_draft_model = cfg.speculative_draft_model
ec.speculative_dflash_block_size = cfg.speculative_dflash_block_size
ec.speculative_draft_window = cfg.speculative_draft_window
assert ec.speculative_algorithm == 'dflash2'
assert ec.speculative_draft_model.endswith('Qwen3.8-27B-DFlash2')
assert ec.speculative_dflash_block_size == 8
assert ec.speculative_draft_window == 2048
print('DFLASH_NATIVE_CONFIG_PASS')
PY
[ "${PIPESTATUS[0]}" -eq 0 ] || exit 3
echo DFLASH_CONFIG_COMPLETE
