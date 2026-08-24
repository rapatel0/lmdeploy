#!/bin/bash
# Build LMDeploy for V100 only.
#
# The master specification pins CUDA 12.8.1 and SM70. Read
# docs/v100/toolchain.md before you change any value here.
#
# CMakeLists.txt drops 70-real when the CUDA compiler version is 13 or later.
# This script fails closed on CUDA 13 rather than build a wheel without SM70
# machine code.
#
# The script pins CMAKE_CUDA_ARCHITECTURES to 70-real. setup.py passes no
# architecture flag, so this environment value reaches CMake and replaces the
# default seven-architecture list. That cuts build time and guarantees that
# the wheel targets V100 only.
set -euo pipefail

ARCH_TARGET="${ARCH_TARGET:-70-real}"
WHEEL_DIR="${WHEEL_DIR:-/wheels}"

echo "=== toolchain check ==="

if ! command -v nvcc >/dev/null 2>&1; then
    echo "FAIL: nvcc not found" >&2
    exit 1
fi

nvcc --version | tail -2

CUDA_MAJOR="$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
CUDA_FULL="$(nvcc --version | sed -n 's/.*release \([0-9.]*\),.*/\1/p')"

if [ -z "${CUDA_MAJOR}" ]; then
    echo "FAIL: cannot parse the CUDA version" >&2
    exit 1
fi

# Fail closed. CUDA 13 removes SM70 support from the default list.
if [ "${CUDA_MAJOR}" -ge 13 ]; then
    echo "FAIL: CUDA ${CUDA_FULL} does not support SM70" >&2
    echo "the specification requires CUDA 12.8.1" >&2
    exit 1
fi

if [ "${CUDA_MAJOR}" -lt 12 ]; then
    echo "FAIL: CUDA ${CUDA_FULL} is older than the pinned 12.8.1" >&2
    exit 1
fi

echo "CUDA ${CUDA_FULL} accepted"
echo "target architecture: ${ARCH_TARGET}"

mkdir -p "${WHEEL_DIR}"

if [[ "${CUDA_VERSION_SHORT:-cu128}" == cu13* ]]; then
    echo "FAIL: CUDA_VERSION_SHORT reports a CUDA 13 runtime" >&2
    exit 1
fi

pip install nvidia-nccl-cu12

echo "=== build ==="

export CMAKE_CUDA_ARCHITECTURES="${ARCH_TARGET}"
export CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

python3 -m build -w -o "${WHEEL_DIR}" -v .

echo "=== verify SM70 machine code ==="

# A PTX-only build is not acceptable. Every retained CUDA library must contain
# real SM70 machine code, so cuobjdump must report an sm_70 ELF section.
bash "$(dirname "$0")/verify_sm70.sh" "${WHEEL_DIR}"
