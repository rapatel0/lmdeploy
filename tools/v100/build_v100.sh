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
# The script pins the architecture to 70-real.
#
# Exporting CMAKE_CUDA_ARCHITECTURES alone does NOT work. `python -m build`
# creates an isolated build environment, and setup.py passes an explicit
# cmake_configure_options list that omits the architecture. A verified build
# proved this: nvcc emitted seven --generate-code flags and the wheel carried
# sm_70, sm_75, sm_80, sm_86, sm_89, sm_90a, sm_100a, and sm_120a.
#
# CMAKE_ARGS does not work either. cmake_build_extension 0.6.1 builds its
# argument list from cmake_configure_options and its own -D options, and it
# never reads CMAKE_ARGS.
#
# CUDAARCHS is the correct channel. CMake reads that environment variable
# natively when it enables the CUDA language, and it survives the isolated
# build environment. CMakeLists.txt guards its default list with
# `if (NOT CMAKE_CUDA_ARCHITECTURES)`, so a value from CUDAARCHS suppresses
# the seven-architecture default.
#
# The script verifies the result instead of trusting it.
set -euo pipefail

cd /src

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

# Remove any stale build tree. A CMakeCache.txt from an earlier pod pins
# absolute paths such as Python3_ROOT_DIR to a temporary directory that no
# longer exists, so a retry fails on a cache that still looks valid. The build
# must be idempotent, because Kubernetes retries a failed Job.
if [ -d build ]; then
    echo "removing a stale build tree"
    rm -rf build
fi
rm -rf lmdeploy.egg-info

# CUDAARCHS is the channel CMake reads natively.
export CUDAARCHS="${ARCH_TARGET}"
# Keep this for any consumer that reads it directly. It is not sufficient on
# its own through `python -m build`.
export CMAKE_CUDA_ARCHITECTURES="${ARCH_TARGET}"
export CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

echo "CUDAARCHS=${CUDAARCHS}"

python3 -m build -w -o "${WHEEL_DIR}" -v .

echo "=== verify the architecture pin ==="

# Trust nothing. Read the generated ninja file and confirm that nvcc received
# exactly one architecture.
# Scan EVERY ninja file under the turbomind build tree, not just one. A
# `find | head -1` picks an arbitrary dependency subbuild such as
# concurrentqueue-subbuild, which contains no CUDA flags at all and produces a
# false failure. Aggregating across all files also catches a dependency that
# compiles for an unwanted architecture.
BUILD_ROOT="$(find build -maxdepth 1 -type d -name '*__turbomind*' 2>/dev/null | head -1)"
if [ -z "${BUILD_ROOT}" ]; then
    echo "FAIL: no turbomind build tree found, cannot verify the pin" >&2
    exit 1
fi

ARCHES="$(find "${BUILD_ROOT}" -name build.ninja -exec \
    grep -ohE 'arch=compute_[0-9]+a?' {} + 2>/dev/null \
    | sort -u | sed 's/.*compute_//')"
COUNT="$(printf '%s\n' "${ARCHES}" | grep -c . || true)"

echo "architectures compiled: ${ARCHES:-none} (count ${COUNT})"

if [ "${COUNT}" -eq 0 ]; then
    echo "FAIL: no CUDA architecture flags found, cannot confirm the pin" >&2
    exit 1
fi
if [ "${COUNT}" -ne 1 ] || [ "${ARCHES}" != "70" ]; then
    echo "FAIL: expected exactly compute_70, found '${ARCHES}'" >&2
    echo "the CUDAARCHS architecture pin did not take effect" >&2
    exit 1
fi
echo "PASS: single architecture, compute_70"

echo "=== verify SM70 machine code ==="

# A PTX-only build is not acceptable. Every retained CUDA library must contain
# real SM70 machine code, so cuobjdump must report an sm_70 ELF section.
bash "$(dirname "$0")/verify_sm70.sh" "${WHEEL_DIR}"
