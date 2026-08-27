#!/bin/bash
# Incremental V100 build for the debug loop.
#
# build_v100.sh is the release path: clean tree, isolated PEP 517 env,
# full architecture audit. It costs ~25 minutes of nvcc per run, and the
# debug loop was paying that for one changed .cc file.
#
# Why the release path cannot be incremental:
#   `python -m build` creates its isolated env under /tmp with a random
#   name. CMakeCache.txt pins Python3_ROOT_DIR to that path. The next run
#   gets a different path, so the cache is stale by construction. The
#   `rm -rf build` in build_v100.sh treats the symptom.
#
# This script removes the cause instead:
#   - --no-build-isolation: the interpreter is the container's own
#     python3, whose path is identical across runs of the same image.
#   - the build/ tree is KEPT. /src is a hostPath mount, so ninja state
#     survives pod restarts. A one-file change recompiles one file.
#   - ccache in front of nvcc and the host compiler. This is the layer
#     that survives what the other two cannot: sync_src.sh deletes the
#     whole source tree on the node (rm -rf /dest/lmdeploy, a deliberate
#     anti-staleness measure), which destroys build/ with it. ccache
#     keys on preprocessed content, not paths or mtimes, so even a
#     from-scratch configure recompiles only the files whose content
#     changed. CCACHE_DIR must point OUTSIDE the source tree.
#
# CMake reads CMAKE_CUDA_COMPILER_LAUNCHER / CMAKE_CXX_COMPILER_LAUNCHER
# from the environment natively (same mechanism as CUDAARCHS), so the
# launcher survives cmake_build_extension's fixed argument list.
#
# Fallback: if configure fails (image changed, toolchain moved, cache
# genuinely stale), wipe build/ and retry once, clean. With a warm
# ccache the clean retry costs minutes, not 25.
#
# The SM70 pin is still verified on every run. The check reads the ninja
# files and the wheel; it does not need a clean tree to be meaningful.
set -euo pipefail

ARCH_TARGET="${ARCH_TARGET:-70-real}"
WHEEL_DIR="${WHEEL_DIR:-/wheels}"

# Self-locating: build the tree this script lives in, whatever the caller's
# cwd. The first caller that forgot `cd /src` produced
#   ERROR Source . does not appear to be a Python project
# and burned a GPU job on it. The script knows where its repo root is;
# requiring every caller to know it too is one invariant per caller instead
# of one total.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
[ -f pyproject.toml ] || [ -f setup.py ] || {
    echo "FAIL: $(pwd) has no pyproject.toml/setup.py; script moved?" >&2
    exit 1
}

echo "=== toolchain check ==="
command -v nvcc >/dev/null 2>&1 || { echo "FAIL: nvcc not found" >&2; exit 1; }
CUDA_MAJOR="$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
[ -n "${CUDA_MAJOR}" ] || { echo "FAIL: cannot parse the CUDA version" >&2; exit 1; }
if [ "${CUDA_MAJOR}" -ge 13 ] || [ "${CUDA_MAJOR}" -lt 12 ]; then
    echo "FAIL: CUDA major ${CUDA_MAJOR} is not the pinned 12.x" >&2
    exit 1
fi

# Build deps that the isolated env used to provide. Idempotent and fast
# when already satisfied; kept explicit so --no-build-isolation cannot
# fail on a missing backend halfway into a debug session.
pip install --quiet build cmake ninja wheel setuptools cmake_build_extension pybind11 nvidia-nccl-cu12

# ccache: install if the image lacks it, then put it in front of both
# compilers. CCACHE_DIR defaults to a hostPath-backed location so the
# cache outlives the pod AND the source tree (sync_src.sh deletes the
# latter). If ccache cannot be obtained the build still works, just slow.
export CCACHE_DIR="${CCACHE_DIR:-/wheels/.ccache}"
if ! command -v ccache >/dev/null 2>&1; then
    { apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq ccache >/dev/null 2>&1; } \
        || conda install -y -q ccache >/dev/null 2>&1 \
        || pip install --quiet ccache-bin >/dev/null 2>&1 \
        || echo "WARN: ccache unavailable; building without compile cache" >&2
fi
if command -v ccache >/dev/null 2>&1; then
    mkdir -p "${CCACHE_DIR}"
    ccache --set-config max_size=20G
    ccache --set-config sloppiness=include_file_ctime,include_file_mtime,time_macros
    export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    ccache --zero-stats >/dev/null
fi

export CUDAARCHS="${ARCH_TARGET}"
export CMAKE_CUDA_ARCHITECTURES="${ARCH_TARGET}"
export CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"

mkdir -p "${WHEEL_DIR}"

build_once() {
    python3 -m build -w -o "${WHEEL_DIR}" --no-isolation -v .
}

echo "=== build (incremental) ==="
if ! build_once; then
    # A stale cache fails at configure time, within seconds. Anything that
    # fails after configure is a real compile error, and a clean retry
    # would burn 25 minutes to report the same diagnostic -- so only a
    # missing/poisoned cache justifies the retry.
    if [ -d build ] && ! grep -qs "Python3_ROOT_DIR" build/*/CMakeCache.txt 2>/dev/null; then
        echo "configure failed with no usable cache; retrying clean" >&2
    else
        echo "incremental build failed; retrying clean once" >&2
    fi
    rm -rf build lmdeploy.egg-info
    build_once
fi

if command -v ccache >/dev/null 2>&1; then
    echo "=== ccache stats for this build ==="
    ccache --show-stats | head -8
fi

echo "=== verify the architecture pin ==="
BUILD_ROOT="$(find build -maxdepth 1 -type d -name '*__turbomind*' 2>/dev/null | head -1)"
[ -n "${BUILD_ROOT}" ] || { echo "FAIL: no turbomind build tree found" >&2; exit 1; }
ARCHES="$(find "${BUILD_ROOT}" -name build.ninja -exec \
    grep -ohE 'arch=compute_[0-9]+a?' {} + 2>/dev/null | sort -u | sed 's/.*compute_//')"
COUNT="$(printf '%s\n' "${ARCHES}" | grep -c . || true)"
echo "architectures compiled: ${ARCHES:-none} (count ${COUNT})"
if [ "${COUNT}" -ne 1 ] || [ "${ARCHES}" != "70" ]; then
    echo "FAIL: expected exactly compute_70, found '${ARCHES}'" >&2
    exit 1
fi
echo "PASS: single architecture, compute_70"

echo "=== verify SM70 machine code ==="
bash "$(dirname "$0")/verify_sm70.sh" "${WHEEL_DIR}"
