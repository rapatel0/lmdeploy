#!/bin/bash
# Verify that a built wheel contains real SM70 machine code.
#
# The master specification states two rules:
#   - Verify that each retained CUDA library contains SM70 machine code.
#   - Do not accept a PTX-only SM70 build.
#
# PTX is forward-compatible source that the driver compiles at load time. A
# PTX-only wheel starts on a V100, but it hides a missing SM70 target and it
# pays a compile cost on every load. This script fails closed when it finds
# PTX for SM70 without a matching ELF section.
set -euo pipefail

TARGET="${1:-/wheels}"
SM="${SM:-70}"

if ! command -v cuobjdump >/dev/null 2>&1; then
    echo "FAIL: cuobjdump not found, cannot verify machine code" >&2
    exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

shopt -s nullglob

# Accept a wheel, a directory of wheels, or a directory of shared objects.
LIBS=()
if [ -d "${TARGET}" ]; then
    WHEELS=("${TARGET}"/*.whl)
    if [ "${#WHEELS[@]}" -gt 0 ]; then
        for wheel in "${WHEELS[@]}"; do
            echo "unpacking $(basename "${wheel}")"
            # A wheel is a zip, but `unzip` is absent from the slim CUDA build
            # image while Python is always present. A missing unzip previously
            # aborted this script with rc=127 *after* a successful build, which
            # looks like a build failure and is not one.
            python3 -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' \
                "${wheel}" "${WORK}/unpacked"
        done
        while IFS= read -r lib; do LIBS+=("${lib}"); done \
            < <(find "${WORK}/unpacked" -name '*.so' -o -name '*.so.*')
    else
        while IFS= read -r lib; do LIBS+=("${lib}"); done \
            < <(find "${TARGET}" -name '*.so' -o -name '*.so.*')
    fi
elif [ -f "${TARGET}" ]; then
    LIBS=("${TARGET}")
else
    echo "FAIL: ${TARGET} not found" >&2
    exit 1
fi

if [ "${#LIBS[@]}" -eq 0 ]; then
    echo "FAIL: no shared library found under ${TARGET}" >&2
    exit 1
fi

echo "=== scanning ${#LIBS[@]} libraries for sm_${SM} machine code ==="

FOUND_ELF=0
PTX_ONLY=()

for lib in "${LIBS[@]}"; do
    name="$(basename "${lib}")"

    # cuobjdump exits non-zero on a library with no device code. That is not
    # an error, because many libraries are host-only.
    if ! elf="$(cuobjdump -lelf "${lib}" 2>/dev/null)"; then
        continue
    fi
    ptx="$(cuobjdump -lptx "${lib}" 2>/dev/null || true)"

    has_elf=0
    has_ptx=0
    [[ "${elf}" == *"sm_${SM}"* ]] && has_elf=1
    [[ "${ptx}" == *"sm_${SM}"* ]] && has_ptx=1

    if [ "${has_elf}" -eq 1 ]; then
        FOUND_ELF=$((FOUND_ELF + 1))
        count="$(echo "${elf}" | grep -c "sm_${SM}" || true)"
        echo "  OK   ${name}: ${count} sm_${SM} ELF sections"
    elif [ "${has_ptx}" -eq 1 ]; then
        PTX_ONLY+=("${name}")
        echo "  FAIL ${name}: sm_${SM} PTX present, no ELF"
    fi
done

echo

if [ "${#PTX_ONLY[@]}" -gt 0 ]; then
    echo "FAIL: ${#PTX_ONLY[@]} libraries carry PTX-only sm_${SM} code" >&2
    printf '  %s\n' "${PTX_ONLY[@]}" >&2
    echo "the specification forbids a PTX-only SM70 build" >&2
    exit 1
fi

if [ "${FOUND_ELF}" -eq 0 ]; then
    echo "FAIL: no library contains sm_${SM} machine code" >&2
    exit 1
fi

echo "PASS: ${FOUND_ELF} libraries contain sm_${SM} machine code"
