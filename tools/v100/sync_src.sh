#!/bin/bash
# Sync the working tree to /localpool/lmdeploy-v100-next/src/lmdeploy on gpu-01.
#
# Why this script exists.
#
# The staged source and the repository drifted apart, and nothing detected it.
# A wheel built at 13:01 from a stale tree produced only exclamation marks, and
# that was read as an FP8 numerics fault for most of a day. The fix was already
# committed at 08:10.
#
# A later job patched tools/v100/verify_sm70.sh directly on the node. The staged
# file then matched the repository by content while sharing no history with it,
# so an inspection of the node looked like a successful sync and was not one.
#
# This script makes the sync explicit and verifiable. It refuses to run from a
# dirty tree, records the commit it shipped, and reads back a marker so the
# caller learns what actually landed rather than what was intended.
set -euo pipefail

NS="${NS:-llm}"
NODE="${NODE:-gpu-01}"
DEST="${DEST:-/localpool/lmdeploy-v100-next/src}"
REPO_ROOT="$(git rev-parse --show-toplevel)"

cd "${REPO_ROOT}"

# Fail closed on a dirty tree. Shipping uncommitted work produces a node state
# that no commit describes, which is the condition this script exists to end.
if [ -n "$(git status --porcelain)" ]; then
    echo "FAIL: the working tree is dirty" >&2
    echo "commit or stash before syncing, so the node matches a known commit" >&2
    git status --short >&2
    exit 1
fi

SHA="$(git rev-parse HEAD)"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "=== syncing ${BRANCH} at ${SHA:0:12} to ${NODE}:${DEST}/lmdeploy ==="

# git archive ships exactly the tracked content of one commit. It cannot carry
# an uncommitted edit or a stray build artifact, unlike a recursive copy.
TARBALL="$(mktemp -t lmdeploy-src-XXXXXX.tar)"
trap 'rm -f "${TARBALL}"' EXIT
git archive --format=tar HEAD >"${TARBALL}"

# Stamp the commit into the payload so the node can be interrogated later.
STAMP_DIR="$(mktemp -d)"
trap 'rm -f "${TARBALL}"; rm -rf "${STAMP_DIR}"' EXIT
{
    echo "commit=${SHA}"
    echo "branch=${BRANCH}"
    echo "synced_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "synced_from=$(hostname)"
} >"${STAMP_DIR}/SOURCE_STAMP"
tar -rf "${TARBALL}" -C "${STAMP_DIR}" SOURCE_STAMP

SIZE="$(wc -c <"${TARBALL}" | tr -d ' ')"
echo "payload: ${SIZE} bytes"

POD="src-sync-$$"
cleanup_pod() { kubectl -n "${NS}" delete pod "${POD}" --ignore-not-found >/dev/null 2>&1 || true; }
trap 'rm -f "${TARBALL}"; rm -rf "${STAMP_DIR}"; cleanup_pod' EXIT

# A long-lived pod, because kubectl cp needs a running target.
kubectl -n "${NS}" run "${POD}" \
    --restart=Never \
    --image=busybox:1.36 \
    --overrides="{\"spec\":{\"nodeName\":\"${NODE}\",\"containers\":[{\"name\":\"c\",\"image\":\"busybox:1.36\",\"command\":[\"sleep\",\"600\"],\"volumeMounts\":[{\"name\":\"s\",\"mountPath\":\"/dest\"}]}],\"volumes\":[{\"name\":\"s\",\"hostPath\":{\"path\":\"${DEST}\",\"type\":\"DirectoryOrCreate\"}}]}}" \
    >/dev/null

kubectl -n "${NS}" wait --for=condition=Ready "pod/${POD}" --timeout=120s >/dev/null

# Replace rather than merge. A merge leaves files that the commit deleted, and
# a stale leftover is precisely the failure this script prevents.
# Refuse to sync while a job is building from this tree.
#
# The next line deletes /dest/lmdeploy outright. Doing that under a running
# build makes the compiler fail on files that vanished beneath it:
#
#   Fatal error: can't create .../kv_cache_utils_v2.cu.o: No such file or directory
#
# which looks exactly like a code defect in the job's log and cost two GPU
# runs to recognise as a race with my own sync.
if ACTIVE=$(kubectl -n "${NS}" get jobs -o jsonpath='{range .items[?(@.status.active)]}{.metadata.name} {end}' 2>/dev/null) \
    && [ -n "${ACTIVE// /}" ]; then
    echo "REFUSING: jobs still active: ${ACTIVE}" >&2
    echo "  Wait for them or delete them; syncing now would delete the tree they are building." >&2
    exit 3
fi

kubectl -n "${NS}" exec "${POD}" -- sh -c 'rm -rf /dest/lmdeploy && mkdir -p /dest/lmdeploy'
kubectl cp "${TARBALL}" "${NS}/${POD}:/tmp/src.tar"
kubectl -n "${NS}" exec "${POD}" -- sh -c 'tar -xf /tmp/src.tar -C /dest/lmdeploy && rm -f /tmp/src.tar'

echo "=== read back what landed ==="
kubectl -n "${NS}" exec "${POD}" -- cat /dest/lmdeploy/SOURCE_STAMP

LANDED="$(kubectl -n "${NS}" exec "${POD}" -- sh -c \
    "sed -n 's/^commit=//p' /dest/lmdeploy/SOURCE_STAMP" | tr -d '\r\n')"

if [ "${LANDED}" != "${SHA}" ]; then
    echo "FAIL: node reports ${LANDED}, expected ${SHA}" >&2
    exit 1
fi

echo "PASS: ${NODE} now serves ${SHA:0:12}"
