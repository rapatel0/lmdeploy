#!/usr/bin/env bash
# Run a script on the free V100 island, inside the prebuilt campaign image.
#
# The image already contains the toolchain, torch and the LMDeploy runtime
# requirements, so a job starts working immediately.
#
# Usage:
#   tools/v100/run_island2.sh <script.sh> [job-name] [extra-file ...]
#
# Extra files land beside work.sh in /job. Pass a Python driver this way rather
# than embedding it in a heredoc: LMDeploy spawns workers, and multiprocessing
# spawn re-imports __main__ by path, so a script arriving on stdin is named
# '<stdin>', which no worker can open. That failure surfaces as
# BrokenProcessPool and looks like an engine fault.
#
# The script runs with:
#   /src     the staged campaign source
#   /wheels  built wheels
#   /models  read-only model store
#
# The island guard runs first. It aborts before any allocation if an island-1
# GPU is visible, if the visible count is not 4, or if island 2 is not idle.
set -euo pipefail

# Campaign defaults. The island is four V100s, so tp=4 is the only sane value
# and the MTP work runs at draft depth 4. Exported into the pod below so a job
# script reads one number rather than repeating a literal.
TP="${TP:-4}"
NUM_DRAFT_TOKENS="${NUM_DRAFT_TOKENS:-4}"
MODEL_DIR="${MODEL_DIR:-/models/Qwen3.8-27B-FP8}"
TM_BATCH_COPY_DURING_API_CALL="${TM_BATCH_COPY_DURING_API_CALL:-0}"
TM_MTP_LOCAL_TOP1="${TM_MTP_LOCAL_TOP1:-0}"
TM_MTP_FROZEN_KV="${TM_MTP_FROZEN_KV:-0}"
TM_MTP_EAGLE_ROTATION="${TM_MTP_EAGLE_ROTATION:-0}"
TM_MTP_FORCE_REJECT="${TM_MTP_FORCE_REJECT:-0}"
TM_SPEC_TRACE="${TM_SPEC_TRACE:-0}"
CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-0}"
CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-}"

# Refuse to launch while a job of the same name is already active.
#
# Two launchers -- an interactive one and a background watcher that chains
# "wait, then launch" -- raced and produced two results directories four seconds
# apart, both incomplete, with two containers competing for the same four GPUs.
# Neither run was usable and the collision was invisible until the directory
# listing showed duplicate timestamps.
#
# kubectl create fails on a duplicate name, but the watcher deletes the job
# first, so the window is real. Checking here makes it explicit.
if [ -n "${2:-}" ] && kubectl -n "${NS:-llm}" get "job/${2}" >/dev/null 2>&1; then
    if [ -n "$(kubectl -n "${NS:-llm}" get "job/${2}" -o jsonpath='{.status.active}' 2>/dev/null)" ]; then
        echo "REFUSING: job/${2} is already active. Delete it or wait." >&2
        exit 3
    fi
fi

SCRIPT="${1:?usage: run_island2.sh <script.sh> [job-name] [extra-file ...]}"
JOB="${2:-lmdeploy-v100-island2}"
shift 2 2>/dev/null || shift $#
EXTRA=("$@")
NS=llm
IMAGE="${IMAGE:-localhost:32000/lmdeploy-v100-base:v2}"

ISLAND2="GPU-c1aa8bd9-f642-328d-96f8-79d7c38ce61e,GPU-c275f176-56b9-eb8d-14d7-a3fd15a0ec03,GPU-d5302639-be81-61f0-5e1a-b8f421bb100b,GPU-dd6f7287-63a3-17c4-64c2-1eb597391f4b"
ISLAND1="GPU-3ceb3a71-cd56-6d10-075e-0300bd506c22 GPU-aa23eb12-62a1-161f-b566-3a8b5d0c6278 GPU-f364b813-c606-1786-40be-f6645f3c33eb GPU-07e14590-993d-404b-4a47-f65d0c4b23e0"

[ -f "$SCRIPT" ] || {
    echo "no such script: $SCRIPT" >&2
    exit 2
}

kubectl -n "$NS" delete job "$JOB" --ignore-not-found >/dev/null 2>&1 || true
kubectl -n "$NS" delete configmap "${JOB}-script" --ignore-not-found >/dev/null 2>&1 || true
CM_ARGS=(--from-file=work.sh="$SCRIPT")
for f in "${EXTRA[@]}"; do
    [ -f "$f" ] || {
        echo "no such extra file: $f" >&2
        exit 2
    }
    CM_ARGS+=(--from-file="$(basename "$f")=$f")
done
kubectl -n "$NS" create configmap "${JOB}-script" "${CM_ARGS[@]}" >/dev/null

python3 - "$JOB" "$NS" "$IMAGE" "$ISLAND2" "$ISLAND1" "$TP" "$NUM_DRAFT_TOKENS" "$MODEL_DIR" \
    "$TM_BATCH_COPY_DURING_API_CALL" "$TM_MTP_LOCAL_TOP1" "$TM_MTP_FROZEN_KV" "$TM_MTP_EAGLE_ROTATION" "$TM_MTP_FORCE_REJECT" "$TM_SPEC_TRACE" "$CUDA_LAUNCH_BLOCKING" "$CUBLAS_WORKSPACE_CONFIG" <<'PY' | kubectl apply -f - >/dev/null
import json, sys
job, ns, image, island2, island1, tp, num_draft, model_dir, batch_copy_order, mtp_local_top1, mtp_frozen_kv, mtp_eagle_rotation, mtp_force_reject, spec_trace, launch_blocking, cublas_workspace = sys.argv[1:17]

guard = r'''set -uo pipefail
for BAD in $ISLAND1_UUIDS; do
  if nvidia-smi --query-gpu=uuid --format=csv,noheader | grep -q "$BAD"; then
    echo "FATAL: island-1 GPU $BAD is visible. Refusing to run."; exit 90
  fi
done
COUNT=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | wc -l)
[ "$COUNT" = "4" ] || { echo "FATAL: expected 4 GPUs, saw $COUNT"; exit 91; }
USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s}')
[ "$USED" -lt 2000 ] || { echo "FATAL: island 2 is not idle (${USED} MiB in use)"; exit 92; }
echo "ISLAND_GUARD_PASS: 4 idle GPUs, no island-1 UUID visible"
exec bash /job/work.sh
'''

manifest = {
    "apiVersion": "batch/v1", "kind": "Job",
    "metadata": {"name": job, "namespace": ns,
                 "labels": {"app": "lmdeploy-v100-next", "island": "2"}},
    "spec": {"backoffLimit": 0, "ttlSecondsAfterFinished": 86400,
        "template": {"metadata": {"labels": {"app": "lmdeploy-v100-next", "island": "2"}},
        "spec": {
            "restartPolicy": "Never",
            "nodeSelector": {"kubernetes.io/hostname": "gpu-01"},
            "dnsPolicy": "None",
            "dnsConfig": {"nameservers": ["10.152.183.10", "1.1.1.1", "8.8.8.8"],
                          "searches": ["llm.svc.cluster.local", "svc.cluster.local", "cluster.local"],
                          "options": [{"name": "ndots", "value": "5"}]},
            "containers": [{
                "name": "run", "image": image, "imagePullPolicy": "IfNotPresent",
                "command": ["/bin/bash", "-lc"], "args": [guard],
                "env": [
                    {"name": "NVIDIA_VISIBLE_DEVICES", "value": island2},
                    {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"},
                    {"name": "ISLAND1_UUIDS", "value": island1},
                    # Campaign defaults, set here so every job on this island
                    # inherits the same configuration instead of each script
                    # hardcoding its own. A job may override them, but nothing
                    # silently runs at a different tp or draft depth because a
                    # flag was omitted at the call site.
                    {"name": "TP", "value": tp},
                    {"name": "NUM_DRAFT_TOKENS", "value": num_draft},
                    {"name": "MODEL_DIR", "value": model_dir},
                    {"name": "TM_BATCH_COPY_DURING_API_CALL", "value": batch_copy_order},
                    {"name": "TM_MTP_LOCAL_TOP1", "value": mtp_local_top1},
                    {"name": "TM_MTP_FROZEN_KV", "value": mtp_frozen_kv},
                    {"name": "TM_MTP_EAGLE_ROTATION", "value": mtp_eagle_rotation},
                    {"name": "TM_MTP_FORCE_REJECT", "value": mtp_force_reject},
                    {"name": "TM_SPEC_TRACE", "value": spec_trace},
                    {"name": "CUDA_LAUNCH_BLOCKING", "value": launch_blocking},
                    {"name": "CUBLAS_WORKSPACE_CONFIG", "value": cublas_workspace},
                ],
                # No nvidia.com/gpu request: the device plugin must not
                # reassign devices out from under the UUID pin.
                "resources": {"requests": {"cpu": "16", "memory": "96Gi"},
                              "limits": {"cpu": "32", "memory": "160Gi"}},
                "volumeMounts": [
                    {"name": "job", "mountPath": "/job"},
                    {"name": "src", "mountPath": "/src"},
                    {"name": "wheels", "mountPath": "/wheels"},
                    {"name": "models", "mountPath": "/models", "readOnly": True},
                    {"name": "shm", "mountPath": "/dev/shm"},
                    {"name": "results", "mountPath": "/results"},
                    # Staged from the SGLang V100 image because the lean
                    # LMDeploy build image does not ship Nsight Systems.
                    {"name": "nsys", "mountPath": "/opt/nsys", "readOnly": True},
                ]}],
            "volumes": [
                {"name": "job", "configMap": {"name": f"{job}-script"}},
                {"name": "src", "hostPath": {"path": "/localpool/lmdeploy-v100-next/src/lmdeploy", "type": "Directory"}},
                {"name": "wheels", "hostPath": {"path": "/localpool/lmdeploy-v100-next/wheels", "type": "Directory"}},
                {"name": "models", "hostPath": {"path": "/srv/models", "type": "Directory"}},
                {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"}},
                # Results must outlive the pod. kubectl logs is not storage:
                # once the pod is reclaimed the log is gone, and a bare
                # "SUCCEEDED" with no readable output is not evidence of
                # anything. Jobs copy their artifacts here.
                {"name": "results", "hostPath": {"path": "/localpool/lmdeploy-v100-next/results", "type": "DirectoryOrCreate"}},
                {"name": "nsys", "hostPath": {"path": "/localpool/lmdeploy-v100-next/nsys-cli", "type": "Directory"}},
            ]}}}}
print(json.dumps(manifest))
PY

echo "started job/$JOB in namespace $NS"
echo "follow with: kubectl -n $NS logs -f job/$JOB"
