#!/usr/bin/env bash
# Run a script on the free V100 island, inside the prebuilt campaign image.
#
# The image already contains the toolchain, torch and the LMDeploy runtime
# requirements, so a job starts working immediately.
#
# Usage:
#   tools/v100/run_island2.sh <script.sh> [job-name]
#
# The script runs with:
#   /src     the staged campaign source
#   /wheels  built wheels
#   /models  read-only model store
#
# The island guard runs first. It aborts before any allocation if an island-1
# GPU is visible, if the visible count is not 4, or if island 2 is not idle.
set -euo pipefail

SCRIPT="${1:?usage: run_island2.sh <script.sh> [job-name]}"
JOB="${2:-lmdeploy-v100-island2}"
NS=llm
IMAGE="localhost:32000/lmdeploy-v100-base:v1"

ISLAND2="GPU-c1aa8bd9-f642-328d-96f8-79d7c38ce61e,GPU-c275f176-56b9-eb8d-14d7-a3fd15a0ec03,GPU-d5302639-be81-61f0-5e1a-b8f421bb100b,GPU-dd6f7287-63a3-17c4-64c2-1eb597391f4b"
ISLAND1="GPU-3ceb3a71-cd56-6d10-075e-0300bd506c22 GPU-aa23eb12-62a1-161f-b566-3a8b5d0c6278 GPU-f364b813-c606-1786-40be-f6645f3c33eb GPU-07e14590-993d-404b-4a47-f65d0c4b23e0"

[ -f "$SCRIPT" ] || { echo "no such script: $SCRIPT" >&2; exit 2; }

kubectl -n "$NS" delete job "$JOB" --ignore-not-found >/dev/null 2>&1 || true
kubectl -n "$NS" delete configmap "${JOB}-script" --ignore-not-found >/dev/null 2>&1 || true
kubectl -n "$NS" create configmap "${JOB}-script" --from-file=work.sh="$SCRIPT" >/dev/null

python3 - "$JOB" "$NS" "$IMAGE" "$ISLAND2" "$ISLAND1" <<'PY' | kubectl apply -f - >/dev/null
import json, sys
job, ns, image, island2, island1 = sys.argv[1:6]

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
                ]}],
            "volumes": [
                {"name": "job", "configMap": {"name": f"{job}-script"}},
                {"name": "src", "hostPath": {"path": "/localpool/lmdeploy-v100-next/src/lmdeploy", "type": "Directory"}},
                {"name": "wheels", "hostPath": {"path": "/localpool/lmdeploy-v100-next/wheels", "type": "Directory"}},
                {"name": "models", "hostPath": {"path": "/srv/models", "type": "Directory"}},
                {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"}},
            ]}}}}
print(json.dumps(manifest))
PY

echo "started job/$JOB in namespace $NS"
echo "follow with: kubectl -n $NS logs -f job/$JOB"
