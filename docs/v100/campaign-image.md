# The V100 campaign image

Every job before this rebuilt its own environment: apt packages, torch, then
the LMDeploy runtime requirements. That cost about seven minutes per run and,
worse, made every run depend on the network. One run was destroyed outright
when `download.pytorch.org` failed to resolve part-way through an install.

The image removes both problems.

| | Before | After |
| --- | ---: | ---: |
| apt update and install | ~90s | 0 |
| pip torch and torchvision | ~210s | 0 |
| pip runtime requirements | ~120s | 0 |
| Total before any work | ~420s | ~15s |

Measured: the smoke pod went from `Pending` to `Succeeded` in 24 seconds
including the container start.

The remaining per-run cost is the cmake configure and compile, which is real
work rather than setup.

## What the image contains, and what it does not

It contains the toolchain, torch 2.9.1+cu128 and the LMDeploy runtime
requirements. Verified by running it:

```text
Cuda compilation tools, release 12.8, V12.8.93
cmake version 3.28.3
torch 2.9.1+cu128 | cuda build 12.8
transformers 5.15.1 | numpy 2.5.2
```

It deliberately contains no campaign source. Source is mounted at `/src` at run
time, so editing a kernel never invalidates the image.

## Two failures worth recording

Both were mistakes in the build context rather than in the Dockerfile.

**ConfigMap mounts are symlinks.** A ConfigMap volume presents each key as a
symlink into a timestamped `..data` directory. Buildkit will not follow those
when checksumming a `COPY` source, so the build failed with
`"/runtime_cuda.txt": not found`. Fixed by copying with `cp -L` into a real
directory before invoking buildkit.

**Requirements files include each other.** `runtime_cuda.txt` begins with
`-r common.txt`, and the context shipped only the former, so pip failed with
`Could not open requirements file: /tmp/req/common.txt`. `common.txt` declares
no further includes, so the four files now in the context are the complete
closure.

## The registry needs two names

The buildkit job pushes to the in-cluster service DNS name. That name resolves
inside a pod but not on the node, and the kubelet resolves through host DNS, so
a pod that references it cannot be scheduled:

```text
failed to resolve reference ".../lmdeploy-v100-base:v1":
lookup registry.container-registry.svc.cluster.local on 127.0.0.53:53
```

microk8s containerd only auto-trusts `localhost:32000` for kubelet pulls, so
workloads must use that name. `02-retag-base.yaml` copies the manifest between
the two names with crane on `hostNetwork`, which is the pattern the hermes and
rotorquant pipelines already use. Crane reports `existing manifest`, confirming
both names address the same backing registry, so this is an alias rather than a
layer copy and it finishes in seconds.

## Running a job

```bash
tools/v100/run_island2.sh <script.sh> [job-name]
```

The script is shipped as a ConfigMap and runs with `/src`, `/wheels` and
`/models` mounted. The island guard always runs first and aborts before any
allocation if an island-1 GPU is visible, if the visible GPU count is not four,
or if island 2 is not idle.

End-to-end validation on the island:

```text
ISLAND_GUARD_PASS: 4 idle GPUs, no island-1 UUID visible
Cuda compilation tools, release 12.8, V12.8.93
torch 2.9.1+cu128
sm70 kernel ran
IMAGE_READY
```

The last line matters: an sm_70 kernel was compiled and executed on a real
V100 from this image, so the image is confirmed for campaign work rather than
merely importable.

## Files

| Path | Purpose |
| --- | --- |
| `lmdeploy/docker/v100/Dockerfile` | the image |
| `manifests/lmdeploy-v100-next/01-build-base.yaml` | buildkit build job |
| `manifests/lmdeploy-v100-next/02-retag-base.yaml` | crane re-tag to the pullable name |
| `lmdeploy/tools/v100/run_island2.sh` | run a script on island 2 |
