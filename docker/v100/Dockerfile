# V100 (SM70) test and runtime base for the LMDeploy campaign.
#
# The campaign runs two kinds of job against the free GPU island:
#   1. Kernel fixtures, such as test_attention, built from source with
#      BUILD_TEST=ON.
#   2. Model tests that import the built wheel and drive a pipeline.
#
# Both used to install apt packages, torch and the runtime requirements on
# every run. That cost about eight minutes per job before any real work, and a
# single DNS failure destroyed one whole run.
#
# This image installs all of it once. A job then mounts the source, builds and
# runs. Nothing in this image pins campaign source, so a source change does not
# invalidate it.
#
# Build with the in-cluster buildkit Job in
# manifests/lmdeploy-v100-next/01-build-base.yaml.

FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Toolchain for the CUDA build plus the headers TurboMind needs.
# rapidjson and glog are required by the C++ tree; ninja drives the build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        git \
        libgoogle-glog-dev \
        ninja-build \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        python3-pip \
        rapidjson-dev \
    && rm -rf /var/lib/apt/lists/*

# One virtualenv, on PATH, so a job never needs to activate it.
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN pip install --upgrade pip

# Torch is the single largest install and the most frequent network failure.
# Pin it here so no job downloads it again.
ARG TORCH_VERSION=2.9.1
ARG TORCHVISION_VERSION=0.24.1
ARG CUDA_CHANNEL=cu128
RUN pip install \
        "torch==${TORCH_VERSION}" \
        "torchvision==${TORCHVISION_VERSION}" \
        --index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}"

# LMDeploy runtime dependencies, resolved against the campaign constraints.
# Only these two files are copied, so editing kernel source does not invalidate
# this layer.
# All three files sit at the context root, because the build context is a
# ConfigMap and a ConfigMap has no directories.
#
# common.txt is required, not optional: runtime_cuda.txt begins with
# "-r common.txt". It declares no further includes, so these three files are
# the complete closure.
COPY runtime_cuda.txt /tmp/req/runtime_cuda.txt
COPY common.txt /tmp/req/common.txt
COPY constraints-v100.txt /tmp/req/constraints-v100.txt
RUN pip install \
        -r /tmp/req/runtime_cuda.txt \
        -c /tmp/req/constraints-v100.txt \
        --extra-index-url "https://download.pytorch.org/whl/${CUDA_CHANNEL}" \
    && rm -rf /tmp/req

# Record what this image provides, so a job can verify it instead of assuming.
RUN python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" \
    && cmake --version | head -1 \
    && ninja --version

WORKDIR /work
