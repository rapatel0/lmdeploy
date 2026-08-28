// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/argmax.h"

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/core/common.h"

namespace turbomind {

namespace {

/// One block per row. Each thread scans a strided slice of the row, then the
/// block reduces to a single (value, index) pair.
///
/// Ties resolve to the lowest index, which matters for verification: the
/// target and the draft must agree on the same token for identical logits, and
/// a tie broken differently would look like a rejected draft.
template<class T, int kBlock>
__global__ void argmax_kernel(int* __restrict__ out, const T* __restrict__ logits, int vocab)
{
    __shared__ float s_val[kBlock];
    __shared__ int   s_idx[kBlock];

    const int row = blockIdx.x;
    const T*  src = logits + (size_t)row * vocab;

    float best_v = -INFINITY;
    int   best_i = 0;

    for (int i = threadIdx.x; i < vocab; i += kBlock) {
        const float v = static_cast<float>(src[i]);
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }

    s_val[threadIdx.x] = best_v;
    s_idx[threadIdx.x] = best_i;
    __syncthreads();

    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const int j = threadIdx.x + stride;
            // Strict greater-than keeps the lower index on a tie.
            if (s_val[j] > s_val[threadIdx.x] || (s_val[j] == s_val[threadIdx.x] && s_idx[j] < s_idx[threadIdx.x])) {
                s_val[threadIdx.x] = s_val[j];
                s_idx[threadIdx.x] = s_idx[j];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[row] = s_idx[0];
    }
}

template<class T, int kBlock>
__global__ void local_argmax_kernel(float* __restrict__ candidates,
                                    const T* __restrict__ logits,
                                    int logits_stride,
                                    int valid_vocab,
                                    int token_id_offset)
{
    __shared__ float s_val[kBlock];
    __shared__ int   s_idx[kBlock];

    const int row = blockIdx.x;
    const T*  src = logits + (size_t)row * logits_stride;
    float best_v = -INFINITY;
    int   best_i = 0;
    for (int i = threadIdx.x; i < valid_vocab; i += kBlock) {
        const float v = static_cast<float>(src[i]);
        if (v > best_v || (v == best_v && i < best_i)) {
            best_v = v;
            best_i = i;
        }
    }
    s_val[threadIdx.x] = best_v;
    s_idx[threadIdx.x] = best_i;
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const int j = threadIdx.x + stride;
            if (s_val[j] > s_val[threadIdx.x]
                || (s_val[j] == s_val[threadIdx.x] && s_idx[j] < s_idx[threadIdx.x])) {
                s_val[threadIdx.x] = s_val[j];
                s_idx[threadIdx.x] = s_idx[j];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        candidates[row * 2]     = s_val[0];
        candidates[row * 2 + 1] = static_cast<float>(token_id_offset + s_idx[0]);
    }
}

__global__ void global_argmax_kernel(
    int* __restrict__ out, const float* __restrict__ candidates, int rows, int ranks)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    float best_v = candidates[row * 2];
    int   best_i = static_cast<int>(candidates[row * 2 + 1]);
    for (int rank = 1; rank < ranks; ++rank) {
        const size_t offset = ((size_t)rank * rows + row) * 2;
        const float  value  = candidates[offset];
        const int    token  = static_cast<int>(candidates[offset + 1]);
        if (value > best_v || (value == best_v && token < best_i)) {
            best_v = value;
            best_i = token;
        }
    }
    out[row] = best_i;
}

}  // namespace

void invokeArgmax(Buffer_<int>& out, const Tensor& logits, cudaStream_t st)
{
    TM_CHECK_EQ(logits.ndim(), 2);
    const int rows  = logits.shape(0);
    const int vocab = logits.shape(1);
    // Buffer_::size() is a signed ssize_t, so compare signed to signed.
    TM_CHECK_GE(out.size(), (ssize_t)rows);

    if (rows == 0) {
        return;
    }

    constexpr int kBlock = 256;

    auto dispatch = [&](auto t) {
        using T = decltype(t);
        argmax_kernel<T, kBlock><<<rows, kBlock, 0, st>>>(out.data(), (const T*)logits.raw_data(), vocab);
    };

    switch (logits.dtype()) {
        case kFloat32:
            dispatch(float{});
            break;
        case kFloat16:
            dispatch(half{});
            break;
        case kBfloat16:
            dispatch(nv_bfloat16{});
            break;
        default:
            TM_CHECK(0) << "argmax: unsupported dtype " << to_string(logits.dtype());
    }
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeLocalArgmax(
    Buffer_<float>& candidates, const Tensor& local_logits, int valid_vocab, int token_id_offset, cudaStream_t st)
{
    TM_CHECK_EQ(local_logits.ndim(), 2);
    const int rows   = local_logits.shape(0);
    const int stride = local_logits.shape(1);
    TM_CHECK_GT(valid_vocab, 0);
    TM_CHECK_LE(valid_vocab, stride);
    TM_CHECK_GE(candidates.size(), (ssize_t)rows * 2);
    constexpr int kBlock = 256;
    auto dispatch = [&](auto t) {
        using T = decltype(t);
        local_argmax_kernel<T, kBlock><<<rows, kBlock, 0, st>>>(
            candidates.data(), (const T*)local_logits.raw_data(), stride, valid_vocab, token_id_offset);
    };
    switch (local_logits.dtype()) {
        case kFloat32:
            dispatch(float{});
            break;
        case kFloat16:
            dispatch(half{});
            break;
        case kBfloat16:
            dispatch(nv_bfloat16{});
            break;
        default:
            TM_CHECK(0) << "local argmax: unsupported dtype " << to_string(local_logits.dtype());
    }
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeGlobalArgmax(
    Buffer_<int>& out, const Buffer_<float>& gathered_candidates, int rows, int ranks, cudaStream_t st)
{
    TM_CHECK_GE(out.size(), (ssize_t)rows);
    TM_CHECK_GE(gathered_candidates.size(), (ssize_t)rows * ranks * 2);
    constexpr int kBlock = 128;
    global_argmax_kernel<<<(rows + kBlock - 1) / kBlock, kBlock, 0, st>>>(
        out.data(), gathered_candidates.data(), rows, ranks);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind
