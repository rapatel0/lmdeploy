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

}  // namespace

void invokeArgmax(Buffer_<int>& out, const Tensor& logits, cudaStream_t st)
{
    TM_CHECK_EQ(logits.ndim(), 2);
    const int rows  = logits.shape(0);
    const int vocab = logits.shape(1);
    TM_CHECK_GE(out.size(), (size_t)rows);

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

}  // namespace turbomind
