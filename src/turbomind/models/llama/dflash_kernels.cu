// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/dflash_kernels.h"

#include <cuda_fp16.h>

#include "src/turbomind/core/check.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {
namespace {

__global__ void DFlashGroupedConvHalf(__half*       output,
                                      const __half* input,
                                      const __half* delta,
                                      const __half* base,
                                      int           token_num,
                                      int           hidden,
                                      int           side,
                                      int           block_size,
                                      int           taps,
                                      int           group_size,
                                      int           groups)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= token_num * hidden) {
        return;
    }
    const int token   = index / hidden;
    const int channel = index - token * hidden;
    const int group   = channel / group_size;
    const int pos     = token % block_size;

    float value = 0.f;
    for (int tap = 0; tap < taps; ++tap) {
        if (tap > pos) {
            break;
        }
        const int base_index  = (side * taps + tap) * hidden + channel;
        const int delta_index = ((token * 2 + side) * taps + tap) * groups + group;
        const float coefficient = __half2float(base[base_index]) + __half2float(delta[delta_index]);
        value += coefficient * __half2float(input[(token - tap) * hidden + channel]);
    }
    output[index] = __float2half_rn(value);
}

}  // namespace

void invokeDFlashGroupedConv(Tensor&       output,
                             const Tensor& input,
                             const Tensor& delta,
                             const Tensor& base_kernel,
                             int           side,
                             int           block_size,
                             int           taps,
                             int           group_size,
                             cudaStream_t  stream)
{
    TM_CHECK_EQ(input.dtype(), kHalf) << "DFlash2 V100 convolution currently supports FP16 only";
    TM_CHECK_EQ(output.dtype(), input.dtype());
    TM_CHECK_EQ(delta.dtype(), input.dtype());
    TM_CHECK_EQ(base_kernel.dtype(), input.dtype());
    TM_CHECK_EQ(input.ndim(), 2);
    TM_CHECK_EQ(output.ndim(), 2);
    TM_CHECK_EQ(output.shape(0), input.shape(0));
    TM_CHECK_EQ(output.shape(1), input.shape(1));
    TM_CHECK(side == 0 || side == 1);
    TM_CHECK_GT(block_size, 0);
    TM_CHECK_GT(taps, 0);
    TM_CHECK_EQ(input.shape(1) % group_size, 0);

    const int token_num = input.shape(0);
    const int hidden    = input.shape(1);
    const int groups    = hidden / group_size;
    TM_CHECK_EQ(delta.shape(0), token_num);
    TM_CHECK_EQ(delta.shape(1), 2 * taps * groups);
    TM_CHECK_EQ(base_kernel.size(), (ssize_t)2 * taps * hidden);

    constexpr int threads = 256;
    const int     blocks  = (token_num * hidden + threads - 1) / threads;
    DFlashGroupedConvHalf<<<blocks, threads, 0, stream>>>((__half*)output.raw_data(),
                                                          (const __half*)input.raw_data(),
                                                          (const __half*)delta.raw_data(),
                                                          (const __half*)base_kernel.raw_data(),
                                                          token_num,
                                                          hidden,
                                                          side,
                                                          block_size,
                                                          taps,
                                                          group_size,
                                                          groups);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind
