// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/dflash_kernels.h"

#include <cuda_fp16.h>

#include "src/turbomind/core/check.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {
namespace {

__global__ void BuildDFlashBlock(int* output, const int* anchors, int count, int block_size, int mask_token_id)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = index % block_size == 0 ? anchors[index / block_size] : mask_token_id;
    }
}

__global__ void GatherDFlashPredictionsHalf(__half* output, const __half* input, int count, int hidden, int block_size)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        const int row     = index / hidden;
        const int channel = index - row * hidden;
        const int batch   = row / (block_size - 1);
        const int slot    = row - batch * (block_size - 1);
        output[index]     = input[(batch * block_size + slot + 1) * hidden + channel];
    }
}

__global__ void DFlashTopK16Half(int* ids, float* scores, const __half* logits, int rows, int vocab, float multiplier, float softcap)
{
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    __shared__ float shared_scores[256];
    __shared__ int   shared_ids[256];
    __shared__ int   selected[16];

    for (int rank = 0; rank < 16; ++rank) {
        float best_score = -CUDART_INF_F;
        int   best_id    = 0;
        for (int token = threadIdx.x; token < vocab; token += blockDim.x) {
            bool used = false;
            for (int i = 0; i < rank; ++i) {
                used = used || selected[i] == token;
            }
            if (used) {
                continue;
            }
            float score = __half2float(logits[row * vocab + token]) * multiplier;
            if (softcap > 0.f) {
                score = tanhf(score / softcap) * softcap;
            }
            if (score > best_score || (score == best_score && token < best_id)) {
                best_score = score;
                best_id    = token;
            }
        }
        shared_scores[threadIdx.x] = best_score;
        shared_ids[threadIdx.x]    = best_id;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                const float other_score = shared_scores[threadIdx.x + stride];
                const int   other_id    = shared_ids[threadIdx.x + stride];
                if (other_score > shared_scores[threadIdx.x]
                    || (other_score == shared_scores[threadIdx.x] && other_id < shared_ids[threadIdx.x])) {
                    shared_scores[threadIdx.x] = other_score;
                    shared_ids[threadIdx.x]    = other_id;
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            selected[rank]       = shared_ids[0];
            ids[row * 16 + rank] = shared_ids[0];
            scores[row * 16 + rank] = shared_scores[0];
        }
        __syncthreads();
    }
}

__global__ void DFlashGreedySelectorHalf(int*         output,
                                         const int*   anchors,
                                         const int*   candidates,
                                         const float* unary,
                                         const __half* hidden,
                                         const __half* predecessor,
                                         const __half* successor,
                                         int           batch_size,
                                         int           slots,
                                         int           top_k,
                                         int           rank)
{
    const int batch = blockIdx.x;
    if (batch >= batch_size || threadIdx.x != 0) {
        return;
    }
    int predecessor_id = anchors[batch];
    for (int slot = 0; slot < slots; ++slot) {
        float best_score = -CUDART_INF_F;
        int   best_index = 0;
        const __half* pred = predecessor + predecessor_id * rank;
        const __half* state = hidden + (batch * slots + slot) * rank;
        for (int candidate_index = 0; candidate_index < top_k; ++candidate_index) {
            const int token = candidates[(batch * slots + slot) * top_k + candidate_index];
            const __half* succ = successor + token * rank;
            float score = unary[(batch * slots + slot) * top_k + candidate_index];
            for (int i = 0; i < rank; ++i) {
                score += __half2float(pred[i]) * __half2float(state[i]) * __half2float(succ[i]);
            }
            if (score > best_score || (score == best_score && candidate_index < best_index)) {
                best_score = score;
                best_index = candidate_index;
            }
        }
        predecessor_id = candidates[(batch * slots + slot) * top_k + best_index];
        output[batch * slots + slot] = predecessor_id;
    }
}

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

void invokeBuildDFlashBlock(Buffer_<int>&       output,
                            const Buffer_<int>& anchors,
                            int                 block_size,
                            int                 mask_token_id,
                            cudaStream_t        stream)
{
    TM_CHECK_GT(block_size, 1);
    TM_CHECK_EQ(output.size(), anchors.size() * block_size);
    constexpr int threads = 256;
    const int     blocks  = (output.size() + threads - 1) / threads;
    BuildDFlashBlock<<<blocks, threads, 0, stream>>>(
        output.data(), anchors.data(), output.size(), block_size, mask_token_id);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeGatherDFlashPredictions(Tensor& output, const Tensor& block_hidden, int block_size, cudaStream_t stream)
{
    TM_CHECK_EQ(output.dtype(), kHalf);
    TM_CHECK_EQ(block_hidden.dtype(), kHalf);
    TM_CHECK_EQ(output.ndim(), 2);
    TM_CHECK_EQ(block_hidden.ndim(), 2);
    TM_CHECK_EQ(output.shape(1), block_hidden.shape(1));
    TM_CHECK_EQ(block_hidden.shape(0) % block_size, 0);
    TM_CHECK_EQ(output.shape(0), block_hidden.shape(0) / block_size * (block_size - 1));
    constexpr int threads = 256;
    const int     count   = output.size();
    const int     blocks  = (count + threads - 1) / threads;
    GatherDFlashPredictionsHalf<<<blocks, threads, 0, stream>>>((__half*)output.raw_data(),
                                                                (const __half*)block_hidden.raw_data(),
                                                                count,
                                                                output.shape(1),
                                                                block_size);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeDFlashTopK16(Buffer_<int>& ids,
                        Tensor&       scores,
                        const Tensor& logits,
                        int           valid_vocab,
                        float         output_multiplier,
                        float         softcap,
                        cudaStream_t  stream)
{
    TM_CHECK_EQ(logits.dtype(), kHalf);
    TM_CHECK_EQ(scores.dtype(), kFloat32);
    TM_CHECK_EQ(logits.ndim(), 2);
    TM_CHECK_EQ(scores.ndim(), 2);
    TM_CHECK_EQ(scores.shape(0), logits.shape(0));
    TM_CHECK_EQ(scores.shape(1), 16);
    TM_CHECK_EQ(ids.size(), logits.shape(0) * 16);
    TM_CHECK_EQ(valid_vocab, logits.shape(1));
    DFlashTopK16Half<<<logits.shape(0), 256, 0, stream>>>(ids.data(),
                                                         scores.data<float>(),
                                                         (const __half*)logits.raw_data(),
                                                         logits.shape(0),
                                                         valid_vocab,
                                                         output_multiplier,
                                                         softcap);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeDFlashGreedySelector(Buffer_<int>&       output,
                                 const Buffer_<int>& anchors,
                                 const Buffer_<int>& candidate_ids,
                                 const Tensor&       unary_scores,
                                 const Tensor&       selector_hidden,
                                 const Tensor&       predecessor_codebook,
                                 const Tensor&       successor_codebook,
                                 int                 slots,
                                 int                 top_k,
                                 cudaStream_t        stream)
{
    TM_CHECK_EQ(top_k, 16);
    TM_CHECK_EQ(selector_hidden.dtype(), kHalf);
    TM_CHECK_EQ(predecessor_codebook.dtype(), kHalf);
    TM_CHECK_EQ(successor_codebook.dtype(), kHalf);
    TM_CHECK_EQ(unary_scores.dtype(), kFloat32);
    const int batch_size = anchors.size();
    const int rank = selector_hidden.shape(1);
    TM_CHECK_EQ(output.size(), (ssize_t)batch_size * slots);
    TM_CHECK_EQ(candidate_ids.size(), (ssize_t)batch_size * slots * top_k);
    TM_CHECK_EQ(selector_hidden.shape(0), (ssize_t)batch_size * slots);
    TM_CHECK_EQ(predecessor_codebook.shape(1), rank);
    TM_CHECK_EQ(successor_codebook.shape(1), rank);
    DFlashGreedySelectorHalf<<<batch_size, 1, 0, stream>>>(output.data(),
                                                           anchors.data(),
                                                           candidate_ids.data(),
                                                           unary_scores.data<float>(),
                                                           (const __half*)selector_hidden.raw_data(),
                                                           (const __half*)predecessor_codebook.raw_data(),
                                                           (const __half*)successor_codebook.raw_data(),
                                                           batch_size,
                                                           slots,
                                                           top_k,
                                                           rank);
    TM_CUDA_CHECK(cudaGetLastError());
}

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
