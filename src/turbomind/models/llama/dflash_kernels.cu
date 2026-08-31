// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/dflash_kernels.h"

#include <cuda_fp16.h>
#include <math_constants.h>
#if CUDART_VERSION >= 11000
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include <climits>
#include <cmath>
#include <cstdlib>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/reduce_kernel_utils.cuh"
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

__device__ __forceinline__ float RoundBFloat16(float value)
{
    uint32_t bits = __float_as_uint(value);
    // Round-to-nearest-even at the BF16 mantissa boundary. Preserve Inf/NaN
    // payloads instead of allowing the rounding increment to wrap them.
    if ((bits & 0x7f800000u) != 0x7f800000u) {
        bits += 0x7fffu + ((bits >> 16) & 1u);
    }
    return __uint_as_float(bits & 0xffff0000u);
}

__global__ void DFlashCastToFloat(float* output, const __half* input, int count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = __half2float(input[index]);
    }
}

__global__ void DFlashCastToHalf(__half* output, const float* input, int count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        output[index] = __float2half_rn(input[index]);
    }
}

__global__ void DFlashRoundBFloat16Half(__half* value, int count)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        value[index] = __float2half_rn(RoundBFloat16(__half2float(value[index])));
    }
}

__global__ void DFlashScaleHalf(__half* value, int count, float scale)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
        value[index] = __float2half_rn(__half2float(value[index]) * scale);
    }
}

__global__ void DFlashLagunaSiluHalf(__half*       output,
                                     float*        row_scales,
                                     const __half* gate_up,
                                     int           cols,
                                     float         gate_up_scale)
{
    const int row = blockIdx.x;
    __shared__ float maxima[256];
    float max_abs = 0.f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const int base = row * 2 * cols;
        const float gate = RoundBFloat16(__half2float(gate_up[base + col]) * gate_up_scale);
        const float up = RoundBFloat16(__half2float(gate_up[base + cols + col]) * gate_up_scale);
        const float activated = RoundBFloat16(gate / (1.f + expf(-gate)));
        const float value = RoundBFloat16(activated * up);
        max_abs = fmaxf(max_abs, fabsf(value));
    }
    maxima[threadIdx.x] = max_abs;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_scale = fmaxf(maxima[0] / 32752.f, 1.f);
    const float power_of_two_scale = exp2f(ceilf(log2f(row_scale)));
    if (threadIdx.x == 0) {
        row_scales[row] = power_of_two_scale;
    }
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const int base = row * 2 * cols;
        const float gate = RoundBFloat16(__half2float(gate_up[base + col]) * gate_up_scale);
        const float up = RoundBFloat16(__half2float(gate_up[base + cols + col]) * gate_up_scale);
        const float activated = RoundBFloat16(gate / (1.f + expf(-gate)));
        const float value = RoundBFloat16(activated * up) / power_of_two_scale;
        output[row * cols + col] = __float2half_rn(value);
    }
}

__global__ void DFlashScaleRowsHalf(__half* value, const float* row_scales, int rows, int cols, float scale)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * cols) {
        const int row = index / cols;
        value[index] = __float2half_rn(__half2float(value[index]) * row_scales[row] * scale);
    }
}

__global__ void DFlashInitialRMSNormHalf(__half*       output,
                                         const __half* input,
                                         const __half* weight,
                                         int           hidden,
                                         float         eps,
                                         bool          zero_centered)
{
    constexpr int kThreads = 512;
    constexpr int kVec     = 8;
    const int token = blockIdx.x;
    float accum[kVec]{};
    for (int base = threadIdx.x * kVec; base < hidden; base += kThreads * kVec) {
#pragma unroll
        for (int c = 0; c < kVec; ++c) {
            if (base + c < hidden) {
                const float value = __half2float(input[token * hidden + base + c]);
                accum[c] += value * value;
            }
        }
    }
    float sum = 0.f;
#pragma unroll
    for (int c = 0; c < kVec; ++c) {
        sum += accum[c];
    }
    using BlockReduce = cub::BlockReduce<float, kThreads>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ float inverse_rms;
    sum = BlockReduce(reduce_storage).Sum(sum);
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(sum / hidden + eps);
    }
    __syncthreads();
    for (int channel = threadIdx.x; channel < hidden; channel += kThreads) {
        const int index = token * hidden + channel;
        const float gamma = RoundBFloat16(__half2float(weight[channel])) + (zero_centered ? 1.f : 0.f);
        const float normalized = RoundBFloat16(__half2float(input[index]) * inverse_rms * gamma);
        output[index] = __float2half_rn(normalized);
    }
}

__global__ void DFlashRankOrderedSumHalf(__half* output, const __half* gathered, size_t count, int ranks)
{
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        float sum = 0.f;
        for (int rank = 0; rank < ranks; ++rank) {
            sum += __half2float(gathered[static_cast<size_t>(rank) * count + index]);
        }
        output[index] = __float2half_rn(sum);
    }
}

__global__ void DFlashResidualRMSNormHalf(__half*       output,
                                          float*        residual,
                                          const __half* reduced,
                                          const __half* bias,
                                          const __half* weight,
                                          int           hidden,
                                          float         eps,
                                          bool          zero_centered,
                                          float         reduced_scale)
{
    const int token = blockIdx.x;
    __shared__ float sums[256];
    float sum = 0.f;
    for (int channel = threadIdx.x; channel < hidden; channel += blockDim.x) {
        const int index = token * hidden + channel;
        // Match the BF16-trained model's residual boundary. V100 cannot use
        // BF16 tensor cores, but it can preserve the checkpoint semantics by
        // rounding in FP32 and retaining the rounded value in FP32 storage.
        const float value = RoundBFloat16(
            residual[index] + __half2float(reduced[index]) * reduced_scale
            + (bias ? __half2float(bias[channel]) : 0.f));
        residual[index] = value;
        sum += value * value;
    }
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float scale = rsqrtf(sums[0] / hidden + eps);
    for (int channel = threadIdx.x; channel < hidden; channel += blockDim.x) {
        const int index = token * hidden + channel;
        const float gamma =
            RoundBFloat16(__half2float(weight[channel])) + (zero_centered ? 1.f : 0.f);
        const float normalized = RoundBFloat16(residual[index] * scale * gamma);
        output[index] = __float2half_rn(normalized);
    }
}

__global__ void DFlashTargetResidualRMSNormHalf(__half*       output,
                                                float*        residual,
                                                const __half* reduced,
                                                const __half* bias,
                                                const __half* weight,
                                                int           hidden,
                                                float         eps,
                                                bool          zero_centered)
{
    constexpr int kThreads = 512;
    constexpr int kVec     = 8;
    const int token = blockIdx.x;
    float accum[kVec]{};
    for (int base = threadIdx.x * kVec; base < hidden; base += kThreads * kVec) {
#pragma unroll
        for (int c = 0; c < kVec; ++c) {
            const int channel = base + c;
            if (channel < hidden) {
                const int index = token * hidden + channel;
                const float value = residual[index] + __half2float(reduced[index])
                                    + (bias ? __half2float(bias[channel]) : 0.f);
                residual[index] = value;
                accum[c] += value * value;
            }
        }
    }
    float sum = 0.f;
#pragma unroll
    for (int c = 0; c < kVec; ++c) {
        sum += accum[c];
    }
    using BlockReduce = cub::BlockReduce<float, kThreads>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ float inverse_rms;
    sum = BlockReduce(reduce_storage).Sum(sum);
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(sum / hidden + eps);
    }
    __syncthreads();
    for (int channel = threadIdx.x; channel < hidden; channel += kThreads) {
        const int index = token * hidden + channel;
        const float gamma = __half2float(weight[channel]) + (zero_centered ? 1.f : 0.f);
        output[index] = __float2half_rn(residual[index] * inverse_rms * gamma);
    }
}

__global__ void DFlashResidualRMSNormExactHalf(__half*       output,
                                               float*        residual,
                                               const __half* reduced,
                                               const __half* bias,
                                               const __half* weight,
                                               int           hidden,
                                               float         eps,
                                               bool          zero_centered,
                                               float         reduced_scale)
{
    constexpr int kThreads = 512;
    constexpr int kVec     = 8;
    const int token = blockIdx.x;
    float accum[kVec]{};
    for (int base = threadIdx.x * kVec; base < hidden; base += kThreads * kVec) {
#pragma unroll
        for (int c = 0; c < kVec; ++c) {
            const int channel = base + c;
            if (channel < hidden) {
                const int index = token * hidden + channel;
                const float value = RoundBFloat16(
                    residual[index] + __half2float(reduced[index]) * reduced_scale
                    + (bias ? __half2float(bias[channel]) : 0.f));
                residual[index] = value;
                accum[c] += value * value;
            }
        }
    }
    float sum = 0.f;
#pragma unroll
    for (int c = 0; c < kVec; ++c) {
        sum += accum[c];
    }
    using BlockReduce = cub::BlockReduce<float, kThreads>;
    __shared__ typename BlockReduce::TempStorage reduce_storage;
    __shared__ float inverse_rms;
    sum = BlockReduce(reduce_storage).Sum(sum);
    if (threadIdx.x == 0) {
        inverse_rms = rsqrtf(sum / hidden + eps);
    }
    __syncthreads();
    for (int channel = threadIdx.x; channel < hidden; channel += kThreads) {
        const int index = token * hidden + channel;
        const float gamma = RoundBFloat16(__half2float(weight[channel])) + (zero_centered ? 1.f : 0.f);
        const float normalized = RoundBFloat16(residual[index] * inverse_rms * gamma);
        output[index] = __float2half_rn(normalized);
    }
}

__global__ void DFlashTopK16HalfLegacy(int* ids, float* scores, const __half* logits, int rows, int vocab, int token_id_offset, float multiplier, float softcap)
{
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    // Read every vocabulary score once. The former implementation rescanned
    // the entire shard for each of the 16 ranks, making this tiny selector a
    // 1.58 ms/cycle kernel on V100. Each lane retains its own sorted top-16;
    // lane zero then performs a deterministic 256-way merge.
    float lane_scores[16];
    int   lane_ids[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        lane_scores[i] = -CUDART_INF_F;
        lane_ids[i] = INT_MAX;
    }
    for (int token = threadIdx.x; token < vocab; token += blockDim.x) {
        float score = __half2float(logits[row * vocab + token]) * multiplier;
        if (softcap > 0.f) {
            score = tanhf(score / softcap) * softcap;
        }
        int insert = 16;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (insert == 16
                && (score > lane_scores[i] || (score == lane_scores[i] && token < lane_ids[i]))) {
                insert = i;
            }
        }
#pragma unroll
        for (int i = 15; i > 0; --i) {
            if (i > insert) {
                lane_scores[i] = lane_scores[i - 1];
                lane_ids[i] = lane_ids[i - 1];
            }
        }
        if (insert < 16) {
            lane_scores[insert] = score;
            lane_ids[insert] = token;
        }
    }

    __shared__ float shared_scores[256][16];
    __shared__ int   shared_ids[256][16];
    __shared__ int   cursors[256];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        shared_scores[threadIdx.x][i] = lane_scores[i];
        shared_ids[threadIdx.x][i] = lane_ids[i];
    }
    cursors[threadIdx.x] = 0;
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int rank = 0; rank < 16; ++rank) {
            float best_score = -CUDART_INF_F;
            int   best_id = INT_MAX;
            int   best_lane = 0;
            for (int lane = 0; lane < 256; ++lane) {
                const int cursor = cursors[lane];
                const float candidate_score = shared_scores[lane][cursor];
                const int candidate_id = shared_ids[lane][cursor];
                if (candidate_score > best_score
                    || (candidate_score == best_score && candidate_id < best_id)) {
                    best_score = candidate_score;
                    best_id = candidate_id;
                    best_lane = lane;
                }
            }
            ++cursors[best_lane];
            ids[row * 16 + rank] = best_id + token_id_offset;
            scores[row * 16 + rank] = best_score;
        }
    }
}

__global__ void DFlashTopK16HalfCub(int*          ids,
                                     float*        scores,
                                     const __half* logits,
                                     int           rows,
                                     int           vocab,
                                     int           token_id_offset,
                                     float         multiplier,
                                     float         softcap)
{
    constexpr int kThreads = 256;
    constexpr int kTopK = 16;
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }

    TopK<float, kTopK> partial;
    partial.init();
    for (int token = threadIdx.x; token < vocab; token += kThreads) {
        float score = __half2float(logits[row * vocab + token]) * multiplier;
        if (softcap > 0.f) {
            score = tanhf(score / softcap) * softcap;
        }
        partial.insert(score, token);
    }

    using BlockReduce = cub::BlockReduce<TopK<float, kTopK>, kThreads>;
    __shared__ typename BlockReduce::TempStorage storage;
    const TopK<float, kTopK> total =
        BlockReduce(storage).Reduce(partial, reduce_topk_op<float, kTopK>);
    if (threadIdx.x == 0) {
#pragma unroll
        for (int rank = 0; rank < kTopK; ++rank) {
            ids[row * kTopK + rank] = total.p[rank] + token_id_offset;
            scores[row * kTopK + rank] = total.u[rank];
        }
    }
}

__global__ void DFlashMergeTopK16Kernel(int*         ids,
                                        float*       scores,
                                        const int*   gathered_ids,
                                        const float* gathered_scores,
                                        int          rows,
                                        int          tp_size,
                                        float        multiplier,
                                        float        softcap)
{
    const int row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) {
        return;
    }
    const int width = tp_size * 16;
    bool used[64]{};
    for (int rank = 0; rank < 16; ++rank) {
        float best_score = -CUDART_INF_F;
        int   best_id = 0;
        int   best_index = 0;
        for (int i = 0; i < width; ++i) {
            if (used[i]) {
                continue;
            }
            const int source_rank = i / 16;
            const int source_slot = i % 16;
            const int index = (source_rank * rows + row) * 16 + source_slot;
            const float score = gathered_scores[index];
            const int token = gathered_ids[index];
            if (score > best_score || (score == best_score && token < best_id)) {
                best_score = score;
                best_id = token;
                best_index = i;
            }
        }
        used[best_index] = true;
        if (softcap > 0.f) {
            best_score = tanhf(best_score * multiplier / softcap) * softcap;
        }
        else {
            best_score *= multiplier;
        }
        ids[row * 16 + rank] = best_id;
        scores[row * 16 + rank] = best_score;
    }
}

template<bool TraceScores>
__global__ void DFlashGreedySelectorHalf(int*         output,
                                         const int*   anchors,
                                         const int*   candidates,
                                         const float* unary,
                                         const __half* hidden,
                                         const __half* predecessor,
                                         const __half* successor,
                                         float*        trace_scores,
                                         int           batch_size,
                                         int           slots,
                                         int           top_k,
                                         int           rank,
                                         float         transition_scale)
{
    const int batch = blockIdx.x;
    if (batch >= batch_size) {
        return;
    }
    __shared__ int predecessor_id;
    __shared__ float candidate_scores[16];
    if (threadIdx.x == 0) {
        predecessor_id = anchors[batch];
    }
    __syncthreads();
    for (int slot = 0; slot < slots; ++slot) {
        const int candidate_index = threadIdx.x;
        if (candidate_index < top_k) {
            const int token = candidates[(batch * slots + slot) * top_k + candidate_index];
            const __half* pred = predecessor + predecessor_id * rank;
            const __half* state = hidden + (batch * slots + slot) * rank;
            const __half* succ = successor + token * rank;
            float score = unary[(batch * slots + slot) * top_k + candidate_index];
            // Preserve the former serial accumulation order exactly while
            // evaluating all 16 candidates concurrently.
            for (int i = 0; i < rank; ++i) {
                score += transition_scale * __half2float(pred[i]) * __half2float(state[i])
                         * __half2float(succ[i]);
            }
            candidate_scores[candidate_index] = score;
            if constexpr (TraceScores) {
                trace_scores[(batch * slots + slot) * top_k + candidate_index] = score;
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            float best_score = -CUDART_INF_F;
            int best_index = 0;
            for (int candidate_index = 0; candidate_index < top_k; ++candidate_index) {
                const float score = candidate_scores[candidate_index];
                if (score > best_score || (score == best_score && candidate_index < best_index)) {
                    best_score = score;
                    best_index = candidate_index;
                }
            }
            predecessor_id = candidates[(batch * slots + slot) * top_k + best_index];
            output[batch * slots + slot] = predecessor_id;
        }
        __syncthreads();
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

void invokeDFlashCastToFloat(Tensor& output, const Tensor& input, cudaStream_t stream)
{
    TM_CHECK_EQ(output.dtype(), kFloat32);
    TM_CHECK_EQ(input.dtype(), kHalf);
    TM_CHECK_EQ(output.size(), input.size());
    constexpr int threads = 256;
    const int blocks = (input.size() + threads - 1) / threads;
    DFlashCastToFloat<<<blocks, threads, 0, stream>>>(
        output.data<float>(), (const __half*)input.raw_data(), input.size());
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeDFlashCastToHalf(Tensor& output, const Tensor& input, cudaStream_t stream)
{
    TM_CHECK_EQ(output.dtype(), kHalf);
    TM_CHECK_EQ(input.dtype(), kFloat32);
    TM_CHECK_EQ(output.size(), input.size());
    constexpr int threads = 256;
    const int blocks = (input.size() + threads - 1) / threads;
    DFlashCastToHalf<<<blocks, threads, 0, stream>>>(
        (__half*)output.raw_data(), input.data<float>(), input.size());
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeDFlashRoundBFloat16(Tensor& value, cudaStream_t stream)
{
    TM_CHECK_EQ(value.dtype(), kHalf);
    constexpr int threads = 256;
    const int blocks = (value.size() + threads - 1) / threads;
    DFlashRoundBFloat16Half<<<blocks, threads, 0, stream>>>((__half*)value.raw_data(), value.size());
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeDFlashScale(Tensor& value, float scale, cudaStream_t stream)
{
    TM_CHECK_EQ(value.dtype(), kHalf);
    constexpr int threads = 256;
    const int blocks = (value.size() + threads - 1) / threads;
    DFlashScaleHalf<<<blocks, threads, 0, stream>>>((__half*)value.raw_data(), value.size(), scale);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeDFlashLagunaSilu(Tensor&       output,
                            Tensor&       row_scales,
                            const Tensor& gate_up,
                            float         gate_up_scale,
                            cudaStream_t  stream)
{
    TM_CHECK_EQ(output.dtype(), kHalf);
    TM_CHECK_EQ(gate_up.dtype(), kHalf);
    TM_CHECK_EQ(row_scales.dtype(), kFloat32);
    TM_CHECK_EQ(output.ndim(), 2);
    TM_CHECK_EQ(gate_up.ndim(), 2);
    TM_CHECK_EQ(gate_up.shape(0), output.shape(0));
    TM_CHECK_EQ(gate_up.shape(1), output.shape(1) * 2);
    TM_CHECK_EQ(row_scales.size(), output.shape(0));
    DFlashLagunaSiluHalf<<<output.shape(0), 256, 0, stream>>>((__half*)output.raw_data(),
                                                              row_scales.data<float>(),
                                                              (const __half*)gate_up.raw_data(),
                                                              output.shape(1),
                                                              gate_up_scale);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeDFlashInitialRMSNorm(Tensor&       output,
                                const Tensor& input,
                                const Tensor& weight,
                                float         eps,
                                bool          zero_centered,
                                cudaStream_t  stream)
{
    TM_CHECK_EQ(output.dtype(), kHalf);
    TM_CHECK_EQ(input.dtype(), kHalf);
    TM_CHECK_EQ(weight.dtype(), kHalf);
    TM_CHECK(output.shape() == input.shape());
    TM_CHECK_EQ(output.ndim(), 2);
    TM_CHECK_EQ(output.shape(1), weight.size());
    DFlashInitialRMSNormHalf<<<output.shape(0), 512, 0, stream>>>((__half*)output.raw_data(),
                                                                 (const __half*)input.raw_data(),
                                                                 (const __half*)weight.raw_data(),
                                                                 output.shape(1),
                                                                 eps,
                                                                 zero_centered);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeDFlashScaleRows(Tensor& value, const Tensor& row_scales, float scale, cudaStream_t stream)
{
    TM_CHECK_EQ(value.dtype(), kHalf);
    TM_CHECK_EQ(row_scales.dtype(), kFloat32);
    TM_CHECK_EQ(value.ndim(), 2);
    TM_CHECK_EQ(row_scales.size(), value.shape(0));
    constexpr int threads = 256;
    const int blocks = (value.size() + threads - 1) / threads;
    DFlashScaleRowsHalf<<<blocks, threads, 0, stream>>>((__half*)value.raw_data(),
                                                        row_scales.data<float>(),
                                                        value.shape(0),
                                                        value.shape(1),
                                                        scale);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeDFlashRankOrderedSum(Tensor& output, const Tensor& gathered, int ranks, cudaStream_t stream)
{
    TM_CHECK_EQ(output.dtype(), kHalf);
    TM_CHECK_EQ(gathered.dtype(), kHalf);
    TM_CHECK_GT(ranks, 0);
    TM_CHECK_EQ(gathered.size(), ranks * output.size());
    constexpr int threads = 256;
    const int blocks = (output.size() + threads - 1) / threads;
    DFlashRankOrderedSumHalf<<<blocks, threads, 0, stream>>>((__half*)output.raw_data(),
                                                             (const __half*)gathered.raw_data(),
                                                             output.size(),
                                                             ranks);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeDFlashTargetResidualRMSNorm(Tensor&       output,
                                       Tensor&       residual,
                                       const Tensor& reduced,
                                       const Tensor& bias,
                                       const Tensor& weight,
                                       float         eps,
                                       bool          zero_centered,
                                       cudaStream_t  stream)
{
    TM_CHECK_EQ(output.dtype(), kHalf);
    TM_CHECK_EQ(residual.dtype(), kFloat32);
    TM_CHECK_EQ(reduced.dtype(), kHalf);
    TM_CHECK_EQ(weight.dtype(), kHalf);
    TM_CHECK_EQ(output.ndim(), 2);
    TM_CHECK_EQ(output.size(), residual.size());
    TM_CHECK_EQ(output.size(), reduced.size());
    TM_CHECK_EQ(output.shape(1), weight.size());
    DFlashTargetResidualRMSNormHalf<<<output.shape(0), 512, 0, stream>>>((__half*)output.raw_data(),
                                                                         residual.data<float>(),
                                                                         (const __half*)reduced.raw_data(),
                                                                         (const __half*)bias.data_or((void*)nullptr),
                                                                         (const __half*)weight.raw_data(),
                                                                         output.shape(1),
                                                                         eps,
                                                                         zero_centered);
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeDFlashResidualRMSNorm(Tensor&       output,
                                 Tensor&       residual,
                                 const Tensor& reduced,
                                 const Tensor& bias,
                                 const Tensor& weight,
                                 float         eps,
                                 bool          zero_centered,
                                 float         reduced_scale,
                                 cudaStream_t  stream)
{
    TM_CHECK_EQ(output.dtype(), kHalf);
    TM_CHECK_EQ(residual.dtype(), kFloat32);
    TM_CHECK_EQ(reduced.dtype(), kHalf);
    TM_CHECK_EQ(weight.dtype(), kHalf);
    TM_CHECK_EQ(output.ndim(), 2);
    TM_CHECK_EQ(output.size(), residual.size());
    TM_CHECK_EQ(output.size(), reduced.size());
    TM_CHECK_EQ(output.shape(1), weight.size());
    static const bool full_product_rmsnorm = [] {
        const char* value = std::getenv("TM_DFLASH_FULL_PRODUCT_RMSNORM");
        return value && value[0] == '1';
    }();
    if (full_product_rmsnorm) {
        DFlashResidualRMSNormExactHalf<<<output.shape(0), 512, 0, stream>>>((__half*)output.raw_data(),
                                                                           residual.data<float>(),
                                                                           (const __half*)reduced.raw_data(),
                                                                           (const __half*)bias.data_or((void*)nullptr),
                                                                           (const __half*)weight.raw_data(),
                                                                           output.shape(1),
                                                                           eps,
                                                                           zero_centered,
                                                                           reduced_scale);
    }
    else {
        DFlashResidualRMSNormHalf<<<output.shape(0), 256, 0, stream>>>((__half*)output.raw_data(),
                                                                      residual.data<float>(),
                                                                      (const __half*)reduced.raw_data(),
                                                                      (const __half*)bias.data_or((void*)nullptr),
                                                                      (const __half*)weight.raw_data(),
                                                                      output.shape(1),
                                                                      eps,
                                                                      zero_centered,
                                                                      reduced_scale);
    }
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeDFlashTopK16(Buffer_<int>& ids,
                        Tensor&       scores,
                        const Tensor& logits,
                        int           valid_vocab,
                        int           token_id_offset,
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
    static const bool cub_topk = [] {
        const char* value = std::getenv("TM_DFLASH_CUB_TOPK");
        return !value || value[0] != '0';
    }();
    if (cub_topk) {
        DFlashTopK16HalfCub<<<logits.shape(0), 256, 0, stream>>>(ids.data(),
                                                                scores.data<float>(),
                                                                (const __half*)logits.raw_data(),
                                                                logits.shape(0),
                                                                valid_vocab,
                                                                token_id_offset,
                                                                output_multiplier,
                                                                softcap);
    }
    else {
        DFlashTopK16HalfLegacy<<<logits.shape(0), 256, 0, stream>>>(ids.data(),
                                                                   scores.data<float>(),
                                                                   (const __half*)logits.raw_data(),
                                                                   logits.shape(0),
                                                                   valid_vocab,
                                                                   token_id_offset,
                                                                   output_multiplier,
                                                                   softcap);
    }
    TM_CUDA_CHECK(cudaGetLastError());
}

void invokeDFlashMergeTopK16(Buffer_<int>&       ids,
                             Tensor&             scores,
                             const Buffer_<int>& gathered_ids,
                             const Tensor&       gathered_scores,
                             int                 tp_size,
                             float               output_multiplier,
                             float               softcap,
                             cudaStream_t        stream)
{
    TM_CHECK_GT(tp_size, 0);
    TM_CHECK_LE(tp_size, 4);
    TM_CHECK_EQ(scores.dtype(), kFloat32);
    TM_CHECK_EQ(gathered_scores.dtype(), kFloat32);
    const int rows = scores.shape(0);
    TM_CHECK_EQ(ids.size(), (ssize_t)rows * 16);
    TM_CHECK_EQ(gathered_ids.size(), (ssize_t)tp_size * rows * 16);
    TM_CHECK_EQ(gathered_scores.size(), (ssize_t)tp_size * rows * 16);
    DFlashMergeTopK16Kernel<<<rows, 1, 0, stream>>>(ids.data(),
                                                    scores.data<float>(),
                                                    gathered_ids.data(),
                                                    gathered_scores.data<float>(),
                                                    rows,
                                                    tp_size,
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
                                 cudaStream_t        stream,
                                 Tensor*             trace_scores)
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
    static const float transition_scale = [] {
        const char* value = std::getenv("TM_DFLASH_SELECTOR_TRANSITION_SCALE");
        return value && value[0] ? std::strtof(value, nullptr) : 2.f;
    }();
    TM_CHECK(std::isfinite(transition_scale));
    if (trace_scores) {
        TM_CHECK_EQ(trace_scores->dtype(), kFloat32);
        TM_CHECK_EQ(trace_scores->size(), (ssize_t)batch_size * slots * top_k);
        DFlashGreedySelectorHalf<true><<<batch_size, 32, 0, stream>>>(output.data(),
                                                                     anchors.data(),
                                                                     candidate_ids.data(),
                                                                     unary_scores.data<float>(),
                                                                     (const __half*)selector_hidden.raw_data(),
                                                                     (const __half*)predecessor_codebook.raw_data(),
                                                                     (const __half*)successor_codebook.raw_data(),
                                                                     trace_scores->data<float>(),
                                                                     batch_size,
                                                                     slots,
                                                                     top_k,
                                                                     rank,
                                                                     transition_scale);
    }
    else {
        DFlashGreedySelectorHalf<false><<<batch_size, 32, 0, stream>>>(output.data(),
                                                                      anchors.data(),
                                                                      candidate_ids.data(),
                                                                      unary_scores.data<float>(),
                                                                      (const __half*)selector_hidden.raw_data(),
                                                                      (const __half*)predecessor_codebook.raw_data(),
                                                                      (const __half*)successor_codebook.raw_data(),
                                                                      nullptr,
                                                                      batch_size,
                                                                      slots,
                                                                      top_k,
                                                                      rank,
                                                                      transition_scale);
    }
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
