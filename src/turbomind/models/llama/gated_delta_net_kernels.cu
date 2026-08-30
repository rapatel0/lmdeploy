
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/kernels/core/common.h"
#include "src/turbomind/models/llama/gated_delta_net_kernels.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cuda_bf16.h>
#include <type_traits>

#include "src/turbomind/utils/cuda_utils.h"

#include "src/turbomind/kernels/core/array_ops.h"

namespace turbomind {

template<class T>
__global__ void ComputeBetaGKernel(float*   beta,
                                   float*   g,
                                   int64_t  gate_token_stride,
                                   const T* b,
                                   int64_t  b_token_stride,
                                   const T* a,
                                   int64_t  a_token_stride,
                                   const T* A_log,
                                   const T* dt_bias,
                                   int      token_num,
                                   int      hv)
{
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = int64_t(token_num) * gate_token_stride;
    if (index >= total) {
        return;
    }
    const int token = static_cast<int>(index / gate_token_stride);
    const int head  = static_cast<int>(index % gate_token_stride);
    if (head >= hv) {
        beta[index] = 0.f;
        g[index]    = 0.f;
        return;
    }
    const float b_value  = static_cast<float>(b[token * b_token_stride + head]);
    const float a_value  = static_cast<float>(a[token * a_token_stride + head]);
    beta[index]          = 1.f / (1.f + expf(-b_value));
    const float x        = a_value + static_cast<float>(dt_bias[head]);
    const float softplus = x > 20.f ? x : log1pf(expf(x));
    g[index]             = -expf(static_cast<float>(A_log[head])) * softplus;
}

void ComputeBetaG(core::Tensor&       beta,
                  core::Tensor&       g,
                  const core::Tensor& b,
                  const core::Tensor& a,
                  const core::Tensor& A_log,
                  const core::Tensor& dt_bias,
                  cudaStream_t        stream)
{
    const int     token_num     = static_cast<int>(b.shape(0));
    const int     hv            = static_cast<int>(A_log.size());
    constexpr int kBlockThreads = 256;
    const int64_t gate_capacity = int64_t(token_num) * beta.stride(1);
    const int     gate_blocks   = static_cast<int>((gate_capacity + kBlockThreads - 1) / kBlockThreads);

    auto invoke = [&](auto t) {
        using T = decltype(t);
        ComputeBetaGKernel<T><<<gate_blocks, kBlockThreads, 0, stream>>>(beta.data<float>(),
                                                                         g.data<float>(),
                                                                         beta.stride(1),
                                                                         b.data<T>(),
                                                                         b.stride(0),
                                                                         a.data<T>(),
                                                                         a.stride(0),
                                                                         A_log.data<T>(),
                                                                         dt_bias.data<T>(),
                                                                         token_num,
                                                                         hv);
    };
    TM_DISPATCH_PRIMARY_DTYPES(b.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

template<class T>
__global__ void L2NormalizeQKKernel(T*      q,
                                    int64_t q_batch_stride,
                                    int64_t q_token_stride,
                                    T*      k,
                                    int64_t k_batch_stride,
                                    int64_t k_token_stride,
                                    int     token_num,
                                    int     hq,
                                    float   epsilon)
{
    constexpr int kHeadDim   = 128;
    const int     lane       = threadIdx.x;
    const int     work       = int(blockIdx.x);
    const int     batch      = work / (token_num * hq);
    const int     token_head = work % (token_num * hq);
    const int     token      = token_head / hq;
    const int     head       = token_head % hq;
    T* q_ptr = q + int64_t(batch) * q_batch_stride + int64_t(token) * q_token_stride + int64_t(head) * kHeadDim;
    T* k_ptr = k + int64_t(batch) * k_batch_stride + int64_t(token) * k_token_stride + int64_t(head) * kHeadDim;

    float q_values[4];
    float k_values[4];
    float q_sum = 0.f;
    float k_sum = 0.f;
    PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
        const int d = lane + i * WARP_SIZE;
        q_values[i] = static_cast<float>(q_ptr[d]);
        k_values[i] = static_cast<float>(k_ptr[d]);
        q_sum += q_values[i] * q_values[i];
        k_sum += k_values[i] * k_values[i];
    }
    PRAGMA_UNROLL
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        q_sum += __shfl_xor_sync(0xffffffff, q_sum, mask);
        k_sum += __shfl_xor_sync(0xffffffff, k_sum, mask);
    }
    const float q_inv = rsqrtf(q_sum + epsilon);
    const float k_inv = rsqrtf(k_sum + epsilon);
    PRAGMA_UNROLL
    for (int i = 0; i < 4; ++i) {
        const int d = lane + i * WARP_SIZE;
        q_ptr[d]    = T(q_values[i] * q_inv);
        k_ptr[d]    = T(k_values[i] * k_inv);
    }
}

void invokeL2NormalizeQK(core::Tensor& q, core::Tensor& k, float epsilon, cudaStream_t stream)
{
    constexpr int kWarpThreads = WARP_SIZE;
    const int     work         = static_cast<int>(q.shape(0) * q.shape(1) * q.shape(2));
    auto          invoke       = [&](auto t) {
        using T = decltype(t);
        L2NormalizeQKKernel<T><<<work, kWarpThreads, 0, stream>>>(q.data<T>(),
                                                                  q.stride(0),
                                                                  q.stride(1),
                                                                  k.data<T>(),
                                                                  k.stride(0),
                                                                  k.stride(1),
                                                                  static_cast<int>(q.shape(1)),
                                                                  static_cast<int>(q.shape(2)),
                                                                  epsilon);
    };
    TM_DISPATCH_PRIMARY_DTYPES(q.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// RMSNorm * SiLU-Gate (fused output normalization)
// =============================================================================
template<typename T>
__global__ void rms_norm_gated_kernel(
    T* hidden, const T* gate, const T* weight, float eps, int N, int head_dim, int gate_stride, int num_heads)
{
    const int row = blockIdx.x;
    if (row >= N)
        return;

    T*        h         = hidden + row * head_dim;
    const int token_idx = row / num_heads;
    const int head_idx  = row % num_heads;
    const T*  g         = gate + token_idx * gate_stride + head_idx * head_dim;

    __shared__ float smem[32];
    float            sum_sq = 0.0f;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float val = static_cast<float>(h[d]);
        sum_sq += val * val;
    }
    for (int mask = 16; mask > 0; mask >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
    if ((threadIdx.x & 31) == 0)
        smem[threadIdx.x >> 5] = sum_sq;
    __syncthreads();
    if (threadIdx.x >> 5 == 0) {
        sum_sq = (threadIdx.x < (blockDim.x + 31) / 32) ? smem[threadIdx.x] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1)
            sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, mask);
        if (threadIdx.x == 0)
            smem[0] = sum_sq;
    }
    __syncthreads();
    sum_sq = smem[0];

    float inv_rms = rsqrtf(sum_sq / (float)head_dim + eps);
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float h_val  = static_cast<float>(h[d]) * inv_rms * static_cast<float>(weight[d]);
        float g_val  = static_cast<float>(g[d]);
        float silu_g = g_val / (1.0f + expf(-g_val));
        h[d]         = static_cast<T>(h_val * silu_g);
    }
}

void invokeRMSNormGated(Ref<Tensor> hidden_, const Tensor& gate, const Tensor& weight, float eps, cudaStream_t stream)
{
    auto& hidden = hidden_.get();

    const int N           = hidden.shape(0);
    const int head_dim    = hidden.shape(1);
    const int token_num   = gate.shape(0);
    const int gate_stride = gate.stride(0);
    const int num_heads   = N / token_num;

    if (N == 0)
        return;

    const int threads = std::min(256, head_dim);

    auto invoke = [&](auto t) {
        using T = decltype(t);
        rms_norm_gated_kernel<<<N, threads, 0, stream>>>(
            hidden.data<T>(), gate.data<T>(), weight.data<T>(), eps, N, head_dim, gate_stride, num_heads);
    };
    TM_DISPATCH_PRIMARY_DTYPES(hidden.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

// =============================================================================
// Fused Conv1d + SiLU — persistent batched kernel
//
// Weight layout: [d_conv, conv_dim], State layout: [d_conv, conv_dim] per batch.
//
// Persistent 1D grid. Each block has a fixed channel tile
// (blockIdx.x % num_ch_tiles) and atomically claims single-token work items
// via a global counter. Token-major work ordering with grid size a multiple of
// num_ch_tiles guarantees monotonically increasing tokens and a fixed channel
// tile per block.
// =============================================================================
template<int D_CONV, int CHANNELS_PER_THREAD, int BLOCK_DIM, int NUM_TOKENS, typename T>
__global__ void __launch_bounds__(BLOCK_DIM) fused_conv1d_batched_kernel_v2(T*           out,
                                                                            const T*     in,
                                                                            const T*     weight,
                                                                            const T*     bias,
                                                                            void* const* conv_state_ptrs,
                                                                            const int*   q_offsets,
                                                                            const int*   k_offsets,
                                                                            const bool*  finished,
                                                                            int*         work_counter,
                                                                            int          batch_size,
                                                                            int          conv_dim,
                                                                            int          in_stride,
                                                                            int          num_token_tiles,
                                                                            int          state_layer_offset,
                                                                            int          total_work,
                                                                            int          num_ch_tiles,
                                                                            uint8_t*     state_snapshots,
                                                                            int64_t      snapshot_row_stride,
                                                                            int64_t      snapshot_step_stride,
                                                                            float*       prepare_beta,
                                                                            float*       prepare_g,
                                                                            const T*     prepare_A_log,
                                                                            const T*     prepare_dt_bias,
                                                                            int          prepare_mode,
                                                                            int          prepare_token_num,
                                                                            int          prepare_key_dim,
                                                                            int          prepare_value_heads,
                                                                            int          prepare_value_gate_offset,
                                                                            int          prepare_decay_gate_offset,
                                                                            int          prepare_gate_stride,
                                                                            float        prepare_epsilon)
{
    static_assert(BLOCK_DIM * CHANNELS_PER_THREAD > 0);

    int prev_ch_tile = -1;
    int c_base       = 0;

    Array<T, CHANNELS_PER_THREAD> w_tap[D_CONV];
    Array<T, CHANNELS_PER_THREAD> bias_vals;

    __shared__ int  s_work_id;
    __shared__ int4 s_batch_info;
    extern __shared__ char dynamic_shared[];
    T*                    norm_shared = reinterpret_cast<T*>(dynamic_shared);
    int                   b_start     = 0;

    while (true) {
        if (threadIdx.x == 0)
            s_work_id = atomicAdd(work_counter, 1);
        __syncthreads();

        if (s_work_id >= total_work)
            break;

        const int t_tile  = s_work_id % num_token_tiles;
        const int ch_tile = s_work_id / num_token_tiles;

        if (ch_tile != prev_ch_tile) {
            prev_ch_tile = ch_tile;
            b_start      = 0;
        }

        c_base = (ch_tile * BLOCK_DIM + threadIdx.x) * CHANNELS_PER_THREAD;

        const bool ch_active = (c_base < conv_dim);

        if (ch_active) {
            PRAGMA_UNROLL
            for (int d = 0; d < D_CONV; ++d) {
                Load(w_tap[d], weight + d * conv_dim + c_base);
            }
            if (bias)
                Load(bias_vals, bias + c_base);
        }

        if constexpr (NUM_TOKENS == 1) {
            for (int b = b_start + threadIdx.x; b < batch_size; b += BLOCK_DIM) {
                int lo = __ldg(&q_offsets[b]);
                if (lo > t_tile)
                    break;
                int hi = __ldg(&q_offsets[b + 1]);
                if (t_tile < hi) {
                    int seq      = hi - lo;
                    int hist     = (__ldg(&k_offsets[b + 1]) - __ldg(&k_offsets[b])) - seq;
                    s_batch_info = make_int4(b, lo, seq, hist);
                }
            }
        }
        else {
            for (int b = b_start + threadIdx.x; b < batch_size; b += BLOCK_DIM) {
                int tile_off = __ldg(&q_offsets[b]) / NUM_TOKENS + b;
                if (tile_off > t_tile)
                    break;
                int tile_off_next = __ldg(&q_offsets[b + 1]) / NUM_TOKENS + b + 1;
                if (t_tile < tile_off_next) {
                    int lo       = __ldg(&q_offsets[b]);
                    int seq      = __ldg(&q_offsets[b + 1]) - lo;
                    int hist     = (__ldg(&k_offsets[b + 1]) - __ldg(&k_offsets[b])) - seq;
                    s_batch_info = make_int4(b, lo, seq, hist);
                }
            }
        }
        __syncthreads();

        b_start = s_batch_info.x;

        const int4 bi          = s_batch_info;
        const int  b           = bi.x;
        const int  seq_off     = bi.y;
        const int  seq_len     = bi.z;
        const int  history_len = bi.w;

        int t_local_start;
        int n_tokens;
        if constexpr (NUM_TOKENS == 1) {
            t_local_start = t_tile - seq_off;
            n_tokens      = 1;
        }
        else {
            const int tile_off_b = seq_off / NUM_TOKENS + b;
            t_local_start        = (t_tile - tile_off_b) * NUM_TOKENS;
            if (t_local_start >= seq_len)
                continue;
            n_tokens = min(NUM_TOKENS, seq_len - t_local_start);
        }

        const bool skip_state_store = finished != nullptr && finished[b];
        const int  ring_start       = (history_len + t_local_start + 1) % D_CONV;
        T*         state_base       = (T*)conv_state_ptrs[b] + state_layer_offset;

        if (ch_active) {
            constexpr int                 VALS_SIZE = NUM_TOKENS + D_CONV - 1;
            Array<T, CHANNELS_PER_THREAD> vals[VALS_SIZE];
            const int                     n_vals = n_tokens + D_CONV - 1;

            PRAGMA_UNROLL
            for (int i = 0; i < VALS_SIZE; ++i) {
                if (i < n_vals) {
                    int pos = t_local_start - (D_CONV - 1) + i;
                    if (pos >= 0) {
                        Load(vals[i], in + (seq_off + pos) * in_stride + c_base);
                    }
                    else {
                        int ring_d = (ring_start + i) % D_CONV;
                        Load(vals[i], state_base + ring_d * conv_dim + c_base);
                    }
                }
            }

            PRAGMA_UNROLL
            for (int tok = 0; tok < NUM_TOKENS; ++tok) {
                if (tok < n_tokens) {
                    float acc[CHANNELS_PER_THREAD] = {};
                    PRAGMA_UNROLL
                    for (int d = 0; d < D_CONV; ++d) {
                        PRAGMA_UNROLL
                        for (int ch = 0; ch < CHANNELS_PER_THREAD; ++ch) {
                            acc[ch] += static_cast<float>(vals[tok + d][ch]) * static_cast<float>(w_tap[d][ch]);
                        }
                    }

                    Array<T, CHANNELS_PER_THREAD> out_vals;
                    PRAGMA_UNROLL
                    for (int ch = 0; ch < CHANNELS_PER_THREAD; ++ch) {
                        if (bias)
                            acc[ch] += static_cast<float>(bias_vals[ch]);
                        out_vals[ch] = static_cast<T>(acc[ch] / (1.0f + expf(-acc[ch])));
                    }

                    if (prepare_mode >= 2 && ch_tile * BLOCK_DIM * CHANNELS_PER_THREAD < 2 * prepare_key_dim) {
                        // Preserve the standalone kernel's FP16 boundary and
                        // exact lane-wise reduction order. Each 1024-channel
                        // tile contains eight aligned 128-wide Q or K heads.
                        T* tile = norm_shared + tok * BLOCK_DIM * CHANNELS_PER_THREAD;
                        Store(tile + threadIdx.x * CHANNELS_PER_THREAD, out_vals);
                        __syncthreads();
                        const int lane = threadIdx.x % WARP_SIZE;
                        const int warp = threadIdx.x / WARP_SIZE;
                        PRAGMA_UNROLL
                        for (int group = 0; group < 2; ++group) {
                            const int head = warp + group * (BLOCK_DIM / WARP_SIZE);
                            T*        ptr  = tile + head * 128;
                            float     values[4];
                            float     sum = 0.f;
                            PRAGMA_UNROLL
                            for (int i = 0; i < 4; ++i) {
                                const int d = lane + i * WARP_SIZE;
                                values[i]   = static_cast<float>(ptr[d]);
                                sum += values[i] * values[i];
                            }
                            PRAGMA_UNROLL
                            for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
                                sum += __shfl_xor_sync(0xffffffff, sum, mask);
                            }
                            const float inv = rsqrtf(sum + prepare_epsilon);
                            PRAGMA_UNROLL
                            for (int i = 0; i < 4; ++i) {
                                const int d = lane + i * WARP_SIZE;
                                ptr[d]     = T(values[i] * inv);
                            }
                        }
                        __syncthreads();
                        Load(out_vals, tile + threadIdx.x * CHANNELS_PER_THREAD);
                    }
                    Store(out + (seq_off + t_local_start + tok) * conv_dim + c_base, out_vals);

                    // Verification needs the recurrent frontier at every
                    // accepted prefix, not only after the full submitted run.
                    // The conv state is a ring of the last D_CONV inputs. Save
                    // that ring after this token into slot token+1. Each layer
                    // writes only its own state_layer_offset slice.
                    if (state_snapshots && !skip_state_store) {
                        T* snapshot = reinterpret_cast<T*>(state_snapshots
                                                           + int64_t(b) * snapshot_row_stride
                                                           + int64_t(t_local_start + tok + 1)
                                                                 * snapshot_step_stride)
                                      + state_layer_offset;
                        PRAGMA_UNROLL
                        for (int d = 0; d < D_CONV; ++d) {
                            const int ring_d = (ring_start + tok + d) % D_CONV;
                            Store(snapshot + ring_d * conv_dim + c_base, vals[tok + d]);
                        }
                    }
                }
            }

            if (!skip_state_store && t_local_start + n_tokens >= seq_len) {
                PRAGMA_UNROLL
                for (int i = 0; i < VALS_SIZE; ++i) {
                    int pos = t_local_start - (D_CONV - 1) + i;
                    if (pos >= 0 && pos >= seq_len - D_CONV && pos < seq_len) {
                        int ring_d = (ring_start + i) % D_CONV;
                        Store(state_base + ring_d * conv_dim + c_base, vals[i]);
                    }
                }
            }
        }

        if (prepare_mode >= 1 && ch_tile == 0 && t_tile == 0) {
            const int index = threadIdx.x;
            const int total = prepare_token_num * prepare_gate_stride;
            if (index < total) {
                const int token = index / prepare_gate_stride;
                const int head  = index % prepare_gate_stride;
                if (head < prepare_value_heads) {
                    const float b_value = static_cast<float>(
                        in[token * in_stride + prepare_value_gate_offset + head]);
                    const float a_value = static_cast<float>(
                        in[token * in_stride + prepare_decay_gate_offset + head]);
                    prepare_beta[index] = 1.f / (1.f + expf(-b_value));
                    const float x = a_value + static_cast<float>(prepare_dt_bias[head]);
                    const float softplus = x > 20.f ? x : log1pf(expf(x));
                    prepare_g[index] = -expf(static_cast<float>(prepare_A_log[head])) * softplus;
                }
                else {
                    prepare_beta[index] = 0.f;
                    prepare_g[index]    = 0.f;
                }
            }
        }
    }
}

void invokeFusedConv1dSiLU(Ref<Tensor>           out_,
                           const Tensor&         in,
                           const Tensor&         weight,
                           const Tensor&         bias,
                           const Buffer_<void*>& conv_state_ptrs,
                           const Buffer_<int>&   q_offsets,
                           const Buffer_<int>&   k_offsets,
                           const Buffer_<bool>&  finished,
                           int                   batch_size,
                           int                   state_layer_offset,
                           int                   sm_count,
                           int*                  work_counter,
                           cudaStream_t          stream,
                           uint8_t*              state_snapshots,
                           int64_t               snapshot_row_stride,
                           int64_t               snapshot_step_stride,
                           const FusedGdnPrepareParams* prepare)
{
    auto& out = out_.get();

    const int total_tokens = in.shape(0);
    const int d_conv       = weight.shape(0);
    const int conv_dim     = weight.shape(1);
    const int in_stride    = in.stride(0);

    constexpr int threads = 128;

    auto invoke = [&](auto t) {
        using T = decltype(t);
        if (d_conv == 4) {
            constexpr int kDConv     = 4;
            constexpr int kChPerT    = 8;
            const int     ch_per_blk = threads * kChPerT;
            TM_CHECK(conv_dim % kChPerT == 0);
            const int num_ch_tiles = cdiv(conv_dim, ch_per_blk);

            auto launch = [&](auto num_tok_tag) {
                constexpr int kNumTok         = decltype(num_tok_tag)::value;
                const int     num_token_tiles = (kNumTok == 1) ? total_tokens : total_tokens / kNumTok + batch_size;
                const int     total_work      = num_token_tiles * num_ch_tiles;
                const int     prepare_mode    = prepare ? prepare->mode : 0;
                TM_CHECK(prepare_mode >= 0 && prepare_mode <= 2);
                if (prepare_mode) {
                    TM_CHECK(batch_size == 1 && total_tokens == 8 && std::is_same_v<T, half>);
                    TM_CHECK(conv_dim % ch_per_blk == 0);
                    TM_CHECK(prepare->key_dim % ch_per_blk == 0);
                    TM_CHECK(total_tokens * prepare->gate_stride <= threads);
                }

                auto   kernel       = fused_conv1d_batched_kernel_v2<kDConv, kChPerT, threads, kNumTok, T>;
                size_t shared_bytes = prepare_mode >= 2 ? size_t(kNumTok) * ch_per_blk * sizeof(T) : 0;
                int    blocks_per_sm = 1;
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, threads, shared_bytes);
                int grid = min(total_work, blocks_per_sm * sm_count);

                cudaMemsetAsync(work_counter, 0, sizeof(int), stream);
                kernel<<<grid, threads, shared_bytes, stream>>>(out.data<T>(),
                                                     in.data<T>(),
                                                     weight.data<T>(),
                                                     bias ? bias.data<T>() : (T*)nullptr,
                                                     conv_state_ptrs.data(),
                                                     q_offsets.data(),
                                                     k_offsets.data(),
                                                     finished ? finished.data() : nullptr,
                                                     work_counter,
                                                     batch_size,
                                                     conv_dim,
                                                     in_stride,
                                                     num_token_tiles,
                                                     state_layer_offset,
                                                     total_work,
                                                     num_ch_tiles,
                                                     state_snapshots,
                                                     snapshot_row_stride,
                                                     snapshot_step_stride,
                                                     prepare_mode ? prepare->beta.data<float>() : nullptr,
                                                     prepare_mode ? prepare->g.data<float>() : nullptr,
                                                     prepare_mode ? prepare->A_log.data<T>() : nullptr,
                                                     prepare_mode ? prepare->dt_bias.data<T>() : nullptr,
                                                     prepare_mode,
                                                     total_tokens,
                                                     prepare_mode ? prepare->key_dim : 0,
                                                     prepare_mode ? prepare->value_heads : 0,
                                                     prepare_mode ? prepare->value_gate_offset : 0,
                                                     prepare_mode ? prepare->decay_gate_offset : 0,
                                                     prepare_mode ? prepare->gate_stride : 0,
                                                     prepare_mode ? prepare->epsilon : 0.f);
            };

            int avg_seq = total_tokens / batch_size;
            if (avg_seq >= 4)
                launch(std::integral_constant<int, 5>{});
            else
                launch(std::integral_constant<int, 1>{});
        }
        else {
            TM_LOG_FATAL("Only d_conv == 4 is supported by fused_conv1d_batched_kernel_v2");
        }
    };
    TM_DISPATCH_PRIMARY_DTYPES(out.dtype(), invoke);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind
