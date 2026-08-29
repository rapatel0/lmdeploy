#include "src/turbomind/kernels/linear_attn/kernel/pre_sm90/internal.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "src/turbomind/kernels/core/array_ops.h"
#include "src/turbomind/kernels/core/layout.h"
#include "src/turbomind/kernels/gemm/thread_map.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind::linear_attn::delta_rule::detail {

using namespace gemm;

template<int HeadDim, int ChunkSize, int ValueCols, int BlockDim, class T, class StateT>
__global__ void ChunkedGdrKernel(T*             out,
                                 int64_t        out_batch_stride,
                                 int64_t        out_token_stride,
                                 const T*       q,
                                 int64_t        q_batch_stride,
                                 int64_t        q_token_stride,
                                 const T*       k,
                                 int64_t        k_batch_stride,
                                 int64_t        k_token_stride,
                                 const T*       v,
                                 int64_t        v_batch_stride,
                                 int64_t        v_token_stride,
                                 const float*   beta,
                                 const float*   g,
                                 int64_t        gate_batch_stride,
                                 int64_t        gate_token_stride,
                                 const int64_t* state_ptrs,
                                 const int32_t* q_offsets,
                                 const bool*    finished,
                                 int            physical_token_slots,
                                 int            sequence_count,
                                 int            hq,
                                 int            hv,
                                 int64_t        state_layer_offset,
                                 int            num_head_groups,
                                 int            heads_per_block,
                                 uint8_t*       state_snapshots,
                                 int64_t        snapshot_row_stride,
                                 int64_t        snapshot_step_stride,
                                 int64_t        snapshot_recurrent_offset,
                                 int64_t        snapshot_block_bytes,
                                 int            snapshot_layer_group)
{
    constexpr int C = ChunkSize;
    constexpr int D = HeadDim;
    constexpr int V = ValueCols;
    static_assert(D % V == 0);

    constexpr int value_tiles = D / V;
    const int     head_block   = int(blockIdx.x) / value_tiles;
    const int     value_tile   = int(blockIdx.x) % value_tiles;
    const int     value_begin  = value_tile * V;
    const int     sequence     = head_block / hv;
    const int     value_head   = head_block % hv;
    const int key_head        = value_head / (hv / hq);
    const int token_begin     = q_offsets[sequence];
    const int sequence_length = q_offsets[sequence + 1] - token_begin;

    const int head_group = value_head / heads_per_block;
    const int local_head = value_head % heads_per_block;
    auto* state = reinterpret_cast<StateT*>(static_cast<uintptr_t>(state_ptrs[sequence * num_head_groups + head_group]))
                  + state_layer_offset + int64_t(local_head) * D * D;

    // Match RecurrentGdrKernel's decomposition exactly. Verification uses the
    // chunked path while ordinary decode uses recurrent; differing reduction
    // trees here are enough to flip a near-tied greedy argmax.
    constexpr int tile_k    = 16;
    constexpr int tile_v    = 4;
    constexpr int k_threads = D / tile_k;
    constexpr int v_threads = BlockDim / k_threads;
    constexpr int v_iters   = (V / tile_v + v_threads - 1) / v_threads;

    const int offset_k = threadIdx.x % k_threads;
    const int offset_v = threadIdx.x / k_threads;

    Array<float, tile_v> vec_S[v_iters][tile_k];

    extern __shared__ __align__(16) char smem_buf[];

    // Use the same direct state loads as RecurrentGdrKernel. Besides matching
    // its floating-point path, this supports tile_v=4 for both FP16 and FP32
    // state without routing through an eight-element shared-memory access.
    PRAGMA_UNROLL
    for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
        PRAGMA_UNROLL
        for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
            Array<StateT, tile_v> tmp;
            Load(tmp,
                 &state[(offset_k * tile_k + k_idx) * D
                        + value_begin + (offset_v + v_iter * v_threads) * tile_v]);
            vec_S[v_iter][k_idx] = cast<float>(tmp);
        }
    }

    constexpr int kSmemStride = D + 4;
    constexpr int vSmemStride = V + 4;
    float*        k_smem      = reinterpret_cast<float*>(smem_buf);
    float*        q_smem      = k_smem + C * kSmemStride;
    float*        v_smem      = q_smem + C * kSmemStride;
    float*        beta_vals   = v_smem + C * vSmemStride;
    float*        g_vals      = beta_vals + C;

    constexpr int kThreadsPerToken    = BlockDim / C;
    constexpr int kElementsPerThread  = D / kThreadsPerToken;
    constexpr int vElementsPerThread  = V / kThreadsPerToken;
    const int     load_token          = threadIdx.x / kThreadsPerToken;
    const int     load_lane           = threadIdx.x % kThreadsPerToken;

    const int chunk_count = (sequence_length + C - 1) / C;
    for (int chunk = 0; chunk < chunk_count; ++chunk) {
        const int chunk_start  = token_begin + chunk * C;
        const int valid_tokens = min(C, sequence_length - chunk * C);

        if (load_token < valid_tokens) {
            const int global_token   = chunk_start + load_token;
            const int physical_batch = global_token / physical_token_slots;
            const int token          = global_token % physical_token_slots;
            const T*  q_ptr =
                q + int64_t(physical_batch) * q_batch_stride + int64_t(token) * q_token_stride + int64_t(key_head) * D;
            const T* k_ptr =
                k + int64_t(physical_batch) * k_batch_stride + int64_t(token) * k_token_stride + int64_t(key_head) * D;
            const T* v_ptr = v + int64_t(physical_batch) * v_batch_stride + int64_t(token) * v_token_stride
                             + int64_t(value_head) * D;
            PRAGMA_UNROLL
            for (int element = 0; element < kElementsPerThread; ++element) {
                const int d                          = load_lane * kElementsPerThread + element;
                k_smem[load_token * kSmemStride + d] = float(k_ptr[d]);
                q_smem[load_token * kSmemStride + d] = float(q_ptr[d]);
            }
            PRAGMA_UNROLL
            for (int element = 0; element < vElementsPerThread; ++element) {
                const int d                          = load_lane * vElementsPerThread + element;
                v_smem[load_token * vSmemStride + d] = float(v_ptr[value_begin + d]);
            }
            if (load_lane == 0) {
                beta_vals[load_token] =
                    beta[int64_t(physical_batch) * gate_batch_stride + int64_t(token) * gate_token_stride + value_head];
                g_vals[load_token] =
                    g[int64_t(physical_batch) * gate_batch_stride + int64_t(token) * gate_token_stride + value_head];
            }
        }
        else {
            PRAGMA_UNROLL
            for (int element = 0; element < kElementsPerThread; ++element) {
                const int d                          = load_lane * kElementsPerThread + element;
                k_smem[load_token * kSmemStride + d] = 0.f;
                q_smem[load_token * kSmemStride + d] = 0.f;
            }
            PRAGMA_UNROLL
            for (int element = 0; element < vElementsPerThread; ++element) {
                const int d                          = load_lane * vElementsPerThread + element;
                v_smem[load_token * vSmemStride + d] = 0.f;
            }
        }
        __syncthreads();

        PRAGMA_UNROLL
        for (int token_in_chunk = 0; token_in_chunk < C; ++token_in_chunk) {
            if (token_in_chunk >= valid_tokens) {
                break;
            }

            const float beta_value = beta_vals[token_in_chunk];
            const float decay      = expf(g_vals[token_in_chunk]);
            float       vec_k[tile_k];
            float       vec_q[tile_k];
            PRAGMA_UNROLL
            for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
                vec_k[k_idx] = k_smem[token_in_chunk * kSmemStride + offset_k * tile_k + k_idx];
                vec_q[k_idx] = q_smem[token_in_chunk * kSmemStride + offset_k * tile_k + k_idx];
            }

            float kq = 0.f;
            PRAGMA_UNROLL
            for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
                kq += vec_k[k_idx] * vec_q[k_idx];
            }
            PRAGMA_UNROLL
            for (int mask = k_threads / 2; mask > 0; mask /= 2) {
                kq += __shfl_xor_sync(0xffffffff, kq, mask);
            }

            const int global_token   = chunk_start + token_in_chunk;
            const int physical_batch = global_token / physical_token_slots;
            const int token          = global_token % physical_token_slots;
            T*        out_ptr = out + int64_t(physical_batch) * out_batch_stride + int64_t(token) * out_token_stride
                         + int64_t(value_head) * D;

            PRAGMA_UNROLL
            for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
                const int value_local = (offset_v + v_iter * v_threads) * tile_v;
                const int value_base  = value_begin + value_local;
                float     vec_v[tile_v];
                PRAGMA_UNROLL
                for (int v_idx = 0; v_idx < tile_v; ++v_idx) {
                    vec_v[v_idx] = v_smem[token_in_chunk * vSmemStride + value_local + v_idx];
                }

                Array<T, tile_v> vec_out;
                PRAGMA_UNROLL
                for (int v_idx = 0; v_idx < tile_v; ++v_idx) {
                    float kv_memory = 0.f;
                    float sq        = 0.f;
                    PRAGMA_UNROLL
                    for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
                        const float s_decayed       = vec_S[v_iter][k_idx][v_idx] * decay;
                        vec_S[v_iter][k_idx][v_idx] = s_decayed;
                        kv_memory += s_decayed * vec_k[k_idx];
                        sq += s_decayed * vec_q[k_idx];
                    }
                    PRAGMA_UNROLL
                    for (int mask = k_threads / 2; mask > 0; mask /= 2) {
                        kv_memory += __shfl_xor_sync(0xffffffff, kv_memory, mask);
                        sq += __shfl_xor_sync(0xffffffff, sq, mask);
                    }
                    const float delta = (vec_v[v_idx] - kv_memory) * beta_value;
                    PRAGMA_UNROLL
                    for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
                        vec_S[v_iter][k_idx][v_idx] += vec_k[k_idx] * delta;
                    }
                    vec_out[v_idx] = T((sq + delta * kq) * rsqrtf(float(D)));
                }
                if (offset_k == 0) {
                    Store(&out_ptr[value_base], vec_out);
                }

                // Save this layer/head's recurrent matrix after every input
                // token. Acceptance is known only after the complete target
                // forward, so rollback later selects one of these frontiers.
                if (state_snapshots && !finished[sequence] && value_base < D) {
                    auto* snapshot = reinterpret_cast<StateT*>(
                                         state_snapshots
                                         + int64_t(sequence) * snapshot_row_stride
                                         + int64_t(chunk * C + token_in_chunk + 1) * snapshot_step_stride
                                         + snapshot_recurrent_offset
                                         + int64_t(snapshot_layer_group * num_head_groups + head_group)
                                               * snapshot_block_bytes)
                                     + state_layer_offset + int64_t(local_head) * D * D;
                    PRAGMA_UNROLL
                    for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
                        PRAGMA_UNROLL
                        for (int v_idx = 0; v_idx < tile_v; ++v_idx) {
                            if (value_base + v_idx < D) {
                                // A normal decode stores StateT after every
                                // token and reloads it for the next forward.
                                // Verification processes several positions in
                                // one chunk, so retaining the FP32 accumulator
                                // here would make both later logits and saved
                                // frontiers differ from sequential target
                                // decoding. Round-trip through StateT at every
                                // captured frontier to preserve exact recurrent
                                // semantics across the speculative block.
                                const StateT rounded = StateT(vec_S[v_iter][k_idx][v_idx]);
                                snapshot[(offset_k * tile_k + k_idx) * D + value_base + v_idx] = rounded;
                                vec_S[v_iter][k_idx][v_idx] = float(rounded);
                            }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }

    if (!finished[sequence]) {
        PRAGMA_UNROLL
        for (int v_iter = 0; v_iter < v_iters; ++v_iter) {
            PRAGMA_UNROLL
            for (int k_idx = 0; k_idx < tile_k; ++k_idx) {
                auto tmp = cast<StateT>(vec_S[v_iter][k_idx]);
                Store(&state[(offset_k * tile_k + k_idx) * D
                             + value_begin + (offset_v + v_iter * v_threads) * tile_v],
                      tmp);
            }
        }
    }
}

template<int ValueCols, class T, class StateT>
void RunPreSm90Chunk16(const Arguments& args, const Plan& plan, cudaStream_t stream)
{
    constexpr int kChunkSize = 16;
    constexpr int kBlockDim  = (128 / 16) * (ValueCols / 4);
    static_assert(kBlockDim >= kChunkSize);
    const int    value_tiles = 128 / ValueCols;
    const int    grid        = plan.problem.sequence_num * plan.problem.hv * value_tiles;
    const size_t smem_bytes  = (2 * kChunkSize * (128 + 4) + kChunkSize * (ValueCols + 4)
                               + 2 * kChunkSize)
                              * sizeof(float);
    auto kernel = ChunkedGdrKernel<128, kChunkSize, ValueCols, kBlockDim, T, StateT>;
    if (smem_bytes > (48u << 10)) {
        TM_CUDA_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)));
    }
    kernel<<<grid, kBlockDim, smem_bytes, stream>>>(args.out->data<T>(),
                                                    args.out->stride(0),
                                                    args.out->stride(1),
                                                    args.q.data<T>(),
                                                    args.q.stride(0),
                                                    args.q.stride(1),
                                                    args.k.data<T>(),
                                                    args.k.stride(0),
                                                    args.k.stride(1),
                                                    args.v.data<T>(),
                                                    args.v.stride(0),
                                                    args.v.stride(1),
                                                    args.beta.data<float>(),
                                                    args.g.data<float>(),
                                                    args.g.stride(0),
                                                    args.g.stride(1),
                                                    reinterpret_cast<const int64_t*>(args.state_ptrs.raw_data()),
                                                    args.q_offsets.data<int32_t>(),
                                                    args.finished.data<bool>(),
                                                    plan.problem.token_num,
                                                    plan.problem.sequence_num,
                                                    plan.problem.hq,
                                                    plan.problem.hv,
                                                    args.state_layer_offset,
                                                    plan.problem.num_head_groups,
                                                    plan.problem.heads_per_block,
                                                    args.state_snapshots,
                                                    args.snapshot_row_stride,
                                                    args.snapshot_step_stride,
                                                    args.snapshot_recurrent_offset,
                                                    args.snapshot_block_bytes,
                                                    args.snapshot_layer_group);
    TM_CUDA_CHECK(cudaGetLastError());
}

void RunPreSm90Chunk16F16F16(const Arguments& args, const Plan& plan, cudaStream_t stream)
{
    static const int value_cols = [] {
        const char* value = std::getenv("TM_GDN_SM70_VALUE_COLS");
        if (!value) {
            return 32;
        }
        const int parsed = std::atoi(value);
        return parsed == 16 || parsed == 32 || parsed == 64 || parsed == 128 ? parsed : 32;
    }();
    // This experiment is deliberately narrower than a shape-only Q=8
    // specialization. Ordinary eight-token prefill can have the same token
    // geometry, but only speculative verification supplies per-token rollback
    // frontiers. The remaining predicates pin the audited TP4 Qwen3.8 GDN
    // geometry and its default state-block layout.
    const bool eligible = args.state_snapshots != nullptr && args.snapshot_row_stride > 0
                          && args.snapshot_step_stride > 0 && args.snapshot_recurrent_offset > 0
                          && args.snapshot_block_bytes > 0 && plan.problem.token_num == 8
                          && plan.problem.sequence_num == 1 && plan.problem.hq == 4 && plan.problem.hv == 12
                          && plan.problem.num_head_groups == 1 && plan.problem.heads_per_block == 12;
    if (!eligible) {
        return RunPreSm90Chunk16<128, half, half>(args, plan, stream);
    }
    if (value_cols != 128) {
        int device = -1;
        TM_CUDA_CHECK(cudaGetDevice(&device));
        static std::atomic<bool> logged[16]{};
        TM_CHECK(device >= 0 && device < 16);
        if (!logged[device].exchange(true, std::memory_order_relaxed)) {
            std::fprintf(stderr,
                         "GDN_SM70_VALUE_COLS_ACTIVE device=%d value_cols=%d grid=%d block=%d\n",
                         device,
                         value_cols,
                         plan.problem.hv * (128 / value_cols),
                         (128 / 16) * (value_cols / 4));
            std::fflush(stderr);
        }
    }
    switch (value_cols) {
        case 16:
            return RunPreSm90Chunk16<16, half, half>(args, plan, stream);
        case 32:
            return RunPreSm90Chunk16<32, half, half>(args, plan, stream);
        case 64:
            return RunPreSm90Chunk16<64, half, half>(args, plan, stream);
        default:
            return RunPreSm90Chunk16<128, half, half>(args, plan, stream);
    }
}

#define DEFINE_PRE_SM90_CALLBACK(name, InputT, StateT)                                                                \
    void name(const Arguments& args, const Plan& plan, cudaStream_t stream)                                            \
    {                                                                                                                  \
        RunPreSm90Chunk16<128, InputT, StateT>(args, plan, stream);                                                    \
    }

DEFINE_PRE_SM90_CALLBACK(RunPreSm90Chunk16F16F32, half, float)
DEFINE_PRE_SM90_CALLBACK(RunPreSm90Chunk16Bf16Bf16, __nv_bfloat16, __nv_bfloat16)
DEFINE_PRE_SM90_CALLBACK(RunPreSm90Chunk16Bf16F32, __nv_bfloat16, float)

#undef DEFINE_PRE_SM90_CALLBACK

}  // namespace turbomind::linear_attn::delta_rule::detail
