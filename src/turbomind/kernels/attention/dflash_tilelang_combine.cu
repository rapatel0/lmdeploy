// Generated from SGLang's Apache-2.0 TileLang SM70 DFlash verifier.
// Specialization: B1, Q8, H8/HKV2, D128, FP16 KV, noncausal, 40 split slots.
#include "attention.h"
#include "rotary_embedding.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cuda.h>
#include <cstddef>
#include <mutex>
#include <type_traits>
#include <unordered_map>

#include "dflash_tilelang_ptx.inc"

namespace turbomind {
namespace {

void CheckDriver(CUresult result)
{
    if (result != CUDA_SUCCESS) {
        const char* name{};
        const char* message{};
        cuGetErrorName(result, &name);
        cuGetErrorString(result, &message);
        TM_CHECK(false) << "CUDA driver failure: " << (name ? name : "unknown") << ": "
                        << (message ? message : "unknown");
    }
}

struct DFlashTileLangDriverKernels {
    CUmodule   partial_module{};
    CUmodule   combine_module{};
    CUfunction partial{};
    CUfunction combine{};

    DFlashTileLangDriverKernels()
    {
        CheckDriver(cuModuleLoadData(&partial_module, kDFlashTileLangPartialPtx));
        CheckDriver(cuModuleLoadData(&combine_module, kDFlashTileLangCombinePtx));
        CheckDriver(cuModuleGetFunction(&partial, partial_module, "main_kernel"));
        CheckDriver(cuModuleGetFunction(&combine, combine_module, "main_kernel"));
        CheckDriver(cuFuncSetAttribute(partial, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 59392));
    }
};

DFlashTileLangDriverKernels& GetDFlashTileLangDriverKernels(bool allow_load)
{
    static std::mutex mutex;
    static std::unordered_map<CUcontext, DFlashTileLangDriverKernels*> per_context;
    CUcontext context{};
    CheckDriver(cuCtxGetCurrent(&context));
    TM_CHECK(context) << "TileLang verifier requires an active CUDA context";
    std::lock_guard<std::mutex> lock{mutex};
    auto it = per_context.find(context);
    if (it == per_context.end()) {
        TM_CHECK(allow_load) << "TileLang verifier modules were not prepared before graph execution";
        // The CUDA context owns module lifetime. Intentionally retain this
        // pair until process teardown instead of unloading after ctx destroy.
        it = per_context.emplace(context, new DFlashTileLangDriverKernels{}).first;
    }
    return *it->second;
}

__global__ void PackDFlashTileLangInputs(const half* __restrict__ q,
                                         int64_t q_stride,
                                         const half* __restrict__ flattened_kv,
                                         int flattened_head_stride,
                                         int flattened_value_offset,
                                         half* __restrict__ packed_q,
                                         half* __restrict__ packed_k,
                                         half* __restrict__ packed_v,
                                         int* __restrict__ identity_pages,
                                         const int* __restrict__ cu_k_len,
                                         int captured_context_len,
                                         int dynamic_context,
                                         RopeKernelParam rope_param,
                                         const float* __restrict__ rope_cache,
                                         int q_position_shift,
                                         bool q_pre_rotated)
{
    constexpr int kQElementsPerRow = 8 * 128;
    constexpr int kQPairsPerRow = kQElementsPerRow / 2;
    constexpr int kKVElementsPerToken = 2 * 128;
    const int context_len = dynamic_context ? cu_k_len[1] - cu_k_len[0] : captured_context_len;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < 8 * kQPairsPerRow;
         index += blockDim.x * gridDim.x) {
        const int row = index / kQPairsPerRow;
        const int pair = index % kQPairsPerRow;
        const int head = pair / 64;
        const int head_pair = pair % 64;
        const int src_col = head * 128 + head_pair * 2;
        Array<half, 2> value;
        value[0] = q[row * q_stride + src_col];
        value[1] = q[row * q_stride + src_col + 1];
        if (!q_pre_rotated) {
            const int position = context_len - 8 + q_position_shift + row;
            if (rope_cache) {
                const float cos_v = rope_cache[(int64_t)position * 128 + head_pair];
                const float sin_v = rope_cache[(int64_t)position * 128 + 64 + head_pair];
                const float first = __half2float(value[0]);
                const float second = __half2float(value[1]);
                value[0] = __float2half_rn(cos_v * first - sin_v * second);
                value[1] = __float2half_rn(cos_v * second + sin_v * first);
            }
            else {
                FastRoPE<2> rope(rope_param, 0, std::integral_constant<int, 2>{});
                rope.init(head_pair * 2);
                rope.apply(value, position, row);
            }
            // TurboMind stores each rotary pair adjacently. TileLang consumes
            // SGLang's NeoX layout with the two pair components in separate
            // half-heads, so deinterleave while packing the verifier ABI.
            const int dst_base = row * kQElementsPerRow + head * 128;
            packed_q[dst_base + head_pair] = value[0];
            packed_q[dst_base + 64 + head_pair] = value[1];
        }
        else {
            // Exact SGLang replay is already in the generated kernel's layout.
            packed_q[row * kQElementsPerRow + src_col] = value[0];
            packed_q[row * kQElementsPerRow + src_col + 1] = value[1];
        }
    }
    const int history_len = context_len - 8;
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < context_len * kKVElementsPerToken;
         index += blockDim.x * gridDim.x) {
        const int token = index / kKVElementsPerToken;
        const int rem = index % kKVElementsPerToken;
        const int head = rem / 128;
        const int dim = rem % 128;
        if (!q_pre_rotated && token >= history_len) {
            const int row = token - history_len;
            const int pair = dim % 64;
            const int component = dim / 64;
            const int pair_dim = pair * 2;
            const int k_offset = kQElementsPerRow + head * 128 + pair_dim;
            Array<half, 2> value;
            value[0] = q[row * q_stride + k_offset];
            value[1] = q[row * q_stride + k_offset + 1];
            if (rope_cache) {
                const float cos_v = rope_cache[(int64_t)token * 128 + pair];
                const float sin_v = rope_cache[(int64_t)token * 128 + 64 + pair];
                const float first = __half2float(value[0]);
                const float second = __half2float(value[1]);
                value[0] = __float2half_rn(cos_v * first - sin_v * second);
                value[1] = __float2half_rn(cos_v * second + sin_v * first);
            }
            else {
                FastRoPE<2> rope(rope_param, 0, std::integral_constant<int, 2>{});
                rope.init(pair_dim);
                rope.apply(value, token, row);
            }
            packed_k[index] = value[component];
            packed_v[index] = q[row * q_stride + kQElementsPerRow + 2 * 128 + head * 128 + dim];
        }
        else {
            const int source_k_dim = q_pre_rotated ? dim : (dim % 64) * 2 + dim / 64;
            packed_k[index] = flattened_kv[head * flattened_head_stride + token * 128 + source_k_dim];
            packed_v[index] =
                flattened_kv[head * flattened_head_stride + flattened_value_offset + token * 128 + dim];
        }
    }
    for (int page = blockIdx.x * blockDim.x + threadIdx.x; page < (context_len + 15) / 16;
         page += blockDim.x * gridDim.x) {
        identity_pages[page] = page;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        identity_pages[1024] = context_len;
    }
}

}  // namespace

void prepareDFlashTileLangAttention()
{
    (void)GetDFlashTileLangDriverKernels(true);
}

void dispatchDFlashTileLangAttention(const AttentionParams<half>& params,
                                      int                          context_len,
                                      half*                        packed_workspace,
                                      std::size_t                  packed_workspace_elements,
                                      int*                         metadata_workspace,
                                      bool                         graph_replay_safe,
                                      bool                         q_pre_rotated,
                                      half*                        trace_packed_q)
{
    constexpr int kMaxContext = 16 * 1024;
    constexpr int kQElementsPerRow = 8 * 128;
    constexpr int kPartialSmem = 59392;
    constexpr int kCombineSmem = 160;
    constexpr float kSoftmaxScale = 0.08838834764831845f;
    TM_CHECK_EQ(params.batch_size, 1);
    TM_CHECK_EQ(params.token_num, 8);
    TM_CHECK_EQ(params.num_heads, 8);
    TM_CHECK_EQ(params.num_kv_heads, 2);
    TM_CHECK_EQ(params.size_per_head, 128);
    TM_CHECK(!params.causal);
    TM_CHECK_GT(context_len, 0);
    TM_CHECK_LE(context_len, kMaxContext);

    auto* flattened_kv = reinterpret_cast<const half*>(params.linear_iter_params.kv_cache);
    const std::size_t elements_per_token = std::size_t{params.num_kv_heads} * 128;
    TM_CHECK(flattened_kv);
    TM_CHECK(packed_workspace);
    TM_CHECK(metadata_workspace);
    TM_CHECK_EQ(packed_workspace_elements % (2 * elements_per_token), 0);
    const std::size_t packed_capacity = packed_workspace_elements / (2 * elements_per_token);
    TM_CHECK_LE(static_cast<std::size_t>(context_len), packed_capacity);
    auto* packed_k = packed_workspace;
    // Keep K and V base addresses stable across graph replays whose live
    // context length changes. The pack kernel reads that length from cu_k_len.
    auto* packed_v = packed_k + packed_capacity * elements_per_token;
    auto* packed_q = reinterpret_cast<half*>(params.out);

    const int flattened_head_stride =
        graph_replay_safe ? params.linear_iter_params.stride_h : 2 * context_len * 128;
    const int flattened_value_offset =
        graph_replay_safe ? params.linear_iter_params.key_to_val : context_len * 128;
    PackDFlashTileLangInputs<<<32, 256, 0, params.stream>>>(reinterpret_cast<const half*>(params.q),
                                                            params.stride,
                                                            flattened_kv,
                                                            flattened_head_stride,
                                                            flattened_value_offset,
                                                            packed_q,
                                                            packed_k,
                                                            packed_v,
                                                            metadata_workspace,
                                                            params.cu_k_len,
                                                            context_len,
                                                            graph_replay_safe,
                                                            params.rope_param,
                                                            params.dflash_context_rope_cache,
                                                            params.q_position_shift,
                                                            q_pre_rotated);
    TM_CUDA_CHECK(cudaGetLastError());
    if (trace_packed_q) {
        TM_CUDA_CHECK(cudaMemcpyAsync(trace_packed_q,
                                      packed_q,
                                      8 * kQElementsPerRow * sizeof(half),
                                      cudaMemcpyDeviceToDevice,
                                      params.stream));
    }

    auto& kernels = GetDFlashTileLangDriverKernels(false);
    auto* partial_lse = params.partial_ML;
    auto* partial_o = reinterpret_cast<half*>(params.partial_O);
    auto* block_table = metadata_workspace;
    auto* cache_seqlens = metadata_workspace + 1024;
    auto* query_start_loc = params.cu_q_len;
    int nt = 8;
    float softmax_scale = kSoftmaxScale;
    void* partial_args[] = {&packed_k,
                            &partial_lse,
                            &partial_o,
                            &packed_q,
                            &packed_v,
                            &block_table,
                            &cache_seqlens,
                            &query_start_loc,
                            &nt,
                            &softmax_scale};
    CheckDriver(cuLaunchKernel(kernels.partial,
                               2,
                               40,
                               1,
                               256,
                               1,
                               1,
                               kPartialSmem,
                               reinterpret_cast<CUstream>(params.stream),
                               partial_args,
                               nullptr));

    void* combine_args[] = {&packed_q, &partial_lse, &partial_o, &cache_seqlens, &query_start_loc};
    CheckDriver(cuLaunchKernel(kernels.combine,
                               16,
                               8,
                               1,
                               128,
                               1,
                               1,
                               kCombineSmem,
                               reinterpret_cast<CUstream>(params.stream),
                               combine_args,
                               nullptr));
}

}  // namespace turbomind
