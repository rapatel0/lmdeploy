// Generated from SGLang's Apache-2.0 TileLang SM70 DFlash verifier.
// Specialization: B1, Q8, H8/HKV2, D128, FP16 KV, noncausal, 40 split slots.
#include "attention.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cuda.h>
#include <cstddef>
#include <mutex>
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
                                         const int* __restrict__ cu_k_len)
{
    constexpr int kQElementsPerRow = 8 * 128;
    constexpr int kKVElementsPerToken = 2 * 128;
    const int context_len = cu_k_len[1] - cu_k_len[0];
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < 8 * kQElementsPerRow;
         index += blockDim.x * gridDim.x) {
        const int row = index / kQElementsPerRow;
        const int col = index % kQElementsPerRow;
        packed_q[index] = q[row * q_stride + col];
    }
    for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < context_len * kKVElementsPerToken;
         index += blockDim.x * gridDim.x) {
        const int token = index / kKVElementsPerToken;
        const int rem = index % kKVElementsPerToken;
        const int head = rem / 128;
        const int dim = rem % 128;
        packed_k[index] = flattened_kv[head * flattened_head_stride + token * 128 + dim];
        packed_v[index] =
            flattened_kv[head * flattened_head_stride + flattened_value_offset + token * 128 + dim];
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
                                      int*                         metadata_workspace)
{
    constexpr int kMaxContext = 16 * 1024;
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

    PackDFlashTileLangInputs<<<32, 256, 0, params.stream>>>(reinterpret_cast<const half*>(params.q),
                                                            params.stride,
                                                            flattened_kv,
                                                            params.linear_iter_params.stride_h,
                                                            params.linear_iter_params.key_to_val,
                                                            packed_q,
                                                            packed_k,
                                                            packed_v,
                                                            metadata_workspace,
                                                            params.cu_k_len);
    TM_CUDA_CHECK(cudaGetLastError());

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
