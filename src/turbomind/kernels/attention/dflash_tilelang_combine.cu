// Generated from SGLang's Apache-2.0 TileLang SM70 DFlash verifier.
// Specialization: B1, Q8, H8/HKV2, D128, FP16 KV, noncausal, 40 split slots.
#include "attention.h"
#include "src/turbomind/utils/cuda_utils.h"

#include <cuda.h>
#include <cstddef>
#include <mutex>

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

DFlashTileLangDriverKernels& GetDFlashTileLangDriverKernels()
{
    constexpr int kMaxDevices = 32;
    static std::mutex mutex;
    static DFlashTileLangDriverKernels* per_device[kMaxDevices]{};
    int device = -1;
    TM_CUDA_CHECK(cudaGetDevice(&device));
    TM_CHECK_GE(device, 0);
    TM_CHECK_LT(device, kMaxDevices);
    std::lock_guard<std::mutex> lock{mutex};
    if (!per_device[device]) {
        // Driver modules and function handles belong to the CUDA context in
        // which they were loaded. TurboMind TP ranks share one process but use
        // separate device contexts, so each device needs its own module pair.
        per_device[device] = new DFlashTileLangDriverKernels{};
    }
    return *per_device[device];
}

__global__ void PackDFlashTileLangInputs(const half* __restrict__ q,
                                         int64_t q_stride,
                                         const half* __restrict__ flattened_kv,
                                         half* __restrict__ packed_q,
                                         half* __restrict__ packed_k,
                                         half* __restrict__ packed_v,
                                         int* __restrict__ identity_pages,
                                         int context_len)
{
    constexpr int kQElementsPerRow = 8 * 128;
    constexpr int kKVElementsPerToken = 2 * 128;
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
        const int head_stride = 2 * context_len * 128;
        packed_k[index] = flattened_kv[head * head_stride + token * 128 + dim];
        packed_v[index] = flattened_kv[head * head_stride + context_len * 128 + token * 128 + dim];
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

void dispatchDFlashTileLangAttention(const AttentionParams<half>& params,
                                      int context_len,
                                      std::size_t workspace_elements)
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

    auto* workspace = reinterpret_cast<half*>(const_cast<void*>(params.linear_iter_params.kv_cache));
    const std::size_t flattened_elements = std::size_t{2} * params.num_kv_heads * context_len * 128;
    const std::size_t packed_elements = std::size_t{2} * context_len * params.num_kv_heads * 128;
    TM_CHECK_LE(flattened_elements + packed_elements, workspace_elements);
    auto* packed_k = workspace + flattened_elements;
    auto* packed_v = packed_k + context_len * params.num_kv_heads * 128;
    auto* packed_q = reinterpret_cast<half*>(params.out);

    PackDFlashTileLangInputs<<<32, 256, 0, params.stream>>>(reinterpret_cast<const half*>(params.q),
                                                            params.stride,
                                                            workspace,
                                                            packed_q,
                                                            packed_k,
                                                            packed_v,
                                                            params.split_cnt,
                                                            context_len);
    TM_CUDA_CHECK(cudaGetLastError());

    auto& kernels = GetDFlashTileLangDriverKernels();
    auto* partial_lse = params.partial_ML;
    auto* partial_o = reinterpret_cast<half*>(params.partial_O);
    auto* block_table = params.split_cnt;
    auto* cache_seqlens = params.split_cnt + 1024;
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
