// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/rejection_sampling.h"
#include "src/turbomind/kernels/reduce_kernel_utils.cuh"

#include <climits>
#include <cstdlib>

#include <cub/block/block_reduce.cuh>
#include <cuda_fp16.h>
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

namespace turbomind {

// Greedy rejection sampling kernel: one block per batch element.
// Each block processes K+1 logit vectors sequentially, computing argmax for each,
// then compares against draft tokens to find the first mismatch.
template<typename T, bool OnePassAmbiguity>
__global__ void GreedyRejectKernel(int*       num_accepted,    // [batch]
                                   int*       bonus_tokens,    // [batch]
                                   int*       bonus_ambiguous, // [batch]
                                   const T*   logits,         // [batch, K+1, vocab_size]
                                   const int* draft_tokens,   // [batch, K]
                                   int        K,
                                   int        vocab_size,
                                   int        vocab_size_padded,
                                   const int* eos_ids,
                                   int        eos_ids_size,
                                   const int* eos_enable_positions,
                                   float      ambiguity_margin)
{
    const int b = blockIdx.x;

    // Stride by the PADDED size: that is how the logits are allocated.
    const T*   batch_logits = logits + (size_t)b * (K + 1) * vocab_size_padded;
    const int* batch_drafts = draft_tokens + (size_t)b * K;

    // Shared memory for the legacy argmax reduction and the one-pass top-2 arm.
    __shared__ float s_max_val[32];
    __shared__ int   s_max_idx[32];
    using BlockTop2 = cub::BlockReduce<TopK<float, 2>, 256>;
    __shared__ typename BlockTop2::TempStorage s_top2_storage;
    // Shared memory for broadcasting the selected token and ambiguity boundary.
    __shared__ int   s_argmax_token;
    __shared__ float s_argmax_value;
    __shared__ int   s_second_token;
    __shared__ float s_second_value;

    const int warp_id   = threadIdx.x / 32;
    const int lane_id   = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    int accepted       = K;  // assume all accepted, will be overwritten on mismatch
    int bonus          = 0;
    int ambiguous      = 0;
    int path_ambiguous = 0;

    for (int pos = 0; pos <= K; ++pos) {
        const T* pos_logits = batch_logits + (size_t)pos * vocab_size_padded;

        // Match the ordinary logits processor: EOS does not participate in
        // argmax before min_new_tokens. Rollback-time EOS handling is too late,
        // because the verifier already selected EOS instead of the canonical
        // next-best token.
        const bool eos_disabled = pos < eos_enable_positions[b];

        int target_ambiguous = 0;
        if constexpr (OnePassAmbiguity) {
            // The second entry of a deterministic top-2 reduction is exactly
            // the strongest alternative needed by the ambiguity predicate.
            // This removes the second full-vocabulary scan.
            TopK<float, 2> partial;
            partial.init();
            for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
                bool skip = false;
                if (eos_disabled) {
                    for (int e = 0; e < eos_ids_size; ++e) {
                        skip |= i == eos_ids[b * eos_ids_size + e];
                    }
                }
                if (!skip) {
                    partial.insert(static_cast<float>(pos_logits[i]), i);
                }
            }
            const TopK<float, 2> total =
                BlockTop2(s_top2_storage).Reduce(partial, reduce_topk_op<float, 2>);
            if (threadIdx.x == 0) {
                s_argmax_token = total.p[0] < 0 ? 0 : total.p[0];
                s_argmax_value = total.u[0];
                s_second_token = total.p[1];
                s_second_value = total.u[1];
            }
            __syncthreads();
            target_ambiguous = s_second_token >= 0 && s_second_value >= s_argmax_value - ambiguity_margin;
        }
        else {
            // Each thread finds its local maximum over strided vocabulary rows.
            float max_val = -1e30f;
            int   max_idx = INT_MAX;
            for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
                bool skip = false;
                if (eos_disabled) {
                    for (int e = 0; e < eos_ids_size; ++e) {
                        skip |= i == eos_ids[b * eos_ids_size + e];
                    }
                }
                if (skip) {
                    continue;
                }
                const float val = static_cast<float>(pos_logits[i]);
                if (val > max_val) {
                    max_val = val;
                    max_idx = i;
                }
            }

            for (int mask = 16; mask > 0; mask >>= 1) {
                const float other_val = __shfl_xor_sync(0xffffffff, max_val, mask);
                const int   other_idx = __shfl_xor_sync(0xffffffff, max_idx, mask);
                if (other_val > max_val || (other_val == max_val && other_idx < max_idx)) {
                    max_val = other_val;
                    max_idx = other_idx;
                }
            }

            if (lane_id == 0) {
                s_max_val[warp_id] = max_val;
                s_max_idx[warp_id] = max_idx;
            }
            __syncthreads();

            if (threadIdx.x < num_warps) {
                max_val = s_max_val[threadIdx.x];
                max_idx = s_max_idx[threadIdx.x];
            }
            else {
                max_val = -1e30f;
                max_idx = INT_MAX;
            }

            if (threadIdx.x < 32) {
                for (int mask = 16; mask > 0; mask >>= 1) {
                    const float other_val = __shfl_xor_sync(0xffffffff, max_val, mask);
                    const int   other_idx = __shfl_xor_sync(0xffffffff, max_idx, mask);
                    if (other_val > max_val || (other_val == max_val && other_idx < max_idx)) {
                        max_val = other_val;
                        max_idx = other_idx;
                    }
                }
            }

            if (threadIdx.x == 0) {
                s_argmax_token = max_idx == INT_MAX ? 0 : max_idx;
                s_argmax_value = max_val;
            }
            __syncthreads();

            bool tied = false;
            for (int i = threadIdx.x; i < vocab_size; i += blockDim.x) {
                bool skip = false;
                if (eos_disabled) {
                    for (int e = 0; e < eos_ids_size; ++e) {
                        skip |= i == eos_ids[b * eos_ids_size + e];
                    }
                }
                if (!skip && i != s_argmax_token
                    && static_cast<float>(pos_logits[i]) >= s_argmax_value - ambiguity_margin) {
                    tied = true;
                    break;
                }
            }
            target_ambiguous = __syncthreads_or(tied);
        }
        // Exact replay requires every committed decision in the prefix to be
        // numerically stable, not only the final bonus row. A draft can match
        // a chunked verifier's near-tied argmax while the canonical one-token
        // path chooses the other token.
        path_ambiguous |= target_ambiguous;

        // The ordinary top-k sampler uses the same value-descending,
        // id-ascending total order, so exact FP16 ties resolve identically.
        const int target_token = s_argmax_token;

        // Compare with draft token (positions 0..K-1)
        // Position K is the "bonus" position — no draft to compare against
        if (pos < K) {
            int draft = batch_drafts[pos];
            if (draft != target_token && accepted == K) {
                // First mismatch found
                accepted  = pos;
                bonus     = target_token;
                ambiguous = path_ambiguous;
            }
        }
        else {
            // pos == K: this is the bonus token position.
            if (accepted == K) {
                // All K drafts matched; bonus = argmax(logits[K])
                bonus     = target_token;
                ambiguous = path_ambiguous;
            }
        }

        // Early exit: once we found a mismatch and computed the bonus token,
        // no need to process remaining positions
        if (accepted < K) {
            break;
        }
    }

    if (threadIdx.x == 0) {
        num_accepted[b]   = accepted;
        bonus_tokens[b]   = bonus;
        bonus_ambiguous[b] = ambiguous;
    }
}

RejectionResult GreedyReject(const void*  verification_logits,
                              const int*   draft_tokens,
                              int          batch_size,
                              int          K,
                              int          vocab_size,
                              int          vocab_size_padded,
                              const int*   eos_ids,
                              int          eos_ids_size,
                              const int*   eos_enable_positions,
                              float        ambiguity_margin,
                              DataType     dtype,
                              cudaStream_t stream)
{
    RejectionResult result;
    result.num_accepted    = Buffer_<int>(batch_size, kDEVICE);
    result.bonus_tokens    = Buffer_<int>(batch_size, kDEVICE);
    result.bonus_ambiguous = Buffer_<int>(batch_size, kDEVICE);

    if (batch_size == 0 || K == 0) {
        return result;
    }

    constexpr int block = 256;
    const int     grid  = batch_size;

    int* d_num_accepted    = result.num_accepted.data();
    int* d_bonus_tokens    = result.bonus_tokens.data();
    int* d_bonus_ambiguous = result.bonus_ambiguous.data();

    static const bool one_pass_ambiguity = [] {
        const char* value = std::getenv("TM_DFLASH_ONE_PASS_REJECT");
        return !value || value[0] != '0';
    }();

#define TM_LAUNCH_GREEDY_REJECT(Type)                                                                                  \
    do {                                                                                                               \
        if (one_pass_ambiguity) {                                                                                      \
            GreedyRejectKernel<Type, true><<<grid, block, 0, stream>>>(d_num_accepted,                                \
                                                                        d_bonus_tokens,                                \
                                                                        d_bonus_ambiguous,                             \
                                                                        (const Type*)verification_logits,              \
                                                                        draft_tokens,                                  \
                                                                        K,                                             \
                                                                        vocab_size,                                    \
                                                                        vocab_size_padded,                             \
                                                                        eos_ids,                                       \
                                                                        eos_ids_size,                                  \
                                                                        eos_enable_positions,                          \
                                                                        ambiguity_margin);                             \
        }                                                                                                              \
        else {                                                                                                         \
            GreedyRejectKernel<Type, false><<<grid, block, 0, stream>>>(d_num_accepted,                               \
                                                                         d_bonus_tokens,                               \
                                                                         d_bonus_ambiguous,                            \
                                                                         (const Type*)verification_logits,             \
                                                                         draft_tokens,                                 \
                                                                         K,                                            \
                                                                         vocab_size,                                   \
                                                                         vocab_size_padded,                            \
                                                                         eos_ids,                                      \
                                                                         eos_ids_size,                                 \
                                                                         eos_enable_positions,                         \
                                                                         ambiguity_margin);                            \
        }                                                                                                              \
    } while (0)

    if (dtype == DataType::kFloat16) {
        TM_LAUNCH_GREEDY_REJECT(half);
    }
#ifdef ENABLE_BF16
    else if (dtype == DataType::kBfloat16) {
        TM_LAUNCH_GREEDY_REJECT(__nv_bfloat16);
    }
#endif
    else {
        TM_LAUNCH_GREEDY_REJECT(float);
    }
#undef TM_LAUNCH_GREEDY_REJECT

    return result;
}

}  // namespace turbomind
