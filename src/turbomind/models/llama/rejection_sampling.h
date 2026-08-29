// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/core/core.h"

#include <cuda_runtime.h>

namespace turbomind {

struct RejectionResult {
    Buffer_<int> num_accepted;    // [batch] number of accepted drafts per request (0 ≤ N ≤ K)
    Buffer_<int> bonus_tokens;    // [batch] bonus token at first mismatch position
    Buffer_<int> bonus_ambiguous; // [batch] selected bonus shared the exact maximum with another token
};

/// Greedy rejection sampling: compare draft tokens against target logits.
///
/// For each request in the batch:
///   target[i] = argmax(verification_logits[i])
///   Accept draft[i] if draft[i] == target[i], for consecutive i starting from 0
///   First mismatch at position p: num_accepted = p, bonus_token = target[p]
///   All K match: num_accepted = K, bonus_token = argmax(logits[K])
///
/// @param verification_logits  [batch, K+1, vocab_size_padded] verification logits
/// @param draft_tokens         [batch, K] draft token IDs
/// @param batch_size           number of requests
/// @param K                    number of draft tokens per request
/// @param vocab_size           real vocabulary size; argmax searches only this far
/// @param vocab_size_padded    row stride of the logits, >= vocab_size
/// @param eos_ids              [batch, eos_ids_size] EOS token IDs, padded with -1
/// @param eos_ids_size         row stride of eos_ids
/// @param eos_enable_positions [batch] first verification position where EOS is valid
/// @param dtype                data type of logits (kFloat16, kBfloat16, or kFloat32)
/// @param stream               CUDA stream
///
/// vocab_size and vocab_size_padded are separate for the same reason the
/// sampler separates them: the logits are allocated as output_dim * tp_size,
/// which exceeds the true vocabulary whenever it does not divide evenly. The
/// padding is never written by the projection, so an argmax that searches it
/// can return an id outside the vocabulary; and striding by the unpadded size
/// would misalign every row after the first.
/// Reduce each rank-local FP16 logit row to exact top-2 candidates.
/// Output rows contain [score0, global_id0, score1, global_id1] as FP32.
void LocalTop2(const void*  local_logits,
               float*       candidates,
               int          rows,
               int          local_vocab_size,
               int          valid_vocab_size,
               int          token_id_offset,
               int          K,
               const int*   eos_ids,
               int          eos_ids_size,
               const int*   eos_enable_positions,
               DataType     dtype,
               cudaStream_t stream);

/// Launch compact rejection into caller-owned device output arrays.
void GreedyRejectTop2Raw(int*         num_accepted,
                         int*         bonus_tokens,
                         int*         bonus_ambiguous,
                         const float* gathered_candidates,
                         const int*   draft_tokens,
                         int          batch_size,
                         int          K,
                         int          tp_size,
                         float        ambiguity_margin,
                         cudaStream_t stream);

/// Reject drafts from rank-major gathered local top-2 candidates.
RejectionResult GreedyRejectTop2(const float* gathered_candidates,
                                 const int*   draft_tokens,
                                 int          batch_size,
                                 int          K,
                                 int          tp_size,
                                 float        ambiguity_margin,
                                 cudaStream_t stream);

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
                             cudaStream_t stream);

}  // namespace turbomind
