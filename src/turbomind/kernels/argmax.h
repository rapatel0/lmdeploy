// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"

namespace turbomind {

/// Greedy token selection: `out[i] = argmax_j logits[i][j]`.
///
/// Ties resolve to the lowest index. That is required rather than incidental:
/// draft verification compares the draft's token against the target's, so both
/// sides must break a tie the same way or an agreeing pair looks like a
/// rejection.
void invokeArgmax(Buffer_<int>& out, const Tensor& logits, cudaStream_t st);

/// Produce `[score, global_token_id]` for each row's local vocabulary shard.
/// Token ids are stored as float because all supported vocabularies are far
/// below FP32's exact-integer limit; this packs each candidate into one tiny
/// homogeneous TP all-gather.
void invokeLocalArgmax(
    Buffer_<float>& candidates, const Tensor& local_logits, int valid_vocab, int token_id_offset, cudaStream_t st);

/// Select the global winner from candidates laid out `[rank, row, 2]`.
void invokeGlobalArgmax(
    Buffer_<int>& out, const Buffer_<float>& gathered_candidates, int rows, int ranks, cudaStream_t st);

}  // namespace turbomind
