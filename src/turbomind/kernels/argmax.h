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

}  // namespace turbomind
