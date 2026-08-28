// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <cuda_runtime.h>

#include "src/turbomind/core/core.h"

namespace turbomind {

void invokeBuildDFlashBlock(Buffer_<int>&       output,
                            const Buffer_<int>& anchors,
                            int                 block_size,
                            int                 mask_token_id,
                            cudaStream_t        stream);

/// Apply one side of DFlash2's dynamic grouped depthwise convolution.
///
/// input/output: [token_num, hidden]
/// delta: [token_num, 2, taps, hidden / group_size]
/// base_kernel: [2, taps, hidden]
void invokeDFlashGroupedConv(Tensor&       output,
                             const Tensor& input,
                             const Tensor& delta,
                             const Tensor& base_kernel,
                             int           side,
                             int           block_size,
                             int           taps,
                             int           group_size,
                             cudaStream_t  stream);

}  // namespace turbomind
