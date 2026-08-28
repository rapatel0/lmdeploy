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

void invokeGatherDFlashPredictions(Tensor& output, const Tensor& block_hidden, int block_size, cudaStream_t stream);

void invokeDFlashCastToFloat(Tensor& output, const Tensor& input, cudaStream_t stream);

/// Round FP16 activations through BF16 while retaining FP16 storage. DFlash2
/// was trained with BF16 residual/norm boundaries, which affect draft quality.
void invokeDFlashRoundBFloat16(Tensor& value, cudaStream_t stream);

void invokeDFlashResidualRMSNorm(Tensor&       output,
                                 Tensor&       residual,
                                 const Tensor& reduced,
                                 const Tensor& bias,
                                 const Tensor& weight,
                                 float         eps,
                                 bool          zero_centered,
                                 cudaStream_t  stream);

void invokeDFlashTopK16(Buffer_<int>& ids,
                        Tensor&       scores,
                        const Tensor& logits,
                        int           valid_vocab,
                        float         output_multiplier,
                        float         softcap,
                        cudaStream_t  stream);

void invokeDFlashGreedySelector(Buffer_<int>&       output,
                                 const Buffer_<int>& anchors,
                                 const Buffer_<int>& candidate_ids,
                                 const Tensor&       unary_scores,
                                 const Tensor&       selector_hidden,
                                 const Tensor&       predecessor_codebook,
                                 const Tensor&       successor_codebook,
                                 int                 slots,
                                 int                 top_k,
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
