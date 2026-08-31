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

void invokeDFlashCastToHalf(Tensor& output, const Tensor& input, cudaStream_t stream);

/// Round FP16 activations through BF16 while retaining FP16 storage. DFlash2
/// was trained with BF16 residual/norm boundaries, which affect draft quality.
void invokeDFlashRoundBFloat16(Tensor& value, cudaStream_t stream);

void invokeDFlashScale(Tensor& value, float scale, cudaStream_t stream);

/// Reproduce Laguna's overflow-safe SM70 SwiGLU path. gate_up is the W13
/// projection of an input divided by 32; output is dynamically row-scaled for
/// W2 and row_scales records the factor restored after W2.
void invokeDFlashLagunaSilu(Tensor&       output,
                            Tensor&       row_scales,
                            const Tensor& gate_up,
                            float         gate_up_scale,
                            cudaStream_t  stream);

void invokeDFlashScaleRows(Tensor& value, const Tensor& row_scales, float scale, cudaStream_t stream);

/// Match SGLang Laguna's SM70 no-residual RMSNorm. Keep the normalized
/// activation and weight product in FP32, then round the output through BF16.
void invokeDFlashInitialRMSNorm(Tensor&       output,
                                const Tensor& input,
                                const Tensor& weight,
                                float         eps,
                                bool          zero_centered,
                                cudaStream_t  stream);

/// Sum rank-major gathered FP16 values in rank order with an FP32 accumulator,
/// matching SGLang's one-shot custom all-reduce arithmetic.
void invokeDFlashRankOrderedSum(Tensor& output, const Tensor& gathered, int ranks, cudaStream_t stream);

/// Match SGLang's target-model fused add RMSNorm contract: preserve the
/// updated residual in FP32 while emitting FP16 normalized activations.
void invokeDFlashTargetResidualRMSNorm(Tensor&       output,
                                       Tensor&       residual,
                                       const Tensor& reduced,
                                       const Tensor& bias,
                                       const Tensor& weight,
                                       float         eps,
                                       bool          zero_centered,
                                       cudaStream_t  stream);

void invokeDFlashResidualRMSNorm(Tensor&       output,
                                 Tensor&       residual,
                                 const Tensor& reduced,
                                 const Tensor& bias,
                                 const Tensor& weight,
                                 float         eps,
                                 bool          zero_centered,
                                 float         reduced_scale,
                                 cudaStream_t  stream);

void invokeDFlashTopK16(Buffer_<int>& ids,
                        Tensor&       scores,
                        const Tensor& logits,
                        int           valid_vocab,
                        int           token_id_offset,
                        float         output_multiplier,
                        float         softcap,
                        cudaStream_t  stream);

void invokeDFlashMergeTopK16(Buffer_<int>&       ids,
                             Tensor&             scores,
                             const Buffer_<int>& gathered_ids,
                             const Tensor&       gathered_scores,
                             int                 tp_size,
                             float               output_multiplier,
                             float               softcap,
                             cudaStream_t        stream);

void invokeDFlashGreedySelector(Buffer_<int>&       output,
                                 const Buffer_<int>& anchors,
                                 const Buffer_<int>& candidate_ids,
                                 const Tensor&       unary_scores,
                                 const Tensor&       selector_hidden,
                                 const Tensor&       predecessor_codebook,
                                 const Tensor&       successor_codebook,
                                 int                 slots,
                                 int                 top_k,
                                 cudaStream_t        stream,
                                 Tensor*             trace_scores = nullptr);

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
