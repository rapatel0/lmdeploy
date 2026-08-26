// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/mtp_predictor.h"

#include <cuda_runtime.h>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/mtp_weight.h"

namespace turbomind {

MTPPredictor::MTPPredictor(const MTPLayerWeight&  weights,
                           UnifiedAttentionLayer& attn_layer,
                           int                    attn_index,
                           const EngineParam&     engine,
                           const Context&         ctx):
    weights_{weights},
    attn_layer_{attn_layer},
    attn_index_{attn_index},
    hidden_units_{weights.fc ? weights.fc->input_dim / 2 : 0},
    tp_size_{engine.attn_tp_size},
    tp_rank_{engine.attn_dp_rank},
    dtype_{engine.data_type},
    linear_{*ctx.linear},
    ctx_{ctx}
{
    // The fc projection consumes the two normalised halves concatenated, so
    // its input is exactly twice the hidden size. Deriving hidden_units_ from
    // the weight rather than from config keeps the two from disagreeing.
    TM_CHECK_GT(hidden_units_, 0) << "MTP fc weight missing or malformed";
    TM_CHECK_GE(attn_index_, 0) << "MTP predictor built without a KV slot";

    TM_LOG_INFO("[MTP] predictor ready: hidden=%d attn_index=%d tp=%d", hidden_units_, attn_index_, tp_size_);
}

MTPPredictor::~MTPPredictor() = default;

Tensor MTPPredictor::Project(const Tensor& embedding, const Tensor& hidden_states, int batch_size)
{
    const auto stream = ctx_.stream->handle();

    // Two independent RMSNorms, one per input. They use their own eps and
    // zero_centered flags: Qwen3.5 is zero-centered, and passing the wrong
    // flag shifts every value by one without failing, so read both from the
    // weight rather than assuming.
    Tensor normed_emb{{batch_size, hidden_units_}, dtype_, kDEVICE};
    invokeRMSNorm(normed_emb,
                  embedding,
                  weights_.pre_fc_norm_embedding->weight,
                  weights_.pre_fc_norm_embedding->norm_eps_,
                  weights_.pre_fc_norm_embedding->zero_centered_,
                  stream);
    sync_check_cuda_error();

    Tensor normed_hidden{{batch_size, hidden_units_}, dtype_, kDEVICE};
    invokeRMSNorm(normed_hidden,
                  hidden_states,
                  weights_.pre_fc_norm_hidden->weight,
                  weights_.pre_fc_norm_hidden->norm_eps_,
                  weights_.pre_fc_norm_hidden->zero_centered_,
                  stream);
    sync_check_cuda_error();

    // Concatenate along the feature axis into [batch, 2 * hidden].
    //
    // The halves must be laid out per row, not per tensor: fc expects each
    // row to be [emb_row, hidden_row]. Copying one whole tensor after the
    // other would produce [all emb rows, all hidden rows], which is the same
    // bytes in the wrong order and yields silently wrong projections for any
    // batch above one.
    Tensor fused{{batch_size, hidden_units_ * 2}, dtype_, kDEVICE};
    {
        const size_t row_bytes = byte_size(dtype_, hidden_units_);
        auto* dst = static_cast<char*>(fused.raw_data());
        const auto* src_e = static_cast<const char*>(normed_emb.raw_data());
        const auto* src_h = static_cast<const char*>(normed_hidden.raw_data());
        // A strided 2D copy expresses this as two calls instead of 2*batch.
        TM_CUDA_CHECK(cudaMemcpy2DAsync(
            dst, row_bytes * 2, src_e, row_bytes, row_bytes, batch_size, cudaMemcpyDeviceToDevice, stream));
        TM_CUDA_CHECK(cudaMemcpy2DAsync(dst + row_bytes,
                                        row_bytes * 2,
                                        src_h,
                                        row_bytes,
                                        row_bytes,
                                        batch_size,
                                        cudaMemcpyDeviceToDevice,
                                        stream));
    }
    sync_check_cuda_error();

    Tensor projected{{batch_size, hidden_units_}, dtype_, kDEVICE};
    linear_.Forward(fused, *weights_.fc, projected);
    sync_check_cuda_error();

    return projected;
}

}  // namespace turbomind
