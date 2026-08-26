// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/mtp_predictor.h"

#include <cuda_runtime.h>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/argmax.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/mtp_weight.h"

namespace turbomind {

MTPPredictor::MTPPredictor(const MTPLayerWeight&  weights,
                           UnifiedAttentionLayer& attn_layer,
                           int                    attn_index,
                           const EngineParam&     engine,
                           const Context&         ctx,
                           EmbedFn                embed,
                           LogitsFn               logits):
    weights_{weights},
    attn_layer_{attn_layer},
    attn_index_{attn_index},
    hidden_units_{weights.fc ? weights.fc->input_dim / 2 : 0},
    tp_size_{engine.attn_tp_size},
    tp_rank_{engine.attn_dp_rank},
    dtype_{engine.data_type},
    linear_{*ctx.linear},
    ctx_{ctx},
    embed_fn_{std::move(embed)},
    logits_fn_{std::move(logits)}
{
    TM_CHECK(embed_fn_) << "MTP predictor needs the target's embedding lookup";
    TM_CHECK(logits_fn_) << "MTP predictor needs the target's lm_head";
    // The fc projection consumes the two normalised halves concatenated, so
    // its input is exactly twice the hidden size. Deriving hidden_units_ from
    // the weight rather than from config keeps the two from disagreeing.
    TM_CHECK_GT(hidden_units_, 0) << "MTP fc weight missing or malformed";
    TM_CHECK_GE(attn_index_, 0) << "MTP predictor built without a KV slot";

    // fmt placeholders, not printf: TM_LOG_INFO forwards to fmt::format, so a
    // %d would be printed literally and the value would never appear.
    // The MTP layer in this checkpoint carries a dense mlp.{gate,up,down}_proj
    // rather than an expert set, so the draft path uses LlamaFfnLayer. The two
    // `mtp.layers.0.mlp.gate` names in the config's exclude list have no
    // corresponding tensor on disk, which is consistent with that.
    ffn_layer_ = std::make_unique<LlamaFfnLayer>(ctx);

    TM_LOG_INFO("[MTP] predictor ready: hidden={} attn_index={} tp={}", hidden_units_, attn_index_, tp_size_);
}

MTPPredictor::~MTPPredictor() = default;

Tensor MTPPredictor::Project(const Tensor& embedding, const Tensor& hidden_states, int batch_size)
{
    const auto stream = ctx_.stream;

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

Tensor MTPPredictor::DecodeStep(Tensor hidden, int phase, TensorMap& env)
{
    // Context::stream is already a cudaStream_t; core_stream is the wrapper.
    const auto  stream = ctx_.stream;
    const auto& layer  = *weights_.decoder_layer;

    const int batch_size = hidden.shape(0);

    // Pre-attention norm, then attention. The residual is kept explicitly:
    // the attention layer writes its output in place, so the input has to be
    // preserved before the call rather than recovered after it.
    Tensor residual{{batch_size, hidden_units_}, dtype_, kDEVICE};
    TM_CUDA_CHECK(cudaMemcpyAsync(residual.raw_data(),
                                  hidden.raw_data(),
                                  byte_size(dtype_, (size_t)batch_size * hidden_units_),
                                  cudaMemcpyDeviceToDevice,
                                  stream));

    invokeRMSNorm(hidden,
                  hidden,
                  layer.attention_norm->weight,
                  layer.attention_norm->norm_eps_,
                  layer.attention_norm->zero_centered_,
                  stream);
    sync_check_cuda_error();

    // Passing the MTP layer's own AttentionWeight is what routes the keys and
    // values into the MTP KV slot: the cache offset is read from
    // `weights.cache_block_offset`, which registration stamped onto this very
    // weight. `attn_index_` only names the layer for debug output.
    attn_layer_.Forward({phase, hidden, hidden, layer.attention.get(), attn_index_});
    sync_check_cuda_error();

    // Fused residual-add plus the pre-FFN norm. This writes the post-add value
    // back into `residual` and the normalised value into `hidden`, so both are
    // carried forward correctly for the next residual.
    invokeResidualBiasRMSNorm(hidden.raw_data(),
                              residual.raw_data(),
                              layer.ffn_norm->weight.raw_data(),
                              nullptr,
                              dtype_,
                              hidden_units_,
                              batch_size,
                              layer.ffn_norm->norm_eps_,
                              layer.ffn_norm->zero_centered_,
                              stream);
    TM_CUDA_CHECK(cudaGetLastError());

    ffn_layer_->forward({hidden, hidden, layer.feed_forward.get(), attn_index_});
    sync_check_cuda_error();

    // Final residual add followed by the MTP block's own final_norm, leaving
    // `hidden` ready for the shared lm_head.
    invokeResidualBiasRMSNorm(hidden.raw_data(),
                              residual.raw_data(),
                              weights_.final_norm->weight.raw_data(),
                              nullptr,
                              dtype_,
                              hidden_units_,
                              batch_size,
                              weights_.final_norm->norm_eps_,
                              weights_.final_norm->zero_centered_,
                              stream);
    TM_CUDA_CHECK(cudaGetLastError());

    return hidden;
}

MTPPredictor::DraftResult MTPPredictor::Draft(int                 batch_size,
                                             const Tensor&       hidden_states,
                                             const Buffer_<int>& last_tokens,
                                             int                 num_draft_tokens,
                                             TensorMap&          env)
{
    TM_CHECK_GT(batch_size, 0);
    TM_CHECK_GT(num_draft_tokens, 0);

    const auto stream = ctx_.stream;

    DraftResult result;
    result.draft_tokens = Buffer_<int>{(size_t)batch_size * num_draft_tokens, kDEVICE};
    result.num_drafts   = 0;

    // The checkpoint ships a single MTP layer, so depth beyond one is reached
    // by running that layer again with its own previous output: the token it
    // just drafted becomes the next input, and the hidden state it produced
    // becomes the next context. Each pass appends one entry to the MTP KV
    // slot, exactly as a real decode step would.
    Buffer_<int> cur_tokens = last_tokens;
    Tensor       cur_hidden = hidden_states;

    for (int step = 0; step < num_draft_tokens; ++step) {
        // Embedding of the previous token, through the target's shared table.
        Tensor embedding = embed_fn_(cur_tokens);

        Tensor projected = Project(embedding, cur_hidden, batch_size);

        // Decode phase: the draft always extends one token per sequence.
        Tensor out = DecodeStep(std::move(projected), /*phase=*/0, env);

        Tensor logits = logits_fn_(out);

        // Write this step's tokens into column `step` of the [batch, K]
        // result. A per-step view keeps the layout batch-major, which is what
        // the verifier reads.
        Buffer_<int> step_out{result.draft_tokens.data() + (size_t)step * batch_size, (size_t)batch_size, kDEVICE};
        invokeArgmax(step_out, logits, stream);

        result.num_drafts = step + 1;

        // Feed forward for the next iteration.
        cur_tokens = step_out;
        cur_hidden = std::move(out);
    }

    return result;
}

}  // namespace turbomind
