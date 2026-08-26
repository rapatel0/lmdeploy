// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/mtp_predictor.h"

#include <cuda_runtime.h>

#include <algorithm>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/argmax.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_kernels.h"
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
    block_seq_len_{engine.cache_block_seq_len},
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

    // A MoE MTP layer carries both `moe_ffn` and `feed_forward` (qwen3_5.py
    // assigns the pair). DecodeStep runs only LlamaFfnLayer, which skips the
    // router, the routed experts and the combine. That does not fail -- it
    // produces drafts from the wrong computation, so acceptance would simply
    // be poor and look like a bad draft model. Refuse instead of guessing.
    TM_CHECK(!weights_.decoder_layer->moe_ffn)
        << "MTP: this checkpoint has a MoE draft layer; the predictor runs only the dense FFN, "
           "which would silently draft from the wrong computation";

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
    // The ACTIVE stream, not ctx_.stream. ctx_.stream is created once when the
    // model is constructed; each inference batch runs on the stream that
    // ModelExecutor installs. CUDA imposes no ordering between the two, so
    // launching predictor work on the construction stream can read inputs the
    // executor has not finished writing, and can free a temporary while a
    // kernel on the other stream is still using it.
    const auto stream = core::Context::stream().handle();

    // Report the real shapes before asserting on them. Guessing which operand
    // disagreed cost a full build-and-run cycle; the kernel's check names
    // neither tensor nor its extents.
    TM_CHECK(embedding.ndim() == 2 && hidden_states.ndim() == 2)
        << "MTP Project: expected 2D inputs, got embedding.ndim=" << embedding.ndim()
        << " hidden_states.ndim=" << hidden_states.ndim();
    TM_CHECK(embedding.shape(0) == batch_size && embedding.shape(1) == hidden_units_
             && hidden_states.shape(0) == batch_size && hidden_states.shape(1) == hidden_units_)
        << "MTP Project: shape mismatch. batch=" << batch_size << " hidden=" << hidden_units_
        << " embedding=[" << embedding.shape(0) << "," << embedding.shape(1) << "]"
        << " hidden_states=[" << hidden_states.shape(0) << "," << hidden_states.shape(1) << "]";

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
    TM_CUDA_CHECK(cudaGetLastError());

    Tensor normed_hidden{{batch_size, hidden_units_}, dtype_, kDEVICE};
    invokeRMSNorm(normed_hidden,
                  hidden_states,
                  weights_.pre_fc_norm_hidden->weight,
                  weights_.pre_fc_norm_hidden->norm_eps_,
                  weights_.pre_fc_norm_hidden->zero_centered_,
                  stream);
    TM_CUDA_CHECK(cudaGetLastError());

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
    TM_CUDA_CHECK(cudaGetLastError());

    Tensor projected{{batch_size, hidden_units_}, dtype_, kDEVICE};
    linear_.Forward(fused, *weights_.fc, projected);
    TM_CUDA_CHECK(cudaGetLastError());

    return projected;
}

Tensor MTPPredictor::DecodeStep(Tensor hidden, int phase, TensorMap& env)
{
    // The executor's active stream; see the note in Project().
    const auto  stream = core::Context::stream().handle();
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
    TM_CUDA_CHECK(cudaGetLastError());

    // Passing the MTP layer's own AttentionWeight is what routes the keys and
    // values into the MTP KV slot: the cache offset is read from
    // `weights.cache_block_offset`, which registration stamped onto this very
    // weight. `attn_index_` only names the layer for debug output.
    attn_layer_.Forward({phase, hidden, hidden, layer.attention.get(), attn_index_});
    TM_CUDA_CHECK(cudaGetLastError());

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
    TM_CUDA_CHECK(cudaGetLastError());

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
                                             int                 phase,
                                             const int*          seq_lens,
                                             TensorMap&          env)
{
    TM_CHECK_GT(batch_size, 0);
    TM_CHECK_GT(num_draft_tokens, 0);
    // `last_tokens` must already be sliced to the batch. Its source buffer is
    // allocated at max_batch_size and reused, so an unsliced view carries the
    // capacity instead of this step's batch, and the mismatch would otherwise
    // surface as a shape error deep inside an RMSNorm.
    TM_CHECK_EQ((int)last_tokens.size(), batch_size) << "MTP Draft: last_tokens must be sliced to the batch";

    // The executor's active stream; see the note in Project().
    const auto stream = core::Context::stream().handle();

    DraftResult result;
    // Buffer_ takes a signed ssize_t extent.
    result.draft_tokens = Buffer_<int>{(ssize_t)batch_size * num_draft_tokens, kDEVICE};
    result.num_drafts   = 0;

    // The checkpoint ships a single MTP layer, so depth beyond one is reached
    // by running that layer again with its own previous output: the token it
    // just drafted becomes the next input, and the hidden state it produced
    // becomes the next context. Each pass appends one entry to the MTP KV
    // slot, exactly as a real decode step would.
    Buffer_<int> cur_tokens = last_tokens;
    Tensor       cur_hidden = hidden_states;

    // The KV positions the draft attends to and writes.
    //
    // `k_offsets` is the target's cumulative key-length array, produced fresh
    // each Forward and consumed by attention as `cu_k_len`. The attention layer
    // BORROWS it, so mutating this buffer changes what the draft attends to.
    //
    // Without advancing it, every draft step wrote its K/V to the same position
    // and read a history ending at the target's last real token: step 2
    // attended as though step 1 had never happened. Measured, that produced
    // 56.2% / 16.1% / 3.3% / 0.0% across four steps -- step 1 correct by
    // accident, everything after it blind.
    //
    // Safe to mutate in place because this runs at the END of Forward and the
    // buffer is rebuilt by PrefixSum at the start of the next one. It is
    // restored below regardless, so nothing downstream inherits drafted
    // positions.
    Buffer_<int> k_offsets = env.at("k_offsets").buffer().view<int>();
    int          advanced  = 0;

    // How far the drafted positions may safely walk.
    //
    // The scheduler allocates KV blocks for `seq_len`, not `seq_len + K`. The
    // block iterator indexes `block_ptrs_[block_id_]` with no bounds check, so
    // advancing past the last allocated block reads a pointer belonging to the
    // next sequence -- or past the end of the array for the final row. That is
    // silent cross-sequence corruption, not a crash.
    //
    // Only the slack inside each row's last block is safe, and only the
    // smallest such slack across the batch, since one array bounds all rows.
    // A sequence sitting exactly on a boundary has zero slack, and then no
    // advance is permitted at all.
    // K advances, not K-1: the sampled token needs a position too.
    //
    // `k_offsets` is a PrefixSum over `sequence_length_` taken at
    // language_model.cc:615, which runs BEFORE sampling increments
    // sequence_length (sampling_kernels.cu:64). So on entry it describes a key
    // length of L covering positions 0..L-1, while the token being drafted
    // FROM -- the one sampling just produced -- belongs at position L.
    //
    // With q_len=1 attention derives history_len = cu_k_len - q_len = L-1 and
    // writes there, on top of the target's own last token, and every later
    // step inherits the shift. Step 1 still scored 64.1% because it attends to
    // a history that is correct up to L-1 and merely misplaces one entry; the
    // damage concentrates at depth, which is where the measured rates fell off
    // fastest.
    const int block_len  = block_seq_len_;
    int       max_extend = num_draft_tokens;
    if (block_len > 0) {
        for (int i = 0; i < batch_size; ++i) {
            const int len   = seq_lens[i];
            const int slack = (len % block_len == 0) ? 0 : block_len - (len % block_len);
            max_extend      = std::min(max_extend, slack);
        }
    }
    else {
        max_extend = 0;
    }
    if (max_extend < num_draft_tokens - 1 && !extend_warned_) {
        extend_warned_ = true;
        TM_LOG_WARNING(
            "[MTP] drafted positions limited to {} of {} steps by KV block capacity; "
            "steps beyond that reuse the last valid position and will draft poorly",
            max_extend,
            num_draft_tokens);
    }

    for (int step = 0; step < num_draft_tokens; ++step) {
        // Step 0 advances too, placing the sampled token at L instead of L-1.
        if (advanced < max_extend) {
            // Steps 1.. attend to the tokens the previous steps produced, so
            // each row's key length grows by one per step.
            AdvanceCuSeqLens(k_offsets.data(), batch_size, 1, stream);
            ++advanced;
        }

        // Embedding of the previous token, through the target's shared table.
        Tensor embedding = embed_fn_(cur_tokens);

        Tensor projected = Project(embedding, cur_hidden, batch_size);

        // Decode phase: the draft always extends one token per sequence.
        Tensor out = DecodeStep(std::move(projected), phase, env);

        Tensor logits = logits_fn_(out);

        // Write this step's tokens as the contiguous run for step `step`.
        //
        // The layout is [step][batch], not [batch][step]: one argmax per
        // sequence is produced per iteration, so a step is contiguous and a
        // sequence is strided by `batch_size`. The header documents this, and
        // the verifier must apply the stride when reading a single sequence's
        // drafts in order.
        Buffer_<int> step_out{
            result.draft_tokens.data() + (ssize_t)step * batch_size, (ssize_t)batch_size, kDEVICE};
        invokeArgmax(step_out, logits, stream);

        result.num_drafts = step + 1;

        // Feed forward for the next iteration.
        cur_tokens = step_out;
        cur_hidden = std::move(out);
    }

    // Restore the target's offsets. The next Forward recomputes them anyway,
    // but leaving a mutated buffer behind would make any future reader of
    // k_offsets after this point silently wrong, and that reader would have no
    // reason to suspect the draft.
    if (advanced) {
        AdvanceCuSeqLens(k_offsets.data(), batch_size, -advanced, stream);
    }

    return result;
}

}  // namespace turbomind
