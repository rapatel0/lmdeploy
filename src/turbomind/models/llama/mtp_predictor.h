// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"

namespace turbomind {

class MTPLayerWeight;
class LlamaFfnLayer;
class MoeFfnLayer;
class LlamaLinear;

/// Draft-token generator for Multi-Token Prediction.
///
/// The checkpoint ships one MTP layer, so a draft depth greater than one runs
/// that same layer repeatedly, feeding each drafted token back in as the next
/// input. The target then verifies the whole run in a single forward pass.
///
/// One draft step, from the reference implementation:
///
///   embed(last_token)      shared embed_tokens, not owned here
///     -> pre_fc_norm_embedding
///   hidden_state           from the target's final layer
///     -> pre_fc_norm_hidden
///   concat both            [batch, 2 * hidden]
///     -> fc                [hidden, 2 * hidden], BF16 in this checkpoint
///     -> attention         full attention, uses the MTP KV slot
///     -> ffn or moe
///     -> final_norm
///     -> lm_head           shared with the target
///     -> argmax            one draft token
///
/// Two things this class deliberately does not own. `embed_tokens` and
/// `lm_head` belong to the target, because the checkpoint sets
/// `mtp_use_dedicated_embeddings` to false. And the KV slot is allocated by
/// UnifiedDecoder during registration, not here; this class only addresses it.
class MTPPredictor {
public:
    struct DraftResult {
        /// Drafted token ids, laid out [step][batch]: step `k` occupies the
        /// contiguous run `draft_tokens[k * batch .. (k + 1) * batch)`.
        ///
        /// Step-major, not batch-major. Each iteration writes one argmax per
        /// sequence, so a step is naturally contiguous and a sequence is
        /// strided. The verifier walks steps in order for a given sequence and
        /// must apply that stride.
        Buffer_<int> draft_tokens;
        /// Number actually produced, which can be fewer than requested.
        int num_drafts{0};
    };

    /// Look up token embeddings through the target's shared table.
    using EmbedFn = std::function<Tensor(const Buffer_<int>&)>;
    /// Project hidden states to vocabulary logits through the target's lm_head.
    using LogitsFn = std::function<Tensor(const Tensor&)>;

    /// `embed` and `logits` are supplied by the target rather than owned here,
    /// because this checkpoint sets `mtp_use_dedicated_embeddings` to false:
    /// the draft layer shares the target's embedding table and lm_head. Taking
    /// them as callbacks keeps the tensor-parallel gather logic in the one
    /// place that already implements it.
    MTPPredictor(const MTPLayerWeight&  weights,
                 UnifiedAttentionLayer& attn_layer,
                 int                    attn_index,
                 int                    attn_phase,
                 const EngineParam&     engine,
                 const Context&         ctx,
                 EmbedFn                embed,
                 LogitsFn               logits);

    ~MTPPredictor();

    /// Produce up to `num_draft_tokens` drafts for each sequence in the batch.
    ///
    /// `hidden_states` is the target's final hidden state, [batch, hidden].
    /// `last_tokens` is the last accepted token per sequence, [batch].
    /// `phase` must be the caller's phase: with async_=1 two batches are in
    /// flight and the attention layer keeps per-phase state, so a hard-coded
    /// phase reads another batch's offsets and block pointers.
    DraftResult Draft(int                 batch_size,
                      const Tensor&       hidden_states,
                      const Buffer_<int>& last_tokens,
                      int                 num_draft_tokens,
                      int                 phase,
                      const int*          seq_lens,
                      /// Blocks allocated per row; bounds how far drafting may walk.
                      const int*          block_counts,
                      TensorMap&          env);

    /// Prepare the MTP attention layer's dispatch state for a decode-only batch.
    ///
    /// This must run before Draft(). Without it the draft attention reads
    /// uninitialised KV entries: UnifiedDecoder::Forward iterates
    /// weights_.layers_list(), which returns only the target's `layers` child,
    /// so nothing ever populates the MTP layer's own slot. Drafting from an
    /// empty cache is why the measured acceptance rate could not be trusted --
    /// the draft was attending to nothing.
    ///
    /// `env` must carry the batch, block pointers, q/k offsets and finished
    /// flags for the decode shape, one query token per sequence.
    /// Register the draft layer's KV slot in the draft's OWN phase slot.
    ///
    /// Needs the engine's setup-time env, the only one carrying `requests`.
    /// Runs against attn_phase_, not the target's phase, so the target's
    /// attention state is untouched.
    void SetupAttention(TensorMap& env);

    /// Build the draft's decode-shaped attention plan in its own phase slot.
    ///
    /// The draft submits one token per row while a verification forward submits
    /// K+1, and AttentionData is per-phase. Sharing the target's slot means one
    /// shape overwrites the other, which aborts on
    /// `d.prefill.q_sum + d.decode.n == q_count`.
    void PrepareAttention(TensorMap& env);

private:
    /// Normalise both inputs, concatenate them per row, and project back down
    /// to hidden size. Returns [batch, hidden].
    Tensor Project(const Tensor& embedding, const Tensor& hidden_states, int batch_size);

    /// Run the draft decoder block over `hidden`: norm, attention against the
    /// MTP KV slot, residual, norm, FFN, residual, then the MTP final_norm.
    /// Returns the value the shared lm_head consumes.
    Tensor DecodeStep(Tensor hidden, int phase, TensorMap& env);

    const MTPLayerWeight& weights_;

    /// Owned by UnifiedDecoder. The draft layer shares that instance and
    /// addresses its own KV slot through `attn_index_`.
    UnifiedAttentionLayer& attn_layer_;

    /// Warn once when KV block capacity truncates the drafted positions.
    // The last reported capacity cap. Initialised to a value max_extend can
    // never take so the first computation always reports, then reports only on
    // change. A once-only bool hid that the cap binds on the first draft and
    // never again.
    int last_warned_extend_{-1};

    /// Position of the draft layer in the attention weight list.
    ///
    /// This is passed to `ForwardParam::layer_id`, which is used for debug
    /// naming and warm-up bookkeeping. It is deliberately not what routes the
    /// KV write: the cache offset is read from `weights.cache_block_offset`,
    /// which registration stamped onto the MTP attention weight itself. So
    /// handing the attention layer the MTP weight pointer is what sends the
    /// draft's keys and values to their own slot.
    const int attn_index_;
    /// Phase slot owned by the draft, so its decode-shaped attention plan does
    /// not overwrite the target's.
    const int attn_phase_;

    const int      hidden_units_;
    const int      tp_size_;

    /// KV block length, needed to bound how far drafted positions may walk.
    const int block_seq_len_;
    const int      tp_rank_;
    const DataType dtype_;

    LlamaLinear&   linear_;
    const Context& ctx_;

    /// The MTP layer in this checkpoint has a dense mlp.{gate,up,down}_proj
    /// rather than an expert set, so the draft path is a plain FFN. The two
    /// `mtp.layers.0.mlp.gate` entries in the config's exclude list have no
    /// tensor on disk, consistent with there being no MoE here.
    std::unique_ptr<LlamaFfnLayer> ffn_layer_;

    const EmbedFn  embed_fn_;
    const LogitsFn logits_fn_;
};

}  // namespace turbomind
