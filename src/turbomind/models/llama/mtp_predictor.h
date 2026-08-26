// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

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
        /// Drafted token ids, [batch, num_drafts].
        Buffer_<int> draft_tokens;
        /// Number actually produced, which can be fewer than requested.
        int num_drafts{0};
    };

    MTPPredictor(const MTPLayerWeight& weights,
                 UnifiedAttentionLayer& attn_layer,
                 int                    attn_index,
                 const EngineParam&     engine,
                 const Context&         ctx);

    ~MTPPredictor();

    /// Produce up to `num_draft_tokens` drafts for each sequence in the batch.
    ///
    /// `hidden_states` is the target's final hidden state, [batch, hidden].
    /// `last_tokens` is the last accepted token per sequence, [batch].
    DraftResult Draft(int                 batch_size,
                      const Tensor&       hidden_states,
                      const Buffer_<int>& last_tokens,
                      int                 num_draft_tokens,
                      TensorMap&          env);

private:
    const MTPLayerWeight& weights_;

    /// Owned by UnifiedDecoder. The draft layer shares that instance and
    /// addresses its own KV slot through `attn_index_`.
    UnifiedAttentionLayer& attn_layer_;

    /// Position of the draft layer in the attention weight list, which selects
    /// its `cache_block_offset` and therefore its KV space.
    const int attn_index_;

    const int      hidden_units_;
    const int      tp_size_;
    const int      tp_rank_;
    const DataType dtype_;

    LlamaLinear& linear_;
    const Context& ctx_;
};

}  // namespace turbomind
