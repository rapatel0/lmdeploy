// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/module.h"

namespace turbomind::core {

struct MTPLayerConfig: ModuleConfig {
    MTPLayerConfig(): ModuleConfig{"MTPLayerWeight"} {}
    template<typename Visitor>
    static void for_each(Visitor&&)
    {
    }
};

}  // namespace turbomind::core

namespace turbomind {

class DecoderLayerWeight;
class LinearWeight;
class NormWeight;

/// Weights for one Multi-Token Prediction layer.
///
/// Qwen3.5 and Qwen3.8 ship this module inside the target checkpoint under the
/// `mtp.` prefix, so it is not a separate draft model. The layer predicts token
/// t+1 from the target's final hidden state, and the target then verifies the
/// prediction.
///
/// Topology, from the checkpoint:
///
///   mtp.pre_fc_norm_embedding   normalize the token embedding branch
///   mtp.pre_fc_norm_hidden      normalize the hidden-state branch
///   mtp.fc                      concat both, project hidden*2 -> hidden
///   mtp.layers.0.*              one ordinary decoder layer
///   mtp.norm                    final norm before the shared lm_head
///
/// The decoder layer is always full attention, never linear attention. Both
/// reference implementations force this: SGLang sets `full_attention_interval`
/// to 1 for the one-layer MTP config, and the lmdeploy-v100 fork gives its
/// predictor a plain UnifiedAttentionLayer. Its projection shapes are identical
/// to a main full-attention layer, so it reuses DecoderLayerWeight unchanged.
///
/// `embed_tokens` and `lm_head` are deliberately absent. The checkpoint sets
/// `mtp_use_dedicated_embeddings` to false, so the MTP layer shares the
/// target's embedding and output projection rather than owning copies.
class MTPLayerWeight: public core::Module {
public:
    const char* type() const override
    {
        return "MTPLayerWeight";
    }

    MTPLayerWeight() = default;
    MTPLayerWeight(const core::ModuleConfig&);

    ~MTPLayerWeight() override;  // defined in .cc where child types are complete

    bool verify(std::vector<std::string>& missing) override;

    // --- X-macro field lists ---
#define MTP_LAYER_WEIGHT_CHILDREN(X)                                                                                   \
    X(NormWeight, pre_fc_norm_embedding)                                                                               \
    X(NormWeight, pre_fc_norm_hidden)                                                                                  \
    X(LinearWeight, fc)                                                                                                \
    X(DecoderLayerWeight, decoder_layer)                                                                               \
    X(NormWeight, final_norm)

#define MTP_LAYER_WEIGHT_PARAMS(X)

    TM_MODULE_DECLARE(MTPLayerWeight, MTP_LAYER_WEIGHT_CHILDREN, MTP_LAYER_WEIGHT_PARAMS)
};

}  // namespace turbomind
