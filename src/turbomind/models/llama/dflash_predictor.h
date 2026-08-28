// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <memory>
#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class DFlashConvWeight;
class DFlashWeight;
class LlamaFfnLayer;
class LlamaLinear;
class UnifiedAttentionLayer;

/// Runtime for the separate five-layer DFlash2 draft.
///
/// This initial slice owns the target-context projection. Later slices add
/// prompt KV materialization, the parallel draft block, and candidate choice.
class DFlashPredictor {
public:
    DFlashPredictor(const DFlashWeight&        weights,
                    UnifiedAttentionLayer&    attention,
                    std::vector<int>           attention_indices,
                    int                        attention_phase_base,
                    const EngineParam&         engine,
                    const Context&             ctx);
    ~DFlashPredictor();

    void SetupAttention(int phase, TensorMap& env);
    void PrepareAttention(int phase, TensorMap& env);

    /// Project [tokens, context_features * hidden] target residuals to the
    /// draft hidden width, then apply hidden_norm.
    Tensor ProjectContext(const Tensor& target_hidden) const;

    struct ConvState {
        Tensor output;
        Tensor delta;
    };

    ConvState PrepareGroupedConv(const Tensor& input, const DFlashConvWeight& weights) const;
    Tensor FinishGroupedConv(const Tensor& input, const Tensor& delta, const DFlashConvWeight& weights) const;

    /// Apply one convolution side for diagnostics.
    Tensor ApplyGroupedConv(const Tensor& input, const DFlashConvWeight& weights, int side) const;

    /// Execute the five-layer draft backbone after its attention plan is prepared.
    Tensor RunDraftLayers(Tensor hidden, int phase) const;

    /// Temporary correctness path: run full attention for each draft layer on
    /// projected target context, preserving its K/V in dedicated cache slots.
    void MaterializeContextKV(int target_phase, const Tensor& context) const;

private:
    const DFlashWeight& weights_;
    int                 hidden_units_{};
    int                 num_context_features_{};
    DataType              dtype_{};
    LlamaLinear&          linear_;
    UnifiedAttentionLayer& attention_;
    std::vector<int>       attention_indices_;
    int                    attention_phase_base_{-1};
    const Context&         ctx_;
    std::unique_ptr<LlamaFfnLayer> ffn_layer_;
};

}  // namespace turbomind
