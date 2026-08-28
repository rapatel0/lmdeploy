// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include <cstdint>
#include <functional>
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
    using EmbedFn = std::function<Tensor(const Buffer_<int>&)>;
    using LogitsFn = std::function<Tensor(const Tensor&)>;
    using CandidatesFn =
        std::function<void(Buffer_<int>&, Tensor&, const Tensor&, float, float)>;

    DFlashPredictor(const DFlashWeight&        weights,
                    UnifiedAttentionLayer&    attention,
                    std::vector<int>           attention_indices,
                    int                        attention_phase_base,
                    const EngineParam&         engine,
                    int                        phases,
                    const Context&             ctx,
                    EmbedFn                    embed,
                    LogitsFn                   logits,
                    CandidatesFn               candidates);
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

    /// Build [anchor, mask x 7] per row and execute one parallel draft block.
    Tensor DraftBlock(const Buffer_<int>& anchors, int phase, TensorMap& env) const;

    Buffer_<int> SelectCandidates(const Tensor& block_hidden, const Buffer_<int>& anchors, int phase) const;

    /// Associate the next context projection with an eligible request.
    void ArmParityContext(uint64_t uid) const;

    /// Start the optional first-real-block parity capture.
    void BeginParityBlock(const Buffer_<int>& anchors, uint64_t uid, int seq_len, int input_len) const;

    /// Flush a complete context, draft, and selector capture.
    /// Returns true when the flush synchronized the model stream.
    bool FinishParityBlock() const;

    bool ParityActive() const;

    /// Temporary correctness path: run full attention for each draft layer on
    /// projected target context, preserving its K/V in dedicated cache slots.
    void MaterializeContextKV(int target_phase, const Tensor& context) const;

private:
    struct ParityTrace;

    struct Workspace {
        Buffer_<int> block_ids;
        Tensor       prediction_hidden;
        Buffer_<int> candidate_ids;
        Tensor       unary_scores;
        Tensor       selector_hidden;
        Buffer_<int> selected_ids;
        Tensor       selector_scores;
    };

    void CaptureParityTensor(const char* name, const Tensor& value) const;
    void PrepareParityContext(const Tensor& target_hidden, const Tensor& projected, const Tensor& normalized) const;

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
    EmbedFn                        embed_fn_;
    LogitsFn                       logits_fn_;
    CandidatesFn                   candidates_fn_;
    bool                           persistent_workspace_{};
    mutable std::vector<Workspace> workspaces_;
    mutable std::unique_ptr<ParityTrace> parity_trace_;
};

}  // namespace turbomind
