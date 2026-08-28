// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"

namespace turbomind {

class DFlashWeight;
class LlamaLinear;

/// Runtime for the separate five-layer DFlash2 draft.
///
/// This initial slice owns the target-context projection. Later slices add
/// prompt KV materialization, the parallel draft block, and candidate choice.
class DFlashPredictor {
public:
    DFlashPredictor(const DFlashWeight& weights, const EngineParam& engine, const Context& ctx);
    ~DFlashPredictor();

    /// Project [tokens, context_features * hidden] target residuals to the
    /// draft hidden width, then apply hidden_norm.
    Tensor ProjectContext(const Tensor& target_hidden) const;

private:
    const DFlashWeight& weights_;
    int                 hidden_units_{};
    int                 num_context_features_{};
    DataType            dtype_{};
    LlamaLinear&        linear_;
};

}  // namespace turbomind
