// Copyright (c) OpenMMLab. All rights reserved.
#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/core/module.h"

namespace turbomind::core {

struct DFlashConvConfig: ModuleConfig {
    DFlashConvConfig(): ModuleConfig{"DFlashConvWeight"} {}

#define DFLASH_CONV_FIELDS(X)                                                                                          \
    X(DataType, data_type)                                                                                             \
    X(int, taps, 0)                                                                                                    \
    X(int, group_size, 0)

    DFLASH_CONV_FIELDS(TM_MEMBER)
    TM_FOR_EACH(DFlashConvConfig, DFLASH_CONV_FIELDS)

#undef DFLASH_CONV_FIELDS
};

struct DFlashSelectorConfig: ModuleConfig {
    DFlashSelectorConfig(): ModuleConfig{"DFlashSelectorWeight"} {}

#define DFLASH_SELECTOR_FIELDS(X)                                                                                      \
    X(DataType, data_type)                                                                                             \
    X(int, state_rank, 0)                                                                                              \
    X(int, top_k, 0)

    DFLASH_SELECTOR_FIELDS(TM_MEMBER)
    TM_FOR_EACH(DFlashSelectorConfig, DFLASH_SELECTOR_FIELDS)

#undef DFLASH_SELECTOR_FIELDS
};

struct DFlashWeightConfig: ModuleConfig {
    DFlashWeightConfig(): ModuleConfig{"DFlashWeight"} {}

#define DFLASH_WEIGHT_FIELDS(X)                                                                                        \
    X(DataType, data_type)                                                                                             \
    X(int, block_size, 8)                                                                                              \
    X(int, draft_window, 2048)                                                                                         \
    X(int, num_context_features, 0)

    DFLASH_WEIGHT_FIELDS(TM_MEMBER)
    TM_FOR_EACH(DFlashWeightConfig, DFLASH_WEIGHT_FIELDS)

#undef DFLASH_WEIGHT_FIELDS
};

}  // namespace turbomind::core

namespace turbomind {

class LinearWeight;
class NormWeight;

/// One grouped dynamic depthwise convolution adapter.
class DFlashConvWeight: public core::Module {
public:
    const char* type() const override
    {
        return "DFlashConvWeight";
    }

    DFlashConvWeight() = default;
    explicit DFlashConvWeight(const core::DFlashConvConfig& cfg);
    ~DFlashConvWeight() override;

#define DFLASH_CONV_CHILDREN(X) X(LinearWeight, kernel_projection)
#define DFLASH_CONV_PARAMS(X) X(base_kernel)

    TM_MODULE_DECLARE(DFlashConvWeight, DFLASH_CONV_CHILDREN, DFLASH_CONV_PARAMS)

    int taps{};
    int group_size{};
};

/// Candidate-path selector tables and hidden-state projection.
class DFlashSelectorWeight: public core::Module {
public:
    const char* type() const override
    {
        return "DFlashSelectorWeight";
    }

    DFlashSelectorWeight() = default;
    explicit DFlashSelectorWeight(const core::DFlashSelectorConfig& cfg);
    ~DFlashSelectorWeight() override;

#define DFLASH_SELECTOR_CHILDREN(X) X(LinearWeight, hidden_projection)
#define DFLASH_SELECTOR_PARAMS(X)                                                                                      \
    X(predecessor_codebook)                                                                                            \
    X(successor_codebook)

    TM_MODULE_DECLARE(DFlashSelectorWeight, DFLASH_SELECTOR_CHILDREN, DFLASH_SELECTOR_PARAMS)

    int state_rank{};
    int top_k{};
};

/// Root for a separate DFlash2 checkpoint.
///
/// The target owns token embeddings and lm_head. This module owns the draft's
/// context projection, five decoder layers, local convolution adapters, final
/// norm, and candidate selector. The parallel ModuleLists use matching indices
/// for decoder, attention-conv, and MLP-conv weights.
class DFlashWeight: public core::Module {
public:
    const char* type() const override
    {
        return "DFlashWeight";
    }

    DFlashWeight() = default;
    explicit DFlashWeight(const core::DFlashWeightConfig& cfg);
    ~DFlashWeight() override;

#define DFLASH_WEIGHT_CHILDREN(X)                                                                                      \
    X(LinearWeight, fc)                                                                                                \
    X(NormWeight, hidden_norm)                                                                                         \
    X(NormWeight, final_norm)                                                                                          \
    X(core::ModuleList, layers)                                                                                        \
    X(core::ModuleList, attention_convs)                                                                               \
    X(core::ModuleList, mlp_convs)                                                                                     \
    X(DFlashSelectorWeight, selector)
#define DFLASH_WEIGHT_PARAMS(X)

    TM_MODULE_DECLARE(DFlashWeight, DFLASH_WEIGHT_CHILDREN, DFLASH_WEIGHT_PARAMS)

    int block_size{};
    int draft_window{};
    int num_context_features{};
};

}  // namespace turbomind
