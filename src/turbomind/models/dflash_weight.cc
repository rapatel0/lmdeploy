// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/dflash_weight.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/core/registry.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"

namespace turbomind {

DFlashConvWeight::DFlashConvWeight(const core::DFlashConvConfig& cfg):
    taps(cfg.taps), group_size(cfg.group_size)
{
    TM_CHECK_GT(taps, 0);
    TM_CHECK_GT(group_size, 0);
}

DFlashConvWeight::~DFlashConvWeight() = default;

DFlashSelectorWeight::DFlashSelectorWeight(const core::DFlashSelectorConfig& cfg):
    state_rank(cfg.state_rank), top_k(cfg.top_k)
{
    TM_CHECK_GT(state_rank, 0);
    TM_CHECK_GT(top_k, 0);
}

DFlashSelectorWeight::~DFlashSelectorWeight() = default;

DFlashWeight::DFlashWeight(const core::DFlashWeightConfig& cfg):
    block_size(cfg.block_size), draft_window(cfg.draft_window), num_context_features(cfg.num_context_features)
{
    TM_CHECK_GE(block_size, 2);
    TM_CHECK_GT(draft_window, 0);
    TM_CHECK_GT(num_context_features, 0);
}

DFlashWeight::~DFlashWeight() = default;

TM_MODULE_REGISTER(DFlashConvWeight, core::DFlashConvConfig);
TM_MODULE_METHODS(DFlashConvWeight, DFLASH_CONV_CHILDREN, DFLASH_CONV_PARAMS)

TM_MODULE_REGISTER(DFlashSelectorWeight, core::DFlashSelectorConfig);
TM_MODULE_METHODS(DFlashSelectorWeight, DFLASH_SELECTOR_CHILDREN, DFLASH_SELECTOR_PARAMS)

TM_MODULE_REGISTER(DFlashWeight, core::DFlashWeightConfig);
TM_MODULE_METHODS(DFlashWeight, DFLASH_WEIGHT_CHILDREN, DFLASH_WEIGHT_PARAMS)

}  // namespace turbomind
