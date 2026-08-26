// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/mtp_weight.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"

#include "src/turbomind/core/registry.h"

namespace turbomind {

MTPLayerWeight::MTPLayerWeight(const core::ModuleConfig&) {}

MTPLayerWeight::~MTPLayerWeight() = default;

bool MTPLayerWeight::verify(std::vector<std::string>& missing)
{
    Module::verify(missing);

    // Every child is required. A partially loaded MTP layer is worse than an
    // absent one: the target still verifies each drafted token, so wrong
    // weights do not corrupt the output. They only cause every draft to be
    // rejected, which appears as a slowdown rather than as a fault. Fail at
    // load time instead.
    if (!pre_fc_norm_embedding) {
        missing.push_back(full_path() + ": missing pre_fc_norm_embedding");
    }
    if (!pre_fc_norm_hidden) {
        missing.push_back(full_path() + ": missing pre_fc_norm_hidden");
    }
    if (!fc) {
        missing.push_back(full_path() + ": missing fc");
    }
    if (!decoder_layer) {
        missing.push_back(full_path() + ": missing decoder_layer");
    }
    if (!final_norm) {
        missing.push_back(full_path() + ": missing final_norm");
    }
    return missing.empty();
}

TM_MODULE_REGISTER(MTPLayerWeight, core::ModuleConfig);

TM_MODULE_METHODS(MTPLayerWeight, MTP_LAYER_WEIGHT_CHILDREN, MTP_LAYER_WEIGHT_PARAMS)

}  // namespace turbomind
