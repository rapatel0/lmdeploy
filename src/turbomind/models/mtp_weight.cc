// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/mtp_weight.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/norm_weight.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/core/registry.h"

namespace turbomind {

MTPLayerWeight::MTPLayerWeight(const core::ModuleConfig&) {}

MTPLayerWeight::~MTPLayerWeight() = default;

/// Note: nothing calls this today.
///
/// `Module::verify` is declared in core/module.h and overridden here, in
/// ModelWeight and in DecoderLayerWeight, but no C++ caller invokes it and it
/// is not bound into Python. The two upstream overrides arrived dormant in
/// #4557 and remain so.
///
/// The override is kept rather than deleted because it is the natural home for
/// these checks once a caller exists, and because deleting it would diverge
/// from the two sibling modules that also define one. Its log line is
/// therefore not evidence of loading: the Python loader in qwen3_5.py emits
/// the record that is actually observed.
bool MTPLayerWeight::verify(std::vector<std::string>& missing)
{
    // `missing` is a single vector threaded through the whole weight tree, so
    // it may already hold entries from an unrelated module. Record the length
    // on entry and judge this layer only by what it appends.
    const size_t before = missing.size();

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

    // Announce the layer once, at the only point where its presence is proven.
    //
    // MTP is optional at every level: ModelWeight::verify does not require it,
    // and the Python loader returns None when the checkpoint carries no `mtp.`
    // tensors. So a model with no MTP layer loads, runs, and prints nothing to
    // distinguish itself from one that has a layer that silently failed to
    // load. Both look identical in the log, and the difference shows up only as
    // an unexplained absence of speedup.
    //
    // Log after the checks rather than in the constructor, because the module
    // is constructed before its children are attached; a message there would
    // report presence that is not yet established.
    const bool ok = missing.size() == before;
    if (ok) {
        TM_LOG_INFO("[MTP] speculation weights loaded: {} (1 decoder layer, shared embedding and lm_head)",
                    full_path());
    }
    else {
        // Name the absent children, so the log distinguishes a checkpoint
        // without MTP from one whose MTP layer failed to load.
        TM_LOG_WARNING("[MTP] speculation DISABLED: {} is incomplete, {} weight(s) missing",
                       full_path(),
                       missing.size() - before);
    }

    return ok;
}

TM_MODULE_REGISTER(MTPLayerWeight, core::ModuleConfig);

TM_MODULE_METHODS(MTPLayerWeight, MTP_LAYER_WEIGHT_CHILDREN, MTP_LAYER_WEIGHT_PARAMS)

}  // namespace turbomind
