// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/dflash_predictor.h"

#include <cuda_runtime.h>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/dflash_weight.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

DFlashPredictor::DFlashPredictor(const DFlashWeight& weights, const EngineParam& engine, const Context& ctx):
    weights_(weights),
    hidden_units_(weights.fc ? weights.fc->output_dim : 0),
    num_context_features_(weights.num_context_features),
    dtype_(engine.data_type),
    linear_(*ctx.linear)
{
    TM_CHECK(weights_.fc) << "DFlash2 context projection is missing";
    TM_CHECK(weights_.hidden_norm) << "DFlash2 hidden_norm is missing";
    TM_CHECK_GT(hidden_units_, 0);
    TM_CHECK_GT(num_context_features_, 0);
    TM_CHECK_EQ(weights_.fc->input_dim, hidden_units_ * num_context_features_);
    TM_LOG_INFO("[DFlash2] context projector ready: features={} input={} hidden={}",
                num_context_features_,
                weights_.fc->input_dim,
                hidden_units_);
}

DFlashPredictor::~DFlashPredictor() = default;

Tensor DFlashPredictor::ProjectContext(const Tensor& target_hidden) const
{
    TM_CHECK_EQ(target_hidden.ndim(), 2);
    TM_CHECK_EQ(target_hidden.shape(1), (ssize_t)num_context_features_ * hidden_units_);
    TM_CHECK_EQ(target_hidden.dtype(), dtype_);

    const int token_num = target_hidden.shape(0);
    Tensor    projected{{token_num, hidden_units_}, dtype_, kDEVICE};
    linear_.Forward(target_hidden, *weights_.fc, projected);
    TM_CUDA_CHECK(cudaGetLastError());

    Tensor normalized{{token_num, hidden_units_}, dtype_, kDEVICE};
    invokeRMSNorm(normalized,
                  projected,
                  weights_.hidden_norm->weight,
                  weights_.hidden_norm->norm_eps_,
                  weights_.hidden_norm->zero_centered_,
                  core::Context::stream().handle());
    TM_CUDA_CHECK(cudaGetLastError());
    return normalized;
}

}  // namespace turbomind
