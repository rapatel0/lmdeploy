// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/dflash_predictor.h"

#include <cuda_runtime.h>

#include <utility>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/dflash_weight.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/dflash_kernels.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

DFlashPredictor::DFlashPredictor(const DFlashWeight&     weights,
                                 UnifiedAttentionLayer& attention,
                                 std::vector<int>        attention_indices,
                                 int                     attention_phase_base,
                                 const EngineParam&      engine,
                                 const Context&          ctx):
    weights_(weights),
    hidden_units_(weights.fc ? weights.fc->output_dim : 0),
    num_context_features_(weights.num_context_features),
    dtype_(engine.data_type),
    linear_(*ctx.linear),
    attention_(attention),
    attention_indices_(std::move(attention_indices)),
    attention_phase_base_(attention_phase_base)
{
    TM_CHECK(weights_.fc) << "DFlash2 context projection is missing";
    TM_CHECK(weights_.hidden_norm) << "DFlash2 hidden_norm is missing";
    TM_CHECK_GT(hidden_units_, 0);
    TM_CHECK_GT(num_context_features_, 0);
    TM_CHECK_EQ(weights_.fc->input_dim, hidden_units_ * num_context_features_);
    if (engine.num_draft_tokens > 0) {
        TM_CHECK_EQ(attention_indices_.size(), 5);
        TM_CHECK_GE(attention_phase_base_, 0);
    }
    TM_LOG_INFO("[DFlash2] context projector ready: features={} input={} hidden={}",
                num_context_features_,
                weights_.fc->input_dim,
                hidden_units_);
}

DFlashPredictor::~DFlashPredictor() = default;

void DFlashPredictor::SetupAttention(int phase, TensorMap& env)
{
    TM_CHECK_GE(attention_phase_base_, 0);
    Buffer_<int> block_size{1, kCPU};
    block_size[0] = weights_.block_size;
    env.produce("attn_draft_block_size", std::move(block_size));
    attention_.Run(BatchOp::kSetup, attention_phase_base_ + phase, env);
    env.consume("attn_draft_block_size");
}

void DFlashPredictor::PrepareAttention(int phase, TensorMap& env)
{
    TM_CHECK_GE(attention_phase_base_, 0);
    attention_.Run(BatchOp::kPrepare, attention_phase_base_ + phase, env);
}

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

void DFlashPredictor::MaterializeContextKV(int target_phase, const Tensor& context) const
{
    TM_CHECK_EQ(attention_indices_.size(), 5);
    TM_CHECK_EQ(context.ndim(), 2);
    TM_CHECK_EQ(context.shape(1), hidden_units_);

    for (int i = 0; i < (int)attention_indices_.size(); ++i) {
        auto* layer = TM_CHECK_NOTNULL(weights_.layer(i));
        TM_CHECK(layer->attention);
        Tensor discarded{{context.shape(0), hidden_units_}, dtype_, kDEVICE};
        attention_.Forward({target_phase, context, discarded, layer->attention.get(), attention_indices_[i]});
        TM_CUDA_CHECK(cudaGetLastError());
    }
    TM_LOG_INFO("[DFlash2] materialized context KV: tokens={} layers={} phase={}",
                context.shape(0),
                attention_indices_.size(),
                target_phase);
}

Tensor DFlashPredictor::ApplyGroupedConv(const Tensor& input, const DFlashConvWeight& weights, int side) const
{
    TM_CHECK(weights.kernel_projection) << "DFlash2 convolution projection is missing";
    TM_CHECK(weights.base_kernel) << "DFlash2 base kernel is missing";
    TM_CHECK_EQ(input.ndim(), 2);
    TM_CHECK_EQ(input.shape(1), hidden_units_);

    Tensor delta{{input.shape(0), weights.kernel_projection->output_dim}, dtype_, kDEVICE};
    linear_.Forward(input, *weights.kernel_projection, delta);
    TM_CUDA_CHECK(cudaGetLastError());

    Tensor output{input.shape(), dtype_, kDEVICE};
    invokeDFlashGroupedConv(output,
                            input,
                            delta,
                            weights.base_kernel,
                            side,
                            weights_.block_size,
                            weights.taps,
                            weights.group_size,
                            core::Context::stream().handle());
    return output;
}

}  // namespace turbomind
