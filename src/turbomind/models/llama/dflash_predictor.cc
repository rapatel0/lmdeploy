// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/dflash_predictor.h"

#include <cuda_runtime.h>

#include <utility>
#include <vector>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/dflash_weight.h"
#include "src/turbomind/models/linear_weight.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/dflash_kernels.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/models/norm_weight.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

DFlashPredictor::DFlashPredictor(const DFlashWeight&     weights,
                                 UnifiedAttentionLayer& attention,
                                 std::vector<int>        attention_indices,
                                 int                     attention_phase_base,
                                 const EngineParam&      engine,
                                 const Context&          ctx,
                                 EmbedFn                 embed,
                                 LogitsFn                logits):
    weights_(weights),
    hidden_units_(weights.fc ? weights.fc->output_dim : 0),
    num_context_features_(weights.num_context_features),
    dtype_(engine.data_type),
    linear_(*ctx.linear),
    attention_(attention),
    attention_indices_(std::move(attention_indices)),
    attention_phase_base_(attention_phase_base),
    ctx_(ctx),
    embed_fn_(std::move(embed)),
    logits_fn_(std::move(logits))
{
    TM_CHECK(embed_fn_) << "DFlash2 needs the target token embedding";
    TM_CHECK(logits_fn_) << "DFlash2 needs the target lm_head";
    TM_CHECK(weights_.fc) << "DFlash2 context projection is missing";
    TM_CHECK(weights_.hidden_norm) << "DFlash2 hidden_norm is missing";
    TM_CHECK_GT(hidden_units_, 0);
    TM_CHECK_GT(num_context_features_, 0);
    TM_CHECK_EQ(weights_.fc->input_dim, hidden_units_ * num_context_features_);
    if (engine.num_draft_tokens > 0) {
        TM_CHECK_EQ(attention_indices_.size(), 5);
        TM_CHECK_GE(attention_phase_base_, 0);
    }
    ffn_layer_ = std::make_unique<LlamaFfnLayer>(ctx);
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

DFlashPredictor::ConvState DFlashPredictor::PrepareGroupedConv(const Tensor& input,
                                                                 const DFlashConvWeight& weights) const
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
                            0,
                            weights_.block_size,
                            weights.taps,
                            weights.group_size,
                            core::Context::stream().handle());
    return {std::move(output), std::move(delta)};
}

Tensor DFlashPredictor::FinishGroupedConv(const Tensor&             input,
                                           const Tensor&             delta,
                                           const DFlashConvWeight& weights) const
{
    Tensor output{input.shape(), dtype_, kDEVICE};
    invokeDFlashGroupedConv(output,
                            input,
                            delta,
                            weights.base_kernel,
                            1,
                            weights_.block_size,
                            weights.taps,
                            weights.group_size,
                            core::Context::stream().handle());
    return output;
}

Tensor DFlashPredictor::ApplyGroupedConv(const Tensor& input, const DFlashConvWeight& weights, int side) const
{
    auto prepared = PrepareGroupedConv(input, weights);
    return side == 0 ? std::move(prepared.output) : FinishGroupedConv(input, prepared.delta, weights);
}

Tensor DFlashPredictor::DraftBlock(const Buffer_<int>& anchors, int phase, TensorMap& env) const
{
    const int batch_size = anchors.size();
    TM_CHECK_GT(batch_size, 0);
    TM_CHECK_GE(attention_phase_base_, 0);

    Buffer_<int> block_ids{(ssize_t)batch_size * weights_.block_size, kDEVICE};
    invokeBuildDFlashBlock(block_ids,
                           anchors,
                           weights_.block_size,
                           weights_.mask_token_id,
                           core::Context::stream().handle());
    Tensor hidden = embed_fn_(block_ids);

    Buffer_<int> k_offsets = env.at("k_offsets").buffer().view<int>();
    AdvanceCuSeqLens(k_offsets.data(), batch_size, weights_.block_size, core::Context::stream().handle());
    hidden = RunDraftLayers(std::move(hidden), phase);
    AdvanceCuSeqLens(k_offsets.data(), batch_size, -weights_.block_size, core::Context::stream().handle());
    return hidden;
}

Buffer_<int> DFlashPredictor::SelectCandidates(const Tensor& block_hidden, const Buffer_<int>& anchors) const
{
    auto* selector = TM_CHECK_NOTNULL(weights_.selector.get());
    TM_CHECK(selector->hidden_projection);
    const int batch_size = anchors.size();
    const int slots      = weights_.block_size - 1;
    const int rows       = batch_size * slots;

    Tensor prediction_hidden{{rows, hidden_units_}, dtype_, kDEVICE};
    invokeGatherDFlashPredictions(
        prediction_hidden, block_hidden, weights_.block_size, core::Context::stream().handle());

    Tensor logits = logits_fn_(prediction_hidden);
    Buffer_<int> candidate_ids{(ssize_t)rows * selector->top_k, kDEVICE};
    Tensor unary_scores{{rows, selector->top_k}, kFloat32, kDEVICE};
    invokeDFlashTopK16(candidate_ids,
                       unary_scores,
                       logits,
                       logits.shape(1),
                       weights_.output_multiplier,
                       weights_.final_logit_softcapping,
                       core::Context::stream().handle());

    Tensor selector_hidden{{rows, selector->state_rank}, dtype_, kDEVICE};
    linear_.Forward(prediction_hidden, *selector->hidden_projection, selector_hidden);
    TM_CUDA_CHECK(cudaGetLastError());

    Buffer_<int> output{(ssize_t)rows, kDEVICE};
    invokeDFlashGreedySelector(output,
                               anchors,
                               candidate_ids,
                               unary_scores,
                               selector_hidden,
                               selector->predecessor_codebook,
                               selector->successor_codebook,
                               slots,
                               selector->top_k,
                               core::Context::stream().handle());
    return output;
}

Tensor DFlashPredictor::RunDraftLayers(Tensor hidden, int phase) const
{
    TM_CHECK_EQ(attention_indices_.size(), 5);
    const int  attention_phase = attention_phase_base_ >= 0 ? attention_phase_base_ + phase : phase;
    const int  token_num       = hidden.shape(0);
    const auto stream          = core::Context::stream().handle();

    Tensor residual{hidden.shape(), dtype_, kDEVICE};
    TM_CUDA_CHECK(cudaMemcpyAsync(
        residual.raw_data(), hidden.raw_data(), hidden.byte_size(), cudaMemcpyDeviceToDevice, stream));

    auto* first_layer = TM_CHECK_NOTNULL(weights_.layer(0));
    invokeRMSNorm(hidden,
                  hidden,
                  first_layer->attention_norm->weight,
                  first_layer->attention_norm->norm_eps_,
                  first_layer->attention_norm->zero_centered_,
                  stream);
    TM_CUDA_CHECK(cudaGetLastError());

    auto residual_norm = [&](Tensor& value, Tensor& res, const Tensor& bias, const NormWeight& norm) {
        if (ctx_.comm.d_comm) {
            ctx_.comm.d_comm->AllreduceResidualBiasRMSnorm(value.raw_data(),
                                                           res.raw_data(),
                                                           bias.data_or((void*)nullptr),
                                                           norm.weight.raw_data(),
                                                           norm.norm_eps_,
                                                           norm.zero_centered_,
                                                           hidden_units_,
                                                           token_num,
                                                           dtype_,
                                                           ctx_.comm.d_tp_group,
                                                           stream);
        }
        else {
            invokeResidualBiasRMSNorm(value.raw_data(),
                                      res.raw_data(),
                                      norm.weight.raw_data(),
                                      bias.data_or((void*)nullptr),
                                      dtype_,
                                      hidden_units_,
                                      token_num,
                                      norm.norm_eps_,
                                      norm.zero_centered_,
                                      stream);
        }
        TM_CUDA_CHECK(cudaGetLastError());
    };

    for (int i = 0; i < (int)attention_indices_.size(); ++i) {
        auto* layer     = TM_CHECK_NOTNULL(weights_.layer(i));
        auto* attn_conv = TM_CHECK_NOTNULL(weights_.attention_conv(i));
        auto* mlp_conv  = TM_CHECK_NOTNULL(weights_.mlp_conv(i));
        TM_CHECK(layer->attention && layer->feed_forward);

        auto attn_input = PrepareGroupedConv(hidden, *attn_conv);
        Tensor attn_output{{token_num, hidden_units_}, dtype_, kDEVICE};
        attention_.Forward({attention_phase,
                            attn_input.output,
                            attn_output,
                            layer->attention.get(),
                            attention_indices_[i]});
        attn_output = FinishGroupedConv(attn_output, attn_input.delta, *attn_conv);
        residual_norm(attn_output, residual, layer->attention->wo->bias, *layer->ffn_norm);
        hidden = std::move(attn_output);

        auto mlp_input = PrepareGroupedConv(hidden, *mlp_conv);
        ffn_layer_->forward({mlp_input.output, mlp_input.output, layer->feed_forward.get(), attention_indices_[i]});
        Tensor mlp_output = FinishGroupedConv(mlp_input.output, mlp_input.delta, *mlp_conv);

        const NormWeight& output_norm =
            i + 1 < (int)attention_indices_.size() ? *weights_.layer(i + 1)->attention_norm : *weights_.final_norm;
        residual_norm(mlp_output, residual, {}, output_norm);
        hidden = std::move(mlp_output);
    }
    return hidden;
}

}  // namespace turbomind
