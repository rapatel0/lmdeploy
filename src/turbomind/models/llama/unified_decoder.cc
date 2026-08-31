

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <optional>

#include <cuda_runtime.h>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/core/math.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/delta_net_weight.h"
#include "src/turbomind/models/dflash_weight.h"
#include "src/turbomind/models/llama/dflash_kernels.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/mtp_weight.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/moe_ffn_layer.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

#include "src/turbomind/engine/request.h"

// #include "dbg.h"

namespace turbomind {

void UnifiedDecoder::Run(BatchOp op, int phase, TensorMap& env)
{
    attn_layer_->Run(op, phase, env);
    if (linear_attn_layer_) {
        linear_attn_layer_->Run(op, phase, env);
    }
}

UnifiedDecoder::UnifiedDecoder(CacheRegistry&     registry,
                               const EngineParam& engine,
                               const Context&     ctx,
                               int                phases,
                               const ModelWeight& model_weight):
    layer_num_(model_weight.num_layer),
    hidden_units_(model_weight.hidden_units),
    output_norm_zero_centered_(model_weight.norm->zero_centered_),
    attn_tp_size_(engine.attn_tp_size),
    attn_dp_size_(engine.attn_dp_size),
    attn_dp_rank_(engine.attn_dp_rank),
    mlp_tp_size_(engine.mlp_tp_size),
    attn_tp_group_(ctx.comm.d_tp_group),
    d_comm_(ctx.comm.d_comm),
    tune_layer_num_(engine.tune_layer_num),
    dflash_target_layer_ids_(model_weight.dflash ? model_weight.dflash->target_layer_ids : std::vector<int>{}),
    is_warm_up_{*ctx.is_warm_up}
{
    std::vector<MoeWeight*>       moe_weights;
    std::vector<FfnWeight*>       ffn_weights;
    std::vector<DeltaNetWeight*>  gdn_weights;
    std::vector<AttentionWeight*> attn_weights;

    for (int i = 0; i < model_weight.num_layer; ++i) {
        auto layer = model_weight.layer(i);
        if (layer->moe_ffn) {
            moe_weights.push_back(layer->moe_ffn.get());
        }
        if (layer->linear_attn) {
            gdn_weights.push_back(layer->linear_attn.get());
        }
        if (layer->attention) {
            attn_weights.push_back(layer->attention.get());
        }
        if (layer->feed_forward) {
            ffn_weights.push_back(layer->feed_forward.get());
        }
    }
    target_attention_count_ = static_cast<int>(attn_weights.size());

    // Give the Multi-Token Prediction layer its own KV slot.
    //
    // UnifiedAttentionLayer assigns each weight in this list a distinct
    // `cache_block_offset` inside one registered cache block, then registers
    // the summed size once. Appending the MTP attention weight after the
    // target's layers therefore reserves KV space for the draft layer without
    // disturbing any existing offset: the target's layers keep the offsets
    // they would have had, because the append happens last.
    //
    // The draft layer needs its own slot rather than sharing one. It attends
    // over the same sequence as the target, so writing into a target layer's
    // slot would overwrite entries the target still needs.
    //
    // This is the one piece with no reference to copy. The v0.12.0 fork picks
    // a KV index arithmetically, `mtp_attn_layer_offset_ + mtp_layer_idx`,
    // because that version indexes the cache by layer number. Here the
    // registry owns allocation, so the slot is obtained by participating in
    // registration instead.
    if (engine.num_draft_tokens > 0 && engine.speculative_algorithm == "mtp" && model_weight.mtp
        && model_weight.mtp->decoder_layer) {
        auto* mtp_layer = model_weight.mtp->decoder_layer.get();
        if (mtp_layer->attention) {
            attn_weights.push_back(mtp_layer->attention.get());
            mtp_attn_index_ = static_cast<int>(attn_weights.size()) - 1;
            // TM_LOG_INFO formats with fmt, not printf: a %d is emitted
            // literally and the value never appears.
            TM_LOG_INFO("[MTP] registered a KV slot for the draft layer at attention index {} of {}",
                        mtp_attn_index_,
                        (int)attn_weights.size());
        }
    }

    static const bool force_dflash_capture = [] {
        const char* value = std::getenv("TM_DFLASH_CAPTURE");
        return value && value[0] == '1';
    }();
    if ((engine.num_draft_tokens > 0 || force_dflash_capture) && engine.speculative_algorithm == "dflash2"
        && model_weight.dflash && model_weight.dflash->layers) {
        for (int i = 0; i < (int)model_weight.dflash->layers->size(); ++i) {
            auto* layer = model_weight.dflash->layer(i);
            TM_CHECK(layer && layer->attention) << "DFlash2 layer " << i << " has no attention weights";
            attn_weights.push_back(layer->attention.get());
            dflash_attn_indices_.push_back(static_cast<int>(attn_weights.size()) - 1);
        }
        TM_CHECK_EQ(dflash_attn_indices_.size(), 5) << "published DFlash2 checkpoint requires five layers";
        TM_LOG_INFO("[DFlash2] registered {} dedicated KV slots at attention indices {}..{}",
                    dflash_attn_indices_.size(),
                    dflash_attn_indices_.front(),
                    dflash_attn_indices_.back());
    }

    if (!moe_weights.empty()) {
        moe_ffn_layer_ = std::make_unique<MoeFfnLayer>(engine, ctx);
    }

    if (!ffn_weights.empty()) {
        ffn_layer_ = std::make_unique<LlamaFfnLayer>(ctx);
    }

    if (!attn_weights.empty()) {
        // One extra phase slot for the MTP draft.
        //
        // The draft shares this layer with the target but has a different
        // shape: it submits one token per row while a verification forward
        // submits K+1. AttentionData is per-phase, so without a slot of its own
        // the draft runs against the target's plan and aborts:
        //
        //   Check failed: d.prefill.q_sum + d.decode.n == q_count (15 vs. 3)
        //
        // where 15 is the plan's expected token count and 3 the draft's actual
        // one. Preparing the draft's shape into the target's slot instead would
        // corrupt the target -- that is the failure the previous commit
        // introduced by removing PrepareAttention.
        // ONE DRAFT SLOT PER PHASE, not one in total.
        //
        // The target's phase alternates because two batches are in flight at
        // once: batch A uses slot 0 while batch B uses slot 1. A single draft
        // slot shared by both means Setup(B) overwrites the plan that batch A's
        // kDraft has not read yet -- the executor is still working on A when
        // the main thread sets up B.
        //
        // That is why the draft saw the target's K+1 shape: not a lost marker,
        // but the NEXT batch's Setup landing in the slot before the current
        // batch finished with it.
        int attention_phases = phases;
        if (engine.num_draft_tokens > 0 && engine.speculative_algorithm == "mtp" && mtp_attn_index_ >= 0) {
            mtp_phase_base_ = attention_phases;
            attention_phases += phases;
        }
        if ((engine.num_draft_tokens > 0 || force_dflash_capture) && engine.speculative_algorithm == "dflash2"
            && !dflash_attn_indices_.empty()) {
            dflash_phase_base_ = attention_phases;
            attention_phases += phases;
        }
        attn_layer_ = std::make_unique<UnifiedAttentionLayer>(attn_weights, registry, engine, ctx, attention_phases);
    }

    if (!gdn_weights.empty()) {
        linear_attn_layer_ = std::make_unique<GatedDeltaNetLayer>(gdn_weights,  //
                                                                  registry,
                                                                  engine,
                                                                  ctx,
                                                                  phases);
    }

    TM_CHECK(!(moe_weights.empty() && engine.ep_size > 1)) << "Dense model is not supported with ep_size > 1";
}

void UnifiedDecoder::AllreduceResidualRMSnorm(Tensor&       hidden_states,
                                              Tensor&       residual,
                                              const Tensor& bias,
                                              const Tensor& weight,
                                              float         eps,
                                              bool          zero_centered,
                                              int           token_num,
                                              int           group0,
                                              int           group1,
                                              const int*    local_token_nums)
{
    const auto dtype = hidden_states.dtype();

    const auto stream = core::Context::stream().handle();

    if (0) {}
    else if (group0 || group1) {
        d_comm_->AllreduceResidualBiasRMSnormEx(hidden_states.raw_data(),
                                                residual.data_or((void*)nullptr),
                                                bias.data_or((void*)nullptr),
                                                weight.raw_data(),
                                                eps,
                                                zero_centered,
                                                hidden_units_,
                                                dtype,
                                                group0,
                                                group1,
                                                local_token_nums,
                                                stream);
        TM_CUDA_CHECK(cudaGetLastError());
    }
    else if (d_comm_) {
        d_comm_->AllreduceResidualBiasRMSnorm(hidden_states.raw_data(),
                                              residual.data_or((void*)nullptr),
                                              bias.data_or((void*)nullptr),
                                              weight.raw_data(),
                                              eps,
                                              zero_centered,
                                              hidden_units_,
                                              token_num,
                                              dtype,
                                              0,
                                              stream);
        TM_CUDA_CHECK(cudaGetLastError());
    }
    else {
        invokeResidualBiasRMSNorm(hidden_states.raw_data(),
                                  residual.data_or((void*)nullptr),
                                  weight.raw_data(),
                                  bias.data_or((void*)nullptr),
                                  dtype,
                                  hidden_units_,
                                  token_num,
                                  eps,
                                  zero_centered,
                                  stream);
        TM_CUDA_CHECK(cudaGetLastError());
    }
}

void UnifiedDecoder::Forward(int phase, TensorMap& args, const std::vector<WeightType*>& weights)
{
    TM_FUNCTION_SCOPE();
    /**
     * input tensors:
     *   \param decoder_input [token_num, hidden_units], float
     *   \param output_norm_weight [hidden_dims], float
     *   \param cu_block_counts [batch_size+1], int
     *   \param finished [batch_size], bool
     *   \param rope_theta [batch_size], float
     *   \param h_q_len [batch_size], int on cpu
     *   \param h_k_len [batch_size], int on cpu
     *   \param pf_batch_size [1], int on cpu
     *   \param dc_batch_size [1], int on cpu
     *
     * output tensors:
     *   \param decoder_output [num_token, hidden_units],
     *   \param last_token_hidden_units [batch_size, hidden_units]
     *   \param block_ptrs [total_block_counts], void*
     */

    constexpr auto device = kDEVICE;

    Tensor      decoder_input    = args.try_consume("input_embeds");
    const auto& local_token_nums = args.at("batch").data<BatchData*>()[0]->local_token_num;
    const bool  use_dflash_target_workspace = args.try_("dflash_target_workspace") != nullptr;

    const auto local_token_num  = decoder_input.shape(0);
    const auto global_token_num = std::accumulate(local_token_nums.begin(), local_token_nums.end(), ssize_t{});

    TM_CHECK_EQ(local_token_num, local_token_nums[attn_dp_rank_]);

    const DataType dtype = decoder_input.dtype();
    static const bool enable_dflash_target_fp32_residual = [] {
        const char* value = std::getenv("TM_DFLASH_TARGET_FP32_RESIDUAL");
        return value && value[0] == '1';
    }();
    const bool use_dflash_target_fp32_residual = enable_dflash_target_fp32_residual
                                                  && !dflash_target_layer_ids_.empty()
                                                  && dtype == kHalf && attn_dp_size_ == 1;
    Tensor local_residual = use_dflash_target_fp32_residual ?
                                Tensor{{local_token_num, (ssize_t)hidden_units_}, kFloat32, kDEVICE} :
                                decoder_input;
    if (use_dflash_target_fp32_residual) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            TM_LOG_INFO("DFLASH_TARGET_FP32_RESIDUAL_ACTIVE phase={} rows={} hidden={}",
                        phase,
                        local_token_num,
                        hidden_units_);
        }
    }

    Tensor global_hidden_states;
    if (d_comm_) {
        Buffer symm_buf      = args.at("symm_buf").buffer();
        global_hidden_states = {symm_buf.view(dtype), {global_token_num, (int)hidden_units_}};
    }
    else {
        global_hidden_states = {{global_token_num, (int)hidden_units_}, dtype, kDEVICE};
    }

    Tensor local_hidden_states;
    if (attn_dp_size_ > 1) {  // Offset hidden states buffer for mixed DP
        TM_CHECK_EQ(local_token_nums.size(), attn_dp_size_);
        std::vector offsets(attn_dp_size_ + 1, 0);
        std::inclusive_scan(local_token_nums.data(), local_token_nums.data() + attn_dp_size_, offsets.begin() + 1);
        const int offset    = offsets[attn_dp_rank_];
        local_hidden_states = global_hidden_states.slice({offset, 0}, {local_token_num, -1});

        // dbg(attn_dp_size_, attn_dp_rank_, local_token_nums, local_token_num, global_token_num);
    }
    else {
        local_hidden_states = global_hidden_states;
    }

    TM_LOG_DEBUG("local_token_num=%d, global_token_num=%d", (int)local_token_num, (int)global_token_num);

    TM_DEBUG_TENSOR(local_residual, "res", 1);

    const auto stream = core::Context::stream().handle();

    auto* target_trajectory = args.try_("dflash_target_trajectory");
    auto capture_target_trajectory = [&](int slot, const Tensor& value) {
        if (!target_trajectory || local_token_num == 0) {
            return;
        }
        TM_CHECK_EQ(target_trajectory->dtype(), dtype);
        TM_CHECK_EQ(target_trajectory->shape(0), 38);
        TM_CHECK_EQ(target_trajectory->shape(1), (ssize_t)hidden_units_);
        TM_CHECK_EQ(value.shape(1), (ssize_t)hidden_units_);
        const auto row_bytes = byte_size(dtype, hidden_units_);
        TM_CUDA_CHECK(cudaMemcpyAsync((char*)target_trajectory->raw_data()
                                         + slot * byte_size(dtype, target_trajectory->stride(0)),
                                     (const char*)value.raw_data()
                                         + (value.shape(0) - 1) * byte_size(dtype, value.stride(0)),
                                     row_bytes,
                                     cudaMemcpyDeviceToDevice,
                                     stream));
    };
    capture_target_trajectory(0, decoder_input);

    auto target_residual_norm = [&](Tensor&       hidden_states,
                                    const Tensor& bias,
                                    const Tensor& weight,
                                    float         eps,
                                    bool          zero_centered) {
        TM_CHECK(use_dflash_target_fp32_residual);
        if (d_comm_) {
            d_comm_->AllReduceSum(hidden_states.raw_data(),
                                  hidden_states.raw_data(),
                                  hidden_states.size(),
                                  dtype,
                                  attn_tp_group_,
                                  stream);
            TM_CUDA_CHECK(cudaGetLastError());
        }
        invokeDFlashTargetResidualRMSNorm(hidden_states,
                                           local_residual,
                                           hidden_states,
                                           bias,
                                           weight,
                                           eps,
                                           zero_centered,
                                           false,
                                           stream);
    };

    const auto& first_norm = *weights.at(0)->attention_norm;
    if (use_dflash_target_fp32_residual) {
        invokeDFlashTargetResidualRMSNorm(local_hidden_states,
                                           local_residual,
                                           decoder_input,
                                           {},
                                           first_norm.weight,
                                           first_norm.norm_eps_,
                                           first_norm.zero_centered_,
                                           true,
                                           stream);
    }
    else {
        invokeRMSNorm(local_hidden_states,
                      local_residual,
                      first_norm.weight,
                      first_norm.norm_eps_,
                      first_norm.zero_centered_,
                      stream);
    }

    TM_CUDA_CHECK(cudaGetLastError());

    TM_DEBUG_TENSOR(local_hidden_states, Concat("norm0", 0), 2);
    capture_target_trajectory(1, local_hidden_states);

    // auto stack_alloc{core::Context::device_alloc().adapt<core::StackAllocatorImpl>()};
    // core::ContextGuard ctx{Allocator{stack_alloc}};

    for (int layer = 0; layer < layer_num_; ++layer) {

        std::string _tm_layer_name = Concat("layer", layer);
        TM_SCOPE(_tm_layer_name.c_str());

        // stack_alloc->iter();

        if (global_token_num == 0) {
            break;
        }

        if (is_warm_up_ && layer >= tune_layer_num_) {
            continue;
        }

        /////////////////////////////////////////////
        /// self-attention or linear-attention
        if (weights.at(layer)->linear_attn) {
            linear_attn_layer_->Forward(
                {phase, local_hidden_states, local_hidden_states, weights.at(layer)->linear_attn.get()});
        }
        else {
            auto* attn = weights.at(layer)->attention.get();
            attn_layer_->Forward(
                {phase, local_hidden_states, local_hidden_states, attn, layer, 1.f, false, use_dflash_target_workspace});
        }

        TM_DEBUG_TENSOR(local_hidden_states, Concat("attn_block", layer), 2);
        if (layer < 6) {
            capture_target_trajectory(2 + layer * 6, local_hidden_states);
        }

        // For gated delta networks, we may need a different output.bias name or it doesn't have it.
        // We will just use `output.bias` from either layer.
        Tensor out_bias;
        if (weights.at(layer)->linear_attn) {
            out_bias = weights.at(layer)->linear_attn->out_proj->bias;
        }
        else {
            out_bias = weights.at(layer)->attention->wo->bias;
        }

        if (use_dflash_target_fp32_residual) {
            target_residual_norm(global_hidden_states,
                                 out_bias,
                                 weights.at(layer)->ffn_norm->weight,
                                 weights.at(layer)->ffn_norm->norm_eps_,
                                 weights.at(layer)->ffn_norm->zero_centered_);
        }
        else {
            AllreduceResidualRMSnorm(global_hidden_states,
                                     local_residual,
                                     out_bias,
                                     weights.at(layer)->ffn_norm->weight,
                                     weights.at(layer)->ffn_norm->norm_eps_,
                                     weights.at(layer)->ffn_norm->zero_centered_,
                                     local_token_num,
                                     attn_tp_group_,
                                     0,
                                     local_token_nums.data());
        }

        TM_DEBUG_TENSOR(local_residual, Concat("residual0", layer), 2);
        TM_DEBUG_TENSOR(local_hidden_states, Concat("norm1", layer), 2);
        if (layer < 6) {
            capture_target_trajectory(3 + layer * 6, local_residual);
            capture_target_trajectory(4 + layer * 6, local_hidden_states);
        }

        // Diagnostic only: replay SGLang's exact last-row layer-0 MLP input
        // after native attention/residual execution. FFN execution is
        // token-independent at inference, so this isolates target-FFN
        // arithmetic without invalid last-row-only replay of recurrent GDN.
        if (layer == 0) {
            if (const auto* replay = args.try_("dflash_target_mlp_replay")) {
                TM_CHECK_EQ(replay->dtype(), dtype);
                TM_CHECK_EQ(replay->shape(0), 1);
                TM_CHECK_EQ(replay->shape(1), (ssize_t)hidden_units_);
                TM_CHECK_GT(local_token_num, 0);
                TM_CUDA_CHECK(cudaMemcpyAsync((char*)local_hidden_states.raw_data()
                                                  + (local_hidden_states.shape(0) - 1)
                                                        * byte_size(dtype, local_hidden_states.stride(0)),
                                              replay->raw_data(),
                                              byte_size(dtype, hidden_units_),
                                              cudaMemcpyDeviceToDevice,
                                              stream));
                capture_target_trajectory(4, local_hidden_states);
                static bool replay_logged = false;
                if (!replay_logged) {
                    replay_logged = true;
                    TM_LOG_INFO("[DFlash2] target layer-0 MLP input replay active");
                }
            }
        }

        ////////////////////////////////////////////
        /// feed-forward network

        std::optional<MoeFfnLayer::ForwardParam> moe_fwd_param;

        if (weights.at(layer)->moe_ffn) {
            moe_fwd_param = MoeFfnLayer::ForwardParam{global_hidden_states,
                                                      global_hidden_states,
                                                      weights.at(layer)->moe_ffn.get(),
                                                      weights.at(layer)->feed_forward ? 1.f : 0.f,
                                                      layer,
                                                      (const bool*)args.at("token_mask").buffer().raw_data()};
            moe_ffn_layer_->Forward(*moe_fwd_param);
        }

        if (ffn_layer_ && weights.at(layer)->feed_forward) {
            auto ffn_input_shared =
                moe_ffn_layer_ ? moe_ffn_layer_->GetShardFfnInput(global_hidden_states) : global_hidden_states;
            if (ffn_input_shared.shape(0) > 0) {
                ffn_layer_->forward({ffn_input_shared,
                                     ffn_input_shared,
                                     weights.at(layer)->feed_forward.get(),
                                     (int)layer,
                                     phase,
                                     use_dflash_target_workspace,
                                     layer == 0 ? args.try_("dflash_target_mlp_activation") : nullptr,
                                     layer == 0 ? args.try_("dflash_target_mlp_activation_replay") : nullptr});
            }
        }

        if (moe_fwd_param) {
            moe_ffn_layer_->Combine(*moe_fwd_param);
        }

        TM_DEBUG_TENSOR(global_hidden_states, Concat("ffn_block", layer), 2);
        if (layer < 6) {
            capture_target_trajectory(5 + layer * 6, global_hidden_states);
        }

        const bool last = layer == layer_num_ - 1;

        auto&      scale_weight = !last ? weights.at(layer + 1)->attention_norm->weight : args.at("output_norm_weight");
        const bool scale_zero_centered =
            !last ? weights.at(layer + 1)->attention_norm->zero_centered_ : output_norm_zero_centered_;

        if (use_dflash_target_fp32_residual) {
            target_residual_norm(global_hidden_states,
                                 {},
                                 scale_weight,
                                 weights.at(layer)->ffn_norm->norm_eps_,
                                 scale_zero_centered);
        }
        else {
            AllreduceResidualRMSnorm(global_hidden_states,
                                     local_residual,
                                     {},
                                     scale_weight,
                                     weights.at(layer)->ffn_norm->norm_eps_,
                                     scale_zero_centered,
                                     local_token_num,
                                     0,
                                     attn_tp_group_,
                                     local_token_nums.data());
        }
        TM_CUDA_CHECK(cudaGetLastError());

        TM_DEBUG_TENSOR(local_residual, Concat("residual1", layer), 2);
        TM_DEBUG_TENSOR(local_hidden_states, Concat("norm0", layer + 1), 2);
        if (layer < 6) {
            capture_target_trajectory(6 + layer * 6, local_residual);
            capture_target_trajectory(7 + layer * 6, local_hidden_states);
        }

        // DFlash2 checkpoint layer IDs name post-layer residuals. Preserve the
        // per-token feature order [token, feature, hidden], matching SGLang's
        // concatenation before the context projection. cudaMemcpy2D expresses
        // the column insertion without a bespoke kernel.
        if (auto* capture = args.try_("dflash_target_hidden"); capture && !dflash_target_layer_ids_.empty()) {
            auto it = std::find(dflash_target_layer_ids_.begin(), dflash_target_layer_ids_.end(), layer);
            if (it != dflash_target_layer_ids_.end()) {
                const int     feature   = static_cast<int>(it - dflash_target_layer_ids_.begin());
                const ssize_t row_bytes = byte_size(dtype, hidden_units_);
                TM_CHECK_EQ(capture->dtype(), dtype);
                TM_CHECK_EQ(capture->shape(0), local_token_num);
                TM_CHECK_EQ(capture->shape(1), (ssize_t)dflash_target_layer_ids_.size() * hidden_units_);
                Tensor capture_source = local_residual;
                Tensor residual_half;
                if (use_dflash_target_fp32_residual) {
                    residual_half = {{local_token_num, (ssize_t)hidden_units_}, kHalf, kDEVICE};
                    invokeDFlashCastToHalf(residual_half, local_residual, stream);
                    capture_source = residual_half;
                }
                TM_CUDA_CHECK(cudaMemcpy2DAsync((char*)capture->raw_data() + feature * row_bytes,
                                                byte_size(dtype, capture->stride(0)),
                                                capture_source.raw_data(),
                                                byte_size(dtype, capture_source.stride(0)),
                                                row_bytes,
                                                local_token_num,
                                                cudaMemcpyDeviceToDevice,
                                                stream));
            }
        }

        // if (layer == layer_num_ - 1) {
        //     args.at("batch").data<BatchData*>()[0]->Notify();
        // }
    }

    // Token indices selected for decoding
    const Buffer selected_pos = args.consume("selected_token_pos").buffer();
    // dbg(selected_pos);
    // When there are no prefill sequences, token selection is not needed
    const bool reuse_hidden_states = selected_pos.size() == local_token_num;

    const bool output_hidden_states = args.try_("output_hidden_states");

    Tensor hidden_states{local_hidden_states};

    Tensor stable_hidden_states;
    if (d_comm_ && (output_hidden_states || reuse_hidden_states)) {
        // The full `hidden_states` buffer is needed for output but it's a ref into `symm_buf` atm.
        // Keep FP16 output storage separate from the optional FP32 target residual.
        if (use_dflash_target_fp32_residual) {
            stable_hidden_states = {{local_token_num, (ssize_t)hidden_units_}, dtype, kDEVICE};
            Copy(hidden_states, stable_hidden_states);
            hidden_states = stable_hidden_states;
        }
        else {
            Copy(hidden_states, local_residual);
            hidden_states = local_residual;
        }
    }

    Tensor selected_states;
    if (reuse_hidden_states) {
        selected_states = hidden_states;
    }
    else {
        selected_states = {{selected_pos.size(), (int)hidden_units_}, dtype, kDEVICE};
        CollectHiddenStates(hidden_states, selected_pos, selected_states, stream);
    }
    args.produce("hidden_states", selected_states);

    // TM_DEBUG_TENSOR(selected_states.slice(0, selected_pos.size()), "out", 1);

    if (output_hidden_states) {
        args.produce("full_hidden_states", hidden_states);
    }
}

}  // namespace turbomind
