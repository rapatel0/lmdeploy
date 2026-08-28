// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/dflash_predictor.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

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
namespace {

bool TraceDFlashSelector()
{
    static const bool enabled = [] {
        const char* value = std::getenv("TM_DFLASH_TRACE_SELECTOR");
        return value && value[0] == '1';
    }();
    return enabled;
}

bool UseLocalDFlashTopK()
{
    static const bool enabled = [] {
        const char* value = std::getenv("TM_DFLASH_LOCAL_TOPK");
        return !value || value[0] != '0';
    }();
    return enabled;
}

void ReportDFlashTensor(const std::string& name, const Tensor& value)
{
    Tensor host{{1, value.shape(1)}, kHalf, kCPU};
    Copy(value.slice(0, 1), host);
    core::Context::stream().Sync();
    const auto* data = (const __half*)host.raw_data();
    ssize_t finite = 0;
    ssize_t nan = 0;
    ssize_t infinite = 0;
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    for (ssize_t i = 0; i < host.size(); ++i) {
        const float item = __half2float(data[i]);
        if (std::isfinite(item)) {
            ++finite;
            minimum = std::min(minimum, item);
            maximum = std::max(maximum, item);
        }
        else {
            nan += std::isnan(item);
            infinite += std::isinf(item);
        }
    }
    TM_LOG_INFO("[DFlash2] tensor {} finite={} nan={} inf={} range=[{},{}]",
                name,
                finite,
                nan,
                infinite,
                minimum,
                maximum);
}

}  // namespace

struct DFlashPredictor::ParityTrace {
    struct Entry {
        std::string name;
        Tensor      host;
    };

    explicit ParityTrace(const Context& ctx, const DFlashWeight& weights):
        root(std::getenv("TM_DFLASH_PARITY_DIR") ? std::getenv("TM_DFLASH_PARITY_DIR") : ""),
        tp_rank(ctx.comm.h_tp_group ? ctx.comm.h_tp_group->rank() : 0),
        tp_size(ctx.comm.h_tp_group ? ctx.comm.h_tp_group->n_ranks() : 1),
        feature_ids(weights.target_layer_ids)
    {
    }

    static std::string JsonString(const char* value)
    {
        std::string out{"\""};
        if (value) {
            for (const char c : std::string{value}) {
                if (c == '\\' || c == '\"') {
                    out += '\\';
                }
                out += c;
            }
        }
        out += '\"';
        return out;
    }

    static const char* Stage(const std::string& name)
    {
        if (name.rfind("target.", 0) == 0 || name.rfind("context.", 0) == 0) {
            return "context";
        }
        if (name.rfind("selector.", 0) == 0) {
            return "selector";
        }
        return "draft";
    }

    void Capture(const char* name, const Tensor& value)
    {
        if (!active) {
            return;
        }
        TM_CHECK(value.is_contiguous()) << "DFlash parity capture requires contiguous tensors: " << name;
        Tensor host{value.layout(), value.dtype(), kCPUpinned};
        Copy(value, host);
        entries.push_back({name, std::move(host)});
    }

    void Flush()
    {
        core::Context::stream().Sync();
        const auto capture_name = "rank-" + std::to_string(tp_rank) + "-device-" + std::to_string(device)
                                  + "-pid-" + std::to_string((long long)getpid());
        const auto dir = std::filesystem::path(root) / "lmdeploy" / capture_name;
        TM_CHECK(!std::filesystem::exists(dir)) << "DFlash parity capture directory already exists: " << dir.string();
        std::filesystem::create_directories(dir);
        std::ofstream manifest{dir / "manifest.jsonl", std::ios::out | std::ios::trunc};
        TM_CHECK(manifest) << "cannot open DFlash parity manifest in " << dir.string();

        for (size_t ordinal = 0; ordinal < entries.size(); ++ordinal) {
            auto& entry = entries[ordinal];
            std::ostringstream prefix;
            prefix << std::setw(6) << std::setfill('0') << ordinal << "-" << entry.name;
            const std::string file_name = prefix.str() + ".bin";
            std::ofstream data{dir / file_name, std::ios::binary};
            TM_CHECK(data) << "cannot open DFlash parity tensor " << file_name;
            data.write((const char*)entry.host.raw_data(), entry.host.byte_size());
            TM_CHECK(data) << "cannot write DFlash parity tensor " << file_name;

            manifest << "{\"runtime\":\"lmdeploy\",\"ordinal\":" << ordinal << ",\"stage\":"
                     << JsonString(Stage(entry.name)) << ",\"name\":" << JsonString(entry.name.c_str())
                     << ",\"dtype\":" << JsonString(to_string(entry.host.dtype()))
                     << ",\"shape\":[";
            for (int i = 0; i < entry.host.ndim(); ++i) {
                manifest << (i ? "," : "") << entry.host.shape(i);
            }
            manifest << "],\"strides\":[";
            for (int i = 0; i < entry.host.ndim(); ++i) {
                manifest << (i ? "," : "") << entry.host.stride(i);
            }
            manifest << "],\"byte_order\":\"little\",\"bytes\":" << entry.host.byte_size()
                     << ",\"file\":" << JsonString(file_name.c_str()) << ",\"tp_rank\":" << tp_rank
                     << ",\"tp_size\":" << tp_size << ",\"device\":" << device << ",\"uid\":" << uid
                     << ",\"sequence_length\":" << seq_len << ",\"input_length\":" << input_len
                     << ",\"target_feature_ids\":[";
            for (int i = 0; i < (int)feature_ids.size(); ++i) {
                manifest << (i ? "," : "") << feature_ids[i];
            }
            manifest << "],\"policies\":{";
            constexpr const char* flags[] = {"TM_DFLASH_LOCAL_TOPK",
                                             "TM_DFLASH_CUB_TOPK",
                                             "TM_DFLASH_REDUCE_BEFORE_CONV",
                                             "TM_DFLASH_CONTEXT_BF16_ROUND",
                                             "TM_DFLASH_LEGACY_ATTENTION_POLICY",
                                             "TM_DFLASH_PER_LAYER_ROPE"};
            for (int i = 0; i < (int)(sizeof(flags) / sizeof(flags[0])); ++i) {
                manifest << (i ? "," : "") << JsonString(flags[i]) << ":" << JsonString(std::getenv(flags[i]));
            }
            manifest << "}}\n";
        }
        manifest.flush();
        TM_CHECK(manifest) << "cannot write DFlash parity manifest";
        TM_LOG_INFO("[DFlash2] parity trace wrote {} tensors to {}", entries.size(), dir.string());
        entries.clear();
        ClearPending();
        context_armed = false;
        active = false;
        completed = true;
    }

    void ClearPending()
    {
        pending_target = {};
        pending_fc = {};
        pending_norm = {};
        context_pending = false;
        context_armed = false;
        armed_uid = 0;
        pending_uid = 0;
    }

    std::string root;
    int         tp_rank{};
    int         tp_size{1};
    int         device{};
    uint64_t    uid{};
    int         seq_len{};
    int         input_len{};
    bool        active{};
    bool        completed{};
    bool        context_armed{};
    bool        context_pending{};
    uint64_t    armed_uid{};
    uint64_t    pending_uid{};
    Tensor      pending_target;
    Tensor      pending_fc;
    Tensor      pending_norm;
    std::vector<int> feature_ids;
    std::vector<Entry> entries;
};

DFlashPredictor::DFlashPredictor(const DFlashWeight&     weights,
                                 UnifiedAttentionLayer& attention,
                                 std::vector<int>        attention_indices,
                                 int                     attention_phase_base,
                                 const EngineParam&      engine,
                                 const Context&          ctx,
                                 EmbedFn                 embed,
                                 LogitsFn                logits,
                                 CandidatesFn            candidates):
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
    logits_fn_(std::move(logits)),
    candidates_fn_(std::move(candidates))
{
    TM_CHECK(embed_fn_) << "DFlash2 needs the target token embedding";
    TM_CHECK(logits_fn_) << "DFlash2 needs the target lm_head";
    TM_CHECK(candidates_fn_) << "DFlash2 needs the sharded target lm_head";
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

void DFlashPredictor::ArmParityContext(uint64_t uid) const
{
    const char* root = std::getenv("TM_DFLASH_PARITY_DIR");
    if (!root || !root[0]) {
        return;
    }
    if (!parity_trace_) {
        parity_trace_ = std::make_unique<ParityTrace>(ctx_, weights_);
    }
    auto& trace = *parity_trace_;
    if (trace.completed || trace.active) {
        return;
    }
    trace.ClearPending();
    trace.context_armed = true;
    trace.armed_uid = uid;
}

void DFlashPredictor::PrepareParityContext(const Tensor& target_hidden,
                                           const Tensor& projected,
                                           const Tensor& normalized) const
{
    if (!parity_trace_) {
        return;
    }
    auto& trace = *parity_trace_;
    if (trace.completed || trace.active || !trace.context_armed || *ctx_.is_warm_up) {
        return;
    }
    trace.pending_target = target_hidden;
    trace.pending_fc = projected;
    trace.pending_norm = normalized;
    trace.pending_uid = trace.armed_uid;
    trace.context_pending = true;
    trace.context_armed = false;
}

void DFlashPredictor::BeginParityBlock(const Buffer_<int>& anchors,
                                       uint64_t             uid,
                                       int                  seq_len,
                                       int                  input_len) const
{
    if (!parity_trace_ || parity_trace_->completed || parity_trace_->active || !parity_trace_->context_pending) {
        return;
    }
    auto& trace = *parity_trace_;
    if (trace.pending_uid != uid) {
        trace.ClearPending();
        return;
    }
    trace.active = true;
    trace.uid = uid;
    trace.seq_len = seq_len;
    trace.input_len = input_len;
    TM_CUDA_CHECK(cudaGetDevice(&trace.device));
    trace.Capture("target.post_layer_residual", trace.pending_target);
    trace.Capture("context.fc", trace.pending_fc);
    trace.Capture("context.norm", trace.pending_norm);
    trace.Capture("block.anchors", Tensor{anchors, {(ssize_t)anchors.size()}});
}

bool DFlashPredictor::FinishParityBlock() const
{
    if (!ParityActive()) {
        return false;
    }
    parity_trace_->Flush();
    return true;
}

bool DFlashPredictor::ParityActive() const
{
    return parity_trace_ && parity_trace_->active;
}

void DFlashPredictor::CaptureParityTensor(const char* name, const Tensor& value) const
{
    if (parity_trace_) {
        parity_trace_->Capture(name, value);
    }
}

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
    // SGLang configures this boundary with scaled_residual_stream=False,
    // which uses ordinary FP16 RMSNorm and does not apply Laguna's BF16
    // residual-stream rounding. Keep the old behavior only as an A/B control.
    static const bool context_bf16_round = [] {
        const char* value = std::getenv("TM_DFLASH_CONTEXT_BF16_ROUND");
        return value && value[0] == '1';
    }();
    if (context_bf16_round) {
        invokeDFlashRoundBFloat16(normalized, core::Context::stream().handle());
        TM_CUDA_CHECK(cudaGetLastError());
    }
    PrepareParityContext(target_hidden, projected, normalized);
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
        attention_.Forward(
            {target_phase, context, discarded, layer->attention.get(), attention_indices_[i], 1.f, true});
        TM_CUDA_CHECK(cudaGetLastError());
    }
    static std::atomic<bool> logged{false};
    if (!logged.exchange(true)) {
        TM_LOG_INFO("[DFlash2] materialized context KV: tokens={} layers={} phase={}",
                    context.shape(0),
                    attention_indices_.size(),
                    target_phase);
    }
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
    CaptureParityTensor("block.ids", Tensor{block_ids, {(ssize_t)batch_size, weights_.block_size}});
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
    CaptureParityTensor("selector.prediction_hidden", prediction_hidden);

    Buffer_<int> candidate_ids{(ssize_t)rows * selector->top_k, kDEVICE};
    Tensor unary_scores{{rows, selector->top_k}, kFloat32, kDEVICE};
    if (UseLocalDFlashTopK() && !TraceDFlashSelector()) {
        candidates_fn_(candidate_ids,
                       unary_scores,
                       prediction_hidden,
                       weights_.output_multiplier,
                       weights_.final_logit_softcapping);
    }
    else {
        Tensor logits = logits_fn_(prediction_hidden);
        static std::atomic<bool> selector_reported{false};
        if (TM_UNLIKELY(TraceDFlashSelector() && !selector_reported.exchange(true))) {
            ReportDFlashTensor("selector.hidden", prediction_hidden);
            ReportDFlashTensor("selector.logits", logits);
        }
        invokeDFlashTopK16(candidate_ids,
                           unary_scores,
                           logits,
                           logits.shape(1),
                           0,
                           weights_.output_multiplier,
                           weights_.final_logit_softcapping,
                           core::Context::stream().handle());
    }

    CaptureParityTensor(
        "selector.candidate_ids", Tensor{candidate_ids, {(ssize_t)rows, selector->top_k}});
    CaptureParityTensor("selector.unary_scores", unary_scores);

    Tensor selector_hidden{{rows, selector->state_rank}, dtype_, kDEVICE};
    linear_.Forward(prediction_hidden, *selector->hidden_projection, selector_hidden);
    TM_CUDA_CHECK(cudaGetLastError());
    CaptureParityTensor("selector.hidden", selector_hidden);

    Tensor selector_scores;
    Tensor* selector_scores_ptr = nullptr;
    if (parity_trace_ && parity_trace_->active) {
        selector_scores = Tensor{{batch_size, slots, selector->top_k}, kFloat32, kDEVICE};
        selector_scores_ptr = &selector_scores;
    }

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
                               core::Context::stream().handle(),
                               selector_scores_ptr);
    if (selector_scores_ptr) {
        CaptureParityTensor("selector.scores", selector_scores);
    }
    CaptureParityTensor("selector.selected_ids", Tensor{output, {(ssize_t)batch_size, slots}});
    return output;
}

Tensor DFlashPredictor::RunDraftLayers(Tensor hidden, int phase) const
{
    TM_CHECK_EQ(attention_indices_.size(), 5);
    const int  attention_phase = attention_phase_base_ >= 0 ? attention_phase_base_ + phase : phase;
    const int  token_num       = hidden.shape(0);
    const auto stream          = core::Context::stream().handle();
    static std::atomic<bool> block_reported{false};
    const bool trace_block  = TraceDFlashSelector() && !block_reported.exchange(true);
    const bool report_block = trace_block || ParityActive();
    auto report = [&](const std::string& stage, const Tensor& value) {
        if (TM_UNLIKELY(trace_block)) {
            ReportDFlashTensor(stage, value);
        }
        CaptureParityTensor(stage.c_str(), value);
    };
    auto report_named = [&](const char* stage, const Tensor& value) {
        if (TM_UNLIKELY(report_block)) {
            report(stage, value);
        }
    };
    auto report_layer = [&](int layer, const char* stage, const Tensor& value) {
        if (TM_UNLIKELY(report_block)) {
            report("layer" + std::to_string(layer) + stage, value);
        }
    };
    report_named("block.embedding", hidden);

    // DFlash2 was trained in BF16 and its unnormalized residual can exceed
    // FP16's 65,504 limit on V100. Keep the residual and TP reduction in FP32,
    // then emit the normalized activation in FP16 for GEMMs.
    Tensor residual{hidden.shape(), kFloat32, kDEVICE};
    invokeDFlashCastToFloat(residual, hidden, stream);
    report_named("block.initial_residual", residual);

    auto* first_layer = TM_CHECK_NOTNULL(weights_.layer(0));
    invokeRMSNorm(hidden,
                  hidden,
                  first_layer->attention_norm->weight,
                  first_layer->attention_norm->norm_eps_,
                  first_layer->attention_norm->zero_centered_,
                  stream);
    TM_CUDA_CHECK(cudaGetLastError());
    // The draft checkpoint's pre-norm boundaries are BF16. Preserve that
    // rounding while storing FP16 activations for V100 GEMMs.
    invokeDFlashRoundBFloat16(hidden, stream);
    report_named("block.initial_norm", hidden);

    constexpr float kResidualScale = 256.f;
    constexpr float kGateUpScale   = 32.f;
    static const bool reduce_before_conv = [] {
        const char* value = std::getenv("TM_DFLASH_REDUCE_BEFORE_CONV");
        return value && value[0] == '1';
    }();

    auto reduce_branch = [&](Tensor& value) {
        if (ctx_.comm.d_comm) {
            ctx_.comm.d_comm->AllReduceSum(value.raw_data(),
                                           value.raw_data(),
                                           value.size(),
                                           kHalf,
                                           ctx_.comm.d_tp_group,
                                           stream);
        }
    };

    auto residual_norm = [&](Tensor&           value,
                             Tensor&           res,
                             const Tensor&     bias,
                             const NormWeight& norm,
                             float             reduced_scale,
                             int               layer,
                             const char*       post_collective_stage) {
        // Laguna transports each 1/256 branch in FP16 and performs the TP
        // reduction in that dtype. The previous FP32 cast added a kernel,
        // doubled collective traffic, and changed reduction rounding relative
        // to the SGLang reference draft.
        if (!reduce_before_conv) {
            reduce_branch(value);
            report_layer(layer, post_collective_stage, value);
        }
        invokeDFlashResidualRMSNorm(value,
                                    res,
                                    value,
                                    bias,
                                    norm.weight,
                                    norm.norm_eps_,
                                    norm.zero_centered_,
                                    reduced_scale,
                                    stream);
    };

    for (int i = 0; i < (int)attention_indices_.size(); ++i) {
        auto* layer     = TM_CHECK_NOTNULL(weights_.layer(i));
        auto* attn_conv = TM_CHECK_NOTNULL(weights_.attention_conv(i));
        auto* mlp_conv  = TM_CHECK_NOTNULL(weights_.mlp_conv(i));
        TM_CHECK(layer->attention && layer->feed_forward);

        report_layer(i, ".input.hidden", hidden);
        report_layer(i, ".input.residual", residual);
        auto attn_input = PrepareGroupedConv(hidden, *attn_conv);
        report_layer(i, ".attention.conv_delta", attn_input.delta);
        report_layer(i, ".attention.conv_side0", attn_input.output);
        Tensor attn_output{{token_num, hidden_units_}, dtype_, kDEVICE};
        attention_.Forward({attention_phase,
                            attn_input.output,
                            attn_output,
                            layer->attention.get(),
                            attention_indices_[i],
                            1.f / kResidualScale});
        report_layer(i, ".attention.wo_local", attn_output);
        if (reduce_before_conv) {
            reduce_branch(attn_output);
            report_layer(i, ".attention.tp_reduced_pre_conv", attn_output);
        }
        attn_output = FinishGroupedConv(attn_output, attn_input.delta, *attn_conv);
        report_layer(i, ".attention.conv_side1", attn_output);
        residual_norm(attn_output,
                      residual,
                      layer->attention->wo->bias,
                      *layer->ffn_norm,
                      kResidualScale,
                      i,
                      ".attention.tp_reduced_post_conv");
        report_layer(i, ".attention.norm_output", attn_output);
        report_layer(i, ".attention.residual_state", residual);
        hidden = std::move(attn_output);

        auto mlp_input = PrepareGroupedConv(hidden, *mlp_conv);
        report_layer(i, ".mlp.conv_delta", mlp_input.delta);
        report_layer(i, ".mlp.conv_side0", mlp_input.output);

        // Match SGLang's Laguna SM70 MLP exactly: shrink W13's input, restore
        // gate/up in FP32 with BF16 rounding, dynamically scale SwiGLU by row
        // for W2, then transport W2's result divided by the residual scale.
        auto& mlp = *TM_CHECK_NOTNULL(layer->feed_forward.get());
        auto* fused = TM_CHECK_NOTNULL(mlp.w1w3.get());
        TM_CHECK(fused->weight) << "DFlash2 Laguna path requires fused gate/up weights";
        TM_CHECK(!mlp.is_fused_silu) << "DFlash2 Laguna path requires unfused SwiGLU activation";
        invokeDFlashScale(mlp_input.output, 1.f / kGateUpScale, stream);
        report_layer(i, ".mlp.scaled_input", mlp_input.output);
        Tensor gate_up;
        linear_.Forward(mlp_input.output, *fused, gate_up);
        report_layer(i, ".mlp.gate_up", gate_up);
        TM_CHECK_EQ(gate_up.shape(1), 2 * mlp.inter_size);
        Tensor activated{{token_num, mlp.inter_size}, dtype_, kDEVICE};
        Tensor activation_scales{{token_num}, kFloat32, kDEVICE};
        invokeDFlashLagunaSilu(activated, activation_scales, gate_up, kGateUpScale, stream);
        report_layer(i, ".mlp.activated", activated);
        report_layer(i, ".mlp.activation_scales", activation_scales);
        linear_.Forward(activated, *mlp.w2, mlp_input.output);
        report_layer(i, ".mlp.w2_local", mlp_input.output);
        if (reduce_before_conv) {
            reduce_branch(mlp_input.output);
            report_layer(i, ".mlp.tp_reduced_pre_scale", mlp_input.output);
        }
        invokeDFlashScaleRows(mlp_input.output, activation_scales, 1.f / kResidualScale, stream);
        report_layer(i, ".mlp.scaled_w2", mlp_input.output);
        Tensor mlp_output = FinishGroupedConv(mlp_input.output, mlp_input.delta, *mlp_conv);
        report_layer(i, ".mlp.conv_side1", mlp_output);

        const NormWeight& output_norm =
            i + 1 < (int)attention_indices_.size() ? *weights_.layer(i + 1)->attention_norm : *weights_.final_norm;
        residual_norm(mlp_output,
                      residual,
                      {},
                      output_norm,
                      kResidualScale,
                      i,
                      ".mlp.tp_reduced_post_conv");
        report_layer(i, ".mlp.norm_output", mlp_output);
        report_layer(i, ".mlp.residual_state", residual);
        hidden = std::move(mlp_output);
    }
    return hidden;
}

}  // namespace turbomind
