
#include "src/turbomind/models/language_model.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#include "src/turbomind/comm/device_comm.h"
#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/core/interval.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/core/state.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/cache_registry.h"
#include "src/turbomind/engine/request.h"
#include "src/turbomind/generation/generation.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/models/input_processor.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/mtp_predictor.h"
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/models/model_weight.h"
#include "src/turbomind/models/mtp_weight.h"
#include "src/turbomind/models/output_processor.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"
#include "src/turbomind/utils/string_utils.h"

// #include "dbg.h"

namespace turbomind {

using std::vector;
using std::unique_ptr;
using std::shared_ptr;

struct LanguageModel::Impl {
    const Communicators& comm_;
    const ModelWeight&   weights_;
    LlamaLinear&         linear_;

    const int  tp_size_;
    const int  tp_rank_;
    const bool use_ag2d_;

    const int attn_dp_size_;
    const int attn_dp_rank_;
    const int max_batch_size_;

    const bool debug_;

    Buffer_<bool> false_;

    // mutable state
    State finished_;
    State sequence_length_;  // length of known tokens
    // immutable state
    Buffer_<int> autoreg_ids_;
    // Buffer_<int> autoreg_ids_offsets_;

    // Symmetric buffer for holding global hidden states or logits
    Buffer_<uint8_t> symm_buf_;

    // Global (all attention DP ranks) per-token validity mask, built at Forward time;
    // consumed by the attention layers (their DP-local slice) and, eventually, the MoE router.
    Buffer_<bool> token_mask_;

    // Symmetric gather buffer for the per-rank `[q_offsets | finished]` metadata blocks
    // ([attn_dp_size, meta_bytes], 16B-aligned rows for the in-place AllGather).
    // Only allocated when attn_dp > 1.
    Tensor_<uint8_t> symm_token_meta_;

    // Max chunk size for compute / output full logits
    int max_logits_len_ = 0;

    Buffer_<int>  sequence_length_buf_;
    Buffer_<int>  readonly_block_num_buf_;  // {max_batch_size}, kCPUpinned
    Buffer_<bool> finished_buf_;

    struct Data {
        Buffer_<int>  sequence_length;
        Buffer_<int>  readonly_block_num;
        Buffer_<bool> finished;

        Buffer_<bool> autoregres;
        Buffer_<bool> generating;

        int  n_generating;
        bool all_decode;

        /// Per-row sequence identity, captured in Setup because `requests` is
        /// gone from the map by Forward. Used to confirm that a draft is being
        /// scored against the same sequence that produced it.
        std::vector<uint64_t> uids;

        /// Per-row key length for this step, host-side, all rows.
        std::vector<int> seq_lens;
    };

    vector<Data> data_;

    std::optional<InputProcessor>   input_processor_;
    std::unique_ptr<UnifiedDecoder> unified_decoder_;

    /// Draft-token generator, present only when the checkpoint carries an MTP
    /// layer and the decoder registered a KV slot for it.
    std::unique_ptr<MTPPredictor> mtp_predictor_;

    /// Drafts from the previous decode step, held on the host so the next step
    /// can score them against the token the target actually sampled.
    ///
    /// This measures acceptance without acting on it. Speculation proper needs
    /// the target to verify K+1 positions in one forward, which is a scheduler
    /// change. But the ordinary decode loop already produces exactly one
    /// ground-truth token per step, so drafting at step t and comparing at
    /// t+1 yields the real step-1 accept rate with no scheduler involvement
    /// and no risk to output.
    ///
    /// It is deliberately step 1 only. Scoring draft k>0 would require the
    /// target to have produced k tokens along the drafted path, which is
    /// precisely what non-speculative decoding does not do -- assuming
    /// otherwise would report an accept length the model never earned.
    std::vector<int>      mtp_prev_draft_;  ///< step-0 draft per row, batch order
    std::vector<uint64_t> mtp_prev_uids_;   ///< the sequences those drafts belong to
    std::vector<int>      mtp_prev_input_;  ///< token each draft was conditioned on
    bool                  mtp_prev_valid_{false};

    /// One in-flight draft set awaiting its ground truth.
    ///
    /// A draft made at step t predicts the tokens the target will sample at
    /// t+1 .. t+K. Those are only known K steps later, so each set is held and
    /// scored one position per subsequent step. `age` counts how many steps
    /// have elapsed, which selects the draft step being scored.
    struct PendingDraft {
        std::vector<int>      tokens;  ///< [step][batch], as Draft produces
        std::vector<uint64_t> uids;
        /// Per row: is every draft step scored so far still correct? Gates the
        /// conditional statistic, and latches false on the first miss.
        std::vector<char> prefix_ok;
        int                   bsz{0};
        int                   num_drafts{0};
        int                   age{0};
    };
    std::vector<PendingDraft> mtp_pending_;

    /// Per-step-index acceptance: matched/scored for draft step k.
    std::vector<size_t> mtp_step_matched_;
    std::vector<size_t> mtp_step_scored_;

    /// Conditional acceptance: step k scored only where prefix 0..k-1 was
    /// correct. Unlike the raw rate this does not decay with depth.
    std::vector<size_t> mtp_cond_matched_;
    std::vector<size_t> mtp_cond_scored_;

    /// Cumulative step-1 acceptance, reported periodically.
    size_t mtp_scored_{0};
    size_t mtp_matched_{0};
    // Distinct drafted token ids; see the note at the insert site.
    std::set<int> mtp_distinct_;
    size_t mtp_echo_{0};    ///< draft equalled the token it was conditioned on
    size_t mtp_repeat_{0};  ///< target itself repeated that token

    /// Engine parameters, retained for the speculation settings.
    const EngineParam engine_param_;

    /// Emit the drafting record once rather than on every decode step.
    bool mtp_logged_{false};
    std::optional<OutputProcessor>  output_processor_;
    std::unique_ptr<Generation>     generation_;  // token generator

    void Run(BatchOp op, int phase, TensorMap& env)
    {
        switch (op) {
            case BatchOp::kSetup:
                return Setup(phase, env);
            case BatchOp::kPrepare:
                return Prepare(phase, env);
            case BatchOp::kForward:
                return Forward(phase, env);
            case BatchOp::kUnprep:
                return Unprep(phase, env);
            case BatchOp::kFetch:
                return Fetch(phase, env);
            default:
                input_processor_->Run(op, phase, env);
                unified_decoder_->Run(op, phase, env);
                generation_->Run(op, phase, env);
                output_processor_->Run(op, phase, env);
        }
    }

    Impl(
        CacheRegistry& registry, const EngineParam& engine, const Context& ctx, const ModelWeight& weights, int phases);

    Tensor LookupEmbedding(const Buffer_<int>& input_ids, Buffer symm_buf);
    Tensor PostEmbedding(const Tensor& features, Buffer symm_buf);

    // Build the global per-token validity mask for this pass (see `token_mask_`).
    void BuildTokenMask(const bool* finished, const int* q_offsets, const BatchData& b);

    void Setup(int phase, TensorMap& env);
    void Prepare(int phase, TensorMap& env);
    void Forward(int phase, TensorMap& env);
    void Unprep(int phase, TensorMap& env);
    void Fetch(int phase, TensorMap& env);
};

LanguageModel::Impl::Impl(
    CacheRegistry& registry, const EngineParam& engine, const Context& ctx, const ModelWeight& weights, int phases):
    comm_{ctx.comm},
    weights_{weights},
    linear_{*ctx.linear},
    tp_size_{comm_.h_tp_group->n_ranks()},
    tp_rank_{comm_.h_tp_group->rank()},
    use_ag2d_{comm_.d_comm && comm_.d_comm->Query(comm::kHasAllGather2D)},
    attn_dp_size_{engine.attn_dp_size},
    attn_dp_rank_{engine.attn_dp_rank},
    max_batch_size_{engine.max_batch_size},
    debug_{isDebug()},
    engine_param_{engine}
{

    false_ = {engine.max_batch_size, kDEVICE};
    Clear(false_);

    finished_buf_ = {engine.max_batch_size, kCPUpinned};
    finished_     = {{engine.max_batch_size}, kBool, kDEVICE};

    autoreg_ids_ = {engine.max_batch_size, kDEVICE};
    // autoreg_ids_offsets_ = {engine.max_batch_size + 1, kCPU};
    // std::fill_n(autoreg_ids_offsets_.data(), autoreg_ids_offsets_.size(), 0);

    sequence_length_buf_    = {engine.max_batch_size, kCPUpinned};
    readonly_block_num_buf_ = {engine.max_batch_size, kCPUpinned};
    sequence_length_        = {{engine.max_batch_size}, kInt, kDEVICE};
    for (int i = 0; i < phases; ++i) {
        auto& d              = data_.emplace_back();
        d.sequence_length    = empty_like(sequence_length_buf_, kDEVICE);
        d.readonly_block_num = empty_like(readonly_block_num_buf_, kDEVICE);
        d.finished           = empty_like(finished_buf_, kDEVICE);
        d.autoregres         = {engine.max_batch_size, kCPU};
        d.generating         = {engine.max_batch_size, kCPU};
    }

    input_processor_.emplace(engine, weights_.hidden_units, weights_.data_type, phases);

    unified_decoder_ = std::make_unique<UnifiedDecoder>(registry, engine, ctx, phases, weights_);

    // Build the draft predictor when the checkpoint supplies an MTP layer and
    // the decoder gave it a KV slot. Both conditions are required: the weights
    // alone are inert without somewhere to write keys and values.
    if (weights_.mtp && weights_.mtp->decoder_layer && unified_decoder_->mtp_attn_index() >= 0
        && unified_decoder_->attn_layer()) {
        mtp_predictor_ = std::make_unique<MTPPredictor>(
            *weights_.mtp,
            *unified_decoder_->attn_layer(),
            unified_decoder_->mtp_attn_index(),
            engine,
            ctx,
            [this](const Buffer_<int>& ids) { return LookupEmbedding(ids, symm_buf_); },
            [this](const Tensor& h) { return PostEmbedding(h, symm_buf_); });
    }

    const int vocab_size = weights_.output->output_dim * tp_size_;

    generation_ = std::make_unique<Generation>(
        kFloat32, engine.max_batch_size, engine.session_len, weights_.vocab_size, vocab_size, comm_.h_tp_group, phases);

    const ssize_t max_fwd_tokens = engine.max_forward_token_num;

    if (ctx.comm.d_comm) {
        auto symm_alloc = GetSymmAllocator(ctx.comm.d_comm);
        // Native comm fuses allreduce & rmsnorm in token granularity
        TM_CHECK(engine.max_forward_token_num % tp_size_ == 0);

        ssize_t bytes{};
        bytes = std::max(bytes,
                         byte_size(weights_.data_type, max_fwd_tokens * engine.attn_dp_size * weights_.hidden_units));
        bytes = std::max(bytes, byte_size(weights_.data_type, engine.max_batch_size * vocab_size));

        symm_buf_ = {bytes, symm_alloc};
        // Compute max logits length based on symm buffer size
        max_logits_len_ = symm_buf_.view(weights_.data_type).size() / vocab_size;

        if (attn_dp_size_ > 1) {
            const int q_bytes    = (max_batch_size_ + 1) * (int)sizeof(int);
            const int meta_bytes = (q_bytes + max_batch_size_ + 15) / 16 * 16;
            symm_token_meta_     = {{attn_dp_size_, meta_bytes}, symm_alloc};
        }
    }
    else {
        max_logits_len_ = std::max<int>(max_fwd_tokens * weights_.hidden_units / vocab_size, engine.max_batch_size);
    }

    token_mask_ = {max_fwd_tokens * attn_dp_size_, kDEVICE};

    output_processor_.emplace(weights_.vocab_size, max_logits_len_, tp_rank_, phases, [this](const Tensor& hstate) {
        return PostEmbedding(hstate, symm_buf_);
    });
}

Tensor LanguageModel::Impl::LookupEmbedding(const Buffer_<int>& input_ids, Buffer symm_buf)
{
    TM_FUNCTION_SCOPE();
    const auto st = core::Context::stream().handle();

    const int hidden_units = weights_.hidden_units;

    const auto& embedding_table = weights_.tok_embeddings;
    TM_CHECK_EQ(embedding_table.shape(1) * tp_size_, hidden_units);

    const int token_num = input_ids.size();

    Tensor input_embeds{{token_num, hidden_units}, weights_.data_type, kDEVICE};

    if (token_num == 0) {
        return input_embeds;
    }

    if (tp_size_ == 1) {
        invokeEmbeddingLookup(input_embeds, input_ids, embedding_table, st);
        TM_CUDA_CHECK(cudaGetLastError());
    }
    else if (use_ag2d_) {
        const auto local_hidden_units = embedding_table.shape(1);

        Tensor temp{symm_buf.view(weights_.data_type), {token_num, tp_size_, local_hidden_units}};
        Tensor local{temp.slice({0, tp_rank_, 0}, {-1, 1, -1}).squeeze(1)};

        invokeEmbeddingLookup(local, input_ids, embedding_table, st);
        TM_CUDA_CHECK(cudaGetLastError());

        comm_.d_comm->AllGather2D(local.raw_data(),
                                  temp.raw_data(),
                                  hidden_units,
                                  local_hidden_units,
                                  local_hidden_units,
                                  token_num,
                                  local.dtype(),
                                  {true, true},
                                  comm_.d_tp_group,
                                  st);
        TM_CUDA_CHECK(cudaGetLastError());

        Copy(temp.buffer(), input_embeds.buffer());
    }
    else {
        const auto local_hidden_units = embedding_table.shape(1);

        Tensor temp{symm_buf.view(weights_.data_type), {tp_size_, token_num, local_hidden_units}};
        Tensor local{temp.slice(tp_rank_).squeeze(0)};

        invokeEmbeddingLookup(local, input_ids, embedding_table, st);
        TM_CUDA_CHECK(cudaGetLastError());

        comm_.d_comm->AllGather(
            local.raw_data(), temp.raw_data(), local.size(), weights_.data_type, comm_.d_tp_group, st);
        TM_CUDA_CHECK(cudaGetLastError());

        invokeInPlaceTranspose102((uint16_t*)input_embeds.raw_data(),
                                  (uint16_t*)temp.raw_data(),
                                  tp_size_,
                                  token_num,
                                  local_hidden_units,
                                  false,
                                  st);
        TM_CUDA_CHECK(cudaGetLastError());
    }

    return input_embeds;
}

Tensor LanguageModel::Impl::PostEmbedding(const Tensor& features, Buffer symm_buf)
{
    TM_FUNCTION_SCOPE();
    NvtxScope scope("postDecodeEmbedding");

    const auto st = core::Context::stream().handle();

    const int bsz              = features.shape(0);
    const int local_vocab_size = weights_.output->output_dim;
    const int vocab_size       = local_vocab_size * tp_size_;

    if (bsz == 0) {
        return Tensor{{0, vocab_size}, weights_.data_type, kDEVICE};
    }

    if (tp_size_ == 1) {
        Tensor logits{{bsz, vocab_size}, weights_.data_type, kDEVICE};
        TM_SCOPE_CALL(linear_.Forward(features, *weights_.output, logits));
        TM_DEBUG_TENSOR(logits, "logits", 1);
        return logits;
    }
    else if (use_ag2d_) {
        Tensor logits{symm_buf.view(weights_.data_type), {bsz, tp_size_, local_vocab_size}};
        Tensor local = logits.slice({0, tp_rank_, 0}, {-1, 1, -1});
        TM_SCOPE_CALL(linear_.Forward(features, *weights_.output, local.squeeze(1)));
        comm_.d_comm->AllGather2D(local.raw_data(),
                                  logits.raw_data(),
                                  vocab_size,
                                  local_vocab_size,
                                  local_vocab_size,
                                  bsz,
                                  logits.dtype(),
                                  {true, true},
                                  comm_.d_tp_group,
                                  st);
        TM_CUDA_CHECK(cudaGetLastError());
        return logits.view({bsz, -1});
    }
    else {
        Tensor logits{symm_buf.view(weights_.data_type), {tp_size_, bsz, local_vocab_size}};
        Tensor local = logits.slice({tp_rank_, 0, 0}, {1, -1, -1});
        TM_SCOPE_CALL(linear_.Forward(features, *weights_.output, local.squeeze(0)));
        comm_.d_comm->AllGather(local.raw_data(), logits.raw_data(), local.size(), local.dtype(), comm_.d_tp_group, st);
        TM_CUDA_CHECK(cudaGetLastError());
        Tensor out{{bsz, vocab_size}, features.dtype(), features.device()};
        invokeTransposeAxis01(
            (uint16_t*)out.raw_data(), (uint16_t*)logits.raw_data(), tp_size_, bsz, local_vocab_size, st);
        TM_CUDA_CHECK(cudaGetLastError());
        return out;
    }
}

void LanguageModel::Impl::Setup(int phase, TensorMap& env)
{
    input_processor_->Run(BatchOp::kSetup, phase, env);

    auto& d    = data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    Buffer_<Sequence*> rc = env.at("requests").buffer();

    d.n_generating = 0;
    // Pure-decode means every row advances by exactly one token. Recorded here,
    // in Setup, because `requests` is gone from the map by Forward.
    d.all_decode = rc.size() > 0;

    d.uids.resize(rc.size());
    d.seq_lens.resize(rc.size());

    for (int i = 0; i < rc.size(); ++i) {
        auto& c         = *rc[i];
        d.uids[i]       = c.req->unique_id;
        d.autoregres[i] = c.autoregres;
        d.generating[i] = c.generating;
        d.n_generating += c.generating;
        d.all_decode &= c.input_len == 1;
        // The row's true key length this step, captured for ALL rows.
        // sequence_length_buf_ below is written only for non-autoregressive
        // rows -- decode rows carry their length forward on the device -- so it
        // cannot be read here as a host-side length. The MTP draft needs these
        // to know how much slack each row has left in its last KV block.
        d.seq_lens[i] = c.history_len + c.inflight_input_len + c.input_len;
        if (TM_UNLIKELY(!c.autoregres)) {
            sequence_length_buf_[i] = c.history_len + c.inflight_input_len + c.input_len;
        }
        readonly_block_num_buf_[i] = c.readonly_block_num;  // all rows, batch order
    }

    copy(sequence_length_buf_, rc.size(), d.sequence_length);
    copy(readonly_block_num_buf_, rc.size(), d.readonly_block_num);

    unified_decoder_->Run(BatchOp::kSetup, phase, env);
    generation_->Run(BatchOp::kSetup, phase, env);
    output_processor_->Run(BatchOp::kSetup, phase, env);
}

void LanguageModel::Impl::Prepare(int phase, TensorMap& env)
{
    env.emplace("autoreg_ids", autoreg_ids_);

    input_processor_->Run(BatchOp::kPrepare, phase, env);

    auto& d = data_.at(phase);

    auto& b    = *env.at("batch").data<BatchData*>()[0];
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    // core::CopyT copy{};

    if (auto group = copy.group()) {
        for (int i = 0; i < b.bsz; ++i) {
            if (const int j = b.perm[i]; j < b.bs0) {
                copy(finished_.front().data<bool>() + j, 1, finished_.back().data<bool>() + i);
            }
            else {
                copy(false_.data() + i, 1, finished_.back().data<bool>() + i);
            }
        }
        finished_.Swap();
    }

    if (auto group = copy.group()) {
        // Non-autoregressive rows use the submitted prefix length:
        // sequence_length = history_len + inflight_input_len + input_len.
        // Existing autoregressive rows carry the previous sequence_length forward.
        for (int i = 0; i < b.bsz; ++i) {
            if (const int j = b.perm[i]; j < b.bs0 && d.autoregres[i]) {
                copy(sequence_length_.front().data<int>() + j, 1, sequence_length_.back().data<int>() + i);
            }
            else {
                copy(d.sequence_length.data() + i, 1, sequence_length_.back().data<int>() + i);
            }
        }
        sequence_length_.Swap();
    }

    Buffer_<int> k_offsets{b.bsz + 1, kDEVICE};
    // PrefixSum(sequence_length_.front().data<int>(), bsz, k_offsets.data(), core::Context::stream().handle());

    // Buffer_<int> k_offsets_tmp{k_offsets.size(), kCPU};
    // Buffer_<int> sequence_length_tmp{sequence_length_.front().size(), kCPU};

    // Copy(k_offsets, k_offsets_tmp);
    // Copy(sequence_length_.front().buffer(), sequence_length_tmp);

    // core::Context::stream().Sync();

    // dbg(core::to_vector<int>(sequence_length_tmp.slice(0, bsz)));
    // dbg(core::to_vector<int>(k_offsets_tmp.slice(0, bsz + 1)));

    env.produce("finished", finished_.front());
    env.produce("sequence_length", sequence_length_.front());
    env.produce("readonly_block_num", d.readonly_block_num);
    env.produce("k_offsets", k_offsets);
    if (symm_buf_) {
        env.produce("symm_buf", symm_buf_);
    }

    // Produced here so consumers may borrow the pointer at kPrepare; the content is
    // only built at Forward time (`BuildTokenMask`).
    env.produce("token_mask", token_mask_);

    unified_decoder_->Run(BatchOp::kPrepare, phase, env);
    generation_->Run(BatchOp::kPrepare, phase, env);
    output_processor_->Run(BatchOp::kPrepare, phase, env);
}

void LanguageModel::Impl::BuildTokenMask(const bool* finished, const int* q_offsets, const BatchData& b)
{
    TM_FUNCTION_SCOPE();

    if (b.global_token_num == 0) {
        return;
    }

    TM_CHECK_EQ((int)b.local_token_num.size(), attn_dp_size_);
    TM_CHECK_LE(attn_dp_size_, kMaxAttnDPSize);

    const auto st = core::Context::stream().handle();

    // Byte stride between per-rank metadata blocks (0 when attn_dp == 1).
    size_t rank_stride = 0;

    if (attn_dp_size_ > 1) {
        const int q_bytes    = (max_batch_size_ + 1) * (int)sizeof(int);
        const int meta_bytes = symm_token_meta_.shape(1);

        // Stage this rank's metadata into its row of the symmetric buffer; the finished
        // tail is zeroed so padding slots never invalidate tokens.
        TM_CHECK_LE(b.bsz, max_batch_size_);
        char* slot = (char*)symm_token_meta_.data() + (ssize_t)attn_dp_rank_ * meta_bytes;
        core::Copy(q_offsets, b.bsz + 1, (int*)slot);
        core::Copy(finished, b.bsz, (bool*)(slot + q_bytes));
        TM_CUDA_CHECK(cudaMemsetAsync(slot + q_bytes + b.bsz, 0, max_batch_size_ - b.bsz, st));

        // In-place all-gather: the peers read this rank's contribution from its own row.
        comm_.d_comm->AllGather(slot, symm_token_meta_.data(), meta_bytes, kUint8, comm_.d_dp_group, st);

        q_offsets   = (const int*)symm_token_meta_.data();
        finished    = (const bool*)(symm_token_meta_.data() + q_bytes);
        rank_stride = meta_bytes;
    }

    // Rank r's tokens occupy [token_base[r], token_base[r] + local_token_num[r]) of the mask.
    int token_base[kMaxAttnDPSize];
    token_base[0] = 0;
    std::partial_sum(b.local_token_num.begin(), b.local_token_num.end() - 1, token_base + 1);

    invokeBuildTokenMask(token_mask_.data(),
                         finished,
                         q_offsets,
                         rank_stride,
                         token_base,
                         attn_dp_size_,
                         // DP > 1 scans all gathered slots (the finished tail is zeroed);
                         // DP == 1 scans only the active batch — beyond it the local
                         // `finished`/`q_offsets` hold stale data from previous passes.
                         attn_dp_size_ > 1 ? max_batch_size_ : b.bsz,
                         b.global_token_num,
                         st);
}

void LanguageModel::Impl::Forward(int phase, TensorMap& env)
{
    TM_FUNCTION_SCOPE();

    auto& d = data_.at(phase);
    auto& b = *env.at("batch").data<BatchData*>()[0];

    // Must run at Forward time: the `finished`/`q_offsets` H2D copies are only flushed
    // after kPrepare returns. The mask is ready before the decoder (its consumers) runs.
    BuildTokenMask(
        (const bool*)env.at("finished").buffer().raw_data(), (const int*)env.at("q_offsets").buffer().raw_data(), b);

    {
        Buffer_<int> k_offsets = env.at("k_offsets").buffer();
        PrefixSum(sequence_length_.front().data<int>(), b.bsz, k_offsets.data(), core::Context::stream().handle());
    }

    {  // compute input embeddings
        auto input_ids = env.at("input_ids").buffer();

        Tensor input_embeds = LookupEmbedding(input_ids, symm_buf_);
        TM_DEBUG_TENSOR(input_embeds, "embeddings", 1);

        auto& copy = *env.at("copy").data<BatchCopy*>()[0];
        input_processor_->PatchEmbedding(phase, input_embeds, copy, env);
        copy.Run();

        env.produce("input_embeds", std::move(input_embeds));
        // dbg(env);
    }

    env.produce("output_norm_weight", weights_.norm->weight);

    unified_decoder_->Forward(phase, env, weights_.layers_list());

    // env.at("batch").data<BatchData*>()[0]->Notify();

    output_processor_->OutputHiddenStatesAndLogits(phase, env, 2);

    auto& hidden_states = env.at("hidden_states");

    env.produce("logits", PostEmbedding(hidden_states, symm_buf_));

    output_processor_->OutputHiddenStatesAndLogits(phase, env, 1);

    if (d.n_generating) {
        generation_->Run(BatchOp::kForward, phase, env);
        Copy(env.at("output_ids").buffer(), autoreg_ids_);

        // Multi-Token Prediction: draft the next few tokens from the state we
        // already have. `output_ids` is the token just sampled and
        // `hidden_states` is the state that produced it, which are exactly the
        // two inputs the draft layer consumes.
        //
        // The drafts are produced but not yet consumed. Verification and
        // rollback are not implemented, so accepting them would change output.
        // Running the draft here measures its cost and proves the path
        // executes, without altering a single generated token.
        //
        // `hidden_states` carries one row per batch sequence, not one per
        // generating sequence: unified_decoder produces it as
        // {selected_token_pos.size(), hidden_units}, and selected_token_pos is
        // filled for all `bsz` rows. Passing d.n_generating as the batch while
        // handing over the full tensor made the two disagree, and the first
        // RMSNorm aborted on `out.shape() == x.shape()`.
        //
        // Draft only when every row is generating AND every row is a real
        // decode step.
        //
        // `generating` does NOT mean "decode". engine.cc:538 sets it to
        //
        //   resume_len + inflight_input_len + input_len == seq_len + inflight_new_tokens
        //
        // i.e. "this forward reaches the end of the sequence", which is also
        // true for the LAST PREFILL CHUNK, because that is the step on which
        // the first token is sampled. So a prefill row with input_len == 8 is
        // `generating`, and n_generating == bsz held while the batch was still
        // a prefill. The draft then ran against attention statistics that had
        // been prepared for an 8-token prefill:
        //
        //   Check failed: d.prefill.q_sum + d.decode.n == q_count (8 vs. 1)
        //
        // The draft submits one row per sequence, so it is only consistent with
        // the cached stats when the batch is pure decode, i.e. every row has
        // input_len == 1. Anything else reuses another step's shape.
        //
        // A mixed batch would require gathering the decode rows first, which is
        // real work that belongs with verification rather than here, where the
        // drafts are discarded.
        // `b.bsz` and not env.at("requests"): `requests` is placed in the map
        // by Setup and is gone by Forward, so reading it here aborted the run
        // even with speculation disabled, because the lookup ran before the
        // num_draft_tokens guard could short-circuit it.
        const int bsz = b.bsz;
        if (mtp_predictor_ && engine_param_.num_draft_tokens > 0 && d.n_generating == bsz && d.all_decode) {
            const int n_draft = engine_param_.num_draft_tokens;
            //
            // NOTE: the MTP KV slot is never populated. UnifiedDecoder::Forward
            // iterates weights_.layers_list(), which returns only the target's
            // `layers` child, so the draft layer never writes the accepted
            // prompt or decode history into its own slot. The first draft
            // attention therefore reads uninitialised entries, and the loop
            // reuses one set of q/k offsets for every step, so each step writes
            // the same cache position.
            //
            // The drafts are discarded, so this cannot corrupt output today.
            // Seeding and advancing that state is part of the verification
            // work, not a detail: without it accept length is meaningless
            // because the draft is attending to nothing.
            //
            // Which is exactly why the acceptance meter below exists BEFORE
            // that fix rather than after it. It turns "the draft attends to
            // nothing" from an argument into a measurement: with unseeded KV
            // the step-1 accept rate should be near zero, and once the KV is
            // seeded correctly it must rise. Without a number recorded first,
            // a seeding change that does nothing would be indistinguishable
            // from one that works.
            // Slice to bsz: output_ids is `output_ids_buf_`, allocated once at
            // max_batch_size (128) and reused, so its extent is the capacity
            // and not this step's batch. Passing it whole made the draft embed
            // 128 rows and collide with a bsz-row hidden_states inside the
            // first RMSNorm.
            auto ids = env.at("output_ids").buffer().view<int>().slice(0, bsz);

            TM_CHECK_EQ((int)hidden_states.shape(0), bsz)
                << "MTP: hidden_states rows must match the batch";

            // Score the PREVIOUS step's drafts before producing new ones.
            //
            // `ids` is the token the target just sampled, i.e. the ground
            // truth for the draft made one step ago. Comparing them gives the
            // step-1 accept rate directly, with no verification machinery and
            // no effect on output.
            //
            // Only score when row i is the SAME SEQUENCE that produced the
            // draft. Matching on batch size alone is not enough: two unrelated
            // batches can both have bsz==1, and rows are re-permuted as
            // requests join and leave. Scoring across that boundary compares
            // sequence A's draft to sequence B's token and books the mismatch
            // as a rejection -- noise that looks exactly like a weak draft
            // model, which is the failure this whole measurement exists to
            // detect. Compare the captured unique_ids instead.
            // Two control statistics, because a bare accept rate cannot
            // distinguish prediction from repetition.
            //
            //   echo  : draft[t] == input token at t  (the draft copied its
            //           own input; would score well on repeated tokens)
            //   repeat: actual[t] == input token at t (the TARGET repeated;
            //           the share of the accept rate any echo would win free)
            //
            // If accept and echo track each other, the predictor is not
            // predicting -- it is echoing, and the rate is an artefact.
            if (mtp_prev_valid_ && mtp_prev_uids_ == d.uids) {
                // Stage to host and synchronise: `ids` is device memory that
                // the next step overwrites. A pinned buffer plus an event
                // would avoid the stall and is worth doing if this ever stops
                // being instrumentation and becomes a hot path.
                Buffer_<int> actual{bsz, kCPU};
                core::Copy(ids, bsz, actual);
                core::Context::stream().Sync();
                for (int i = 0; i < bsz; ++i) {
                    ++mtp_scored_;
                    mtp_matched_ += (actual[i] == mtp_prev_draft_[i]);
                    // How many DISTINCT tokens the draft head has ever
                    // produced. This separates two states the acceptance rate
                    // cannot: a predictor that is genuinely predicting, and
                    // one that emits a near-constant token which happens to
                    // match often because natural text is repetitive. A
                    // handful of distinct values across hundreds of draws
                    // means the head is inert regardless of what percentage it
                    // scores.
                    mtp_distinct_.insert(mtp_prev_draft_[i]);
                    if (i < (int)mtp_prev_input_.size()) {
                        mtp_echo_ += (mtp_prev_draft_[i] == mtp_prev_input_[i]);
                        mtp_repeat_ += (actual[i] == mtp_prev_input_[i]);
                    }
                }
            }
            mtp_prev_valid_ = false;

            // Score every in-flight draft set that this step supplies ground
            // truth for. A set of age k has its step-k prediction resolved by
            // the token just sampled.
            //
            // This is what separates "the predictor works" from "step 1 works".
            // If per-decode history is sound but the within-Draft position
            // never advances, step 1 should look like the aggregate rate and
            // steps 2..K should collapse. Measuring per step turns that from a
            // claim into evidence.
            {
                Buffer_<int> truth{bsz, kCPU};
                core::Copy(ids, bsz, truth);
                core::Context::stream().Sync();

                for (auto& p : mtp_pending_) {
                    if (p.bsz != bsz || p.uids != d.uids) {
                        continue;  // batch changed; rows no longer comparable
                    }
                    const int k = p.age;  // 0-based draft step now resolvable
                    if (k >= p.num_drafts) {
                        continue;
                    }
                    if ((int)mtp_step_scored_.size() <= k) {
                        mtp_step_scored_.resize(k + 1, 0);
                        mtp_step_matched_.resize(k + 1, 0);
                        mtp_cond_scored_.resize(k + 1, 0);
                        mtp_cond_matched_.resize(k + 1, 0);
                    }
                    // [step][batch]: step k starts at k * bsz.
                    const int base = k * p.bsz;
                    for (int i = 0; i < bsz; ++i) {
                        const bool hit = (truth[i] == p.tokens[base + i]);
                        ++mtp_step_scored_[k];
                        mtp_step_matched_[k] += hit;

                        // Conditional accuracy: score step k only on rows whose
                        // ENTIRE prefix 0..k-1 was correct.
                        //
                        // The raw rate above has a hard ceiling of the previous
                        // step's rate, because step k conditions on its own
                        // predecessors' output. At depth 4 that leaves under
                        // one expected hit, so a raw 0.0% cannot distinguish a
                        // broken step from an unreachable one. The conditional
                        // rate does not shrink with depth, and it is the
                        // quantity that predicts speculative speedup.
                        if (p.prefix_ok[i]) {
                            ++mtp_cond_scored_[k];
                            mtp_cond_matched_[k] += hit;
                            p.prefix_ok[i] = hit;  // prefix stays correct only while hits continue
                        }
                    }
                }
                for (auto& p : mtp_pending_) {
                    ++p.age;
                }
                mtp_pending_.erase(std::remove_if(mtp_pending_.begin(),
                                                  mtp_pending_.end(),
                                                  [](const PendingDraft& p) { return p.age >= p.num_drafts; }),
                                   mtp_pending_.end());
            }

            auto drafts = mtp_predictor_->Draft(bsz,  //
                                                hidden_states,
                                                ids,
                                                n_draft,
                                                phase,
                                                d.seq_lens.data(),
                                                env);

            if (!mtp_logged_) {
                mtp_logged_ = true;
                TM_LOG_INFO("[MTP] drafted {} token(s) for {} sequence(s); drafts are not yet verified",
                            drafts.num_drafts,
                            bsz);
            }

            // Retain step 0 of this batch's drafts for the next step to score.
            // draft_tokens is [step][batch], so step 0 is the first `bsz`
            // entries -- a contiguous run, not a stride.
            if (drafts.num_drafts > 0 && (int)drafts.draft_tokens.size() >= bsz) {
                Buffer_<int> step0{bsz, kCPU};
                core::Copy(drafts.draft_tokens, bsz, step0);
                core::Context::stream().Sync();
                mtp_prev_draft_.assign(step0.begin(), step0.end());
                // The token this draft was conditioned on, kept so the next
                // step can tell prediction apart from echo.
                Buffer_<int> in0{bsz, kCPU};
                core::Copy(ids, bsz, in0);
                core::Context::stream().Sync();
                mtp_prev_input_.assign(in0.begin(), in0.end());
                mtp_prev_uids_  = d.uids;
                mtp_prev_valid_ = true;

                // Retain the whole [step][batch] set so steps beyond the first
                // can be scored as their ground truth arrives.
                const int n_tok = drafts.num_drafts * bsz;
                if ((int)drafts.draft_tokens.size() >= n_tok) {
                    Buffer_<int> all{n_tok, kCPU};
                    core::Copy(drafts.draft_tokens, n_tok, all);
                    core::Context::stream().Sync();
                    PendingDraft p;
                    p.tokens.assign(all.begin(), all.end());
                    p.uids       = d.uids;
                    p.bsz        = bsz;
                    p.num_drafts = drafts.num_drafts;
                    p.age        = 0;
                    p.prefix_ok.assign(bsz, 1);
                    // Bounded by construction: a set is erased once its age
                    // reaches num_drafts, so at most K are ever live. The
                    // guard covers the path where rows stop matching and sets
                    // are skipped rather than scored -- they still age, so
                    // they still retire, but a cheap ceiling means a future
                    // change to that logic cannot leak memory silently.
                    if ((int)mtp_pending_.size() < 4 * n_draft) {
                        mtp_pending_.push_back(std::move(p));
                    }
                }
            }

            // Report periodically. A rate over a handful of tokens is noise,
            // so the first report waits for a sample worth quoting. 32 rather
            // than 64: a 64-token generation scores only 63 pairs (the first
            // step has no prior draft), so a 64 threshold would print nothing
            // at all for the standard run and look like the meter was broken.
            if (mtp_scored_ >= 32 && mtp_scored_ % 32 == 0) {
                TM_LOG_INFO("[MTP] step-1 draft acceptance: {}/{} = {:.1f}% "
                            "(echo {:.1f}%, target-repeat {:.1f}%, distinct drafts {})",
                            mtp_matched_,
                            mtp_scored_,
                            100.0 * (double)mtp_matched_ / (double)mtp_scored_,
                            100.0 * (double)mtp_echo_ / (double)mtp_scored_,
                            100.0 * (double)mtp_repeat_ / (double)mtp_scored_,
                            mtp_distinct_.size());

                for (size_t k = 0; k < mtp_step_scored_.size(); ++k) {
                    if (mtp_step_scored_[k] == 0) {
                        continue;
                    }
                    const size_t cs = k < mtp_cond_scored_.size() ? mtp_cond_scored_[k] : 0;
                    const size_t cm = k < mtp_cond_matched_.size() ? mtp_cond_matched_[k] : 0;
                    TM_LOG_INFO("[MTP]   draft step {}: {}/{} = {:.1f}%  | given correct prefix: {}/{} = {}",
                                k + 1,
                                mtp_step_matched_[k],
                                mtp_step_scored_[k],
                                100.0 * (double)mtp_step_matched_[k] / (double)mtp_step_scored_[k],
                                cm,
                                cs,
                                cs ? fmtstr("%.1f%%", 100.0 * (double)cm / (double)cs) : std::string("n/a"));
                }
            }
        }
    }
}

void LanguageModel::Impl::Unprep(int phase, TensorMap& env)
{
    auto& d    = data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    copy(sequence_length_.front().buffer(), d.sequence_length.size(), d.sequence_length);

    copy(finished_.front().buffer(), d.finished.size(), d.finished);

    unified_decoder_->Run(BatchOp::kUnprep, phase, env);
    generation_->Run(BatchOp::kUnprep, phase, env);
}

void LanguageModel::Impl::Fetch(int phase, TensorMap& env)
{
    auto& d    = data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    copy(d.sequence_length, d.sequence_length.size(), sequence_length_buf_);
    env.produce("sequence_length", sequence_length_buf_);

    copy(d.finished, d.finished.size(), finished_buf_);
    env.produce("finished", finished_buf_);

    env.produce("generating", d.generating);

    generation_->Run(BatchOp::kFetch, phase, env);
}

LanguageModel::~LanguageModel() = default;

LanguageModel::LanguageModel(LanguageModel&&) noexcept = default;

LanguageModel::LanguageModel(
    CacheRegistry& registry, const EngineParam& engine, const Context& ctx, const ModelWeight& weights, int phases)
{
    impl_ = std::make_unique<Impl>(registry, engine, ctx, weights, phases);
}

void LanguageModel::Run(BatchOp op, int phase, TensorMap& env)
{
    return TM_CHECK_NOTNULL(impl_)->Run(op, phase, env);
}

}  // namespace turbomind
