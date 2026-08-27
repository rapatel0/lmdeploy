
#include "src/turbomind/models/language_model.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <numeric>
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
#include "src/turbomind/models/llama/rejection_sampling.h"
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

        /// Row sequence pointers, retained from Setup.
        ///
        /// `requests` is removed from the env map before Forward, but the
        /// speculative ops run after Forward and must write accepted tokens and
        /// the new seq_len back onto the sequence itself. Holding the pointers
        /// here is what lets kReject and kRollback reach them.
        ///
        /// Valid only within the step that filled them.
        std::vector<Sequence*> rows;

        /// Per-row draft count for this step, captured in Setup for the same
        /// reason. num_drafts on the sequence is rewritten by kDraft at the end
        /// of the step, so the value the forward actually verified must be kept
        /// separately or rejection would read the wrong K.
        std::vector<int> num_drafts;
    };

    vector<Data> data_;

    std::optional<InputProcessor>   input_processor_;
    std::unique_ptr<UnifiedDecoder> unified_decoder_;

    /// Draft-token generator, present only when the checkpoint carries an MTP
    /// layer and the decoder registered a KV slot for it.
    std::unique_ptr<MTPPredictor> mtp_predictor_;

    /// Verification logits, held between Forward and kReject because the env
    /// map is rebuilt in between.
    Tensor verify_logits_;

    /// Accept-length accounting. 1 + accepted/steps is the tokens emitted per
    /// forward, which is the number that decides whether speculation pays.
    size_t mtp_accepted_ = 0;
    size_t mtp_steps_    = 0;

    /// Per row, how many drafts the last verification accepted. Selects which
    /// row of a [bsz*(K+1), hidden] block carries the accepted tip.
    std::vector<int> last_accepted_;

    /// Engine parameters, retained for the speculation settings.
    const EngineParam engine_param_;

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
            case BatchOp::kDraft:
                return DraftTokens(phase, env);
            case BatchOp::kReject:
                return RejectDrafts(phase, env);
            case BatchOp::kRollback:
                return Rollback(phase, env);
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

    /// Does this phase's batch carry drafts for the next forward to verify?
    bool HasDraftsToVerify(int phase) const
    {
        const auto& d = data_.at(phase);
        if (d.rows.empty()) {
            return false;
        }

        // EVERY row must carry the SAME draft count, not merely some row.
        //
        // Verification submits 1 + num_drafts tokens per row and reads back
        // logits for each submitted position, so it only works when the batch
        // is uniform. A sequence that joins mid-stream has num_drafts == 0
        // while the others carry K: input_processor then selects a different
        // number of positions per row, the logits are no longer
        // bsz*(K+1) rows, and the shape assertion in RejectDrafts aborts the
        // whole engine.
        //
        // Requiring uniformity turns that abort into one skipped step of
        // speculation: the batch decodes normally, every row drafts again at
        // the end of it, and the next step is uniform. Correctness is never at
        // stake -- only one step's worth of speedup.
        //
        // KNOWN LIMITATION, and it is not one step in the worst case.
        //
        // A row still prefilling is not `generating`, so it blocks speculation
        // for the WHOLE batch until its prompt is consumed. A 4096-token
        // prompt at 1024-token chunks suppresses speculation for four steps
        // across every other sequence, and continuous arrivals can suppress it
        // for much longer. Single-stream decoding -- what the identity and
        // throughput checks measure -- is unaffected, because there is only
        // ever one row.
        //
        // Lifting this means verifying only the rows that carry drafts while
        // the others take their normal path in the same forward, which needs
        // per-row position selection in input_processor rather than the
        // batch-wide flag it uses now, and a rejection kernel that tolerates a
        // ragged [row, positions] layout. That is real work and it belongs
        // after the base case is proven, not before it.
        //
        // The Setup-time snapshot, not the live field: this runs on the
        // executor thread while the main thread may already be scheduling the
        // next batch, so reading the sequence directly would race with Update's
        // publish. The snapshot is what this batch was actually built from, and
        // is what Forward's gate uses too, so the two cannot disagree.
        const int n = d.num_drafts[0];
        if (n <= 0) {
            return false;
        }
        for (size_t i = 0; i < d.rows.size(); ++i) {
            if (!d.rows[i] || !d.rows[i]->generating || d.num_drafts[i] != n) {
                return false;
            }
        }
        return true;
    }

    /// Is this a real decode step? `generating` alone is not enough: the last
    /// prefill chunk is also generating, and drafting from it would submit one
    /// row per sequence against attention statistics prepared for a longer
    /// prefill shape.
    bool CanDraft(int phase) const
    {
        const auto& d = data_.at(phase);
        if (d.rows.empty() || !mtp_predictor_) {
            return false;
        }
        for (size_t i = 0; i < d.rows.size(); ++i) {
            if (!d.rows[i] || !d.rows[i]->generating || !d.autoregres[i]) {
                return false;
            }
        }
        return true;
    }

    /// Propose K drafts from the current tip and store them on each sequence.
    void DraftTokens(int phase, TensorMap& env);
    /// Compare the verification logits against the drafts; produce the accepted
    /// count and bonus token per row.
    void RejectDrafts(int phase, TensorMap& env);
    /// Commit the accepted prefix: write the tokens and set the new seq_len.
    void Rollback(int phase, TensorMap& env);
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
    // num_draft_tokens > 0 is part of the condition, not an afterthought.
    //
    // The predictor is built from the weights, which are present in the
    // checkpoint whether or not speculation is enabled. Without this clause a
    // K=0 run still constructed it, still ran its attention Setup/Prepare, and
    // still paid for a draft layer whose output nothing consumes -- and an
    // error in that path took down plain decoding, which is exactly what
    // happened: K=0 aborted on a missing `finished` key.
    //
    // Speculation off must mean the speculative code does not run at all.
    if (engine.num_draft_tokens > 0 && weights_.mtp && weights_.mtp->decoder_layer
        && unified_decoder_->mtp_attn_index() >= 0 && unified_decoder_->attn_layer()) {
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
    d.rows.resize(rc.size());
    d.num_drafts.resize(rc.size());

    // Reset every step. A stale accepted-count would make the next draft read
    // the wrong row of the hidden block, and on a non-verification step the
    // block is one row per sequence so the offset must be 0.
    last_accepted_.assign(rc.size(), 0);

    for (int i = 0; i < rc.size(); ++i) {
        auto& c         = *rc[i];
        d.rows[i]       = &c;
        d.num_drafts[i] = c.num_drafts;
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

    // Register the draft layer's KV slot.
    //
    // This must happen here rather than at draft time. The MTP attention layer
    // needs `requests` to set up its block table, and only the engine's Setup
    // env carries that; the executor's forward-time env holds just `batch` and
    // `copy`. Calling it from DraftTokens aborted on the missing key.
    //
    // The target's UnifiedDecoder walks only its own layers, so nothing else
    // populates this slot. Without it the draft attends to uninitialised cache
    // entries, which is what made the earlier acceptance measurement worthless.
    //
    // Only the kSetup half belongs here; see PrepareAttention for the rest.
    if (mtp_predictor_) {
        mtp_predictor_->SetupAttention(phase, env);
    }
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

    // The draft layer's attention borrows this step's offsets, which only
    // exist once the tensors above have been produced.
    if (mtp_predictor_) {
        mtp_predictor_->PrepareAttention(phase, env);
    }
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

    // On a verification step the sampler must not run.
    //
    // Generation::Forward samples one token per row and advances seq_len by
    // one. A verification forward has already submitted [bonus, D0..D_{K-1}]
    // and the tokens it will keep are decided by kReject and kRollback, from
    // the logits, not by sampling. Letting the sampler run here would append a
    // token on top of the accepted prefix and corrupt the sequence.
    // The SAME predicate the executor used to choose this path. Deriving it
    // twice from the same snapshot is what keeps the two from disagreeing --
    // and a disagreement here is the `verify_logits_` abort.
    const bool spec_verify = HasDraftsToVerify(phase);
    if (spec_verify) {
        // Keep the logits for kReject; it runs after Unprep, by which point the
        // env map has been rebuilt.
        verify_logits_ = env.at("logits");
        return;
    }

    // The executor decided this was a verification step but Forward did not.
    // Both gates now read the same Setup snapshot, so this cannot happen for
    // the reason it used to -- a race on the live field. Assert it here, where
    // the disagreeing state is still available, rather than in kReject, which
    // sees only a missing tensor.
    TM_CHECK(!HasDraftsToVerify(phase))
        << "executor scheduled a verification step that Forward declined; "
        << "the drafts snapshot and the autoregres snapshot disagree";

    if (d.n_generating) {
        generation_->Run(BatchOp::kForward, phase, env);
        Copy(env.at("output_ids").buffer(), autoreg_ids_);

        // Drafting happens at kDraft, which the executor runs immediately
        // after this forward in the same env.
        //
        // It used to happen here, in a path that drafted, scored the drafts
        // against the next step's sampled token, and discarded them. That path
        // never set `num_drafts`. Once real verification existed the two
        // competed silently: this one ran, threw its work away, and left
        // num_drafts at 0, so the engine never widened the reservation, never
        // took the verify path, and aborted in kReject with no logits stored.
    }
}

void LanguageModel::Impl::DraftTokens(int phase, TensorMap& env)
{
    TM_CHECK_NOTNULL(mtp_predictor_.get());

    auto&     d   = data_.at(phase);
    const int bsz = (int)d.rows.size();
    const int K   = engine_param_.num_draft_tokens;
    if (bsz == 0 || K <= 0) {
        return;
    }

    // The MTP KV slot was already seeded in Setup, which is the only place the
    // `requests` tensor it needs is available.

    // Draft from the accepted tip.
    //
    // On a verification step hidden_states is [bsz*(K+1), hidden], not
    // [bsz, hidden], because Setup selected every submitted position so each
    // draft could be scored against its own logit. The draft layer wants ONE
    // row per sequence: the state that produced the accepted tip, which for a
    // row that accepted n tokens is row i*(K+1) + n of that block.
    //
    // Handing over the whole tensor with batch=bsz makes the two disagree and
    // aborts in the first RMSNorm on `out.shape() == x.shape()`.
    auto hidden = env.at("hidden_states");

    if (hidden.shape(0) != bsz) {
        const int rows_per_seq = (int)hidden.shape(0) / bsz;
        TM_CHECK_EQ(rows_per_seq * bsz, (int)hidden.shape(0))
            << "hidden_states rows are not a whole multiple of the batch";

        Tensor gathered = empty_like(hidden.slice(0, bsz));
        for (int i = 0; i < bsz; ++i) {
            // last_accepted_[i] is the offset of the accepted tip inside this
            // row's block, recorded by Rollback.
            const int off = i * rows_per_seq + last_accepted_[i];
            TM_CHECK_LT(off, (int)hidden.shape(0));
            Copy(hidden.slice(off, 1), gathered.slice(i, 1));
        }
        hidden = std::move(gathered);
    }

    auto ids = autoreg_ids_.slice(0, bsz);

    // Each row's key length after this step's accepted tokens, which is what
    // the draft needs to know how much slack its last KV block still has.
    std::vector<int> seq_lens(bsz);
    std::vector<int> block_counts(bsz);
    for (int i = 0; i < bsz; ++i) {
        seq_lens[i] = d.rows[i]->seq_len;
        // How many KV blocks this row owns. The draft may walk to the end of
        // the last one and no further: the block iterator does not bounds-check
        // block_ptrs_, so overrunning reads the next sequence's pointer.
        block_counts[i] = (int)d.rows[i]->block_ids.size();
    }

    // Never draft past the end of the request's token buffer.
    //
    // token_ids is the request's output_ids allocation, session_len + 1 long.
    // Schedule injects drafts at token_ids[seq_len + k] and Rollback writes
    // token_ids[base + k], so K drafts need K slots beyond the current tip.
    //
    // Ordinary decode is protected by stop_criteria, which finishes a row at
    // max_seq_len. A verification step skips generation entirely, so that guard
    // does not run and nothing else bounds the write -- this would be a heap
    // overflow a few tokens before the session limit, not a clean stop.
    int budget = K;
    for (int i = 0; i < bsz; ++i) {
        const auto& c = *d.rows[i];
        budget        = std::min(budget, c.max_seq_len - c.seq_len - 1);
    }
    if (budget <= 0) {
        for (int i = 0; i < bsz; ++i) {
            d.rows[i]->pending_num_drafts = 0;
        }
        return;
    }

    auto drafts =
        mtp_predictor_->Draft(bsz, hidden, ids, budget, phase, seq_lens.data(), block_counts.data(), env);

    // Store them on the sequences so the next step's Setup can inject them.
    // Layout is [step][batch], so draft k for row i is at k*bsz + i.
    if (drafts.num_drafts > 0) {
        const int n = drafts.num_drafts;
        Buffer_<int> host{n * bsz, kCPU};
        core::Copy(drafts.draft_tokens, n * bsz, host);
        core::Context::stream().Sync();
        for (int i = 0; i < bsz; ++i) {
            auto& c              = *d.rows[i];
            c.pending_num_drafts = n;
            for (int k = 0; k < n; ++k) {
                c.pending_draft_tokens[k] = host[k * bsz + i];
            }
        }
    }
    else {
        // No capacity to draft this step. Zeroing matters: a stale count would
        // make the next Setup inject drafts that were never produced.
        for (int i = 0; i < bsz; ++i) {
            d.rows[i]->pending_num_drafts = 0;
        }
    }
}

void LanguageModel::Impl::RejectDrafts(int phase, TensorMap& env)
{
    TM_CHECK_NOTNULL(mtp_predictor_.get());

    auto&      d   = data_.at(phase);
    const int  bsz = (int)d.rows.size();
    const auto st  = core::Context::stream().handle();

    TM_CHECK(verify_logits_) << "kReject ran without verification logits";

    // The number of drafts THIS forward actually submitted, not the configured
    // maximum. Drafting is clamped near the session limit so a row can carry
    // fewer than num_draft_tokens, and the logits then have 1 + K rows for that
    // smaller K. Using the config value here would read rows that do not exist.
    //
    // Every row in a batch is drafted with the same budget, so this is uniform.
    int K = 0;
    for (int i = 0; i < bsz; ++i) {
        K = std::max(K, d.num_drafts[i]);
    }
    TM_CHECK_GT(K, 0) << "kReject ran with no drafts";

    // Drafts, laid out [bsz, K] to match GreedyReject.
    Buffer_<int> drafts{bsz * K, kDEVICE};
    {
        Buffer_<int> host{bsz * K, kCPU};
        for (int i = 0; i < bsz; ++i) {
            for (int k = 0; k < K; ++k) {
                host[i * K + k] = (k < d.num_drafts[i]) ? d.rows[i]->draft_tokens[k] : -1;
            }
        }
        Copy(host, drafts);
    }

    // The PADDED vocab size, because that is the row stride of the logits.
    //
    // PostEmbedding allocates [bsz, output_dim * tp_size], which is
    // vocab_size_padded, not weights_.vocab_size. Those differ whenever the
    // vocabulary is not a multiple of the tensor-parallel degree, and Qwen3.5
    // is such a model. Striding rows by the unpadded size would walk each
    // successive position a little further off the start of its row -- an
    // argmax over the wrong window, so drafts get rejected for no reason and
    // acceptance quietly collapses toward zero instead of failing.
    const int vocab_stride = weights_.output->output_dim * tp_size_;

    // verify_logits_ is [bsz*(K+1), vocab_stride], because Setup selected every
    // submitted position rather than only the last one.
    TM_CHECK_EQ((int)verify_logits_.shape(1), vocab_stride);
    TM_CHECK_EQ((int)verify_logits_.shape(0), bsz * (K + 1));

    auto result = GreedyReject(verify_logits_.raw_data(),
                               drafts.data(),
                               bsz,
                               K,
                               weights_.vocab_size,  // argmax searches only the real vocabulary
                               vocab_stride,         // rows are strided by the padded size
                               weights_.data_type,
                               st);

    env.produce("num_accepted", result.num_accepted);
    env.produce("bonus_tokens", result.bonus_tokens);

    // Generation::Rollback needs the drafts themselves to append the accepted
    // ones to its own copy of each sequence.
    env.produce("draft_tokens", drafts);

    // The bonus token is the tip the next draft conditions on.
    Copy(result.bonus_tokens, autoreg_ids_.slice(0, bsz));

    verify_logits_ = {};
}

void LanguageModel::Impl::Rollback(int phase, TensorMap& env)
{
    auto&     d   = data_.at(phase);
    const int bsz = (int)d.rows.size();

    Buffer_<int> accepted{bsz, kCPU};
    Buffer_<int> bonus{bsz, kCPU};
    Copy(env.at("num_accepted").buffer().slice(0, bsz), accepted);
    Copy(env.at("bonus_tokens").buffer().slice(0, bsz), bonus);
    core::Context::stream().Sync();

    // Do NOT write token_ids or seq_len here.
    //
    // Engine::Update already owns both. For every generating row it runs
    //
    //   c.token_ids[c.seq_len] = output_ids[j];
    //   c.seq_len              = sequence_length[j];
    //
    // so writing the accepted tokens here as well would append them twice and
    // advance seq_len past its own writes. The fix is not to duplicate that
    // loop but to feed it: publish the accepted prefix through `output_ids`
    // and the new length through `sequence_length`, which are the two buffers
    // Update reads.
    //
    // Update writes exactly one token per row, so only the bonus token travels
    // through output_ids. The accepted drafts are placed into token_ids here at
    // their own offsets and covered by the advanced sequence_length. Update's
    // `new_tokens` calculation then picks up the whole run, because it appends
    // token_ids[seq_len - new_tokens .. seq_len) rather than a single token.
    bool         clamped = false;
    Buffer_<int> seq_lens{bsz, kCPU};
    Buffer_<int> out_ids{bsz, kCPU};
    // sequence_length_.front(), not d.sequence_length. Rollback runs before
    // Unprep, and Unprep is what copies the live buffer into d.sequence_length,
    // so reading d here would give the PREVIOUS step's lengths.
    Copy(sequence_length_.front().buffer().slice(0, bsz), seq_lens);
    Copy(autoreg_ids_.slice(0, bsz), out_ids);
    core::Context::stream().Sync();

    for (int i = 0; i < bsz; ++i) {
        auto& c = *d.rows[i];

        if (!c.generating || d.num_drafts[i] == 0) {
            continue;  // not a verification row; its published length stands
        }

        int n = accepted[i];
        TM_CHECK_GE(n, 0);
        TM_CHECK_LE(n, d.num_drafts[i]);

        // `seq_len` still points at the tip from before this forward, because
        // Update has not run yet for this step.
        const int base = c.seq_len;

        // Never commit past max_seq_len.
        //
        // A verification step commits 1 + n tokens at once, so it can jump
        // over the limit that ordinary decode lands on exactly. The length
        // criterion uses `>=`, so the row still terminates -- but it terminates
        // having emitted more tokens than max_new_tokens allows, and the
        // baseline emits exactly that many. So this is both an API violation
        // and an identity failure, and the identity check would report it as a
        // token-id divergence with no hint that length was the cause.
        //
        // Truncating the accepted run is safe: the dropped tokens are ones the
        // baseline never emitted either, because it stopped first.
        if (const int room = c.max_seq_len - base - 1; n > room) {
            n           = std::max(room, 0);
            accepted[i] = n;
            clamped     = true;
        }

        // The accepted drafts come FIRST, the bonus token LAST.
        //
        // This is forced by what each logit predicts. The forward submitted
        // [last_committed, D0..D_{K-1}] and the logit at submitted position p
        // predicts the token at the NEXT index, so logits[0] is compared
        // against D0, logits[k] against D_k, and logits[K] has no draft to
        // compare against -- it is the bonus. With n drafts accepted the
        // sequence continues D0..D_{n-1} then bonus:
        //
        //   token_ids[base + k] = D_k     for k < n
        //   token_ids[base + n] = bonus
        //
        // I had this backwards, writing the bonus at base and the drafts after
        // it. That yields a valid-looking sequence of exactly the right length
        // with the tokens transposed, so it corrupts output without ever
        // failing a check.
        //
        // Engine::Update writes one token at token_ids[base] from output_ids,
        // so output_ids must carry D0 when anything was accepted, and the
        // bonus only when nothing was. The drafts past that first slot are
        // written here.
        for (int k = 1; k < n; ++k) {
            c.token_ids[base + k] = c.draft_tokens[k];
        }
        c.token_ids[base + n] = bonus[i];

        // What Update puts at token_ids[base].
        out_ids[i] = n > 0 ? c.draft_tokens[0] : bonus[i];

        // The KV entries for the rejected tail stay allocated but are logically
        // dead: the next forward starts at this length and overwrites them.
        seq_lens[i] = base + 1 + n;

        // Where this row's accepted tip sits inside its K+1 hidden block, for
        // the draft that runs next on this same env.
        last_accepted_[i] = n;

        mtp_accepted_ += n;
        mtp_steps_ += 1;
    }

    // Order matters here, and getting it backwards fails silently.
    //
    // Generation::Rollback appends its accepted tokens at base_len + slot, so
    // it needs the length from BEFORE this step's acceptance. Publishing the
    // advanced lengths first would make it write each token one full accepted
    // run too far into its copy of the sequence.
    //
    // Publish the clamped acceptance before Generation reads it.
    //
    // Generation::Rollback appends one token per accepted slot into its own
    // copy of each sequence, driven by the `num_accepted` buffer. If the clamp
    // above only touched the host copy, Generation would append the untruncated
    // run and its sequence would drift longer than the engine's -- wrong
    // repetition penalties and stop criteria, with nothing to assert on.
    if (clamped) {
        Copy(accepted, env.at("num_accepted").buffer().slice(0, bsz));
    }

    // d.sequence_length still holds the base lengths at this point, so let
    // Generation read them and advance its own copy first.
    //
    // NOT env.produce: Prepare already published `sequence_length` into this
    // same env, and produce refuses to overwrite -- it asserts on
    // emplace(...).second. Generation::Rollback reads the key that is already
    // there, so the buffer contents are what must be right, not the mapping.
    generation_->Run(BatchOp::kRollback, phase, env);

    // Only now overwrite with the advanced lengths, for Engine::Update.
    // Generation has already consumed the base values, and the sync inside its
    // Rollback means this write cannot race the read.
    Copy(seq_lens, sequence_length_.front().buffer().slice(0, bsz));

    // The token Engine::Update writes at token_ids[base] for each row. Unprep
    // copies this buffer out, so writing it is enough; producing the key again
    // would abort.
    Copy(out_ids, autoreg_ids_.slice(0, bsz));

    // Accept length is the quantity that decides whether this is worth doing:
    // 1 + mean accepted drafts is the tokens produced per forward.
    if (mtp_steps_ >= 64 && mtp_steps_ % 64 == 0) {
        TM_LOG_INFO("[MTP] accept length {:.2f} tokens/step over {} steps ({} drafts accepted)",
                    1.0 + (double)mtp_accepted_ / (double)mtp_steps_,
                    mtp_steps_,
                    mtp_accepted_);
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

bool LanguageModel::HasDraftsToVerify(int phase) const
{
    return impl_->HasDraftsToVerify(phase);
}

bool LanguageModel::CanDraft(int phase) const
{
    return impl_->CanDraft(phase);
}

void LanguageModel::Run(BatchOp op, int phase, TensorMap& env)
{
    return TM_CHECK_NOTNULL(impl_)->Run(op, phase, env);
}

}  // namespace turbomind
