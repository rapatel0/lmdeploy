
#include "src/turbomind/models/language_model.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
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
#include "src/turbomind/kernels/argmax.h"
#include "src/turbomind/kernels/gpt_kernels.h"
#include "src/turbomind/models/input_processor.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/dflash_kernels.h"
#include "src/turbomind/models/llama/dflash_predictor.h"
#include "src/turbomind/models/llama/mtp_predictor.h"
#include "src/turbomind/models/llama/rejection_sampling.h"
#include "src/turbomind/models/llama/unified_decoder.h"
#include "src/turbomind/models/dflash_weight.h"
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

    // Device-to-host FETCH buffers, shared across phases deliberately: Fetch's
    // copy is consumed by Engine::Update after the batch's done-event, before
    // the other phase's Fetch can be enqueued. The opposite direction from the
    // per-phase host STAGING above, and a different lifecycle.
    Buffer_<int>  sequence_length_buf_;
    Buffer_<bool> finished_buf_;

    struct Data {
        Buffer_<int>  sequence_length;
        Buffer_<int>  readonly_block_num;
        Buffer_<bool> finished;

        // Host staging for the two uploads above; per phase, see constructor.
        Buffer_<int> sequence_length_host;
        Buffer_<int> readonly_block_num_host;

        // Trace-only pinned staging for the production selector candidates.
        Buffer_<int> spec_draft_candidates_host;

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

        /// Per-row submitted token count this step, host-side. The MTP
        /// prefill fill needs the chunk boundaries inside the flat input_ids
        /// tensor, and `requests` is gone from the map by Forward.
        std::vector<int> input_lens;

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

        /// Per row, how many drafts the last verification accepted. Selects
        /// which row of a [bsz*(K+1), hidden] block carries the accepted tip.
        ///
        /// Per-phase for the same reason as skip_draft: Setup runs on the main
        /// thread and would otherwise zero this vector while the executor sits
        /// between Rollback and DraftTokens, making the draft gather the state
        /// from before any draft instead of the accepted tip.
        std::vector<int> last_accepted;

        /// Per row, suppress drafting at the end of THIS step. Set by Rollback
        /// when a verification was rejected, so the row takes an ordinary
        /// decode next step instead of retrying the prediction that failed.
        ///
        /// Per-phase, not a shared member: with async execution the main thread
        /// runs Setup for the next batch while the executor is still between
        /// kRollback and kDraft for this one, so a shared vector would be reset
        /// mid-step -- losing the suppression and racing at the same time.
        std::vector<char> skip_draft;
    };

    vector<Data> data_;

    std::optional<InputProcessor>   input_processor_;
    std::unique_ptr<UnifiedDecoder> unified_decoder_;

    /// Draft-token generators. The selected algorithm owns exactly one path.
    std::unique_ptr<MTPPredictor>    mtp_predictor_;
    std::unique_ptr<DFlashPredictor> dflash_predictor_;

    /// Verification logits, held between Forward and kReject because the env
    /// map is rebuilt in between.
    Tensor verify_logits_;

    /// Diagnostic reference: verifier logits at the first submitted position,
    /// keyed by request uid until the forced-reject replay runs. Populated only
    /// under TM_SPEC_LOGIT_PARITY=1.
    std::unordered_map<uint64_t, std::vector<uint16_t>> replay_ref_logits_;

    /// Accept-length accounting. 1 + accepted/steps is the tokens emitted per
    /// forward, which is the number that decides whether speculation pays.
    size_t mtp_accepted_ = 0;
    /// Tokens actually committed by verification forwards. Distinct from
    /// mtp_accepted_, which counts only drafts: a committed step contributes
    /// 1 + n, a rejected one contributes nothing.
    size_t mtp_committed_ = 0;
    /// Verification steps that accepted every draft. Counted directly rather
    /// than derived as mtp_accepted_/K: that division is only valid while
    /// acceptance is all-or-nothing, which holds for a model with recurrent
    /// state and would silently produce nonsense on one without it.
    size_t mtp_full_accepts_          = 0;
    size_t mtp_steps_                 = 0;
    size_t spec_raw_accepted_         = 0;
    size_t spec_raw_committed_        = 0;
    size_t spec_ambiguous_steps_      = 0;
    size_t spec_ambiguity_discarded_  = 0;

    /// Bounded draft-vs-target alignment dumps in RejectDrafts.
    int reject_dumps_ = 0;

    /// Per row, did this verification commit an EOS token? A verification step
    /// skips Generation, so stop_criteria never runs and Rollback must set
    /// `finished` itself.
    std::vector<char> eos_hit_;

    /// Per row, did this verification commit nothing at all? Set when a draft
    /// was rejected on a model with recurrent state: the snapshot is restored
    /// and the tip does not move, so state and tip stay aligned.
    std::vector<int> no_commit_;

    /// Per row, an accepted draft was EOS. Commit the accepted drafts through
    /// EOS but not the verifier's bonus token after it.
    std::vector<int> no_bonus_;

    /// Per row, must this sequence's recurrent state be rewound? This is the
    /// bounded-memory fallback when the active batch cannot retain every slot.
    std::vector<char> gdn_restore_;

    /// Per row, the verification-input frontier to publish after acceptance.
    /// Negative values keep the final live state unchanged.
    std::vector<int> gdn_state_slots_;

    /// Does this model carry recurrent state that speculative verification
    /// must preserve?
    bool gdn_rollback_{false};

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

    ~Impl()
    {
        if (mtp_steps_ != 0) {
            TM_LOG_INFO("[spec] final commit length {:.3f}, raw {:.3f} over {} verification steps "
                        "({} committed, {} raw committed, {} accepted drafts, {} raw accepted drafts, "
                        "{} ambiguous steps, {} tokens discarded by ambiguity, {} full accepts)",
                        (double)mtp_committed_ / (double)mtp_steps_,
                        (double)spec_raw_committed_ / (double)mtp_steps_,
                        mtp_steps_,
                        mtp_committed_,
                        spec_raw_committed_,
                        mtp_accepted_,
                        spec_raw_accepted_,
                        spec_ambiguous_steps_,
                        spec_ambiguity_discarded_,
                        mtp_full_accepts_);
        }
    }

    Tensor LookupEmbedding(const Buffer_<int>& input_ids, Buffer symm_buf);
    Tensor PostEmbedding(const Tensor& features, Buffer symm_buf);
    void   DraftTop1(Buffer_<int>& out, const Tensor& features);
    void   DraftTopK16(Buffer_<int>& out,
                       Tensor&       scores,
                       const Tensor& features,
                       float         output_multiplier,
                       float         softcap);

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
        if (d.rows.empty() || (!mtp_predictor_ && !dflash_predictor_)) {
            return false;
        }
        for (size_t i = 0; i < d.rows.size(); ++i) {
            const auto* c = d.rows[i];
            if (!c || !c->generating) {
                return false;
            }
            // A one-token decode, or a verification submitting exactly
            // 1 + num_drafts. Anything wider is a genuine prefill chunk, and
            // drafting from it would condition on a sequence whose prompt is
            // still being consumed.
            //
            // Gating on `autoregres` alone is wrong in the other direction: it
            // is false on EVERY verification step, so drafting would stop after
            // the first cycle and speculation would decay to ordinary decoding
            // with no error at all.
            const int expected = 1 + d.num_drafts[i];
            if (!d.autoregres[i] && c->input_len != expected) {
                return false;
            }
        }
        return true;
    }

    /// Should this step fill the draft layer's KV slot for prompt positions?
    ///
    /// True only for a PURE prefill batch: every row non-autoregressive with
    /// no drafts in flight. The fill runs one attention pass over the whole
    /// flat token tensor against the target's phase plan, so it cannot skip
    /// rows -- and in a mixed batch a decode row would get its tip entry
    /// overwritten with a degraded (zero-hidden) one, corrupting KV the
    /// draft already wrote with the correct convention. Skipping mixed
    /// batches costs draft quality for sequences that prefilled alongside
    /// decoding rows, never correctness: verification rejects what the
    /// blind draft proposes.
    bool NeedsPrefillFill(int phase) const
    {
        const auto& d = data_.at(phase);
        if (!mtp_predictor_ || d.rows.empty()) {
            return false;
        }
        for (size_t i = 0; i < d.rows.size(); ++i) {
            if (d.autoregres[i] || d.num_drafts[i] != 0) {
                return false;
            }
        }
        return true;
    }

    /// Propose K drafts from the current tip and store them on each sequence.
    void DraftTokens(int phase, TensorMap& env);
    void DraftDFlashTokens(int phase, TensorMap& env);
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

    sequence_length_buf_ = {engine.max_batch_size, kCPUpinned};
    finished_buf_        = {engine.max_batch_size, kCPUpinned};
    finished_     = {{engine.max_batch_size}, kBool, kDEVICE};

    autoreg_ids_ = {engine.max_batch_size, kDEVICE};
    // autoreg_ids_offsets_ = {engine.max_batch_size + 1, kCPU};
    // std::fill_n(autoreg_ids_offsets_.data(), autoreg_ids_offsets_.size(), 0);

    sequence_length_ = {{engine.max_batch_size}, kInt, kDEVICE};
    // Host staging for the per-phase device buffers is itself per phase: the
    // upload reads the pinned source at stream-execution time, so a shared
    // buffer lets a later phase's Setup refill it before an earlier phase's
    // queued copy has run. Proven in UnifiedAttentionLayer, then observed
    // again in GatedDeltaNetLayer when that fix moved the fault here-adjacent.
    // (finished_buf_ stays shared: it is a device-to-host fetch consumed
    // before the next Setup, the opposite direction and lifecycle.)
    for (int i = 0; i < phases; ++i) {
        auto& d                   = data_.emplace_back();
        d.sequence_length_host       = {engine.max_batch_size, kCPUpinned};
        d.readonly_block_num_host    = {engine.max_batch_size, kCPUpinned};
        d.spec_draft_candidates_host = {engine.max_batch_size * Sequence::kMaxDraftTokens, kCPUpinned};
        d.sequence_length    = empty_like(d.sequence_length_host, kDEVICE);
        d.readonly_block_num = empty_like(d.readonly_block_num_host, kDEVICE);
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
    if (engine.speculative_algorithm == "dflash2") {
        TM_CHECK(weights_.dflash) << "DFlash2 selected but separate draft weights are absent";
        dflash_predictor_ = std::make_unique<DFlashPredictor>(*weights_.dflash,
                                                              *TM_CHECK_NOTNULL(unified_decoder_->attn_layer()),
                                                              unified_decoder_->dflash_attn_indices(),
                                                              unified_decoder_->dflash_phase(0),
                                                              engine,
                                                              phases,
                                                              ctx,
                                                              [this](const Buffer_<int>& ids) {
                                                                  return LookupEmbedding(ids, symm_buf_);
                                                              },
                                                              [this](const Tensor& hidden) {
                                                                  return PostEmbedding(hidden, symm_buf_);
                                                              },
                                                              [this](Buffer_<int>& ids,
                                                                     Tensor&       scores,
                                                                     const Tensor& hidden,
                                                                     float         multiplier,
                                                                     float         softcap) {
                                                                  DraftTopK16(ids, scores, hidden, multiplier, softcap);
                                                              });
        if (engine.num_draft_tokens > 0) {
            TM_CHECK_EQ(engine.num_draft_tokens, weights_.dflash->block_size - 1);
            gdn_rollback_ = unified_decoder_->has_recurrent_state();
        }
    }
    else if (engine.num_draft_tokens > 0 && engine.speculative_algorithm == "mtp" && weights_.mtp
             && weights_.mtp->decoder_layer && unified_decoder_->mtp_attn_index() >= 0
             && unified_decoder_->attn_layer()) {
        gdn_rollback_  = unified_decoder_->has_recurrent_state();
        mtp_predictor_ = std::make_unique<MTPPredictor>(
            *weights_.mtp,
            *unified_decoder_->attn_layer(),
            unified_decoder_->mtp_attn_index(),
            unified_decoder_->mtp_phase(0),  // base; the per-phase offset is added per call
            engine,
            ctx,
            [this](const Buffer_<int>& ids) { return LookupEmbedding(ids, symm_buf_); },
            [this](Buffer_<int>& out, const Tensor& h) { DraftTop1(out, h); });
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

    // TM_SPEC_VALIDATE_IDS=1: read back the ids this lookup is ABOUT to
    // dereference and bounds-check them on the host. The launch-blocking abort
    // blames embeddingLookupKernel, but its cudaGetLastError also inherits any
    // sticky error from the un-checked batched memcpy that ran just before the
    // forward -- launch blocking serializes kernels, not memcpys. This check
    // separates the two: ids valid here + kernel still faults means the ids
    // were never the problem and the earlier copy (or the table pointer) is.
    static const bool validate_ids = [] {
        const auto v = std::getenv("TM_SPEC_VALIDATE_IDS");
        return v && v[0] == '1';
    }();
    if (TM_UNLIKELY(validate_ids) && token_num > 0) {
        Buffer_<int> h_ids{token_num, kCPU};
        core::Copy(input_ids, token_num, h_ids);
        core::Context::stream().Sync();
        const auto limit = embedding_table.shape(0);
        for (int i = 0; i < token_num; ++i) {
            TM_CHECK(0 <= h_ids[i] && h_ids[i] < limit)
                << "[spec] input_ids[" << i << "] = " << h_ids[i] << " outside the embedding table of " << limit
                << " rows (token_num " << token_num << ")";
        }
    }

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

void LanguageModel::Impl::DraftTop1(Buffer_<int>& out, const Tensor& features)
{
    static const bool local_top1 = [] {
        const char* value = std::getenv("TM_MTP_LOCAL_TOP1");
        return value && value[0] == '1';
    }();
    const auto st = core::Context::stream().handle();
    if (!local_top1 || tp_size_ == 1) {
        auto logits = PostEmbedding(features, symm_buf_);
        invokeArgmax(out, logits, st);
        return;
    }

    const int rows             = features.shape(0);
    const int local_vocab_size = weights_.output->output_dim;
    const int token_id_offset  = tp_rank_ * local_vocab_size;
    const int valid_vocab      = std::min(local_vocab_size, weights_.vocab_size - token_id_offset);
    TM_CHECK_GT(valid_vocab, 0);

    // Match SGLang's V100 greedy TP-top1 route: each rank computes its local
    // LM-head shard, then exchanges one [score, global token id] pair per row.
    // This replaces a full-vocabulary all-gather and transpose on every serial
    // draft step with a tiny homogeneous FP32 all-gather. FP32 represents all
    // supported token ids exactly.
    Tensor local_logits{{rows, local_vocab_size}, weights_.data_type, kDEVICE};
    TM_SCOPE_CALL(linear_.Forward(features, *weights_.output, local_logits));

    Buffer_<float> local_candidates{(ssize_t)rows * 2, kDEVICE};
    Buffer_<float> gathered_candidates{(ssize_t)tp_size_ * rows * 2, kDEVICE};
    invokeLocalArgmax(local_candidates, local_logits, valid_vocab, token_id_offset, st);
    comm_.d_comm->AllGather(local_candidates.data(),
                            gathered_candidates.data(),
                            local_candidates.size(),
                            kFloat32,
                            comm_.d_tp_group,
                            st);
    TM_CUDA_CHECK(cudaGetLastError());
    invokeGlobalArgmax(out, gathered_candidates, rows, tp_size_, st);
}

void LanguageModel::Impl::DraftTopK16(Buffer_<int>& out,
                                      Tensor&       scores,
                                      const Tensor& features,
                                      float         output_multiplier,
                                      float         softcap)
{
    const auto st = core::Context::stream().handle();
    const int rows = features.shape(0);
    const int local_vocab_size = weights_.output->output_dim;
    const int token_id_offset = tp_rank_ * local_vocab_size;
    const int valid_vocab = std::min(local_vocab_size, weights_.vocab_size - token_id_offset);
    TM_CHECK_GT(valid_vocab, 0);

    Tensor local_logits{{rows, local_vocab_size}, weights_.data_type, kDEVICE};
    TM_SCOPE_CALL(linear_.Forward(features, *weights_.output, local_logits));
    if (tp_size_ == 1) {
        invokeDFlashTopK16(out,
                           scores,
                           local_logits,
                           valid_vocab,
                           token_id_offset,
                           output_multiplier,
                           softcap,
                           st);
        return;
    }

    Buffer_<int> local_ids{(ssize_t)rows * 16, kDEVICE};
    Tensor local_scores{{rows, 16}, kFloat32, kDEVICE};
    invokeDFlashTopK16(local_ids, local_scores, local_logits, valid_vocab, token_id_offset, 1.f, 0.f, st);

    Buffer_<int> gathered_ids{(ssize_t)tp_size_ * rows * 16, kDEVICE};
    Tensor gathered_scores{{tp_size_, rows, 16}, kFloat32, kDEVICE};
    comm_.d_comm->AllGather(
        local_ids.data(), gathered_ids.data(), local_ids.size(), kInt32, comm_.d_tp_group, st);
    comm_.d_comm->AllGather(local_scores.raw_data(),
                            gathered_scores.raw_data(),
                            local_scores.size(),
                            kFloat32,
                            comm_.d_tp_group,
                            st);
    TM_CUDA_CHECK(cudaGetLastError());
    invokeDFlashMergeTopK16(
        out, scores, gathered_ids, gathered_scores, tp_size_, output_multiplier, softcap, st);
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
    d.input_lens.resize(rc.size());
    d.rows.resize(rc.size());
    d.num_drafts.resize(rc.size());

    // Reset every step. A stale accepted-count would make the next draft read
    // the wrong row of the hidden block, and on a non-verification step the
    // block is one row per sequence so the offset must be 0.
    d.last_accepted.assign(rc.size(), 0);

    // Sized here, not in Rollback, because DraftTokens reads it on EVERY step
    // while Rollback runs only on verification steps. Left to Rollback it would
    // carry the previous step's length: too short is an out-of-range read, too
    // long suppresses drafting for rows that never rejected anything.
    d.skip_draft.assign(rc.size(), 0);

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
        // d.sequence_length_host below is written only for non-autoregressive
        // rows -- decode rows carry their length forward on the device -- so it
        // cannot be read here as a host-side length. The MTP draft needs these
        // to know how much slack each row has left in its last KV block.
        d.seq_lens[i]   = c.history_len + c.inflight_input_len + c.input_len;
        d.input_lens[i] = c.input_len;
        if (TM_UNLIKELY(!c.autoregres)) {
            d.sequence_length_host[i] = c.history_len + c.inflight_input_len + c.input_len;
        }
        d.readonly_block_num_host[i] = c.readonly_block_num;  // all rows, batch order
    }

    copy(d.sequence_length_host, rc.size(), d.sequence_length);
    copy(d.readonly_block_num_host, rc.size(), d.readonly_block_num);

    unified_decoder_->Run(BatchOp::kSetup, phase, env);
    generation_->Run(BatchOp::kSetup, phase, env);
    output_processor_->Run(BatchOp::kSetup, phase, env);

    // Set up the draft's attention in ITS OWN phase slot.
    //
    // The draft shares the layer with the target but not the phase:
    // AttentionData is per-phase, and unified_decoder allocated one extra slot
    // for exactly this. Running Setup against the target's phase is what
    // corrupted the target's plan; running it against the draft's own does not.
    if (mtp_predictor_) {
        // An earlier version flushed `copy` here and claimed that made the
        // shared host staging safe. It did not: BatchCopy::Run only ENQUEUES a
        // stream-ordered batched memcpy (SRC_ACCESS_ORDER_STREAM), so the
        // host-side source read happens when the stream reaches it, not when
        // Run returns. The draft's Setup could still refill the staging before
        // the target's queued copy executed, and the target then attended
        // through the draft's block pointers. That was the illegal memory
        // access: absent at K=0, absent under CUDA_LAUNCH_BLOCKING=1 (the
        // drain closed the window), present otherwise.
        //
        // The staging is now per phase, on AttentionData, so the draft's fill
        // cannot alias the target's. The flush is kept only to bound the queue
        // depth; correctness no longer depends on it.
        copy.Run();

        mtp_predictor_->SetupAttention(phase, env);
    }
    if (dflash_predictor_ && unified_decoder_->dflash_phase(phase) >= 0) {
        copy.Run();
        dflash_predictor_->SetupAttention(phase, env);
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

    // Ask the decoder for the full hidden states when this step will fill
    // the draft's KV slot for prompt positions. The decoder produces
    // `full_hidden_states` only when this key exists; output_processor may
    // already have produced it for a user request, hence the guard.
    if (NeedsPrefillFill(phase) && !env.try_("output_hidden_states")) {
        env.produce("output_hidden_states", Tensor{});
    }

    // Build the draft's decode-shaped plan, in the draft's own phase.
    //
    // The draft submits one token per row; a verification forward submits K+1.
    // Both shapes must exist at once, because kDraft runs immediately after the
    // verification forward on the same step. Two phase slots is what makes that
    // possible -- sharing one meant whichever prepared last won, and the other
    // aborted on
    //
    //   Check failed: d.prefill.q_sum + d.decode.n == q_count (15 vs. 3)
    //
    // where 15 is the plan's expected token count and 3 the draft's actual one.
    if (mtp_predictor_) {
        mtp_predictor_->PrepareAttention(phase, env);
    }
    if (dflash_predictor_ && unified_decoder_->dflash_phase(phase) >= 0) {
        dflash_predictor_->PrepareAttention(phase, env);
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

        // TM_SPEC_TRACE=1: log what THIS forward actually consumes, per row.
        //
        // Two hypothesis-driven fixes in a row have not moved the identity
        // divergence at position 2-4. This replaces the next guess with a
        // measurement: the K=0 and K=4 arms each emit one line per forward,
        // and diffing the traces names the first step whose INPUT differs
        // -- which is the step whose predecessor leaked, whatever the
        // mechanism. Bounded to small batches; identity runs use 5 prompts.
        static const bool spec_trace = [] {
            const char* s = std::getenv("TM_SPEC_TRACE");
            return s && s[0] == '1';
        }();
        // Skip prompt/warm-up slabs: the trace is about cross-step decode
        // transitions, and dumping thousands of input ids hides those lines.
        if (TM_UNLIKELY(spec_trace) && d.rows.size() <= 8 && input_ids.size() <= 64) {
            Buffer_<int> h_ids{input_ids.size(), kCPU};
            core::Copy(input_ids, input_ids.size(), h_ids);
            core::Context::stream().Sync();
            int off = 0;
            for (size_t i = 0; i < d.rows.size(); ++i) {
                const int len = d.input_lens[i];
                std::string toks;
                for (int k = 0; k < len; ++k) {
                    toks += std::to_string(h_ids[off + k]) + (k + 1 < len ? "," : "");
                }
                TM_LOG_WARNING("[trace] fwd uid={} seq_len={} in_len={} drafts={} in=[{}]",
                               (long)d.uids[i],
                               d.seq_lens[i],
                               len,
                               d.num_drafts[i],
                               toks);
                off += len;
            }
        }

        Tensor input_embeds = LookupEmbedding(input_ids, symm_buf_);
        TM_DEBUG_TENSOR(input_embeds, "embeddings", 1);

        auto& copy = *env.at("copy").data<BatchCopy*>()[0];
        input_processor_->PatchEmbedding(phase, input_embeds, copy, env);
        copy.Run();

        env.produce("input_embeds", std::move(input_embeds));
        // dbg(env);
    }

    // Capture DFlash2's five target residual features when its runtime is
    // active. TM_DFLASH_CAPTURE permits a K=0 diagnostic before proposal
    // execution is enabled; ordinary target-only and MTP runs allocate none.
    static const bool force_dflash_capture = [] {
        const char* value = std::getenv("TM_DFLASH_CAPTURE");
        return value && value[0] == '1';
    }();
    if (weights_.dflash && engine_param_.speculative_algorithm == "dflash2"
        && (engine_param_.num_draft_tokens > 0 || force_dflash_capture)) {
        const auto token_num = env.at("input_embeds").shape(0);
        Tensor capture{{token_num,
                        (ssize_t)weights_.dflash->target_layer_ids.size() * weights_.hidden_units},
                       weights_.data_type,
                       kDEVICE};
        Clear(capture);
        env.produce("dflash_target_hidden", std::move(capture));
    }

    env.produce("output_norm_weight", weights_.norm->weight);

    // Save recurrent state before a verification forward, while it still
    // describes only committed tokens. Rollback restores it if any draft is
    // rejected; there is no way to recover it afterwards, because the forward
    // advances it in place.
    static const bool trace_gdn_state = [] {
        const char* s = std::getenv("TM_GDN_TRACE_STATE");
        return s && s[0] == '1';
    }();
    if (gdn_rollback_ && (HasDraftsToVerify(phase) || TM_UNLIKELY(trace_gdn_state))) {
        unified_decoder_->SnapshotGDNState(phase);
    }

    {
        NvtxScope scope(HasDraftsToVerify(phase) ? "targetVerify" : "targetDecode");
        unified_decoder_->Forward(phase, env, weights_.layers_list());
    }

    if (dflash_predictor_ && env.try_("dflash_target_hidden") && !unified_decoder_->is_warm_up()) {
        NvtxScope scope("dflashContextKV");
        if (d.rows.size() == 1 && engine_param_.num_draft_tokens == 7) {
            dflash_predictor_->ArmParityContext(d.uids[0]);
        }
        Tensor context = dflash_predictor_->ProjectContext(env.at("dflash_target_hidden"));
        dflash_predictor_->MaterializeContextKV(phase, context);
        env.produce("dflash_context_hidden", std::move(context));
    }

    if (force_dflash_capture) {
        static bool capture_logged = false;
        if (!capture_logged) {
            auto* capture = env.try_("dflash_target_hidden");
            TM_CHECK(capture);
            Tensor host{{1, capture->shape(1)}, capture->dtype(), kCPU};
            Copy(capture->slice(0, 1), host);
            core::Context::stream().Sync();
            const ssize_t feature_bytes = byte_size(capture->dtype(), weights_.hidden_units);
            const char*   bytes         = (const char*)host.raw_data();
            std::string   nonzero;
            std::string   layer_ids;
            bool          complete = true;
            for (size_t feature = 0; feature < weights_.dflash->target_layer_ids.size(); ++feature) {
                ssize_t count = 0;
                for (ssize_t i = 0; i < feature_bytes; ++i) {
                    count += bytes[feature * feature_bytes + i] != 0;
                }
                complete = complete && count > 0;
                nonzero += std::to_string(count);
                layer_ids += std::to_string(weights_.dflash->target_layer_ids[feature]);
                const char* separator = feature + 1 < weights_.dflash->target_layer_ids.size() ? "," : "";
                nonzero += separator;
                layer_ids += separator;
            }
            // Warm-up executes only tune_layer_num layers, so its intentionally
            // incomplete capture remains zero. Wait for the first full target
            // forward before publishing the one-shot diagnostic.
            if (complete) {
                TM_LOG_INFO("[DFlash2] target capture shape=[{},{}] layer_ids=[{}] nonzero_bytes=[{}]",
                            capture->shape(0),
                            capture->shape(1),
                            layer_ids,
                            nonzero);

                Tensor context = env.at("dflash_context_hidden");
                Tensor context_host{{1, context.shape(1)}, context.dtype(), kCPU};
                Copy(context.slice(0, 1), context_host);
                core::Context::stream().Sync();
                const ssize_t context_bytes = context_host.byte_size();
                const char*   context_data  = (const char*)context_host.raw_data();
                ssize_t       context_nonzero = 0;
                for (ssize_t i = 0; i < context_bytes; ++i) {
                    context_nonzero += context_data[i] != 0;
                }
                TM_LOG_INFO("[DFlash2] context projection shape=[{},{}] nonzero_bytes={}",
                            context.shape(0),
                            context.shape(1),
                            context_nonzero);

                auto* first_conv = TM_CHECK_NOTNULL(weights_.dflash->attention_conv(0));
                Tensor convolved = dflash_predictor_->ApplyGroupedConv(context, *first_conv, 0);
                Tensor convolved_host{{1, convolved.shape(1)}, convolved.dtype(), kCPU};
                Copy(convolved.slice(0, 1), convolved_host);
                core::Context::stream().Sync();
                const char* convolved_data = (const char*)convolved_host.raw_data();
                ssize_t     convolved_nonzero = 0;
                for (ssize_t i = 0; i < convolved_host.byte_size(); ++i) {
                    convolved_nonzero += convolved_data[i] != 0;
                }
                TM_LOG_INFO("[DFlash2] grouped convolution shape=[{},{}] nonzero_bytes={}",
                            convolved.shape(0),
                            convolved.shape(1),
                            convolved_nonzero);

                auto input_ids = env.at("input_ids").buffer().view<int>();
                TM_CHECK_EQ(d.rows.size(), 1) << "TM_DFLASH_CAPTURE block diagnostic requires batch size one";
                Buffer_<int> anchors{1, kDEVICE};
                core::Copy(input_ids.slice(input_ids.size() - 1, 1), 1, anchors);
                Tensor draft_hidden = dflash_predictor_->DraftBlock(anchors, phase, env);
                Tensor draft_host{{1, draft_hidden.shape(1)}, draft_hidden.dtype(), kCPU};
                Copy(draft_hidden.slice(0, 1), draft_host);
                core::Context::stream().Sync();
                const char* draft_data = (const char*)draft_host.raw_data();
                ssize_t     draft_nonzero = 0;
                for (ssize_t i = 0; i < draft_host.byte_size(); ++i) {
                    draft_nonzero += draft_data[i] != 0;
                }
                TM_LOG_INFO("[DFlash2] five-layer draft shape=[{},{}] nonzero_bytes={}",
                            draft_hidden.shape(0),
                            draft_hidden.shape(1),
                            draft_nonzero);

                Buffer_<int> candidates = dflash_predictor_->SelectCandidates(draft_hidden, anchors, phase);
                Buffer_<int> host_candidates{candidates.size(), kCPU};
                core::Copy(candidates, candidates.size(), host_candidates);
                core::Context::stream().Sync();
                std::string candidate_text;
                for (int i = 0; i < host_candidates.size(); ++i) {
                    candidate_text += std::to_string(host_candidates[i]);
                    candidate_text += i + 1 < host_candidates.size() ? "," : "";
                }
                TM_LOG_INFO("[DFlash2] greedy candidates=[{}]", candidate_text);
                capture_logged = true;
            }
        }
    }

    // Fill the draft layer's KV slot for this chunk's prompt positions.
    //
    // Without this the draft attends over uninitialized KV for every prompt
    // position: the MTP layer used to run only inside decode-time Draft(),
    // so nothing ever wrote its slot for the prompt. Junk at seq_len 1108,
    // plausible at 85 -- run 20260827_092923. The upstream head is trained
    // teacher-forced over the full sequence, and vLLM's proposer does this
    // same prefill pass; without it acceptance at long context is zero.
    //
    // BEFORE OutputHiddenStatesAndLogits, which consumes full_hidden_states
    // when a user requested hidden output. Reading env.at here fails loudly
    // if the decoder did not produce it, rather than silently skipping the
    // fill and reintroducing the blind-draft failure as a quality mystery.
    if (NeedsPrefillFill(phase)) {
        mtp_predictor_->PrefillFill(phase,
                                    env.at("full_hidden_states"),
                                    env.at("input_ids").buffer(),
                                    d.input_lens.data(),
                                    (int)d.rows.size());
    }

    // env.at("batch").data<BatchData*>()[0]->Notify();

    output_processor_->OutputHiddenStatesAndLogits(phase, env, 2);

    auto& hidden_states = env.at("hidden_states");

    // The SAME predicate the executor used to choose this path. Deriving it
    // from the phase snapshot keeps sampling, parity diagnostics, and reject
    // handling on one classification.
    const bool spec_verify = HasDraftsToVerify(phase);

    env.produce("logits", PostEmbedding(hidden_states, symm_buf_));

    // Compare the rejected verifier's first-position logits with the ordinary
    // replay from the exact restored state. This is a same-process numeric
    // contract, unlike K=0-vs-K=N token text from independent prefills (the K=0
    // control itself is nondeterministic on a near-tied argmax).
    static const bool trace_logit_parity = [] {
        const char* s = std::getenv("TM_SPEC_LOGIT_PARITY");
        return s && s[0] == '1';
    }();
    if (TM_UNLIKELY(trace_logit_parity && !spec_verify && !replay_ref_logits_.empty())) {
        const int   bsz          = (int)d.rows.size();
        const auto& logits       = env.at("logits");
        const int   vocab_stride = (int)logits.shape(1);
        TM_CHECK_EQ(logits.dtype(), kHalf);
        Buffer_<uint16_t> host{(ssize_t)bsz * vocab_stride, kCPU};
        Copy(Buffer_<uint16_t>{(uint16_t*)logits.raw_data(), host.size(), kDEVICE}, host);
        core::Context::stream().Sync();
        auto half_to_float = [](uint16_t u) {
            const int sign = (u >> 15) & 1;
            const int exp  = (u >> 10) & 0x1f;
            const int man  = u & 0x3ff;
            float val;
            if (exp == 0) val = std::ldexp((float)man, -24);
            else if (exp == 31) val = man ? NAN : INFINITY;
            else val = std::ldexp((float)(man + 1024), exp - 25);
            return sign ? -val : val;
        };
        for (int i = 0; i < bsz; ++i) {
            auto it = replay_ref_logits_.find(d.uids[i]);
            if (it == replay_ref_logits_.end()) continue;
            const auto& ref = it->second;
            TM_CHECK_EQ((int)ref.size(), vocab_stride);
            const uint16_t* got = host.data() + (ssize_t)i * vocab_stride;
            float max_abs = 0.f;
            double sum_sq = 0.;
            int ref_argmax = 0, got_argmax = 0, raw_diff = 0;
            float ref_max = -INFINITY, got_max = -INFINITY;
            for (int v = 0; v < weights_.vocab_size; ++v) {
                const float a = half_to_float(ref[v]);
                const float b = half_to_float(got[v]);
                const float e = std::abs(a - b);
                max_abs = std::max(max_abs, e);
                sum_sq += double(e) * e;
                raw_diff += ref[v] != got[v];
                if (a > ref_max) { ref_max = a; ref_argmax = v; }
                if (b > got_max) { got_max = b; got_argmax = v; }
            }
            TM_LOG_WARNING("[logit-parity] uid={} raw_diff={} max_abs={} rms={} ref_argmax={} "
                           "got_argmax={} ref_at_got={} got_at_ref={}",
                           (long)d.uids[i],
                           raw_diff,
                           max_abs,
                           std::sqrt(sum_sq / weights_.vocab_size),
                           ref_argmax,
                           got_argmax,
                           half_to_float(ref[got_argmax]),
                           half_to_float(got[ref_argmax]));
            replay_ref_logits_.erase(it);
        }
    }

    output_processor_->OutputHiddenStatesAndLogits(phase, env, 1);

    // On a verification step the sampler must not run.
    //
    // Generation::Forward samples one token per row and advances seq_len by
    // one. A verification forward has already submitted [bonus, D0..D_{K-1}]
    // and the tokens it will keep are decided by kReject and kRollback, from
    // the logits, not by sampling. Letting the sampler run here would append a
    // token on top of the accepted prefix and corrupt the sequence.
    // A disagreement here is the `verify_logits_` abort.
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

        // TM_SPEC_TRACE: the token this ordinary decode step produced.
        static const bool spec_trace = [] {
            const char* s = std::getenv("TM_SPEC_TRACE");
            return s && s[0] == '1';
        }();
        if (TM_UNLIKELY(spec_trace) && d.rows.size() <= 8) {
            Buffer_<int> h_out{(ssize_t)d.rows.size(), kCPU};
            Copy(autoreg_ids_.slice(0, (ssize_t)d.rows.size()), h_out);
            core::Context::stream().Sync();
            for (size_t i = 0; i < d.rows.size(); ++i) {
                TM_LOG_WARNING("[trace] gen uid={} seq_len={} out={}",
                               (long)d.uids[i], d.seq_lens[i], h_out[i]);
            }
        }

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

void LanguageModel::Impl::DraftDFlashTokens(int phase, TensorMap& env)
{
    NvtxScope scope("dflashDraftAndSelect");
    auto&     d   = data_.at(phase);
    const int bsz = (int)d.rows.size();
    const int K   = engine_param_.num_draft_tokens;
    if (bsz == 0 || K <= 0) {
        return;
    }
    TM_CHECK_EQ(K, weights_.dflash->block_size - 1);

    // Rollback publishes the committed target frontier. Rebuild the shared
    // key offsets before the next parallel draft block.
    Buffer_<int> k_offsets = env.at("k_offsets").buffer();
    PrefixSum(sequence_length_.front().data<int>(), bsz, k_offsets.data(), core::Context::stream().handle());

    Buffer_<int> tips{bsz, kCPU};
    Copy(sequence_length_.front().buffer().slice(0, bsz), tips);
    core::Context::stream().Sync();
    for (int i = 0; i < bsz; ++i) {
        if (d.rows[i]->max_seq_len - tips[i] - 1 < K) {
            for (auto* row : d.rows) {
                row->pending_num_drafts = 0;
            }
            return;
        }
    }

    auto anchors = autoreg_ids_.slice(0, bsz);
    if (bsz == 1 && K == 7) {
        dflash_predictor_->BeginParityBlock(anchors, d.uids[0], tips[0], d.input_lens[0]);
    }
    Tensor block_hidden = dflash_predictor_->DraftBlock(anchors, phase, env);
    Buffer_<int> candidates = dflash_predictor_->SelectCandidates(block_hidden, anchors, phase);
    Buffer_<int> host = dflash_predictor_->ParityActive() ?
                            d.spec_draft_candidates_host.slice(0, candidates.size()) :
                            Buffer_<int>{candidates.size(), kCPU};
    core::Copy(candidates, candidates.size(), host);
    if (!dflash_predictor_->FinishParityBlock()) {
        core::Context::stream().Sync();
    }

    for (int i = 0; i < bsz; ++i) {
        auto& row = *d.rows[i];
        if (i < (int)d.skip_draft.size() && d.skip_draft[i]) {
            row.pending_num_drafts = 0;
            continue;
        }
        row.pending_num_drafts = K;
        for (int k = 0; k < K; ++k) {
            const int id = host[i * K + k];
            TM_CHECK(0 <= id && id < weights_.vocab_size)
                << "[DFlash2] candidate " << k << " for row " << i << " produced invalid token " << id;
            row.pending_draft_tokens[k] = id;
        }
    }
}

void LanguageModel::Impl::DraftTokens(int phase, TensorMap& env)
{
    if (dflash_predictor_) {
        return DraftDFlashTokens(phase, env);
    }
    TM_CHECK_NOTNULL(mtp_predictor_.get());

    auto&     d   = data_.at(phase);
    const int bsz = (int)d.rows.size();
    const int K   = engine_param_.num_draft_tokens;
    if (bsz == 0 || K <= 0) {
        return;
    }

    // The MTP KV slot was already seeded in Setup, which is the only place the
    // `requests` tensor it needs is available.

    // Rollback has just replaced sequence_length_ with the COMMITTED frontier,
    // but k_offsets was built before verification and still describes the full
    // K-wide verifier input. Rebuild it before either accepted-token repair or
    // the next proposal. Rewinding a stale verifier frontier by the accepted
    // count lands partial accepts among rejected-tail positions; the resulting
    // draft KV is internally consistent but attached to the wrong tokens.
    {
        Buffer_<int> k_offsets = env.at("k_offsets").buffer();
        PrefixSum(sequence_length_.front().data<int>(), bsz, k_offsets.data(), core::Context::stream().handle());
    }

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

    // SGLang's EAGLE lifecycle repairs every accepted draft KV entry with the
    // target verifier hidden state that predicted that token. Proposal-time KV
    // was conditioned on draft hidden states and is only provisional. Keeping
    // it after acceptance poisons later chains even though the accepted token
    // IDs themselves were correct.
    //
    // Start with the batch-one reference path used by the V100 benchmark. It
    // replays accepted entries sequentially; a variable-length extend plan can
    // fuse mixed-batch repair once this A/B establishes the quality effect.
    static const bool repair_accepted = [] {
        const char* value = std::getenv("TM_MTP_ACCEPTED_REPAIR");
        return value && value[0] == '1';
    }();
    if (repair_accepted && bsz == 1 && hidden.shape(0) != bsz && !d.skip_draft[0]) {
        const int n = d.last_accepted[0];
        if (n > 0) {
            Buffer_<int> host_tokens{n, kCPU};
            for (int k = 0; k < n; ++k) {
                host_tokens[k] = d.rows[0]->draft_tokens[k];
            }
            Buffer_<int> device_tokens{n, kDEVICE};
            Copy(host_tokens, device_tokens);
            mtp_predictor_->RepairAcceptedSingle(hidden.slice(0, n), device_tokens, n, phase, env);
        }
    }

    if (hidden.shape(0) != bsz) {
        const int rows_per_seq = (int)hidden.shape(0) / bsz;
        TM_CHECK_EQ(rows_per_seq * bsz, (int)hidden.shape(0))
            << "hidden_states rows are not a whole multiple of the batch";

        Tensor gathered = empty_like(hidden.slice(0, bsz));
        for (int i = 0; i < bsz; ++i) {
            // d.last_accepted[i] is the offset of the accepted tip inside this
            // row's block, recorded by Rollback.
            const int off = i * rows_per_seq + d.last_accepted[i];
            TM_CHECK_LT(off, (int)hidden.shape(0));
            Copy(hidden.slice(off, 1), gathered.slice(i, 1));
        }
        hidden = std::move(gathered);
    }

    auto ids = autoreg_ids_.slice(0, bsz);

    // Each row's key length after this step's accepted tokens, which is what
    // the draft needs to know how much slack its last KV block still has.
    // The POST-rollback tip, not c.seq_len.
    //
    // Rollback publishes the advanced lengths into the sequence_length buffer;
    // c.seq_len is only updated later, by Engine::Update. So on a verification
    // step c.seq_len still points at the tip from BEFORE this step's accepted
    // run, and drafting from it walks positions the verification already
    // committed while advancing k_offsets from a base that does not match the
    // real KV extent. That is the illegal memory access.
    Buffer_<int> tips{bsz, kCPU};
    Copy(sequence_length_.front().buffer().slice(0, bsz), tips);
    core::Context::stream().Sync();

    std::vector<int> seq_lens(bsz);
    std::vector<int> block_counts(bsz);
    for (int i = 0; i < bsz; ++i) {
        seq_lens[i] = tips[i];
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
        // seq_lens[i], not c.seq_len: same staleness as above.
        budget = std::min(budget, d.rows[i]->max_seq_len - seq_lens[i] - 1);
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
            auto& c = *d.rows[i];

            // A row whose verification was just rejected takes an ordinary
            // decode next step. Drafting for it again would retry the same
            // prediction against the same state and fail the same way, and the
            // row would never make progress.
            if (i < (int)d.skip_draft.size() && d.skip_draft[i]) {
                c.pending_num_drafts = 0;
                continue;
            }

            c.pending_num_drafts = n;
            for (int k = 0; k < n; ++k) {
                const int id = host[k * bsz + i];
                // Validate here, where the tokens are already on the host,
                // rather than letting a bad id reach the embedding lookup.
                // Launch-blocking placed the long-prompt illegal access in
                // embeddingLookupKernel during the VERIFICATION forward -- the
                // kernel that dereferences table[token_id]. That kernel faults
                // in exactly one way: an id outside the table. The drafts are
                // the only ids that do not come from the tokenizer, so they
                // are checked at their source; an abort HERE with the value in
                // hand indicts the draft's argmax/logits path, and a fault
                // THERE with clean ids acquits it.
                TM_CHECK(0 <= id && id < weights_.vocab_size)
                    << "[MTP] draft step " << k << " for row " << i << " (uid " << c.req->unique_id
                    << ", seq_len " << seq_lens[i] << ") produced token " << id << " outside the vocab of "
                    << weights_.vocab_size;
                c.pending_draft_tokens[k] = id;
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
    NvtxScope scope("speculativeReject");
    TM_CHECK(mtp_predictor_ || dflash_predictor_);

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

    // TM_MTP_FORCE_REJECT=1 makes position zero reject inside GreedyReject,
    // rather than clearing the accepted count afterwards. Clearing afterwards
    // leaves bonus_tokens at target[n] from the original accepted prefix; once
    // partial commits are enabled that skips target[0] and invalidates the
    // forced-rejection identity diagnostic itself.
    static const bool force_reject = [] {
        const char* s = std::getenv("TM_MTP_FORCE_REJECT");
        return s && s[0] == '1';
    }();
    if (TM_UNLIKELY(force_reject)) {
        Buffer_<int> rejected{bsz * K, kCPU};
        std::fill(rejected.data(), rejected.data() + rejected.size(), -1);
        Copy(rejected, drafts);
    }

    static const float configured_ambiguity_margin = [] {
        const char* s = std::getenv("TM_MTP_AMBIGUITY_MARGIN");
        return s && s[0] ? std::max(0.f, (float)std::atof(s)) : -1.f;
    }();
    // A 0.0625 safety margin replayed whole DFlash prefixes and reduced the
    // audited commit length by 0.22--0.26 tokens/step. Zero margin retains
    // exact-tie replay and passed exact K=0/K=7 identity on the audited
    // 1,000-token prompt. An explicit environment value remains available for
    // conservative diagnostics.
    const float ambiguity_margin = configured_ambiguity_margin >= 0.f ? configured_ambiguity_margin : 0.f;

    // Ordinary decoding masks every EOS ID until min_new_tokens. Verification
    // must use the same candidate set before it compares target and draft
    // tokens. Handling EOS during rollback cannot recover the next-best token
    // that the canonical sampler selected instead.
    int eos_ids_size = 1;
    for (const auto* row : d.rows) {
        eos_ids_size = std::max(eos_ids_size, (int)row->gen_cfg.eos_ids.size());
    }
    Buffer_<int> eos_ids_host{bsz * eos_ids_size, kCPU};
    Buffer_<int> eos_enable_positions_host{bsz, kCPU};
    std::fill(eos_ids_host.data(), eos_ids_host.data() + eos_ids_host.size(), -1);
    for (int i = 0; i < bsz; ++i) {
        const auto& c = *d.rows[i];
        std::copy(c.gen_cfg.eos_ids.begin(),
                  c.gen_cfg.eos_ids.end(),
                  eos_ids_host.data() + i * eos_ids_size);
        eos_enable_positions_host[i] = c.prompt_len + c.gen_cfg.min_new_tokens - c.seq_len - 1;
    }
    Buffer_<int> eos_ids{bsz * eos_ids_size, kDEVICE};
    Buffer_<int> eos_enable_positions{bsz, kDEVICE};
    Copy(eos_ids_host, eos_ids);
    Copy(eos_enable_positions_host, eos_enable_positions);

    auto result = GreedyReject(verify_logits_.raw_data(),
                               drafts.data(),
                               bsz,
                               K,
                               weights_.vocab_size,  // argmax searches only the real vocabulary
                               vocab_stride,         // rows are strided by the padded size
                               eos_ids.data(),
                               eos_ids_size,
                               eos_enable_positions.data(),
                               ambiguity_margin,
                               weights_.data_type,
                               st);

    static const bool trace_logit_parity = [] {
        const char* s = std::getenv("TM_SPEC_LOGIT_PARITY");
        return s && s[0] == '1';
    }();
    if (TM_UNLIKELY(trace_logit_parity)) {
        TM_CHECK_EQ(weights_.data_type, kHalf);
        Buffer_<uint16_t> host{(ssize_t)bsz * vocab_stride, kCPU};
        for (int i = 0; i < bsz; ++i) {
            const ssize_t src_offset = (ssize_t)i * (K + 1) * vocab_stride;
            core::Copy(Buffer_<uint16_t>{(uint16_t*)verify_logits_.raw_data() + src_offset,
                                         vocab_stride,
                                         kDEVICE},
                       vocab_stride,
                       host.slice((ssize_t)i * vocab_stride, vocab_stride));
        }
        core::Context::stream().Sync();
        for (int i = 0; i < bsz; ++i) {
            const uint16_t* row = host.data() + (ssize_t)i * vocab_stride;
            replay_ref_logits_[d.uids[i]] = std::vector<uint16_t>(row, row + vocab_stride);
        }
    }

    // Alignment diagnostic for the first few verifications: the drafts against
    // the argmax the verifier computed at each position. The three failure
    // shapes it separates:
    //   T[k] == D[k]   -> aligned and accepted (healthy)
    //   T[k] == D[k-1] or D[k+1] -> a residual off-by-one inside verification
    //   T and D unrelated -> the drafts themselves are junk (wrong hidden row
    //                        or wrong conditioning tip in kDraft)
    // Each row also prints the ids the forward actually submitted, so the
    // bonus-vs-draft window placement is visible in the same line.
    static const bool dump_rejections = [] {
        const char* value = std::getenv("TM_SPEC_REJECT_DUMP");
        return value && value[0] == '1';
    }();
    if (TM_UNLIKELY(dump_rejections && reject_dumps_ < 6)) {
        ++reject_dumps_;
        // The target argmax per position is not returned by GreedyReject, so
        // recompute it here on the host for the dump. Cost is irrelevant: six
        // dumps, ever.
        Buffer_<int> h_acc{bsz, kCPU};
        Buffer_<int> h_bonus{bsz, kCPU};
        Copy(result.num_accepted, h_acc);
        Copy(result.bonus_tokens, h_bonus);
        // Row 0's K+1 logit rows, to compute the verifier's argmax per
        // position on the host. fp16, ~2.4 MB for six dumps ever.
        const ssize_t   row_elems = (ssize_t)(K + 1) * vocab_stride;
        Buffer_<uint16_t> h_logits{row_elems, kCPU};
        core::Copy(Buffer_<uint16_t>{(uint16_t*)verify_logits_.raw_data(), row_elems, kDEVICE},
                   row_elems,
                   h_logits);
        core::Context::stream().Sync();
        std::string ts;
        for (int p = 0; p <= K; ++p) {
            const uint16_t* row  = h_logits.data() + (ssize_t)p * vocab_stride;
            float           best = -1e30f;
            int             bi   = 0;
            for (int v = 0; v < weights_.vocab_size; ++v) {
                // fp16 bits -> float via __half memcpy trick, host-side.
                uint16_t u = row[v];
                int  sign = (u >> 15) & 1;
                int  exp  = (u >> 10) & 0x1f;
                int  man  = u & 0x3ff;
                float val;
                if (exp == 0) {
                    val = std::ldexp((float)man, -24);
                }
                else if (exp == 31) {
                    val = man ? NAN : INFINITY;
                }
                else {
                    val = std::ldexp((float)(man + 1024), exp - 25);
                }
                if (sign) val = -val;
                if (val > best) {
                    best = val;
                    bi   = v;
                }
            }
            ts += std::to_string(bi) + (p < K ? "," : "");
        }
        for (int i = 0; i < std::min(bsz, 2); ++i) {
            std::string ds;
            for (int k = 0; k < K; ++k) {
                ds += std::to_string(d.rows[i]->draft_tokens[k]) + (k + 1 < K ? "," : "");
            }
            TM_LOG_WARNING("[reject] row {} uid {}: drafts=[{}] targets=[{}] accepted={} bonus={} (seq_len {})",
                           i,
                           (long)d.uids[i],
                           ds,
                           i == 0 ? ts : std::string("-"),
                           h_acc[i],
                           h_bonus[i],
                           d.seq_lens[i]);
        }
    }

    env.produce("num_accepted", result.num_accepted);
    env.produce("bonus_tokens", result.bonus_tokens);
    env.produce("bonus_ambiguous", result.bonus_ambiguous);

    // Generation::Rollback needs the drafts themselves to append the accepted
    // ones to its own copy of each sequence.
    env.produce("draft_tokens", drafts);

    // The bonus token is the tip the next draft conditions on.
    Copy(result.bonus_tokens, autoreg_ids_.slice(0, bsz));

    verify_logits_ = {};
}

void LanguageModel::Impl::Rollback(int phase, TensorMap& env)
{
    NvtxScope scope("speculativeRollback");
    auto&     d   = data_.at(phase);
    const int bsz = (int)d.rows.size();

    Buffer_<int> accepted{bsz, kCPU};
    Buffer_<int> bonus{bsz, kCPU};
    Buffer_<int> ambiguous{bsz, kCPU};
    Copy(env.at("num_accepted").buffer().slice(0, bsz), accepted);
    Copy(env.at("bonus_tokens").buffer().slice(0, bsz), bonus);
    Copy(env.at("bonus_ambiguous").buffer().slice(0, bsz), ambiguous);
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
    bool clamped     = false;
    bool restore_gdn = false;
    bool select_gdn  = false;
    // Parity mode deliberately takes the legacy restore-and-replay branch so
    // it can compare this verifier's first-position logits with the next
    // ordinary recurrent decode in the same process.
    static const bool trace_logit_parity = [] {
        const char* s = std::getenv("TM_SPEC_LOGIT_PARITY");
        return s && s[0] == '1';
    }();
    const bool has_intermediate_gdn = gdn_rollback_ && !trace_logit_parity
                                      && unified_decoder_->has_intermediate_gdn_state(phase);
    eos_hit_.assign(bsz, false);
    gdn_restore_.assign(bsz, 0);
    gdn_state_slots_.assign(bsz, -1);
    no_commit_.assign(bsz, 0);
    no_bonus_.assign(bsz, 0);
    Buffer_<int> seq_lens{bsz, kCPU};
    Buffer_<int> out_ids{bsz, kCPU};
    Buffer_<int> tip_ids{bsz, kCPU};
    // sequence_length_.front(), not d.sequence_length. Rollback runs before
    // Unprep, and Unprep is what copies the live buffer into d.sequence_length,
    // so reading d here would give the PREVIOUS step's lengths.
    Copy(sequence_length_.front().buffer().slice(0, bsz), seq_lens);
    Copy(autoreg_ids_.slice(0, bsz), out_ids);
    core::Context::stream().Sync();
    std::copy_n(out_ids.data(), bsz, tip_ids.data());

    for (int i = 0; i < bsz; ++i) {
        auto& c = *d.rows[i];

        if (!c.generating || d.num_drafts[i] == 0) {
            continue;  // not a verification row; its published length stands
        }

        int n = accepted[i];
        TM_CHECK_GE(n, 0);
        TM_CHECK_LE(n, d.num_drafts[i]);
        const int verifier_n = n;

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

        // Stop at the first EOS inside the accepted run.
        //
        // A verification step skips Generation entirely, so stop_criteria never
        // runs and nothing else inspects the accepted tokens. If a draft that
        // happens to be EOS is accepted, the sequence would continue past the
        // end of its own answer and run to max_new_tokens, while the baseline
        // stops. That is a user-visible divergence -- the model rambles -- and
        // the identity check would report it as a length mismatch with no hint
        // that EOS was the cause.
        //
        // Truncating to k + 1 keeps the EOS token itself and drops the rest,
        // which is what ordinary decode produces.
        const auto& eos = c.gen_cfg.eos_ids;
        if (!eos.empty()) {
            for (int k = 0; k < n; ++k) {
                const bool eos_enabled = base + k + 1 >= c.prompt_len + c.gen_cfg.min_new_tokens;
                if (eos_enabled && std::find(eos.begin(), eos.end(), c.draft_tokens[k]) != eos.end()) {
                    n            = k + 1;
                    accepted[i]  = n;
                    clamped      = true;
                    eos_hit_[i]  = true;
                    no_bonus_[i] = 1;
                    break;
                }
            }
        }

        static const bool zero_accept_replay = [] {
            const char* s = std::getenv("TM_MTP_ZERO_ACCEPT_REPLAY");
            return s && s[0] == '1';
        }();
        static const bool ambiguous_replay = [] {
            const char* s = std::getenv("TM_MTP_AMBIGUOUS_REPLAY");
            return s && s[0] == '1';
        }();
        const bool replay_ambiguous = dflash_predictor_ || ambiguous_replay;
        const bool replay_bonus = (n == 0 && zero_accept_replay) || (replay_ambiguous && ambiguous[i]);
        const int raw_commit = n + (no_bonus_[i] ? 0 : 1);
        spec_raw_accepted_ += n;
        spec_raw_committed_ += raw_commit;
        spec_ambiguous_steps_ += ambiguous[i] != 0;
        if (replay_bonus && ambiguous[i]) {
            spec_ambiguity_discarded_ += raw_commit;
        }
        if (gdn_rollback_) {
            if (has_intermediate_gdn && replay_bonus) {
                // A K+1 verifier forward is numerically close to, but not
                // bit-identical with, the ordinary one-token target path. On
                // a measured FP16 near-tie any accepted-prefix or bonus row can
                // choose the other argmax. Restore slot zero and let the next
                // ordinary forward recompute from the canonical target-only
                // shape. DFlash enables this exactness guard by default; the
                // measured margin keeps replay limited to sensitive decisions.
                clamped             = true;
                gdn_restore_[i]     = 1;
                restore_gdn         = true;
                no_commit_[i]       = 1;
                gdn_state_slots_[i] = -1;
            }
            else if (has_intermediate_gdn) {
                // Verification input is [old tip, D0, ...]. If n drafts and
                // the bonus commit, state must include input positions through
                // D(n-1): slot n+1. An accepted EOS draft omits the bonus and
                // finishes at slot n. Slot zero is the pre-forward state.
                gdn_state_slots_[i] = n + (no_bonus_[i] ? 0 : 1);
                select_gdn          = true;
            }
            else if (verifier_n < d.num_drafts[i]) {
                // The active batch exceeded the fixed snapshot budget. Keep
                // the exact fallback: restore slot zero, commit nothing, and
                // perform an ordinary decode on the next step.
                n                    = 0;
                accepted[i]          = 0;
                clamped              = true;
                eos_hit_[i]          = false;
                no_bonus_[i]         = 0;
                gdn_restore_[i]      = 1;
                restore_gdn          = true;
                no_commit_[i]        = 1;
                gdn_state_slots_[i]  = -1;
            }
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

        // A no-commit row emits nothing: no bonus token, no length change.
        // Its next step decodes normally and produces one token there.
        if (no_commit_[i]) {
            // The next step's input must be the last COMMITTED token, not
            // the bonus.
            //
            // RejectDrafts wrote the bonus into autoreg_ids_ (it is the
            // right conditioning tip for a row that commits), and this
            // branch used to leave it there. The next step is an ordinary
            // autoregressive decode -- skip_draft guarantees that -- and its
            // input token comes from autoreg_ids_, at the position of the
            // committed tip. Feeding the uncommitted bonus there computes
            // the successor of a token the sequence does not contain,
            // overwrites the real tip's KV with the phantom's, and commits
            // the result. The baseline computes the successor of the real
            // tip: output diverges at the first rejected verification and
            // never recovers. This is the position-2 identity divergence
            // the force-reject probe isolated to the verification path.
            //
            // token_ids[base - 1] is the real tip: base = c.seq_len points
            // one past the last committed token, and base >= 1 always,
            // because at minimum the prompt is committed. Re-submitting the
            // tip is exactly correct on both remaining state carriers: its
            // KV write is idempotent (same token, same position), and the
            // restored GDN snapshot predates the tip's own advance -- the
            // snapshot was taken before the verification forward, which is
            // the forward that would have advanced the tip through GDN --
            // so the re-submit performs that advance for the first time.
            //
            // Update also reads this value and writes it at token_ids[base],
            // a scratch slot past the tip: sequence_length is unchanged, so
            // new_tokens is 0, nothing is appended, and the next real
            // commit overwrites the slot. Harmless on that path, load-
            // bearing on the input path.
            out_ids[i] = c.token_ids[base - 1];
            tip_ids[i] = out_ids[i];

            // Rewind the published length to the base. THE leak, measured.
            //
            // seq_lens was pre-filled from sequence_length_.front(), which
            // Prepare set to base + K for this verification row: non-
            // autoregressive rows publish history + input_len, and a
            // verification submits 1 + K tokens. The normal path overwrites
            // the entry with base + 1 + n; this branch used to `continue`
            // with the pre-filled value still in place, so Engine::Update
            // advanced c.seq_len over the re-fed tip AND every rejected
            // draft that Schedule had injected into token_ids -- K + 1
            // junk tokens committed per rejected verification, then decoded
            // on top of. TM_SPEC_TRACE showed it directly:
            //
            //   [trace] verify base=65 accepted=0 new_len=69 no_commit=1
            //
            // where a no-commit row must republish exactly base.
            seq_lens[i] = base;

            // Generation::Rollback also receives no_commit_ below. Keep the
            // accepted-count contract (0 <= n <= K) intact; using -1 as an
            // append sentinel made mixed/retiring batches crash after several
            // hundred steps. The append decision and acceptance count are two
            // different facts, so represent them separately.

            // Count the step BEFORE bailing out.
            //
            // A rejected verification is still a forward that produced zero
            // tokens, and leaving it out makes the meter average over only the
            // steps that accepted everything -- so it would report exactly
            // 1 + K forever, whatever the real rate. That is the ceiling, not a
            // measurement, and the accept-length gate would pass a build that
            // accepts almost nothing.
            mtp_steps_ += 1;

            // Force an ordinary decode next step via the pending field, NOT by
            // writing c.num_drafts here.
            //
            // Two reasons. Rollback runs on the executor thread while the main
            // thread may already be in Schedule(), so touching the live field
            // is the same cross-thread race that commit 2428300a fixed. And
            // kDraft runs after this in the same step and sets
            // pending_num_drafts = K, so a write here would simply be
            // overwritten by Update's publish -- the row would draft again and
            // retry the prediction that just failed.
            //
            // d.skip_draft is honoured by DraftTokens, which runs later in this
            // same step, so the suppression survives to the publish.
            d.skip_draft[i] = 1;
            continue;
        }

        if (!no_bonus_[i]) {
            c.token_ids[base + n] = bonus[i];
        }

        // What Update puts at token_ids[base]. If an accepted draft was EOS,
        // n is at least one and the first draft remains the correct first
        // committed token.
        out_ids[i] = n > 0 ? c.draft_tokens[0] : bonus[i];

        // autoreg_ids conditions the next draft (or the next ordinary decode
        // when the request is too close to its limit to draft again). It must
        // carry the LAST committed token, whereas Generation::Rollback keeps
        // its separate output_ids buffer on the FIRST token for Engine::Update.
        // Conflating those roles only surfaced at the request tail: kDraft
        // normally hid the stale first token, but when its budget reached zero
        // the next decode re-fed D0 instead of the verifier bonus.
        tip_ids[i] = no_bonus_[i] ? c.draft_tokens[n - 1] : bonus[i];

        // The KV entries for the rejected tail stay allocated but are logically
        // dead: the next forward starts at this length and overwrites them.
        // An EOS draft ends the sequence before the verifier's bonus.
        seq_lens[i] = base + n + (no_bonus_[i] ? 0 : 1);
        // Generation is skipped on verification steps, so its ordinary length
        // stop criterion cannot mark the row finished. A higher-acceptance
        // DFlash block exposed the missing transition: the sequence reached
        // max_seq_len exactly, drafted once more, and either published a 257th
        // token or produced an empty terminal stream update after the engine's
        // defensive clamp.
        if (seq_lens[i] >= c.max_seq_len) {
            eos_hit_[i] = true;  // shared finished/skip-draft publication path
        }

        // The bonus token is the last one committed, so EOS there ends the
        // sequence without truncating anything.
        const bool bonus_eos_enabled = seq_lens[i] >= c.prompt_len + c.gen_cfg.min_new_tokens;
        if (bonus_eos_enabled && !eos.empty() && !eos_hit_[i]
            && std::find(eos.begin(), eos.end(), bonus[i]) != eos.end()) {
            eos_hit_[i] = true;
        }

        // Where this row's accepted tip sits inside its K+1 hidden block, for
        // the draft that runs next on this same env.
        d.last_accepted[i] = n;

        mtp_accepted_ += n;
        mtp_committed_ += n + (no_bonus_[i] ? 0 : 1);
        mtp_steps_ += 1;
        if (n == d.num_drafts[i]) {
            ++mtp_full_accepts_;
        }
    }

    // TM_SPEC_TRACE: the verification verdict, per row, before publication.
    {
        static const bool spec_trace = [] {
            const char* s = std::getenv("TM_SPEC_TRACE");
            return s && s[0] == '1';
        }();
        if (TM_UNLIKELY(spec_trace) && bsz <= 8) {
            for (int i = 0; i < bsz; ++i) {
                TM_LOG_WARNING("[trace] verify uid={} base={} accepted={} bonus={} out_id={} "
                               "new_len={} no_commit={} skip_draft={}",
                               (long)d.uids[i],
                               d.rows[i]->seq_len,
                               accepted[i],
                               bonus[i],
                               out_ids[i],
                               seq_lens[i],
                               (int)no_commit_[i],
                               (int)d.skip_draft[i]);
            }
        }
    }

    // Order matters here, and getting it backwards fails silently.
    //
    // Generation::Rollback appends its accepted tokens at base_len + slot, so
    // it needs the length from BEFORE this step's acceptance. Publishing the
    // advanced lengths first would make it write each token one full accepted
    // run too far into its copy of the sequence.
    //
    // Mark EOS-terminated rows finished.
    //
    // Engine::Update reads `finished` to retire a sequence, and on a
    // verification step nothing else sets it: stop_criteria belongs to
    // Generation, which this path skips. Without this the row keeps generating
    // after emitting EOS.
    // A row that just emitted EOS must not draft.
    //
    // It is finished, so its drafts would predict tokens past the end of its
    // own answer, and the next step retires it before they could be verified.
    // Worse, the draft's per-row validity mask is derived from `finished`, and
    // a finished row drafting means reading a mask entry that says otherwise.
    for (int i = 0; i < bsz; ++i) {
        if (eos_hit_[i]) {
            d.skip_draft[i] = 1;
        }
    }

    if (std::find(eos_hit_.begin(), eos_hit_.begin() + bsz, true) != eos_hit_.begin() + bsz) {
        Buffer_<bool> host{bsz, kCPU};
        Copy(finished_.front().buffer().slice(0, bsz), host);
        core::Context::stream().Sync();
        for (int i = 0; i < bsz; ++i) {
            if (eos_hit_[i]) {
                host[i] = true;
            }
        }
        Copy(host, finished_.front().buffer().slice(0, bsz));
    }

    // Rewind the recurrent state, for the rejecting rows ONLY.
    //
    // Restoring the whole batch would corrupt every row that accepted all its
    // drafts: that row's state is correctly advanced, and rewinding strands it
    // behind a tip already committed. The next forward begins at that tip and
    // never re-runs the prefix, so the state would stay K+1 tokens behind for
    // the rest of the sequence.
    //
    // I wrote the opposite in the previous commit -- that a fully-accepting row
    // would be re-advanced for free -- which is the same assumption I had
    // already disproved earlier in this same turn. The next forward starts at
    // resume_len + inflight_input_len, which lands on the new tip.
    if (restore_gdn) {
        unified_decoder_->RestoreGDNState(phase, gdn_restore_.data(), bsz);
    }
    if (select_gdn) {
        unified_decoder_->SelectGDNState(phase, gdn_state_slots_.data(), bsz);
    }

    // Publish the no-commit mask separately from the accepted count.
    //
    // A rejected GDN verification has accepted=0 but must not append the
    // bonus. Generation::Rollback normally appends its slot 0 for accepted=0,
    // so it needs the separate state transition fact. This buffer aliases the
    // member vector and stays valid for the synchronous Rollback call below.
    env.produce("mtp_no_commit", Buffer{no_commit_.data(), bsz, kCPU});
    env.produce("mtp_no_bonus", Buffer{no_bonus_.data(), bsz, kCPU});

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

    // Generation::Rollback owns the separate output_ids buffer fetched by
    // Engine::Update and leaves it on the first committed token. autoreg_ids_
    // instead owns the conditioning tip for kDraft / the next decode, so keep
    // it on the final committed token.
    Copy(tip_ids, autoreg_ids_.slice(0, bsz));

    // Tokens committed per verification forward -- the quantity that decides
    // whether speculation pays for itself.
    //
    // NOT `1 + accepted/steps`. That formula assumes every step commits at
    // least a bonus token, which stopped being true when rejection became a
    // no-op: an all-or-nothing step commits 1 + K tokens or nothing at all.
    // With rejected steps also excluded from the denominator, as they were, the
    // meter averaged over only the steps that accepted everything and therefore
    // reported exactly 1 + K forever -- the ceiling, not a measurement, and
    // enough to pass an accept-length gate on a build that accepts almost
    // nothing.
    //
    // mtp_committed_ counts tokens actually committed; mtp_steps_ counts every
    // verification forward including the rejected ones.
    if (mtp_steps_ >= 64 && mtp_steps_ % 64 == 0) {
        TM_LOG_INFO("[MTP] accept length {:.2f} tokens/step over {} steps "
                    "(raw length {:.2f}, {} committed, {} raw committed, {} raw accepted drafts, "
                    "{} ambiguous steps, {} tokens discarded by ambiguity, {} full accepts)",
                    (double)mtp_committed_ / (double)mtp_steps_,
                    mtp_steps_,
                    (double)spec_raw_committed_ / (double)mtp_steps_,
                    mtp_committed_,
                    spec_raw_committed_,
                    spec_raw_accepted_,
                    spec_ambiguous_steps_,
                    spec_ambiguity_discarded_,
                    mtp_full_accepts_);
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
