
#include <memory>

#include "src/turbomind/generation/generation.h"

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/generation/guided_decoding.h"
#include "src/turbomind/generation/logits_processor.h"
#include "src/turbomind/generation/sampling.h"
#include "src/turbomind/generation/stop_criteria.h"

#include "src/turbomind/kernels/sampling_topk_kernels.h"  // InitializeRandomStates

#include "src/turbomind/models/llama/llama_kernels.h"  // invokePadLastTokenIds
#include "src/turbomind/utils/cuda_utils.h"

// #include "dbg.h"

namespace turbomind {

using std::unique_ptr;
using std::shared_ptr;
using std::vector;

struct GenerationData {
    Buffer_<uint64_t> random_seed;
    Buffer_<bool>     random_init;
    Buffer_<int>      random_state_indices;
    Buffer_<int>      max_seq_len;
    Buffer_<int*>     token_ids_ptrs;
    Buffer_<int>      output_ids;

    bool random_init_needed;
    int  generation_size;
};

struct Generation::Impl {

    // child modules
    unique_ptr<LogitsProcessor> logits_processor_;
    unique_ptr<Sampling>        sampling_;
    shared_ptr<StopCriteria>    stop_criteria_;
    unique_ptr<GuidedDecoding>  guided_decoding_;

    // persistent
    Tensor_<int>     token_ids_;
    Tensor_<uint8_t> random_states_;

    // scheduling states
    vector<int> free_token_rows_;
    vector<int> free_random_state_rows_;

    // immutable states
    Buffer_<int> output_ids_;

    std::vector<std::unique_ptr<GenerationData>> data_;

    // staging buffers
    Buffer_<uint64_t> random_seed_buf_;
    Buffer_<bool>     random_init_buf_;
    Buffer_<int>      random_state_indices_buf_;
    Buffer_<int*>     token_ids_ptrs_buf_;

    /// Speculative rollback scratch; see Rollback.
    Buffer_<int*> token_ids_ptrs_dev_;
    Buffer_<int>  output_pos_;
    Buffer_<int>  scratch_row_;
    Buffer_<int>      token_ids_buf_;
    Buffer_<int>      output_ids_buf_;

    const int max_batch_size_;
    const int session_len_;

    int* RowPtr(int row)
    {
        TM_CHECK_GE(row, 0);
        TM_CHECK_LT(row, max_batch_size_);
        return token_ids_.data() + row * token_ids_.stride(0);
    }

    Impl(DataType              dtype,
         int                   max_batch_size,
         int                   session_len,
         int                   vocab_size,
         int                   vocab_size_padded,
         const comm::HostComm& tp_group,
         int                   phases):
        max_batch_size_{max_batch_size}, session_len_{session_len}
    {
        TM_CHECK_EQ(dtype, kFloat32);
        BaseGenerationParam base{max_batch_size, vocab_size, vocab_size_padded};
        logits_processor_ = std::make_unique<LogitsProcessor>(base, phases);
        sampling_         = std::make_unique<Sampling>(base, phases, tp_group->rank());
        stop_criteria_    = std::make_unique<StopCriteria>(base, phases);
        guided_decoding_  = std::make_unique<GuidedDecoding>(base, tp_group, phases);

        static_assert(sizeof(curandState_t) % alignof(curandState_t) == 0);
        random_states_ = {{max_batch_size_, (int)sizeof(curandState_t)}, kDEVICE};
        token_ids_     = {{max_batch_size_, session_len_}, kDEVICE};
        output_ids_    = {max_batch_size_, kDEVICE};
        for (int i = 0; i < max_batch_size_; ++i) {
            free_token_rows_.push_back(i);
            free_random_state_rows_.push_back(i);
        }

        random_seed_buf_          = {max_batch_size_, kCPUpinned};
        random_init_buf_          = {max_batch_size_, kCPUpinned};
        random_state_indices_buf_ = {max_batch_size_, kCPUpinned};

        token_ids_ptrs_buf_ = {max_batch_size_, kCPUpinned};

        // Speculative rollback scratch.
        //
        // `scratch_row_` is a sink for rows that accepted fewer tokens than the
        // widest row in the batch: pointing their write at it means the kernel
        // runs uniformly while those rows receive no update. It must be at
        // least as long as the furthest slot any row can address.
        token_ids_ptrs_dev_ = {max_batch_size_, kDEVICE};
        output_pos_         = {max_batch_size_, kDEVICE};
        scratch_row_        = {session_len_ + Sequence::kMaxDraftTokens + 1, kDEVICE};
        token_ids_buf_      = {max_batch_size_ * (ssize_t)session_len_, kCPUpinned};

        output_ids_buf_ = {max_batch_size_, kCPUpinned};

        for (int i = 0; i < phases; ++i) {
            auto d = std::make_unique<GenerationData>();

            d->random_seed          = empty_like(random_seed_buf_, kDEVICE);
            d->random_init          = empty_like(random_init_buf_, kDEVICE);
            d->random_state_indices = empty_like(random_state_indices_buf_, kDEVICE);
            d->token_ids_ptrs       = empty_like(token_ids_ptrs_buf_, kDEVICE);
            d->output_ids           = empty_like(output_ids_, kDEVICE);

            data_.push_back(std::move(d));
        }
    }

    void Setup(int phase, TensorMap& env)
    {
        TM_FUNCTION_SCOPE();
        auto& d = *data_.at(phase);

        auto& copy = *env.at("copy").data<BatchCopy*>()[0];

        Buffer_<Sequence*> rc = env.at("requests").buffer();

        // random states
        d.random_init_needed = false;
        std::fill_n(random_init_buf_.data(), max_batch_size_, false);

        int* token_ids_buf   = token_ids_buf_.data();
        int  generation_size = 0;
        for (int i = 0; i < rc.size(); ++i) {
            auto& c = *rc[i];
            if (!c.generating) {
                continue;
            }

            if (c.generation_random_state_row < 0) {
                TM_CHECK(!free_random_state_rows_.empty());

                c.generation_random_state_row = free_random_state_rows_.back();
                free_random_state_rows_.pop_back();

                random_init_buf_[c.generation_random_state_row] = true;
                random_seed_buf_[c.generation_random_state_row] = c.gen_cfg.random_seed;
                d.random_init_needed                            = true;
            }

            if (c.generation_token_ids_row < 0) {
                TM_CHECK(!free_token_rows_.empty());

                c.generation_token_ids_row = free_token_rows_.back();
                free_token_rows_.pop_back();

                auto* dst = RowPtr(c.generation_token_ids_row);
                std::copy_n(c.token_ids, c.seq_len, token_ids_buf);
                copy(token_ids_buf, c.seq_len, dst);
                token_ids_buf += c.seq_len;
            }

            random_state_indices_buf_[generation_size] = c.generation_random_state_row;
            token_ids_ptrs_buf_[generation_size++]     = RowPtr(c.generation_token_ids_row);
        }

        if (d.random_init_needed) {
            copy(random_init_buf_, max_batch_size_, d.random_init);
            copy(random_seed_buf_, max_batch_size_, d.random_seed);
        }

        copy(token_ids_ptrs_buf_, generation_size, d.token_ids_ptrs);
        copy(random_state_indices_buf_, generation_size, d.random_state_indices);
        d.generation_size = generation_size;
        // dbg(d.generation_size);

        logits_processor_->Setup(phase, env);
        sampling_->Setup(phase, env);
        stop_criteria_->Setup(phase, env);
        guided_decoding_->Setup(phase, env);
    }

    void Del(TensorMap& env)
    {
        Buffer_<Sequence*> rc = env.at("requests").buffer();

        for (int i = 0; i < rc.size(); ++i) {
            auto& token_row = rc[i]->generation_token_ids_row;
            if (token_row >= 0) {
                free_token_rows_.push_back(token_row);
                token_row = -1;
            }

            auto& random_row = rc[i]->generation_random_state_row;
            if (random_row >= 0) {
                free_random_state_rows_.push_back(random_row);
                random_row = -1;
            }
        }
    }

    void Prepare(int phase, TensorMap& env)
    {
        TM_FUNCTION_SCOPE();
        (void)phase;
        (void)env;
    }

    void Unprep(int phase, TensorMap& env)
    {
        TM_FUNCTION_SCOPE();
        auto& d    = *data_.at(phase);
        auto& b    = *env.at("batch").data<BatchData*>()[0];
        auto& copy = *env.at("copy").data<BatchCopy*>()[0];

        copy(output_ids_, b.bsz, d.output_ids);
    }

    void Fetch(int phase, TensorMap& env)
    {
        TM_FUNCTION_SCOPE();
        auto& d    = *data_.at(phase);
        auto& copy = *env.at("copy").data<BatchCopy*>()[0];

        copy(d.output_ids, d.output_ids.size(), output_ids_buf_);
        env.produce("output_ids", output_ids_buf_);

        sampling_->Fetch(phase, env);
    }

    void Update(int phase, TensorMap& env)
    {
        TM_FUNCTION_SCOPE();
        sampling_->Update(phase, env);
    }

    /// Publish the outcome of a verification step.
    ///
    /// Ownership is the whole point of this function. Engine::Update is the
    /// only place that advances a sequence:
    ///
    ///   c.token_ids[c.seq_len] = output_ids[j];
    ///   c.seq_len              = sequence_length[j];
    ///
    /// so rejection must not write those fields itself, or every accepted
    /// token lands twice. Rejection instead writes the two buffers Update
    /// reads: the bonus token into `output_ids`, and the post-acceptance
    /// length into `sequence_length`.
    ///
    /// The generation-side copy of each sequence must advance too, because
    /// repetition penalties and stop criteria read it. AppendTokenIds is run
    /// once per accepted slot. Rows that accepted fewer tokens are pointed at
    /// a scratch row for the remaining slots, so they receive no write rather
    /// than a wrong one -- masking by pointer avoids needing a predicated
    /// variant of the kernel.
    void Rollback(int phase, TensorMap& env)
    {
        TM_FUNCTION_SCOPE();

        auto&     d      = *data_.at(phase);
        auto&     b      = *env.at("batch").data<BatchData*>()[0];
        auto      stream = core::Context::stream().handle();
        const int bsz    = b.bsz;
        const int gs     = d.generation_size;
        if (gs == 0) {
            return;
        }

        Buffer_<int> accepted{bsz, kCPU};
        Buffer_<int> bonus{bsz, kCPU};
        Copy(env.at("num_accepted").buffer().slice(0, bsz), accepted);
        Copy(env.at("bonus_tokens").buffer().slice(0, bsz), bonus);

        Buffer_<int> base_len{bsz, kCPU};
        Copy(env.at("sequence_length").buffer().slice(0, bsz), base_len);
        core::Context::stream().Sync();


        // Slot 0 is the bonus token, slots 1..n the accepted drafts.
        //
        // The row stride is K, not Sequence::kMaxDraftTokens: RejectDrafts
        // builds this buffer as [bsz, K] to match the rejection kernel's
        // layout. Reading it with the larger compile-time stride would walk off
        // the end of each row into the next one.
        const auto   draft_buf = env.at("draft_tokens").buffer();
        const int    stride    = bsz > 0 ? (int)draft_buf.size() / bsz : 0;
        Buffer_<int> drafts{draft_buf.size(), kCPU};
        Copy(draft_buf, drafts);
        core::Context::stream().Sync();

        int max_accepted = 0;
        for (int i = 0; i < bsz; ++i) {
            max_accepted = std::max(max_accepted, accepted[i]);
        }

        Buffer_<int*> ptrs{max_batch_size_, kCPU};
        Buffer_<int>  pos{max_batch_size_, kCPU};
        Buffer_<int>  tok{max_batch_size_, kCPU};

        for (int slot = 0; slot <= max_accepted; ++slot) {
            for (int i = 0; i < gs; ++i) {
                const bool active = slot <= accepted[i];
                ptrs[i] = active ? token_ids_ptrs_buf_[i] : scratch_row_.data();
                pos[i]  = active ? base_len[i] + slot : 0;
                // Accepted drafts first, bonus last -- the same order the
                // engine-side sequence uses, because the logit at submitted
                // position p predicts the token after it.
                tok[i] = slot < accepted[i] ? drafts[i * stride + slot] : bonus[i];
            }
            Copy(ptrs.slice(0, gs), token_ids_ptrs_dev_.slice(0, gs));
            Copy(pos.slice(0, gs), output_pos_.slice(0, gs));
            Copy(tok.slice(0, gs), output_ids_.slice(0, gs));
            AppendTokenIds(
                token_ids_ptrs_dev_.data(), output_ids_.data(), output_pos_.data(), gs, stream);
        }

        // Leave output_ids holding the token that goes at token_ids[base]:
        // the first accepted draft, or the bonus when nothing was accepted.
        // Fetch publishes this and Engine::Update writes exactly it.
        Buffer_<int> host_out{max_batch_size_, kCPU};
        for (int i = 0; i < bsz; ++i) {
            host_out[i] = accepted[i] > 0 ? drafts[i * stride] : bonus[i];
        }
        Copy(host_out.slice(0, bsz), output_ids_.slice(0, bsz));
        Copy(output_ids_, bsz, d.output_ids);
    }

    void Forward(int phase, TensorMap& env)
    {
        TM_FUNCTION_SCOPE();
        auto& d = *data_.at(phase);

        const auto stream = core::Context::stream().handle();

        if (d.random_init_needed) {
            InitializeRandomStates((curandState_t*)random_states_.raw_data(),
                                   d.random_seed.data(),
                                   d.random_init.data(),
                                   max_batch_size_,
                                   stream);
        }

        env.emplace("output_ids", output_ids_);       // out
        env.emplace("curand_state", random_states_);  // inout

        if (const int gs = d.generation_size) {

            env.emplace("token_ids_ptrs", d.token_ids_ptrs.slice(0, gs));
            env.emplace("curand_state_indices", d.random_state_indices.slice(0, gs));

            auto logits = env.consume("logits");

            if (logits.dtype() != kFloat32) {
                auto tmp = empty_like(logits, kFloat32);
                TM_SCOPE_CALL(invokeCastFloat2D(logits, tmp, stream));
                logits = std::move(tmp);
            }

            env.produce("logits", logits.slice(0, gs));

            Buffer_<int> output_pos{max_batch_size_, kDEVICE};
            Copy(env.at("sequence_length").buffer(), gs, output_pos);

            logits_processor_->Forward(phase, env);

            guided_decoding_->FillMask(phase, env);
            guided_decoding_->ApplyMask(phase, env);

            sampling_->Forward(phase, env);

            guided_decoding_->ScheduleUpdate(phase, env);

            AppendTokenIds(d.token_ids_ptrs.data(), output_ids_.data(), output_pos.data(), gs, stream);

            stop_criteria_->Forward(phase, env);

            guided_decoding_->FinishUpdate(phase, env);
        }
    }
};

Generation::~Generation() = default;

Generation::Generation(DataType              dtype,
                       int                   max_batch_size,
                       int                   session_len,
                       int                   vocab_size,
                       int                   vocab_size_padded,
                       const comm::HostComm& tp_group,
                       int                   phases):
    impl_{std::make_unique<Impl>(dtype, max_batch_size, session_len, vocab_size, vocab_size_padded, tp_group, phases)}
{
}

void Generation::Run(BatchOp op, int phase, TensorMap& env)
{
    if (op == BatchOp::kSetup) {
        return impl_->Setup(phase, env);
    }
    else if (op == BatchOp::kDel) {
        return impl_->Del(env);
    }
    else if (op == BatchOp::kPrepare) {
        return impl_->Prepare(phase, env);
    }
    else if (op == BatchOp::kForward) {
        return impl_->Forward(phase, env);
    }
    else if (op == BatchOp::kUnprep) {
        return impl_->Unprep(phase, env);
    }
    else if (op == BatchOp::kFetch) {
        return impl_->Fetch(phase, env);
    }
    else if (op == BatchOp::kUpdate) {
        return impl_->Update(phase, env);
    }
    else if (op == BatchOp::kRollback) {
        return impl_->Rollback(phase, env);
    }
}

}  // namespace turbomind
