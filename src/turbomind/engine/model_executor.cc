
#include "src/turbomind/engine/model_executor.h"

#include <memory>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/copy.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/models/language_model.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/vision_model.h"
#include "src/turbomind/utils/anomaly_handler.h"

// #include "dbg.h"

namespace turbomind {

using std::shared_ptr;
using std::unique_ptr;

struct ModelExecutor::Impl {

    LanguageModel& model_;
    VisionModel*   vision_model_;  // nullable: only set for VLM checkpoints
    LlamaLinear&   linear_;

    const int device_id_;

    Queue<unique_ptr<BatchData>>& inbound_;
    Queue<unique_ptr<BatchData>>& outbound_;

    std::thread internal_thread_;

    void InternalThreadEntry()
    {
        TM_FUNCTION_SCOPE();
        TM_CUDA_CHECK(cudaSetDevice(device_id_));

        Stream    stream  = Stream::create();
        Allocator h_alloc = Allocator(kCPU);
        Allocator d_alloc = Allocator(stream, false);

        AnomalyHandler::instance().Init(0, 1000, 0, 1000, stream.handle());

        core::ContextGuard ctx{stream, h_alloc, d_alloc};

        unique_ptr<BatchData> d;

        while (inbound_.pop(d)) {
            TM_CHECK_NOTNULL(d);
            core::Context::stream().Wait(d->ready);

            // Two speculative shapes, distinguished by whether drafts exist.
            //
            // Steady state: the batch already carries drafts from the previous
            // step, so the forward verifies them and the pipeline continues
            // into reject, rollback and a fresh draft.
            //
            // First decode: no drafts yet. Run the ordinary forward, then draft
            // once so the next step has something to verify. Without this the
            // steady state can never be entered.
            // BatchData carries no sequence pointers in this tree, so the two
            // questions -- are there drafts to verify, is there a decode step
            // to draft from -- are answered by the model, which captured that
            // state during Setup.
            if (d->spec_decode && model_.HasDraftsToVerify(d->phase)) {
                RunWithDrafts(*d);
            }
            else {
                // Priming the first drafts happens inside Run, immediately
                // after the forward, because drafting needs the `hidden_states`
                // that forward produced. A separate pass with a fresh env has
                // no access to it.
                Run(*d, /*draft_after_forward=*/d->spec_decode && model_.CanDraft(d->phase));
            }

            d->done.Record(core::Context::stream());
            outbound_.push(std::move(d));
        }
    }

    static void RunCopies(std::vector<ResolvedCopy>& copies)
    {
        for (const auto& c : copies) {
            Copy(Buffer_<uint8_t>{static_cast<uint8_t*>(c.src), static_cast<ssize_t>(c.bytes), kDEVICE},
                 Buffer_<uint8_t>{static_cast<uint8_t*>(c.dst), static_cast<ssize_t>(c.bytes), kDEVICE});
        }
        copies.clear();
    }

    // Steady-state speculative decode: Forward(K+1) -> Reject -> Rollback -> Draft.
    //
    // The forward consumes [bonus, D0..D_{K-1}] in one prefill-shaped pass and
    // emits a logit per position. kReject compares those logits against the
    // drafts and finds the longest correct prefix; kRollback truncates to the
    // first rejection; kDraft proposes again from the new tip.
    //
    // The whole scheme is only worth doing because one K+1-shaped forward costs
    // less than K+1 separate decode forwards.
    void RunWithDrafts(BatchData& d)
    {
        TM_FUNCTION_SCOPE();

        BatchCopy copy;
        TensorMap env{{"batch", d.buf()}, {"copy", copy.buf()}};

        RunCopies(d.restore_copies);

        model_.Run(BatchOp::kPrepare, d.phase, env);
        copy.Run();

        model_.Run(BatchOp::kForward, d.phase, env);

        // Reject, rollback and draft run BEFORE Unprep, not after.
        //
        // All three read tensors that the forward produced into this env --
        // `hidden_states` for drafting, `sequence_length` for rollback -- and
        // Unprep is what tears that state down. Running them afterwards left
        // them reading an env holding only `batch` and `copy`.
        model_.Run(BatchOp::kReject, d.phase, env);
        model_.Run(BatchOp::kRollback, d.phase, env);

        // kDraft is UNCONDITIONAL here, deliberately, while the ordinary path
        // below gates it on CanDraft. That asymmetry looks like an oversight
        // and is not.
        //
        // CanDraft requires every row to be `autoregres`, which keeps a
        // prefill-shaped batch from drafting: the draft submits one row per
        // sequence and would meet attention statistics prepared for a longer
        // prefill. On a verification step autoregres is false BY CONSTRUCTION,
        // because input_len is K+1 rather than 1 -- yet this is exactly the
        // step that must draft again.
        //
        // Gating this call on CanDraft would stop drafting after the first
        // verification, and speculation would run one cycle and then decay to
        // ordinary decoding, quietly and with no error.
        model_.Run(BatchOp::kDraft, d.phase, env);

        model_.Run(BatchOp::kUnprep, d.phase, env);
        copy.Run();

        RunCopies(d.publish_copies);

        AnomalyHandler::instance().Summarize([](...) {});
        AnomalyHandler::instance().Reset();
    }


    /// Ordinary forward. When `draft_after_forward` is set, the first drafts
    /// are proposed at the end of this same env, which is the only place the
    /// `hidden_states` they consume is still alive.
    void Run(BatchData& d, bool draft_after_forward = false)
    {
        TM_FUNCTION_SCOPE();

        BatchCopy copy;
        TensorMap env{{"batch", d.buf()}, {"copy", copy.buf()}};

        // Restore copies first so kPrepare may post-process restored content
        // (a module reset overrides whatever a whole-object restore wrote).
        RunCopies(d.restore_copies);

        // Vision sub-graph runs before the language model in each phase so its
        // env outputs (image embeddings, mrope tensors) are visible downstream.
        if (vision_model_) {
            vision_model_->Run(BatchOp::kPrepare, d.phase, env);
        }
        model_.Run(BatchOp::kPrepare, d.phase, env);
        copy.Run();

        if (vision_model_) {
            vision_model_->Run(BatchOp::kForward, d.phase, env);
        }
        model_.Run(BatchOp::kForward, d.phase, env);

        // Before Unprep, which tears down the forward's env state.
        if (draft_after_forward) {
            model_.Run(BatchOp::kDraft, d.phase, env);
        }

        model_.Run(BatchOp::kUnprep, d.phase, env);
        copy.Run();

        // Publication copies last: kUnprep is the module's final chance to
        // finalize frontier contents before the snapshot.
        RunCopies(d.publish_copies);

        AnomalyHandler::instance().Summarize([](...) {});
        AnomalyHandler::instance().Reset();
    }

    Impl(LanguageModel&                model,
         VisionModel*                  vision_model,
         Context&                      context,
         int                           device_id,
         Queue<unique_ptr<BatchData>>& inbound,
         Queue<unique_ptr<BatchData>>& outbound):
        model_{model},
        vision_model_{vision_model},
        linear_{*context.linear},
        device_id_{device_id},
        inbound_{inbound},
        outbound_{outbound}
    {
    }

    ~Impl()
    {
        if (internal_thread_.joinable()) {
            internal_thread_.join();
        }
    }

    void Start()
    {
        internal_thread_ = std::thread(&Impl::InternalThreadEntry, this);
    }
};

ModelExecutor::~ModelExecutor() = default;

ModelExecutor::ModelExecutor()                         = default;
ModelExecutor::ModelExecutor(ModelExecutor&&) noexcept = default;
ModelExecutor& ModelExecutor::operator=(ModelExecutor&&) noexcept = default;

ModelExecutor::ModelExecutor(LanguageModel&                model,
                             VisionModel*                  vision_model,
                             Context&                      context,
                             int                           device_id,
                             Queue<unique_ptr<BatchData>>& inbound,
                             Queue<unique_ptr<BatchData>>& outbound):
    impl_{std::make_unique<Impl>(model, vision_model, context, device_id, inbound, outbound)}
{
}

void ModelExecutor::Start()
{
    return impl_->Start();
}

}  // namespace turbomind
