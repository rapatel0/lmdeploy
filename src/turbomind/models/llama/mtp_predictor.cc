// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/models/llama/mtp_predictor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>

#include "src/turbomind/core/check.h"
#include "src/turbomind/kernels/argmax.h"
#include "src/turbomind/kernels/norm/rms_norm.h"
#include "src/turbomind/models/decoder_layer_weight.h"
#include "src/turbomind/models/llama/LlamaFfnLayer.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/mtp_weight.h"

namespace turbomind {

MTPPredictor::MTPPredictor(const MTPLayerWeight&  weights,
                           UnifiedAttentionLayer& attn_layer,
                           int                    attn_index,
                           int                    attn_phase_base,
                           const EngineParam&     engine,
                           const Context&         ctx,
                           EmbedFn                embed,
                           Top1Fn                 top1):
    weights_{weights},
    attn_layer_{attn_layer},
    attn_index_{attn_index},
    attn_phase_base_{attn_phase_base},
    hidden_units_{weights.fc ? weights.fc->input_dim / 2 : 0},
    tp_size_{engine.attn_tp_size},
    block_seq_len_{engine.cache_block_seq_len},
    tp_rank_{engine.attn_dp_rank},
    dtype_{engine.data_type},
    linear_{*ctx.linear},
    ctx_{ctx},
    embed_fn_{std::move(embed)},
    top1_fn_{std::move(top1)}
{
    TM_CHECK(embed_fn_) << "MTP predictor needs the target's embedding lookup";
    TM_CHECK(top1_fn_) << "MTP predictor needs the target's greedy lm_head";
    // The fc projection consumes the two normalised halves concatenated, so
    // its input is exactly twice the hidden size. Deriving hidden_units_ from
    // the weight rather than from config keeps the two from disagreeing.
    TM_CHECK_GT(hidden_units_, 0) << "MTP fc weight missing or malformed";
    TM_CHECK_GE(attn_index_, 0) << "MTP predictor built without a KV slot";
    TM_CHECK_GE(attn_phase_base_, 0) << "MTP predictor built without its own attention phases";

    // A MoE MTP layer carries both `moe_ffn` and `feed_forward` (qwen3_5.py
    // assigns the pair). DecodeStep runs only LlamaFfnLayer, which skips the
    // router, the routed experts and the combine. That does not fail -- it
    // produces drafts from the wrong computation, so acceptance would simply
    // be poor and look like a bad draft model. Refuse instead of guessing.
    TM_CHECK(!weights_.decoder_layer->moe_ffn)
        << "MTP: this checkpoint has a MoE draft layer; the predictor runs only the dense FFN, "
           "which would silently draft from the wrong computation";

    // fmt placeholders, not printf: TM_LOG_INFO forwards to fmt::format, so a
    // %d would be printed literally and the value would never appear.
    // The MTP layer in this checkpoint carries a dense mlp.{gate,up,down}_proj
    // rather than an expert set, so the draft path uses LlamaFfnLayer. The two
    // `mtp.layers.0.mlp.gate` names in the config's exclude list have no
    // corresponding tensor on disk, which is consistent with that.
    ffn_layer_ = std::make_unique<LlamaFfnLayer>(ctx);

    TM_LOG_INFO("[MTP] predictor ready: hidden={} attn_index={} tp={}", hidden_units_, attn_index_, tp_size_);
}

MTPPredictor::~MTPPredictor() = default;

void MTPPredictor::SetupAttention(int phase, TensorMap& env)
{
    // Needs `requests`, so this runs from the engine's Setup rather than the
    // executor's forward-time env, which carries only `batch` and `copy`.
    TM_CHECK(env.try_("requests")) << "SetupAttention needs the setup-time env";

    // Build the plan for the DRAFT's shape: one query token per row.
    //
    // Without this the plan is derived from the target's input_len, which is
    // K+1 on a verification step, and the draft's bsz-token forward aborts on
    // `d.prefill.q_sum + d.decode.n == q_count`. The separate phase slot keeps
    // the draft from corrupting the target's plan; this keeps the draft's own
    // plan correct.
    env.produce("attn_decode_shape", Buffer_<int>{1, kCPU});
    attn_layer_.Run(BatchOp::kSetup, attn_phase_base_ + phase, env);
    env.consume("attn_decode_shape");
}

void MTPPredictor::PrepareAttention(int phase, TensorMap& env)
{
    // Separate from SetupAttention because the two ops need different things.
    //
    // kSetup needs `requests`, which exists only in the engine's setup env.
    // kPrepare borrows `finished`, `q_offsets`, `k_offsets` and
    // `readonly_block_num`, none of which exist yet at setup time -- the
    // language model produces them in its own Prepare. Running both together
    // aborted on the missing `finished` key, and did so on the K=0 path too,
    // because Setup runs regardless of whether speculation is active.
    TM_CHECK(env.try_("finished")) << "PrepareAttention needs the prepared env";

    attn_layer_.Run(BatchOp::kPrepare, attn_phase_base_ + phase, env);
}

Tensor MTPPredictor::Project(const Tensor& embedding, const Tensor& hidden_states, int batch_size)
{
    // The ACTIVE stream, not ctx_.stream. ctx_.stream is created once when the
    // model is constructed; each inference batch runs on the stream that
    // ModelExecutor installs. CUDA imposes no ordering between the two, so
    // launching predictor work on the construction stream can read inputs the
    // executor has not finished writing, and can free a temporary while a
    // kernel on the other stream is still using it.
    const auto stream = core::Context::stream().handle();

    // Report the real shapes before asserting on them. Guessing which operand
    // disagreed cost a full build-and-run cycle; the kernel's check names
    // neither tensor nor its extents.
    TM_CHECK(embedding.ndim() == 2 && hidden_states.ndim() == 2)
        << "MTP Project: expected 2D inputs, got embedding.ndim=" << embedding.ndim()
        << " hidden_states.ndim=" << hidden_states.ndim();
    TM_CHECK(embedding.shape(0) == batch_size && embedding.shape(1) == hidden_units_
             && hidden_states.shape(0) == batch_size && hidden_states.shape(1) == hidden_units_)
        << "MTP Project: shape mismatch. batch=" << batch_size << " hidden=" << hidden_units_
        << " embedding=[" << embedding.shape(0) << "," << embedding.shape(1) << "]"
        << " hidden_states=[" << hidden_states.shape(0) << "," << hidden_states.shape(1) << "]";

    // Two independent RMSNorms, one per input. They use their own eps and
    // zero_centered flags: Qwen3.5 is zero-centered, and passing the wrong
    // flag shifts every value by one without failing, so read both from the
    // weight rather than assuming.
    Tensor normed_emb{{batch_size, hidden_units_}, dtype_, kDEVICE};
    invokeRMSNorm(normed_emb,
                  embedding,
                  weights_.pre_fc_norm_embedding->weight,
                  weights_.pre_fc_norm_embedding->norm_eps_,
                  weights_.pre_fc_norm_embedding->zero_centered_,
                  stream);
    TM_CUDA_CHECK(cudaGetLastError());

    Tensor normed_hidden{{batch_size, hidden_units_}, dtype_, kDEVICE};
    invokeRMSNorm(normed_hidden,
                  hidden_states,
                  weights_.pre_fc_norm_hidden->weight,
                  weights_.pre_fc_norm_hidden->norm_eps_,
                  weights_.pre_fc_norm_hidden->zero_centered_,
                  stream);
    TM_CUDA_CHECK(cudaGetLastError());

    // Concatenate along the feature axis into [batch, 2 * hidden].
    //
    // The halves must be laid out per row, not per tensor: fc expects each
    // row to be [emb_row, hidden_row]. Copying one whole tensor after the
    // other would produce [all emb rows, all hidden rows], which is the same
    // bytes in the wrong order and yields silently wrong projections for any
    // batch above one.
    Tensor fused{{batch_size, hidden_units_ * 2}, dtype_, kDEVICE};
    {
        const size_t row_bytes = byte_size(dtype_, hidden_units_);
        auto* dst = static_cast<char*>(fused.raw_data());
        const auto* src_e = static_cast<const char*>(normed_emb.raw_data());
        const auto* src_h = static_cast<const char*>(normed_hidden.raw_data());
        // A strided 2D copy expresses this as two calls instead of 2*batch.
        TM_CUDA_CHECK(cudaMemcpy2DAsync(
            dst, row_bytes * 2, src_e, row_bytes, row_bytes, batch_size, cudaMemcpyDeviceToDevice, stream));
        TM_CUDA_CHECK(cudaMemcpy2DAsync(dst + row_bytes,
                                        row_bytes * 2,
                                        src_h,
                                        row_bytes,
                                        row_bytes,
                                        batch_size,
                                        cudaMemcpyDeviceToDevice,
                                        stream));
    }
    TM_CUDA_CHECK(cudaGetLastError());

    Tensor projected{{batch_size, hidden_units_}, dtype_, kDEVICE};
    linear_.Forward(fused, *weights_.fc, projected);
    TM_CUDA_CHECK(cudaGetLastError());

    return projected;
}

void MTPPredictor::PrefillFill(int                 target_phase,
                               const Tensor&       full_hidden_states,
                               const Buffer_<int>& input_ids,
                               const int*          input_lens,
                               int                 batch_size)
{
    const auto stream    = core::Context::stream().handle();
    const int  token_num = (int)input_ids.size();

    TM_CHECK_GT(batch_size, 0);
    TM_CHECK_EQ((int)full_hidden_states.shape(0), token_num)
        << "MTP PrefillFill: hidden rows must match the submitted tokens";

    // Entry convention, shared with Draft(): the entry at position p is
    // f(embed(token[p]), hidden[p-1]). Draft() writes the sampled token's
    // entry from the tip hidden state -- that is exactly p = S. Here the
    // chunk supplies token[p] directly and hidden[p-1] is the PREVIOUS row
    // of the target's full hidden states, so the hidden half is the chunk
    // shifted down by one row, per row of the batch.
    //
    // Each row's first chunk position has no predecessor hidden in this
    // forward -- it lives in the previous chunk, or nowhere at p = 0. It
    // gets zeros. One degraded entry per chunk per row, against hundreds
    // filled with the trained convention; the alternative of stashing the
    // last hidden row per sequence across chunks is bookkeeping this pass
    // does not need to prove the mechanism.
    Tensor embedding = embed_fn_(input_ids);

    Tensor shifted{{token_num, hidden_units_}, dtype_, kDEVICE};
    {
        const size_t row_bytes = byte_size(dtype_, (size_t)hidden_units_);
        char*        dst       = static_cast<char*>(shifted.raw_data());
        const char*  src       = static_cast<const char*>(full_hidden_states.raw_data());
        int          offset    = 0;
        for (int i = 0; i < batch_size; ++i) {
            const int len = input_lens[i];
            TM_CHECK_GT(len, 0);
            TM_CUDA_CHECK(cudaMemsetAsync(dst + (size_t)offset * row_bytes, 0, row_bytes, stream));
            if (len > 1) {
                TM_CUDA_CHECK(cudaMemcpyAsync(dst + (size_t)(offset + 1) * row_bytes,
                                              src + (size_t)offset * row_bytes,
                                              (size_t)(len - 1) * row_bytes,
                                              cudaMemcpyDeviceToDevice,
                                              stream));
            }
            offset += len;
        }
        TM_CHECK_EQ(offset, token_num) << "MTP PrefillFill: input_lens do not sum to the token count";
    }

    Tensor projected = Project(embedding, shifted, token_num);

    // Norm, then attention against the TARGET's phase plan. The plan already
    // describes this chunk's exact shape -- q_len per row, key lengths, block
    // pointers -- because the target's forward just used it. What routes the
    // KV write to the DRAFT's slot is the MTP attention weight: the cache
    // offset is read from weights.cache_block_offset, stamped at
    // registration. Phase selects the plan; the weight selects the slot.
    //
    // The pipeline stops here. K and V are projections of this layer's
    // INPUT, so once the attention call has written them the FFN, the final
    // norm and the lm_head would only compute an output nothing reads.
    const auto& layer = *weights_.decoder_layer;

    invokeRMSNorm(projected,
                  projected,
                  layer.attention_norm->weight,
                  layer.attention_norm->norm_eps_,
                  layer.attention_norm->zero_centered_,
                  stream);
    TM_CUDA_CHECK(cudaGetLastError());

    attn_layer_.Forward({target_phase, projected, projected, layer.attention.get(), attn_index_});
    TM_CUDA_CHECK(cudaGetLastError());

    // A fault here must name this path, not surface later inside the target's
    // next KV write with a filename that sends the search elsewhere. This is
    // a prefill-rate cost on prompt chunks only, not on decode steps.
    {
        const auto err = cudaStreamSynchronize(stream);
        TM_CHECK_EQ(err, cudaSuccess) << "MTP prefill fill faulted: " << cudaGetErrorString(err)
                                      << " (batch=" << batch_size << " token_num=" << token_num << ")";
    }
}

Tensor MTPPredictor::DecodeStep(Tensor hidden, int phase, TensorMap& env)
{
    // The executor's active stream; see the note in Project().
    const auto  stream = core::Context::stream().handle();
    const auto& layer  = *weights_.decoder_layer;

    const int batch_size = hidden.shape(0);

    // Pre-attention norm, then attention. The residual is kept explicitly:
    // the attention layer writes its output in place, so the input has to be
    // preserved before the call rather than recovered after it.
    Tensor residual{{batch_size, hidden_units_}, dtype_, kDEVICE};
    TM_CUDA_CHECK(cudaMemcpyAsync(residual.raw_data(),
                                  hidden.raw_data(),
                                  byte_size(dtype_, (size_t)batch_size * hidden_units_),
                                  cudaMemcpyDeviceToDevice,
                                  stream));

    invokeRMSNorm(hidden,
                  hidden,
                  layer.attention_norm->weight,
                  layer.attention_norm->norm_eps_,
                  layer.attention_norm->zero_centered_,
                  stream);
    TM_CUDA_CHECK(cudaGetLastError());

    // Passing the MTP layer's own AttentionWeight is what routes the keys and
    // values into the MTP KV slot: the cache offset is read from
    // `weights.cache_block_offset`, which registration stamped onto this very
    // weight. `attn_index_` only names the layer for debug output.
    // The draft's slot for THIS batch phase, not the target's slot.
    attn_layer_.Forward({attn_phase_base_ + phase, hidden, hidden, layer.attention.get(), attn_index_});
    TM_CUDA_CHECK(cudaGetLastError());

    // Fused residual-add plus the pre-FFN norm. This writes the post-add value
    // back into `residual` and the normalised value into `hidden`, so both are
    // carried forward correctly for the next residual.
    invokeResidualBiasRMSNorm(hidden.raw_data(),
                              residual.raw_data(),
                              layer.ffn_norm->weight.raw_data(),
                              nullptr,
                              dtype_,
                              hidden_units_,
                              batch_size,
                              layer.ffn_norm->norm_eps_,
                              layer.ffn_norm->zero_centered_,
                              stream);
    TM_CUDA_CHECK(cudaGetLastError());

    ffn_layer_->forward({hidden, hidden, layer.feed_forward.get(), attn_index_});
    TM_CUDA_CHECK(cudaGetLastError());

    // Final residual add followed by the MTP block's own final_norm, leaving
    // `hidden` ready for the shared lm_head.
    invokeResidualBiasRMSNorm(hidden.raw_data(),
                              residual.raw_data(),
                              weights_.final_norm->weight.raw_data(),
                              nullptr,
                              dtype_,
                              hidden_units_,
                              batch_size,
                              weights_.final_norm->norm_eps_,
                              weights_.final_norm->zero_centered_,
                              stream);
    TM_CUDA_CHECK(cudaGetLastError());

    return hidden;
}

MTPPredictor::DraftResult MTPPredictor::Draft(int                 batch_size,
                                             const Tensor&       hidden_states,
                                             const Buffer_<int>& last_tokens,
                                             int                 num_draft_tokens,
                                             int                 phase,
                                             const int*          seq_lens,
                                             const int*          block_counts,
                                             TensorMap&          env)
{
    TM_CHECK_GT(batch_size, 0);
    TM_CHECK_GT(num_draft_tokens, 0);
    // `last_tokens` must already be sliced to the batch. Its source buffer is
    // allocated at max_batch_size and reused, so an unsliced view carries the
    // capacity instead of this step's batch, and the mismatch would otherwise
    // surface as a shape error deep inside an RMSNorm.
    TM_CHECK_EQ((int)last_tokens.size(), batch_size) << "MTP Draft: last_tokens must be sliced to the batch";

    // The executor's active stream; see the note in Project().
    const auto stream = core::Context::stream().handle();

    DraftResult result;
    // Buffer_ takes a signed ssize_t extent.
    result.draft_tokens = Buffer_<int>{(ssize_t)batch_size * num_draft_tokens, kDEVICE};
    result.num_drafts   = 0;

    // The checkpoint ships a single MTP layer, so depth beyond one is reached
    // by running that layer again with its own previous output: the token it
    // just drafted becomes the next input, and the hidden state it produced
    // becomes the next context. Each pass appends one entry to the MTP KV
    // slot, exactly as a real decode step would.
    Buffer_<int> cur_tokens = last_tokens;
    Tensor       cur_hidden = hidden_states;

    // The KV positions the draft attends to and writes.
    //
    // `k_offsets` is the target's cumulative key-length array, produced fresh
    // each Forward and consumed by attention as `cu_k_len`. The attention layer
    // BORROWS it, so mutating this buffer changes what the draft attends to.
    //
    // Without advancing it, every draft step wrote its K/V to the same position
    // and read a history ending at the target's last real token: step 2
    // attended as though step 1 had never happened. Measured, that produced
    // 56.2% / 16.1% / 3.3% / 0.0% across four steps -- step 1 correct by
    // accident, everything after it blind.
    //
    // Safe to mutate in place because this runs at the END of Forward and the
    // buffer is rebuilt by PrefixSum at the start of the next one. It is
    // restored below regardless, so nothing downstream inherits drafted
    // positions.
    Buffer_<int> k_offsets = env.at("k_offsets").buffer().view<int>();
    int          advanced  = 0;

    // Diagnostic probe: shift the base of every drafted position by a constant.
    //
    // This answers a question the acceptance rate cannot. Giving the sampled
    // token its own position (a uniform +1) left steps 1-3 bit-identical over
    // 64 greedy samples, which is only possible if the entries BELOW the base
    // contribute something exactly invariant to the shift. Zeros do that;
    // arbitrary leftover bytes do not.
    //
    // So shifting the base by a large constant discriminates directly:
    //   zeros   -> still bit-identical, the count of them cannot matter
    //   garbage -> lands on different bytes, output moves
    //
    // Shift NEGATIVE, never positive.
    //
    // The first attempt at this probe used +32 and died with an illegal memory
    // access in argmax: forward slack is only 1..63 and usually below 32, so
    // the base crossed the block boundary and dereferenced a block pointer
    // past the end of the row's allocation. That is precisely the hazard
    // max_extend exists to prevent, and the probe bypassed it deliberately --
    // so the crash confirmed the guard rather than revealing a new defect, but
    // it discriminated nothing because no clean run happened.
    //
    // Shifting backward stays inside memory the row already owns: positions
    // below the base are within the same allocated blocks, being the target's
    // own earlier KV region. So a negative shift is bounded by seq_len rather
    // than by block slack, and it answers the same question -- if the entries
    // under the base are zeros, moving the base among them cannot change a
    // single argmax.
    //
    // Clamped so the base cannot go below 1.
    static const int probe_shift = [] {
        const char* s = std::getenv("TM_MTP_PROBE_SHIFT");
        return s ? std::atoi(s) : 0;
    }();
    if (probe_shift < 0) {
        int shift = probe_shift;
        for (int i = 0; i < batch_size; ++i) {
            shift = std::max(shift, 1 - seq_lens[i]);
        }
        if (shift < 0) {
            AdvanceCuSeqLens(k_offsets.data(), batch_size, shift, stream);
            advanced += shift;
        }
    }

    // How far the drafted positions may safely walk.
    //
    // The scheduler allocates KV blocks for `seq_len`, not `seq_len + K`. The
    // block iterator indexes `block_ptrs_[block_id_]` with no bounds check, so
    // advancing past the last allocated block reads a pointer belonging to the
    // next sequence -- or past the end of the array for the final row. That is
    // silent cross-sequence corruption, not a crash.
    //
    // Only the slack inside each row's last block is safe, and only the
    // smallest such slack across the batch, since one array bounds all rows.
    // A sequence sitting exactly on a boundary has zero slack, and then no
    // advance is permitted at all.
    // K advances, not K-1: the sampled token needs a position too.
    //
    // `k_offsets` is a PrefixSum over `sequence_length_` taken at
    // language_model.cc:615, which runs BEFORE sampling increments
    // sequence_length (sampling_kernels.cu:64). So on entry it describes a key
    // length of L covering positions 0..L-1, while the token being drafted
    // FROM -- the one sampling just produced -- belongs at position L.
    //
    // With q_len=1 attention derives history_len = cu_k_len - q_len = L-1 and
    // writes there, on top of the target's own last token, and every later
    // step inherits the shift.
    //
    // Correcting this changed nothing: steps 1-3 stayed bit-identical over 64
    // greedy samples. That is measured, not predicted, and it means the
    // absolute base does not currently matter -- see the probe below. The fix
    // is kept because it is correct and must precede any seeding of real
    // history, which would otherwise be written to shifted positions.
    // Bound by ALLOCATED capacity, not by slack in the last block.
    //
    // The old rule was `block_len - (seq_len % block_len)`, i.e. only the room
    // left inside the current block, and zero when a row sits exactly on a
    // boundary. That was right while the scheduler allocated for seq_len alone.
    //
    // The reservation now covers seq_len + inflight_new_tokens, so a row that
    // already carries drafts has whole blocks allocated beyond its tip. Capping
    // at the block remainder throws that away: roughly one step in sixteen
    // drafts nothing at all, and three more draft fewer than K, purely because
    // of where the tip happens to fall.
    //
    // The hazard the old rule guarded against is real -- the block iterator
    // indexes block_ptrs_ with no bounds check, so walking past the last
    // allocated block reads another sequence's pointer, silently. So the cap
    // stays; it is now derived from how many blocks the row actually owns.
    static const bool frozen_kv = [] {
        const char* value = std::getenv("TM_MTP_FROZEN_KV");
        return value && value[0] == '1';
    }();

    const int block_len  = block_seq_len_;
    int       max_extend = num_draft_tokens;
    if (block_len > 0 && block_counts) {
        for (int i = 0; i < batch_size; ++i) {
            const int capacity = block_counts[i] * block_len;
            const int slack    = capacity - seq_lens[i];
            // SGLang's NEXTN worker keeps the committed KV prefix frozen while
            // walking the proposal chain. Every draft then reuses one scratch
            // position instead of appending speculative K/V that training did
            // not commit. Keep this as an A/B flag until acceptance proves it.
            max_extend = std::min(max_extend, frozen_kv ? (slack > 0 ? num_draft_tokens : 0) : slack);
        }
        max_extend = std::max(max_extend, 0);
    }
    else if (block_len > 0) {
        // No block counts supplied: fall back to the conservative rule.
        for (int i = 0; i < batch_size; ++i) {
            const int len   = seq_lens[i];
            const int slack = (len % block_len == 0) ? 0 : block_len - (len % block_len);
            max_extend      = std::min(max_extend, slack);
        }
    }
    else {
        max_extend = 0;
    }
    // Report the cap whenever it CHANGES, not once ever.
    //
    // The once-only latch made this warning actively misleading, and I misread
    // it. The verify prompt is 63 tokens, so the first draft runs at seq_len
    // 64 -- exactly on a 64-token block boundary, where slack is 0. The
    // warning fired "limited to 0 of 4" on that first step and then suppressed
    // itself for the whole run, while max_extend was in fact 4 for every step
    // after it. Read as a steady state it says the advance never happens. The
    // truth is that it fails only on the first step.
    if (max_extend != last_warned_extend_) {
        last_warned_extend_ = max_extend;
        TM_LOG_WARNING("[MTP] drafted positions limited to {} of {} steps by KV block capacity "
                       "(seq_len[0]={}); steps beyond that reuse the last valid position",
                       max_extend,
                       num_draft_tokens,
                       batch_size > 0 ? seq_lens[0] : -1);
    }

    // Produce only as many drafts as there are distinct positions for.
    //
    // The loop used to run num_draft_tokens times regardless, and simply stop
    // advancing the key length once max_extend was reached. Every step past
    // that point attended to identical history and wrote the same KV slot, so
    // it was not a prediction of its position at all -- with max_extend=2 and
    // K=4 the positions were [1,2,2,2].
    //
    // Those drafts are harmless to correctness, since the verifier compares
    // against the target's own argmax and rejects whatever disagrees. But they
    // are nearly always rejected, and each one still consumes a slot in the
    // K+1 verification forward. Emitting fewer, real drafts is strictly better
    // than padding with duplicates.
    // Assert the walk stays inside every row's allocation.
    //
    // The draft advances k_offsets once per step and writes MTP KV at each
    // position. The block iterator does not bounds-check block_ptrs_, so
    // overrunning reads a pointer belonging to another sequence and surfaces
    // far away as
    //
    //   kv_cache_utils_v2.cu: CUDA error: an illegal memory access
    //
    // naming neither the row nor the amount. max_extend is supposed to prevent
    // that; this states the invariant it is supposed to enforce, so a wrong
    // bound fails here with the numbers attached instead of in a kernel.
    if (block_len > 0 && block_counts) {
        for (int i = 0; i < batch_size; ++i) {
            const int reach = seq_lens[i]
                              + (frozen_kv ? (max_extend > 0 ? 1 : 0)
                                           : std::min(num_draft_tokens, max_extend));
            const int capacity = block_counts[i] * block_len;
            TM_CHECK_LE(reach, capacity)
                << "MTP draft row " << i << " would reach key length " << reach << " but owns "
                << block_counts[i] << " blocks of " << block_len << " = " << capacity
                << " (seq_len=" << seq_lens[i] << " max_extend=" << max_extend
                << " num_draft_tokens=" << num_draft_tokens << ")";
        }
    }

    const int effective_drafts = std::min(num_draft_tokens, max_extend);
    if (effective_drafts <= 0) {
        // Restore before leaving. The debug probe above can already have moved
        // the offsets, and returning early without undoing that would leave a
        // mutated k_offsets behind for the next reader -- who would have no
        // reason to suspect the draft path.
        if (advanced) {
            AdvanceCuSeqLens(k_offsets.data(), batch_size, -advanced, stream);
        }
        return {};
    }

    // These synchronizations are fault-attribution instrumentation, not part
    // of inference. Leaving them unconditional serializes the autoregressive
    // draft loop and invalidates every performance result.
    static const bool sync_check = [] {
        const char* s = std::getenv("TM_MTP_SYNC_CHECK");
        return s && s[0] == '1';
    }();
    if (TM_UNLIKELY(sync_check)) {
        const auto err = cudaStreamSynchronize(stream);
        TM_CHECK_EQ(err, cudaSuccess)
            << "MTP draft faulted BEFORE its first step: " << cudaGetErrorString(err)
            << " (batch=" << batch_size << " seq_len[0]=" << (batch_size > 0 ? seq_lens[0] : -1)
            << " effective_drafts=" << effective_drafts << ")";
    }

    for (int step = 0; step < effective_drafts; ++step) {
        // Step 0 advances too, placing the sampled token at L instead of L-1.
        // The frozen-KV control deliberately reuses that scratch position on
        // later steps, matching SGLang NEXTN's proposal walk.
        if (step == 0 || !frozen_kv) {
            AdvanceCuSeqLens(k_offsets.data(), batch_size, 1, stream);
            ++advanced;
        }

        // Embedding of the previous token, through the target's shared table.
        Tensor embedding = embed_fn_(cur_tokens);

        Tensor projected = Project(embedding, cur_hidden, batch_size);

        // Decode phase: the draft always extends one token per sequence.
        Tensor out = DecodeStep(std::move(projected), phase, env);

        // Write this step's tokens as the contiguous run for step `step`.
        //
        // The layout is [step][batch], not [batch][step]: one argmax per
        // sequence is produced per iteration, so a step is contiguous and a
        // sequence is strided by `batch_size`. The header documents this, and
        // the verifier must apply the stride when reading a single sequence's
        // drafts in order.
        Buffer_<int> step_out{
            result.draft_tokens.data() + (ssize_t)step * batch_size, (ssize_t)batch_size, kDEVICE};
        top1_fn_(step_out, out);

        result.num_drafts = step + 1;

        // Feed forward for the next iteration.
        cur_tokens = step_out;
        cur_hidden = std::move(out);

        if (TM_UNLIKELY(sync_check)) {
            const auto err = cudaStreamSynchronize(stream);
            TM_CHECK_EQ(err, cudaSuccess)
                << "MTP draft step " << step << " of " << effective_drafts << " faulted: "
                << cudaGetErrorString(err) << " (batch=" << batch_size << " seq_len[0]="
                << (batch_size > 0 ? seq_lens[0] : -1)
                << " blocks[0]=" << (block_counts ? block_counts[0] : -1)
                << " max_extend=" << max_extend << " advanced=" << advanced << ")";
        }
    }

    // Restore the target's offsets. The next Forward recomputes them anyway,
    // but leaving a mutated buffer behind would make any future reader of
    // k_offsets after this point silently wrong, and that reader would have no
    // reason to suspect the draft.
    if (advanced) {
        AdvanceCuSeqLens(k_offsets.data(), batch_size, -advanced, stream);
    }

    return result;
}

}  // namespace turbomind
