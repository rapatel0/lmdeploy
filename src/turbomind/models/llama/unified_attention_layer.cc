/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Modified from
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/attention_layers/GptContextAttentionLayer.cc

#include "src/turbomind/engine/block.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <math.h>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

#include "src/turbomind/core/allocator.h"
#include "src/turbomind/core/check.h"
#include "src/turbomind/core/context.h"
#include "src/turbomind/core/core.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/tensor.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/request.h"

#include "src/turbomind/kernels/attention/attention.h"
#include "src/turbomind/kernels/attention/decoding.h"
#include "src/turbomind/kernels/attention/kv_cache_utils_v2.h"
#include "src/turbomind/kernels/norm/rms_norm.h"

#include "src/turbomind/macro.h"

#include "src/turbomind/kernels/attention/block.h"
#include "src/turbomind/memory/object.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/llama/llama_kernels.h"
#include "src/turbomind/models/llama/llama_rope.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/models/llama/mla_utils.h"
#include "src/turbomind/models/llama/unified_attention_layer.h"
#include "src/turbomind/models/llama/dflash_kernels.h"

#include "src/turbomind/core/logger.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/utils/anomaly_handler.h"
#include "src/turbomind/utils/cuda_utils.h"

// #include "dbg.h"

namespace turbomind {

namespace {
// clang-format off
struct BlockConfig {
    int head_dim_;
    int head_num_;
    int block_len_;
    int t_bits_;
    int q_bits_;
    bool share_kv_;
    int t_bits() const { return t_bits_; }
    int q_bits() const { return q_bits_; }
    int head_dim() const { return head_dim_; }
    int head_num() const { return head_num_; }
    int block_len() const { return block_len_; }
    bool is_share_kv() const { return share_kv_; }
    auto as_tuple() const noexcept {
        return std::tie(head_dim_, head_num_, block_len_, t_bits_, q_bits_, share_kv_);
    }
    friend bool operator==(const BlockConfig& a, const BlockConfig& b) {
        return a.as_tuple() == b.as_tuple();
    }
};
// clang-format on

bool ClaimGroupedQ8Parity(int device, int layer_id)
{
    if (!std::getenv("TM_DFLASH_GROUPED_PAGED_Q8_PARITY_DIR")) {
        return false;
    }
    static std::mutex                    mutex;
    static std::set<std::pair<int, int>> captured;
    std::lock_guard<std::mutex>          lock(mutex);
    return captured.emplace(device, layer_id).second;
}

void CheckGroupedQ8Parity(const Tensor& reference,
                          const Tensor& grouped,
                          int           layer_id,
                          int           context_len,
                          cudaStream_t  stream)
{
    TM_CHECK(reference.shape() == grouped.shape());
    TM_CHECK_EQ(reference.dtype(), kHalf);
    TM_CHECK_EQ(grouped.dtype(), kHalf);

    int device{};
    TM_CUDA_CHECK(cudaGetDevice(&device));
    const char* output_dir = TM_CHECK_NOTNULL(std::getenv("TM_DFLASH_GROUPED_PAGED_Q8_PARITY_DIR"));

    std::vector<half> reference_host(reference.size());
    std::vector<half> grouped_host(grouped.size());
    TM_CUDA_CHECK(cudaMemcpyAsync(reference_host.data(),
                                  reference.raw_data(),
                                  reference.byte_size(),
                                  cudaMemcpyDeviceToHost,
                                  stream));
    TM_CUDA_CHECK(cudaMemcpyAsync(
        grouped_host.data(), grouped.raw_data(), grouped.byte_size(), cudaMemcpyDeviceToHost, stream));
    TM_CUDA_CHECK(cudaStreamSynchronize(stream));

    bool   finite = true;
    double squared_error{};
    float  max_abs{};
    for (size_t i = 0; i < reference_host.size(); ++i) {
        const float expected = __half2float(reference_host[i]);
        const float actual   = __half2float(grouped_host[i]);
        finite               = finite && std::isfinite(expected) && std::isfinite(actual);
        const float error    = std::abs(actual - expected);
        max_abs              = std::max(max_abs, error);
        squared_error += (double)error * error;
    }
    const double rms = std::sqrt(squared_error / reference_host.size());

    constexpr float  kMaxAbsLimit = .125f;
    constexpr double kRmsLimit    = .01;
    const bool pass = finite && max_abs <= kMaxAbsLimit && rms <= kRmsLimit;
    const int  tiles = (context_len + 63) / 64;
    const int  splits = std::min(8, std::max(1, tiles));

    const std::string path = std::string(output_dir) + "/grouped-q8h4-pid-" + std::to_string(getpid())
                             + "-device-" + std::to_string(device) + ".jsonl";
    {
        static std::mutex       output_mutex;
        std::lock_guard<std::mutex> lock(output_mutex);
        std::ofstream out(path, std::ios::app);
        TM_CHECK(out) << "failed to open grouped Q8 parity report " << path;
        out << "{\"device\":" << device << ",\"layer\":" << layer_id << ",\"queries\":8"
            << ",\"local_q_heads\":6,\"local_kv_heads\":1,\"cta_q\":8,\"cta_h\":4"
            << ",\"base_ctas\":2,\"splits\":" << splits << ",\"launched_ctas\":" << 2 * splits
            << ",\"context_len\":" << context_len << ",\"elements\":" << reference_host.size()
            << ",\"finite\":" << (finite ? "true" : "false") << ",\"max_abs\":" << max_abs
            << ",\"rms\":" << rms << ",\"max_abs_limit\":" << kMaxAbsLimit
            << ",\"rms_limit\":" << kRmsLimit << ",\"pass\":" << (pass ? "true" : "false") << "}\n";
    }

    TM_LOG_INFO("[DFlash2] GROUPED_Q8H4_ACTIVE device={} layer={} context={} base_ctas=2 splits={} "
                "launched_ctas={} max_abs={} rms={} pass={}",
                device,
                layer_id,
                context_len,
                splits,
                2 * splits,
                max_abs,
                rms,
                pass);
    TM_CHECK(pass) << "grouped Q8H4 attention parity failed on device " << device << " layer " << layer_id
                   << ": finite=" << finite << " max_abs=" << max_abs << " rms=" << rms;
}

}  // namespace

struct AttentionData {
    struct Stat {
        int n;
        int q_sum;
        int q_max;
        int k_sum;
        int k_max;
    } decode, prefill;

    Buffer_<void*> block_ptrs;
    Buffer_<int>   block_ptrs_offsets;

    // Host staging for the device buffers above, PER PHASE.
    //
    // These were single buffers on the layer, shared by every phase. That was
    // a use-after-overwrite: BatchCopy records the source POINTER and defers
    // the read to stream execution (cuMemcpyBatchAsync with
    // SRC_ACCESS_ORDER_STREAM), so "fill staging, enqueue copy" followed by
    // another phase's "fill staging, enqueue copy" lets the second host fill
    // race the first stream read. The draft's SetupAttention refilled the
    // staging while the target's copy was still queued, and the target then
    // attended through the draft's block pointers.
    //
    // CUDA_LAUNCH_BLOCKING=1 made the crash vanish -- the drain after every
    // launch closed the window -- which is what identified this as an
    // ordering race rather than a bad address. Host staging must follow the
    // same rule as every other per-step value: one slot per in-flight phase.
    Buffer_<void*> block_ptrs_host;
    Buffer_<int>   block_ptrs_offsets_host;
    Buffer_<int>   decode_q_offsets_host;
    Buffer_<float> rope_base_host;

    Buffer_<float> rope_base;

    Buffer_<int> mrope_position_ids;
    Buffer_<int> mrope_position_delta;
    Buffer_<int> mrope_position_offsets;
    Buffer_<int> mrope_length;

    // borrowed from env
    Buffer_<bool> finished;
    Buffer_<int>  q_offsets;
    /// Cumulative one-query-per-row offsets for the MTP draft, and whether this
    /// phase slot is drafting. See the note in Setup.
    Buffer_<int>  decode_q_offsets;
    /// All-true per-row validity mask for the MTP draft; see the note in
    /// kPrepare. Length is the batch size, not the token count.
    Buffer_<bool> decode_token_mask;
    bool          decode_shape{false};
    int           draft_block_size{0};
    /// Row 0's draft count at Setup. Diagnostics only: a prompt's final chunk
    /// can be as short as a verification forward, so token counts alone do not
    /// identify the call.
    int              num_drafts0{-1};
    std::vector<int> draft_k_lens_host;
    bool             assert_draft_metadata{};
    Buffer_<int>     k_offsets;
    Buffer_<int>     readonly_block_num;  // per-request, batch order

    // Global per-token validity mask, owned by LanguageModel and borrowed here; the
    // content is only built at Forward time. Consumed by the attention reduce.
    Buffer_<bool> token_mask;
    int           token_mask_base = 0;  // this DP rank's token offset within the global mask

    struct DFlashWorkspace {
        Tensor qkv;
        Tensor attention;
        Tensor flattened_kv;
        int    max_tokens{};
        int    max_k_sum{};
        int    qkv_width{};
        int    attention_width{};
        int    local_kv_heads{};
        int    head_dim{};
        bool   initialized{};
        bool   traced{};
    } dflash_workspace;

    // int dbg_offset;
    // int dbg_size;
};

UnifiedAttentionLayer::~UnifiedAttentionLayer()
{

    TM_CUDA_CHECK(cudaEventDestroy(aux_event_));
    TM_CUDA_CHECK(cudaEventDestroy(qkv_event_));
    TM_CUDA_CHECK(cudaStreamDestroy(aux_stream_));

    aux_event_ = qkv_event_ = {};
    aux_stream_             = {};
}

void UnifiedAttentionLayer::PoisonVerifierSuffix(int        phase,
                                                 const int* begin_positions,
                                                 const int* end_positions,
                                                 int        row_count,
                                                 int        target_attention_count)
{
    TM_CHECK(phase >= 0 && phase < (int)data_.size());
    TM_CHECK(row_count >= 0);
    TM_CHECK(target_attention_count >= 0 && target_attention_count <= (int)weights_.size());
    auto& d = *data_.at(phase);
    TM_CHECK(row_count + 1 <= d.block_ptrs_offsets_host.size());

    const int dtype_bits = byte_size(engine_param_.data_type, 8);
    TM_CHECK_EQ(quant_policy_, 0) << "DFlash KV poison supports only the audited unquantized cache";
    const auto stream = core::Context::stream().handle();
    size_t     poisoned_positions{};
    size_t     poisoned_bytes{};

    for (int row = 0; row < row_count; ++row) {
        TM_CHECK_GE(begin_positions[row], 0);
        TM_CHECK_GE(end_positions[row], begin_positions[row]);
        for (int pos = begin_positions[row]; pos < end_positions[row]; ++pos) {
            const int block_index = pos / engine_param_.cache_block_seq_len;
            const int block_token = pos % engine_param_.cache_block_seq_len;
            const int pointer_index = d.block_ptrs_offsets_host[row] + block_index;
            TM_CHECK_LT(pointer_index, d.block_ptrs_offsets_host[row + 1]);
            auto* block_ptr = static_cast<char*>(d.block_ptrs_host[pointer_index]);
            TM_CHECK_NOTNULL(block_ptr);

            for (int layer = 0; layer < target_attention_count; ++layer) {
                const auto& weight = *weights_.at(layer);
                BlockConfig config{weight.head_dim,
                                   weight.kv_head_num / weight.tp_size,
                                   engine_param_.cache_block_seq_len,
                                   0,
                                   dtype_bits,
                                   weight.head_dim == 576};
                block::Layout layout{config};
                TM_CHECK(!layout.is_share_kv());
                const size_t token_bytes = layout.token_data_size();
                for (int head = 0; head < config.head_num(); ++head) {
                    char* layer_ptr = block_ptr + weight.cache_block_offset;
                    TM_CUDA_CHECK(cudaMemsetAsync(layer_ptr + layout.k_data(head, block_token),
                                                 0xA5,
                                                 token_bytes,
                                                 stream));
                    TM_CUDA_CHECK(cudaMemsetAsync(layer_ptr + layout.v_data(head, block_token),
                                                 0x5A,
                                                 token_bytes,
                                                 stream));
                    poisoned_bytes += 2 * token_bytes;
                }
            }
            ++poisoned_positions;
        }
    }
    TM_CUDA_CHECK(cudaGetLastError());

    int device = -1;
    TM_CUDA_CHECK(cudaGetDevice(&device));
    static std::atomic<bool> logged[16]{};
    if (poisoned_positions && device >= 0 && device < 16
        && !logged[device].exchange(true, std::memory_order_relaxed)) {
        TM_LOG_INFO("DFLASH_REJECTED_KV_POISON_ACTIVE device={} rows={} positions={} bytes={} target_attention_layers={}",
                    device,
                    row_count,
                    poisoned_positions,
                    poisoned_bytes,
                    target_attention_count);
    }
}

UnifiedAttentionLayer::UnifiedAttentionLayer(std::vector<AttentionWeight*> weights,
                                             CacheRegistry&                registry,
                                             const EngineParam&            engine,
                                             const Context&                context,
                                             int                           phases):
    quant_policy_{engine.quant_policy},
    rope_{weights.at(0)->rope},
    engine_param_{engine},
    cp_fn_ctx_{context.comm.d_comm, context.comm.d_cp_group, engine.attn_cp_size},
    is_warm_up_{*context.is_warm_up},
    context_{context},
    linear_(*context.linear),
    arch_{getSMVersion()},
    weights_{weights}
{
    TM_CHECK_GE(weights.size(), 1);

    const auto dtype = engine.data_type;

    const int dtype_bits = byte_size(dtype, 8);
    const int qaunt_bits = quant_policy_ ? quant_policy_ : dtype_bits;

    auto get_block_config = [&](const AttentionWeight& w) {
        BlockConfig b{w.head_dim,
                      w.kv_head_num / w.tp_size,
                      engine.cache_block_seq_len,
                      dtype_bits == qaunt_bits ? 0 : dtype_bits,
                      qaunt_bits,
                      w.head_dim == 576};
        return b;
    };

    size_t offset = 0;  // byte size (quantization aware)
    for (int i = 0; i < weights.size(); ++i) {
        block::Layout layout{get_block_config(*weights[i])};
        weights[i]->cache_block_offset = offset;
        offset += layout.layer_size();
    }

    const size_t cache_block_byte_size = offset;

    // Record what each layer's KV region costs and where the last one ends.
    //
    // The prefill KV write faults with an illegal access while every
    // block-COUNT check passes, so the remaining suspect is the byte offset
    // WITHIN a block. cache_block_offset accumulates per layer, and the MTP
    // layer is appended last -- if the registry reserved a size that excludes
    // it, every target layer would still fit and only the final region would
    // run past the end.
    //
    // This states the sizes so that hypothesis is confirmed or dismissed from
    // the log rather than argued from the source.
    TM_LOG_INFO("[kv] block layout: {} layers, {} bytes/block, last layer offset {}",
                weights.size(),
                cache_block_byte_size,
                weights.empty() ? size_t{0} : weights.back()->cache_block_offset);
    prefix_cache_offset_               = registry.prefix().Register(cache_block_byte_size, /*alignment=*/1);

    // Room for the draft's extra blocks, and a margin.
    //
    // EnsureBlocks now allocates ceil((seq_len + inflight_new_tokens +
    // num_drafts) / block_len) blocks per row, so a sequence near session_len
    // owns MORE than ceil(session_len / block_len). Sizing this staging buffer
    // from session_len alone lets Setup's unbounded `*blocks++` walk off the
    // end -- a host buffer overrun that then uploads whatever it read as device
    // pointers.
    //
    // The device-side d->block_ptrs already carries a +16 margin; this now
    // matches it, so both are bounded by the same number.
    const auto blocks_per_seq =
        cdiv(engine.session_len, engine.cache_block_seq_len) + cdiv(engine.num_draft_tokens + 1, engine.cache_block_seq_len) + 1;
    const auto max_block_num = engine.max_batch_size * blocks_per_seq;

    TM_CUDA_CHECK(cudaStreamCreateWithFlags(&aux_stream_, cudaStreamNonBlocking));
    TM_CUDA_CHECK(cudaEventCreateWithFlags(&qkv_event_, cudaEventDisableTiming));
    TM_CUDA_CHECK(cudaEventCreateWithFlags(&aux_event_, cudaEventDisableTiming));

    init_rope_kernel_param(rope_, rope_param_);
    for (const auto* weight : weights) {
        RopeKernelParam param{};
        init_rope_kernel_param(weight->rope, param);
        rope_params_.emplace(weight, param);
    }

    const int bsz = engine.max_batch_size;
    const int max_draft_block = engine.speculative_algorithm == "dflash2" ?
                                    engine.speculative_dflash_block_size :
                                    std::max(1, engine.num_draft_tokens + 1);

    if (rope_param_.mrope_mode != MropeMode::kNone) {
        mrope_default_buf_ = Buffer_<int>{std::max(bsz, 3), kDEVICE};
        Clear(mrope_default_buf_);
    }
    // Same bound as the host staging buffer above, including the draft's extra
    // blocks. These two must agree: Setup fills the host buffer and copies
    // `offsets[bsz]` entries into this one, so a device buffer sized from a
    // smaller expression would truncate or overrun the upload.
    const int max_blocks = (int)max_block_num;
    for (int i = 0; i < phases; ++i) {
        auto& d               = data_.emplace_back(std::make_shared<AttentionData>());
        d->block_ptrs         = {max_blocks + 16, kDEVICE};
        d->block_ptrs_offsets = {bsz + 1, kDEVICE};
        d->decode_q_offsets  = {bsz + 1, kDEVICE};
        d->decode_token_mask = {(ssize_t)bsz * max_draft_block, kDEVICE};
        // Host staging is per phase for the same reason the device buffers
        // are: a copy enqueued from one phase's Setup may not have executed
        // when another phase's Setup runs on the host. See AttentionData.
        d->block_ptrs_host         = {max_blocks, kCPUpinned};
        d->block_ptrs_offsets_host = {bsz + 1, kCPUpinned};
        d->decode_q_offsets_host   = {bsz + 1, kCPUpinned};
        if (rope_param_.type == RopeType::kDynamic) {
            d->rope_base_host = {bsz + 1, kCPUpinned};
            d->rope_base      = {bsz + 1, kDEVICE};
        }
    }

    // The draft's validity mask is constant: every row's single query is valid,
    // because rows that finished this step are excluded from drafting upstream
    // by skip_draft. Fill it once here rather than rebuilding it per step.
    {
        const int mask_size = bsz * max_draft_block;
        Buffer_<bool> ones{mask_size, kCPUpinned};
        std::fill_n(ones.data(), mask_size, true);
        for (auto& d : data_) {
            core::Copy(ones, mask_size, d->decode_token_mask);
        }
    }

    // Eagerly initialize workspace buffers (was previously lazy in Init())
    {
        const auto& w              = *weights[0];
        const int   tp_size        = w.tp_size;
        const int   local_head_num = w.head_num / tp_size;
        const int   size_per_head  = w.head_dim;

        TM_CHECK_EQ(w.head_num % tp_size, 0) << w.head_num << " " << tp_size;
        TM_CHECK_EQ(w.head_num % w.kv_head_num, 0) << w.head_num << " " << w.kv_head_num;

        ssize_t   workspace_tokens = kMaxWorkspaceTokens;
        Allocator alloc            = core::Context::device_alloc();
        if (engine_param_.attn_cp_size > 1) {
            alloc = GetSymmAllocator(context_.comm.d_comm);
            workspace_tokens += engine_param_.max_forward_token_num;
        }

        partial_O_  = Tensor_<float>({workspace_tokens, local_head_num, size_per_head}, kDEVICE);
        partial_ML_ = Tensor_<float>({engine_param_.attn_cp_size, workspace_tokens, local_head_num, 2}, alloc);
        split_cnt_  = Tensor_<int>({workspace_tokens}, kDEVICE);

        Clear(split_cnt_.buffer());
    }
}

static void init_dynamic_ntk(Sequence& cache, const core::RopeConfig& rope)
{
    cache.rope_base = rope.base;
    if (auto scaling_factor = rope.factor; scaling_factor > 1.f) {
        const auto max_seq_len = cache.prompt_len;
        const auto max_pos_emb = rope.max_position_embeddings;
        if (max_seq_len > max_pos_emb) {
            scaling_factor = scaling_factor * max_seq_len / max_pos_emb - (scaling_factor - 1);
            cache.rope_base *= powf(scaling_factor, rope.dim / (rope.dim - 2.f));
            // clang-format off
            TM_LOG_INFO("{} rope_scaling_factor: {}, rope_theta = {}",
                        cache.req->id, scaling_factor, cache.rope_base);
            // clang-format on
        }
    }
}

void UnifiedAttentionLayer::Run(BatchOp op, int phase, TensorMap& env)
{
    if (op == BatchOp::kAdd) {
        Buffer_<Sequence*> rc = env.at("requests").buffer();
        if (rope_param_.type == RopeType::kDynamic) {
            for (int i = 0; i < rc.size(); ++i) {
                init_dynamic_ntk(*rc[i], rope_);
            }
        }
    }
    else if (op == BatchOp::kSetup) {
        Setup(phase, env);
    }
    else if (op == BatchOp::kPrepare) {
        // Which env buffers the MTP draft may share with the target, and which
        // it may not, reduces to one question: is the buffer indexed by ROW or
        // by TOKEN?
        //
        // Row-indexed buffers are safe. The draft and the target agree on the
        // batch, so entry i means row i to both -- `finished`, `k_offsets` and
        // `readonly_block_num` are all of this kind.
        //
        // Token-indexed buffers are not, because a verification step submits
        // K+1 tokens per row while the draft submits one. `token_mask` is per
        // token, so row i sits at i*(K+1), and `q_offsets` is per row but holds
        // token counts. Both are replaced below with draft-owned versions; at
        // batch size 1 the two layouts coincide, so sharing them survives a
        // single-sequence test and fails on the second request.
        auto& d               = data_.at(phase);
        d->finished           = env.at("finished").buffer().borrow();
        // The MTP draft submits one query per row, so it cannot borrow the
        // target's q_offsets: those describe the target's input_len, which is
        // K+1 on a verification step.
        //
        // cu_q_len is not merely a size hint. attention_universal indexes the
        // query tensor with it directly --
        //
        //   qi_begin = cu_q_len[b] + query_idx;  qi_end = cu_q_len[b + 1];
        //
        // -- so a borrowed K+1 span would walk 5 query rows out of a 1-row
        // tensor. The decode-shape flag fixes the plan's counts; this fixes the
        // offsets the kernel actually dereferences.
        // decode_q_offsets is filled during Setup, which is the only op with
        // `requests` in env; Prepare carries just the prepared tensors.
        if (d->decode_shape) {
            d->q_offsets = d->decode_q_offsets;
        }
        else {
            d->q_offsets = env.at("q_offsets").buffer().borrow();
        }
        d->k_offsets          = env.at("k_offsets").buffer().borrow();
        d->readonly_block_num = env.at("readonly_block_num").buffer().borrow();

        // Borrow the global mask owned by LanguageModel (pointer only; its content is
        // built at Forward time) and resolve this rank's token offset within it.
        // The MTP draft needs a per-ROW mask, not the target's per-TOKEN one.
        //
        // reduce.cu indexes it as token_mask[query_idx]. The target's mask has
        // one entry per submitted token, so on a verification step row i owns
        // entry i*(K+1). The draft has one query per row and would read entry
        // i -- which for i > 0 belongs to an earlier row's later token. At
        // bsz == 1 the two happen to coincide, which is why this survives a
        // single-sequence test and breaks as soon as a second request arrives.
        //
        // The draft's queries are all valid: rows that finished this step are
        // excluded from drafting by skip_draft, and CanDraft requires every row
        // generating. So an all-true mask of length bsz is correct, and stays
        // correct because the exclusion happens upstream rather than here.
        if (d->decode_shape) {
            d->token_mask      = d->decode_token_mask;
            d->token_mask_base = 0;
        }
        else {
            d->token_mask      = env.at("token_mask").buffer().borrow();
            d->token_mask_base = 0;
        }
        if (engine_param_.attn_dp_size > 1) {
            const auto& local_token_num = env.at("batch").data<BatchData*>()[0]->local_token_num;
            TM_CHECK_EQ((int)local_token_num.size(), engine_param_.attn_dp_size);
            d->token_mask_base =
                std::accumulate(local_token_num.begin(), local_token_num.begin() + engine_param_.attn_dp_rank, 0);
        }
    }
}

void UnifiedAttentionLayer::ValidateDFlashDraftMetadata(int        phase,
                                                         const int* committed_seq_lens,
                                                         int        batch_size,
                                                         int        block_size,
                                                         bool       rebuild,
                                                         bool       assert_exact)
{
    TM_CHECK_GE(phase, 0);
    TM_CHECK_LT(phase, (int)data_.size());
    TM_CHECK_NOTNULL(committed_seq_lens);
    TM_CHECK_GT(batch_size, 0);
    TM_CHECK_GT(block_size, 0);

    auto& d                  = *data_.at(phase);
    d.assert_draft_metadata = assert_exact;
    TM_CHECK(d.decode_shape) << "DFlash metadata validation requires a draft-shaped attention phase";
    TM_CHECK_EQ(d.draft_block_size, block_size);

    const auto old_prefill = d.prefill;
    static const bool anchor_inclusive_frontier = [] {
        const char* value = std::getenv("TM_DFLASH_ANCHOR_INCLUSIVE_FRONTIER");
        return value && value[0] == '1';
    }();
    const int frontier_adjust = anchor_inclusive_frontier ? -1 : block_size;
    int       expected_k_sum{};
    int       expected_k_max{};
    for (int i = 0; i < batch_size; ++i) {
        TM_CHECK_GE(committed_seq_lens[i] + frontier_adjust, 0);
        // The DFlash block includes the freshly generated anchor as row zero,
        // but SGLang's DFlash attention is frozen-KV: all eight parallel
        // queries read only the context before that anchor. The corrected
        // contract is frontier-1. The legacy control remains frontier+block.
        const int expected = committed_seq_lens[i] + frontier_adjust;
        expected_k_sum += expected;
        expected_k_max = std::max(expected_k_max, expected);
        if (rebuild) {
            if ((int)d.draft_k_lens_host.size() != batch_size) {
                d.draft_k_lens_host.resize(batch_size);
            }
            d.draft_k_lens_host[i] = expected;
        }
    }

    if (rebuild) {
        d.decode  = {};
        d.prefill = {batch_size,
                     batch_size * block_size,
                     block_size,
                     expected_k_sum,
                     expected_k_max};
    }

    if (assert_exact) {
        TM_CHECK_EQ((int)d.draft_k_lens_host.size(), batch_size);
        for (int i = 0; i < batch_size; ++i) {
            TM_CHECK_EQ(d.draft_k_lens_host[i], committed_seq_lens[i] + frontier_adjust)
                << "DFlash draft metadata row " << i << " does not match the post-rollback committed frontier";
        }
        TM_CHECK_EQ(d.decode.n, 0);
        TM_CHECK_EQ(d.prefill.n, batch_size);
        TM_CHECK_EQ(d.prefill.q_sum, batch_size * block_size);
        TM_CHECK_EQ(d.prefill.q_max, block_size);
        TM_CHECK_EQ(d.prefill.k_sum, expected_k_sum);
        TM_CHECK_EQ(d.prefill.k_max, expected_k_max);
    }

    if (rebuild || assert_exact) {
        int device = -1;
        TM_CUDA_CHECK(cudaGetDevice(&device));
        TM_LOG_INFO("DFLASH_METADATA_{}_ACTIVE device={} phase={} rows={} block={} old_q_sum={} old_k_sum={} "
                    "new_q_sum={} new_k_sum={} assert={}",
                    rebuild ? "REBUILD" : "ASSERT",
                    device,
                    phase,
                    batch_size,
                    block_size,
                    old_prefill.q_sum,
                    old_prefill.k_sum,
                    d.prefill.q_sum,
                    d.prefill.k_sum,
                    (int)assert_exact);
    }
}

void UnifiedAttentionLayer::AssertDFlashDraftKeySpans(int phase, int batch_size)
{
    TM_CHECK_GE(phase, 0);
    TM_CHECK_LT(phase, (int)data_.size());
    auto& d = *data_.at(phase);
    if (!d.assert_draft_metadata) {
        return;
    }

    TM_CHECK_EQ((int)d.draft_k_lens_host.size(), batch_size);
    Buffer_<int> offsets_host{batch_size + 1, kCPU};
    Copy(d.k_offsets.slice(0, batch_size + 1), offsets_host);
    core::Context::stream().Sync();
    for (int i = 0; i < batch_size; ++i) {
        const int actual = offsets_host[i + 1] - offsets_host[i];
        TM_CHECK_EQ(actual, d.draft_k_lens_host[i])
            << "DFlash live key span for row " << i << " differs from rebuilt draft metadata";
    }

    int device = -1;
    TM_CUDA_CHECK(cudaGetDevice(&device));
    TM_LOG_INFO("DFLASH_METADATA_OFFSETS_ASSERT_ACTIVE device={} phase={} rows={} first_span={}",
                device,
                phase,
                batch_size,
                offsets_host[1] - offsets_host[0]);
}

void UnifiedAttentionLayer::Setup(int phase, TensorMap& env)
{
    // const auto& rc  = env.at("batch").data<BatchData*>()[0]->rc;  // active requests
    Buffer_<Sequence*> rc = env.at("requests").buffer();

    const int bsz = rc.size();

    auto& d    = *data_.at(phase);
    auto& copy = *env.at("copy").data<BatchCopy*>()[0];

    {  /// Upload KV cache ptrs
        auto blocks  = d.block_ptrs_host.data();
        auto offsets = d.block_ptrs_offsets_host.data();

        offsets[0] = 0;
        for (int i = 0; i < rc.size(); ++i) {
            const auto& r = *rc[i];
            for (const auto& h : r.block_ids) {
                const CacheBlock& cb = *h->prefix;
                TM_CHECK_NOTNULL(cb.allocation.a);
                // Bounded. The unchecked `*blocks++` was safe only while every
                // row owned at most ceil(session_len / block_len) blocks, which
                // the draft headroom broke.
                TM_CHECK_LT(blocks - d.block_ptrs_host.data(), d.block_ptrs_host.size())
                    << "block pointer staging overflow at row " << i;
                *blocks++ = cb.base(0) + prefix_cache_offset_;
            }
            offsets[i + 1] = offsets[i] + r.block_ids.size();
        }

        copy(d.block_ptrs_host, d.block_ptrs_offsets_host[bsz], d.block_ptrs);
        copy(d.block_ptrs_offsets_host, bsz + 1, d.block_ptrs_offsets);
    }

    /// prepare Q/K stats for decode/prefill
    d.decode = d.prefill = {};

    // The MTP draft always submits exactly one token per row, whatever the
    // target's input_len is for this step.
    //
    // On a verification step input_len is K+1, so deriving the draft's plan
    // from it describes bsz*(K+1) tokens while the draft submits bsz. That is
    // the `d.prefill.q_sum + d.decode.n == q_count` abort, and a separate phase
    // slot alone does not fix it: the slot stops the draft corrupting the
    // target's plan, but the draft still needs its OWN shape rather than a copy
    // of the target's.
    const bool decode_shape = env.try_("attn_decode_shape") != nullptr;
    auto*      block_shape  = env.try_("attn_draft_block_size");
    TM_CHECK(!(decode_shape && block_shape)) << "only one speculative attention shape can be active";
    d.draft_block_size = decode_shape ? 1 : (block_shape ? block_shape->data<int>()[0] : 0);
    d.decode_shape     = d.draft_block_size > 0;

    // Row 0's draft count, so a diagnostic can say what kind of call it is
    // looking at instead of inferring it from token counts. A prompt's final
    // chunk can be as short as a verification forward, and the two are then
    // indistinguishable by q_sum alone.
    d.num_drafts0 = bsz > 0 ? rc[0]->num_drafts : -1;

    // Report each Setup once per phase, so the log shows whether the draft's
    // phase is being set up at all and with which shape. Four rounds of
    // reasoning have not settled that, and one line of evidence will.
    if (!setup_logged_.count(phase)) {
        setup_logged_.insert(phase);
        TM_LOG_INFO("[attn] first Setup on phase {}: bsz={} decode_shape={} input_len[0]={}",
                    phase,
                    bsz,
                    d.draft_block_size,
                    bsz > 0 ? rc[0]->input_len : -1);
    }

    // One query token per row, cumulative: [0, 1, 2, ... bsz].
    //
    // Built here because Setup is the only op with `requests` in env, and used
    // by Prepare, which carries just the prepared tensors. cu_q_len indexes the
    // query tensor directly in attention_universal, so the draft cannot borrow
    // the target's offsets: those describe input_len, which is K+1 on a
    // verification step.
    if (d.decode_shape) {
        for (int i = 0; i <= bsz; ++i) {
            d.decode_q_offsets_host[i] = i * d.draft_block_size;
        }
        copy(d.decode_q_offsets_host, bsz + 1, d.decode_q_offsets);
    }

    d.decode.n = d.draft_block_size == 1 ?
                     bsz :
                     std::find_if(rc.begin(), rc.end(), [](auto r) { return r->input_len > 1; }) - rc.begin();
    if (d.draft_block_size > 1) {
        d.decode.n = 0;
    }
    d.prefill.n = bsz - d.decode.n;

    // d.dbg_offset = d.dbg_size = 0;

    for (int i = 0; i < bsz; ++i) {
        const auto& c = *rc[i];

        // if (c.request->id == 4 && c.input_len > 1) {
        //     d.dbg_offset = d.decode.q_sum + d.prefill.q_sum;
        //     d.dbg_size   = c.input_len;
        // }

        // One query token per row for the draft; the key length is unchanged,
        // because the draft attends over the same accepted history.
        // The KV this row will write must fit the blocks it owns.
        //
        // invokeProcessKV_v2 writes up to k_len tokens into block_ptrs, which
        // was uploaded from block_ids above. Overrunning dereferences a pointer
        // belonging to another sequence, or past the array, and surfaces as
        //
        //   kv_cache_utils_v2.cu: CUDA error: an illegal memory access
        //
        // naming neither the row nor the amount. Checking it here, where both
        // numbers are in hand, turns that into a precise failure.
        {
            const int bs = engine_param_.cache_block_seq_len;
            if (bs > 0) {
                // The keys THIS Setup describes. The draft's own walk is
                // bounded separately, in MTPPredictor::Draft, against the same
                // block count -- adding num_drafts here would assert on a
                // length no kernel writes, which is exactly what it did.
                const int draft_extra = !is_warm_up_ && engine_param_.num_draft_tokens > 0 && d.draft_block_size > 1 ?
                                            d.draft_block_size :
                                            0;
                const int k_len_row = c.history_len + c.inflight_input_len + c.input_len + draft_extra;
                const int need = (k_len_row + bs - 1) / bs;
                TM_CHECK_LE(need, (int)c.block_ids.size())
                    << "row " << i << " needs " << need << " KV blocks for " << k_len_row
                    << " keys but owns " << c.block_ids.size() << " (history=" << c.history_len
                    << " inflight=" << c.inflight_input_len << " input=" << c.input_len
                    << " drafts=" << c.num_drafts << " decode_shape=" << (int)decode_shape << ")";
            }
        }

        const int q_len = d.decode_shape ? d.draft_block_size : c.input_len;

        // NOT widened by num_drafts, though the draft does walk that far.
        //
        // I added `+ num_drafts` here as a free correctness improvement, on the
        // reasoning that k_max should describe the furthest key the draft
        // reaches. It is not free: MTPPredictor clamps the walk to
        // `min(K, capacity - seq_len)`, so the draft often stops short, and
        // claiming the unclamped length made this row assert at 130 keys
        // against a 128-key allocation -- a bound violation invented by the
        // hint, not by any actual write.
        //
        // Setup cannot know the clamped value: max_extend is computed later, in
        // Draft. And under-counting is harmless, as established when this was
        // first considered -- cu_k_len drives the real iteration, k_max only
        // sizes the split-K grid, so the cost is parallelism in the rare case
        // where the extra keys cross a CTA_S boundary.
        const int draft_extra = !is_warm_up_ && engine_param_.num_draft_tokens > 0 && d.draft_block_size > 1 ?
                                    d.draft_block_size :
                                    0;
        const int k_len = c.history_len + c.inflight_input_len + c.input_len + draft_extra;
        static const bool track_dflash_metadata = [] {
            const char* rebuild = std::getenv("TM_DFLASH_REBUILD_METADATA_AFTER_ROLLBACK");
            const char* check   = std::getenv("TM_DFLASH_ASSERT_DRAFT_METADATA");
            return (!rebuild || rebuild[0] != '0') || (check && check[0] == '1');
        }();
        if (track_dflash_metadata) {
            if ((int)d.draft_k_lens_host.size() != bsz) {
                d.draft_k_lens_host.resize(bsz);
            }
            d.draft_k_lens_host[i] = k_len;
        }

        auto& s = i < d.decode.n ? d.decode : d.prefill;
        s.q_sum += q_len;
        s.k_sum += k_len;
        s.q_max = std::max(s.q_max, q_len);
        s.k_max = std::max(s.k_max, k_len);
    }

    // auto &D = d.decode, &P = d.prefill;
    // dbg(D.n, D.k_sum, D.k_max, P.n, P.q_sum, P.q_max, P.k_sum, P.k_max);

    /// handling different RoPE types
    if (rope_param_.type == RopeType::kDynamic) {
        for (int i = 0; i < bsz; ++i) {
            d.rope_base_host[i] = rc[i]->rope_base;
        }
        copy(d.rope_base_host, bsz, d.rope_base);
    }
    else if (rope_param_.mrope_mode != MropeMode::kNone) {
        auto* mrope_length           = env.try_("mrope_length");
        auto* mrope_position_delta   = env.try_("mrope_position_delta");
        auto* mrope_position_offsets = env.try_("mrope_position_offsets");
        auto* mrope_position_ids     = env.try_("mrope_position_ids");
        if (mrope_length || mrope_position_delta || mrope_position_offsets || mrope_position_ids) {
            TM_CHECK(mrope_length) << "MRoPE requires native vision-produced mrope_length";
            TM_CHECK(mrope_position_delta) << "MRoPE requires native vision-produced mrope_position_delta";
            TM_CHECK(mrope_position_offsets) << "MRoPE requires native vision-produced mrope_position_offsets";
            TM_CHECK(mrope_position_ids) << "MRoPE requires native vision-produced mrope_position_ids";

            d.mrope_length           = mrope_length->buffer().borrow();
            d.mrope_position_delta   = mrope_position_delta->buffer().borrow();
            d.mrope_position_offsets = mrope_position_offsets->buffer().borrow();
            d.mrope_position_ids     = mrope_position_ids->buffer().borrow();
        }
        else {
            d.mrope_length           = mrope_default_buf_.borrow();
            d.mrope_position_delta   = mrope_default_buf_.borrow();
            d.mrope_position_offsets = mrope_default_buf_.borrow();
            d.mrope_position_ids     = mrope_default_buf_.borrow();
        }
    }
}

void UnifiedAttentionLayer::Forward(ForwardParam p)
{
    TM_FUNCTION_SCOPE();

    /////////////////////////////////////////////
    /// parse inputs
    const int token_num = p.input.shape(0);

    if (token_num == 0) {
        return;
    }

    const int layer_id = p.layer_id;

    const auto& weights = *p.weights;

    TM_LOG_DEBUG("layer=%d, token_num=%d", layer_id, token_num);

    Tensor qkv;

    auto& d = *data_.at(p.phase);

    static const bool persistent_dflash_workspace = [] {
        const char* value = std::getenv("TM_DFLASH_PERSISTENT_WORKSPACE");
        return !value || value[0] != '0';
    }();
    AttentionData::DFlashWorkspace* dflash_workspace = nullptr;
    const int dflash_batch = d.decode.n + d.prefill.n;
    if (p.use_dflash_workspace && persistent_dflash_workspace && dflash_batch == 1) {
        TM_CHECK(!p.kv_only);
        TM_CHECK(engine_param_.speculative_algorithm == "dflash2");
        TM_CHECK(weights.w_qkv && weights.w_qkv->output_dim);
        TM_CHECK(!weights.is_mla());

        auto& workspace = d.dflash_workspace;
        const int draft_block = std::max(1,
                                         engine_param_.speculative_dflash_block_size > 0 ?
                                             engine_param_.speculative_dflash_block_size :
                                             engine_param_.num_draft_tokens + 1);
        const int local_heads    = weights.head_num / weights.tp_size;
        const int local_kv_heads = weights.kv_head_num / weights.tp_size;
        const int qkv_width      = weights.w_qkv->output_dim;
        const int attention_width = local_heads * weights.head_dim;
        const int max_tokens      = draft_block;
        // Graph replay is restricted to batch one. FlattenKV materializes the
        // full logical key span before the attention kernel applies its window,
        // so reserve the configured session plus the proposal block.
        const int max_k_sum = engine_param_.session_len + draft_block;
        TM_CHECK_EQ(token_num, draft_block);

        if (!workspace.initialized) {
            workspace.qkv = {{max_tokens, qkv_width}, engine_param_.data_type, kDEVICE};
            workspace.attention = {{max_tokens, attention_width}, engine_param_.data_type, kDEVICE};
            workspace.flattened_kv = {{local_kv_heads,
                                       2,
                                       max_k_sum + MAX_CTA_S,
                                       weights.head_dim},
                                      engine_param_.data_type,
                                      kDEVICE};
            workspace.max_tokens = max_tokens;
            workspace.max_k_sum = max_k_sum;
            workspace.qkv_width = qkv_width;
            workspace.attention_width = attention_width;
            workspace.local_kv_heads = local_kv_heads;
            workspace.head_dim = weights.head_dim;
            workspace.initialized = true;
        }

        TM_CHECK_EQ(workspace.qkv_width, qkv_width);
        TM_CHECK_EQ(workspace.attention_width, attention_width);
        TM_CHECK_EQ(workspace.local_kv_heads, local_kv_heads);
        TM_CHECK_EQ(workspace.head_dim, weights.head_dim);
        TM_CHECK_LE(token_num, workspace.max_tokens);

        static const bool trace_workspace = [] {
            const char* value = std::getenv("TM_DFLASH_WORKSPACE_TRACE");
            return value && value[0] == '1';
        }();
        if (trace_workspace && !workspace.traced) {
            workspace.traced = true;
            TM_LOG_INFO("[DFlash2] attention workspace phase={} qkv={} attention={} flattened_kv={} "
                        "max_tokens={} max_k_sum={} qkv_width={} attention_width={} kv_heads={} head_dim={}",
                        p.phase,
                        (uintptr_t)workspace.qkv.raw_data(),
                        (uintptr_t)workspace.attention.raw_data(),
                        (uintptr_t)workspace.flattened_kv.raw_data(),
                        workspace.max_tokens,
                        workspace.max_k_sum,
                        workspace.qkv_width,
                        workspace.attention_width,
                        workspace.local_kv_heads,
                        workspace.head_dim);
        }
        dflash_workspace = &workspace;
    }
    // ForwardParam is by value. Clear eligibility for unsupported batches so
    // core_attention cannot accidentally reuse an initialized batch-one arena.
    p.use_dflash_workspace = dflash_workspace != nullptr;

    // if (d.dbg_size) {
    //     DebugTensor(p.input.slice(d.dbg_offset, d.dbg_size), Concat("attn_in", p.layer_id), 0);
    // }

    if (weights.w_qkv && weights.w_qkv->output_dim) {
        // [token_num, hidden_dim] -> [token_num, local_q_kv_head_num, head_dim]
        if (dflash_workspace) {
            qkv = dflash_workspace->qkv.slice(0, token_num);
        }
        TM_SCOPE_CALL(linear_.Forward(p.input, *weights.w_qkv, qkv));
        if (p.trace_fn && p.trace_qkv_projection) {
            p.trace_fn(p.trace_context, p.trace_qkv_projection, qkv);
        }

        qk_norm(qkv, weights);
        if (p.q_replay || p.k_replay) {
            TM_CHECK(p.q_replay && p.k_replay);
            const int local_head_num    = weights.head_num / weights.tp_size;
            const int local_kv_head_num = weights.kv_head_num / weights.tp_size;
            const int q_width           = local_head_num * weights.head_dim;
            const int k_width           = local_kv_head_num * weights.head_dim;
            TM_CHECK_EQ(p.q_replay.ndim(), 2);
            TM_CHECK_EQ(p.k_replay.ndim(), 2);
            TM_CHECK_EQ(p.q_replay.shape(0), qkv.shape(0));
            TM_CHECK_EQ(p.k_replay.shape(0), qkv.shape(0));
            TM_CHECK_EQ(p.q_replay.shape(1), q_width);
            TM_CHECK_EQ(p.k_replay.shape(1), k_width);
            TM_CHECK_EQ(p.q_replay.dtype(), qkv.dtype());
            TM_CHECK_EQ(p.k_replay.dtype(), qkv.dtype());
            const size_t elem_bytes = byte_size(qkv.dtype(), 1);
            auto* qkv_bytes = static_cast<char*>(qkv.raw_data());
            TM_CUDA_CHECK(cudaMemcpy2DAsync(qkv_bytes,
                                            qkv.stride(0) * elem_bytes,
                                            p.q_replay.raw_data(),
                                            p.q_replay.stride(0) * elem_bytes,
                                            q_width * elem_bytes,
                                            qkv.shape(0),
                                            cudaMemcpyDeviceToDevice,
                                            core::Context::stream().handle()));
            TM_CUDA_CHECK(cudaMemcpy2DAsync(qkv_bytes + q_width * elem_bytes,
                                            qkv.stride(0) * elem_bytes,
                                            p.k_replay.raw_data(),
                                            p.k_replay.stride(0) * elem_bytes,
                                            k_width * elem_bytes,
                                            qkv.shape(0),
                                            cudaMemcpyDeviceToDevice,
                                            core::Context::stream().handle()));
        }
        if (p.trace_fn && p.trace_qkv_pre) {
            p.trace_fn(p.trace_context, p.trace_qkv_pre, qkv);
        }
    }
    else {
        qkv = forward_mla(p.input, weights);
    }

    TM_DEBUG_TENSOR(qkv, Concat("qkv", layer_id), 3);

    auto invoke = [&](auto t) -> Tensor {
        using T = decltype(t);
        return core_attention<T>(qkv, p, weights);
    };

    Tensor attn = [&]() -> Tensor { TM_DISPATCH_PRIMARY_DTYPES_RET(qkv.dtype(), invoke); }();
    if (p.trace_fn && p.trace_qkv_post) {
        p.trace_fn(p.trace_context, p.trace_qkv_post, qkv);
    }
    if (!p.kv_only && p.trace_fn && p.trace_attention) {
        p.trace_fn(p.trace_context, p.trace_attention, attn);
    }

    // Context materialization for DFlash only needs projected, normalized and
    // rotated K/V in the cache. SGLang uses kv_proj_only here; running full
    // attention and Wo for five draft layers discarded their outputs and cost
    // several milliseconds every speculative cycle.
    if (p.kv_only) {
        return;
    }

    // Apply sigmoid gating: attn *= sigmoid(gate)
    // Gate is stored at the end of each token's QKV: [Q|K|V|Gate]
    if (weights.output_gate) {
        const int  tp_size           = weights.tp_size;
        const int  local_head_num    = weights.head_num / tp_size;
        const int  local_kv_head_num = weights.kv_head_num / tp_size;
        const int  size_per_head     = weights.head_dim;
        const int  q_count           = qkv.shape(0);
        const int  attn_dim          = local_head_num * size_per_head;
        const int  gate_offset       = (local_head_num + 2 * local_kv_head_num) * size_per_head;
        const int  qkv_stride        = (2 * local_head_num + 2 * local_kv_head_num) * size_per_head;
        const auto stream            = core::Context::stream().handle();
        invokeSigmoidGateMultiply(attn.raw_data(),
                                  (const char*)qkv.raw_data() + gate_offset * byte_size(qkv.dtype(), 1),
                                  attn_dim,
                                  qkv_stride,
                                  q_count,
                                  qkv.dtype(),
                                  stream);
        TM_CUDA_CHECK(cudaGetLastError());
    }

    TM_DEBUG_TENSOR(attn, Concat("attn", layer_id), 3);

    // if (d.dbg_size) {
    //     DebugTensor(attn.slice(d.dbg_offset, d.dbg_size), Concat("attn_out", p.layer_id), 0);
    // }

    //////////////////////////////////////////////
    /// output gemm <Bs,HD> -> <Bs,HD>
    if (p.output_input_scale != 1.f) {
        // DFlash's SM70 compatibility path transports wide attention branches
        // through FP16 at 1/256 scale before Wo, then restores the branch in
        // FP32 residual+norm. Scaling after Wo changes FP16 rounding and draft
        // predictions, even though it is algebraically equivalent in FP32.
        invokeDFlashScale(attn, p.output_input_scale, core::Context::stream().handle());
    }
    TM_SCOPE_CALL(linear_.Forward(attn, *weights.wo, p.output));
}

template<class T>
Tensor UnifiedAttentionLayer::core_attention(Tensor& qkv, const ForwardParam& p, const WeightType& weights)
{
    TM_FUNCTION_SCOPE();
    const int tp_size           = weights.tp_size;
    const int local_head_num    = weights.head_num / tp_size;
    const int local_kv_head_num = weights.kv_head_num / tp_size;
    const int size_per_head     = weights.head_dim;

    const auto device = qkv.device();
    const auto dtype  = qkv.dtype();

    auto& d = *data_.at(p.phase);

    const int batch_size = d.decode.n + d.prefill.n;
    const int q_count    = qkv.shape(0);

    // Report the state, not just the mismatch.
    //
    // This assertion has fired five times from four different causes, and each
    // time the two bare numbers were consistent with several stories. Printing
    // the phase and the plan's composition distinguishes them immediately:
    // a draft phase should show decode.n == batch and prefill.q_sum == 0, and
    // anything else means the decode-shape marker did not reach Setup.
    if (d.prefill.q_sum + d.decode.n != q_count) {
        TM_LOG_ERROR("[attn] shape mismatch on phase {}: decode.n={} decode.q_sum={} "
                     "prefill.n={} prefill.q_sum={} q_count={} decode_shape={}",
                     p.phase,
                     d.decode.n,
                     d.decode.q_sum,
                     d.prefill.n,
                     d.prefill.q_sum,
                     q_count,
                     (int)d.decode_shape);
    }
    TM_CHECK_EQ(d.prefill.q_sum + d.decode.n, q_count);

    const int local_q_kv_head_num = local_head_num + 2 * local_kv_head_num;

    const bool use_dflash_workspace = p.use_dflash_workspace && d.dflash_workspace.initialized;
    if (use_dflash_workspace) {
        TM_CHECK_LE(q_count, d.dflash_workspace.max_tokens);
        TM_CHECK_LE(d.prefill.k_sum, d.dflash_workspace.max_k_sum);
    }
    Tensor attn = use_dflash_workspace ? d.dflash_workspace.attention.slice(0, q_count) :
                                         Tensor{{q_count, local_head_num * size_per_head}, dtype, device};

    const bool is_mla = weights.is_mla();
    static const bool enable_dflash_paged_q8 = [] {
        const char* value = std::getenv("TM_DFLASH_PAGED_Q8");
        return value && value[0] == '1';
    }();
    static const bool enable_dflash_grouped_paged_q8 = [] {
        const char* value = std::getenv("TM_DFLASH_GROUPED_PAGED_Q8");
        return !value || value[0] != '0';
    }();
    static const bool enable_dflash_draft_graph = [] {
        const char* value = std::getenv("TM_DFLASH_DRAFT_GRAPH");
        return value && value[0] == '1';
    }();
    const int dflash_block = engine_param_.speculative_dflash_block_size > 0 ?
                                 engine_param_.speculative_dflash_block_size :
                                 engine_param_.num_draft_tokens + 1;
    const bool dflash_paged_q8_eligible = engine_param_.speculative_algorithm == "dflash2" && arch_ == 70 &&
                                          !is_mla && weights.causal && quant_policy_ == 0 && d.decode.n == 0 &&
                                          d.prefill.n == 1 && d.prefill.q_sum == dflash_block;
    // Keep this specialization strict. The audited target has 6Q:1KV per TP
    // rank and sixteen full-attention layers. Sliding and draft layers retain
    // their existing paths.
    const bool use_dflash_grouped_paged_q8 = enable_dflash_grouped_paged_q8 && dflash_paged_q8_eligible &&
                                             dtype == kHalf && dflash_block == 8 && q_count == 8 &&
                                             local_head_num == 6 && local_kv_head_num == 1 &&
                                             size_per_head == 256 && weights.window_size == 0;
    const bool use_dflash_paged_q8 = (enable_dflash_paged_q8 || use_dflash_grouped_paged_q8) &&
                                     dflash_paged_q8_eligible;
    const bool use_fixed_dflash_graph_geometry =
        use_dflash_paged_q8 && enable_dflash_draft_graph && p.use_dflash_workspace;
    const int dflash_graph_k_bound = weights.window_size > 0 ?
                                         std::min(engine_param_.session_len, weights.window_size) :
                                         engine_param_.session_len;

    Tensor tmp_kv;
    if (!use_dflash_paged_q8) {
        tmp_kv = use_dflash_workspace ? d.dflash_workspace.flattened_kv :
                                        Tensor{{local_kv_head_num,
                                                is_mla ? 1 : 2,
                                                d.prefill.k_sum + MAX_CTA_S,
                                                size_per_head},
                                               dtype,
                                               device};
    }

    auto CreateParams = [&](int offset, AttentionData::Stat stat, int max_kv_splits, cudaStream_t stream) {
        AttentionParams<T> params{};

        // Batch offset for `out` and `q` are computed inside the kernel
        params.out = (T*)attn.raw_data();

        params.q = (T*)qkv.raw_data();
        params.k = params.q + local_head_num * size_per_head;
        if (is_mla) {
            params.v      = params.k;
            params.stride = (local_head_num + 1 * local_kv_head_num) * size_per_head;
        }
        else {
            params.v = params.k + local_kv_head_num * size_per_head;
            // When attn_output_gate, QKV layout is [Q|K|V|Gate] per token
            // stride must account for the extra gate portion at the end
            if (weights.output_gate) {
                params.stride = (2 * local_head_num + 2 * local_kv_head_num) * size_per_head;
            }
            else {
                params.stride = (local_head_num + 2 * local_kv_head_num) * size_per_head;
            }
        }

        if (!is_mla && weights.w_qkv && weights.w_qkv->bias) {
            params.q_bias = (T*)weights.w_qkv->bias.data_or<T>(nullptr);
            params.k_bias = params.q_bias + local_head_num * size_per_head;
            params.v_bias = params.k_bias + local_kv_head_num * size_per_head;
        }

        params.batch_size = stat.n;

        params.token_num = stat.q_sum;
        params.max_q_len = stat.q_max;
        params.max_k_len = stat.k_max;
        if (use_fixed_dflash_graph_geometry) {
            TM_CHECK_GE(dflash_graph_k_bound, dflash_block);
            // CUDA graph kernel parameters retain their capture-time scalar
            // values. Use a fixed sliding-window/session bound for host launch
            // geometry, while the paged kernel continues to read each row's
            // actual context length from device cu_k_len. ProcessKV already
            // launches from the fixed max_q_len=8. The current Q=8 attention
            // path uses one split. If split-K is enabled later, inactive
            // overprovisioned splits exit and the last active split publishes
            // the count consumed by invokeReduceV3.
            params.max_k_len = dflash_graph_k_bound;
        }

        TM_CHECK_LE(weights.cache_block_offset, INT_MAX);

        // decode only
        params.block_iter_params = BlockIteratorParams{(char**)d.block_ptrs.data(),  //
                                                       d.block_ptrs_offsets.data() + offset,
                                                       (int)weights.cache_block_offset,
                                                       engine_param_.cache_block_seq_len};

        // Prefill only. The direct paged path constructs its iterator from the
        // block table above and must not dereference an intentionally empty
        // flattened-KV tensor while assembling otherwise shared parameters.
        if (use_dflash_paged_q8) {
            params.linear_iter_params = LinearIteratorParams{nullptr, 0, 0};
        }
        else if (is_mla) {
            params.linear_iter_params = LinearIteratorParams{
                tmp_kv.raw_data(),           // flattened KV
                stat.k_sum * size_per_head,  // stride to next head
                0                            // stride from K to V
            };
        }
        else {
            params.linear_iter_params = LinearIteratorParams{
                tmp_kv.raw_data(),               // flattened KV
                stat.k_sum * size_per_head * 2,  // stride to next head
                stat.k_sum * size_per_head       // stride from K to V
            };
        }

        params.finished = d.finished.data() + offset;
        // decode rows: base; prefill rows: + decode.n (this rank's slice of the global mask)
        params.token_mask         = d.token_mask.data() + d.token_mask_base + offset;
        params.cu_q_len           = d.q_offsets.data() + offset;
        params.cu_k_len           = d.k_offsets.data() + offset;
        params.readonly_block_num = d.readonly_block_num.data() + offset;

        params.num_heads        = local_head_num;
        params.num_kv_heads     = local_kv_head_num;
        params.size_per_head    = size_per_head;
        params.q_position_shift = p.frozen_kv ? stat.q_max : 0;

        double scaling = 1.;
        if (weights.softmax_scale) {  // model predefined softmax scale
            scaling *= weights.softmax_scale;
        }
        else {  // default value
            scaling /= std::sqrt((float)params.size_per_head);
        }
        params.inv_sqrt_dh = scaling * std::log2(std::exp(1.));

        params.sinks       = weights.sinks ? weights.sinks.data_or((T*)nullptr) : (T*)nullptr;
        params.scale_sinks = scaling;

        // DFlash2 draft attention is non-causal inside its parallel proposal
        // block. AttentionWeight already carries the checkpoint contract; do
        // not leave AttentionParams at its causal default.
        params.causal      = weights.causal;
        params.window_size = weights.window_size;
        if (!params.window_size) {
            params.window_size = 256 << 20;  // 256 M
        }

        static const bool per_layer_rope = [] {
            const char* value = std::getenv("TM_DFLASH_PER_LAYER_ROPE");
            return !value || value[0] != '0';
        }();
        params.rope_param = per_layer_rope ? rope_params_.at(&weights) : rope_param_;
        if (params.rope_param.type == RopeType::kDynamic) {
            params.rope_param.base = d.rope_base.data() + offset;
        }
        if (params.rope_param.mrope_mode != MropeMode::kNone) {
            params.rope_param.mrope.position_delta   = d.mrope_position_delta.data() + offset;
            params.rope_param.mrope.position_offsets = d.mrope_position_offsets.data() + offset;
            params.rope_param.mrope.length           = d.mrope_length.data() + offset;
            params.rope_param.mrope.position_ids     = d.mrope_position_ids.data();
        }

        // logn attn
        params.use_logn_attn           = weights.use_logn_attn;
        params.max_position_embeddings = weights.rope.max_position_embeddings;

        // Decoding use only for now
        params.split_cnt   = split_cnt_.data();
        params.partial_ML  = partial_ML_.data();
        params.partial_O   = partial_O_.data();
        params.max_split_k = std::min(std::max(1, kMaxWorkspaceTokens / params.token_num), max_kv_splits);

        // context parallel
        params.cp_rank = engine_param_.attn_cp_rank;
        params.cp_size = engine_param_.attn_cp_size;
        if (params.cp_size > 1) {
            params.cp_size = cutlass::FastDivmod(params.cp_size);

            // update ML,O offset if both prefill and decode present
            const int offset_ML_stage =
                engine_param_.attn_cp_size * (offset ? kMaxWorkspaceTokens * local_head_num * 2 : 0);
            const int offset_ML_rank = params.cp_rank * params.token_num * local_head_num * params.max_split_k * 2;
            const int offset_O       = offset ? kMaxWorkspaceTokens * local_head_num * size_per_head : 0;

            params.partial_ML = partial_ML_.data() + offset_ML_stage + offset_ML_rank;
            params.partial_O  = partial_O_.data() + offset_O;
            params.offset_q   = offset;

            // postprocess func
            params.cp_fn          = CpPost;
            params.cp_fn_ctx      = (void*)&cp_fn_ctx_;
            cp_fn_ctx_.cp_rank    = params.cp_rank;
            cp_fn_ctx_.count      = params.token_num * local_head_num * params.max_split_k * 2;
            cp_fn_ctx_.partial_ML = partial_ML_.data() + offset_ML_stage;
            cp_fn_ctx_.stream     = stream;
        }

        params.arch   = arch_;
        params.stream = stream;

        params.quant_policy = quant_policy_;
        return params;
    };

    const cudaStream_t stream = core::Context::stream().handle();

    cudaStream_t pf_stream = stream;
    cudaStream_t dc_stream = stream;

    if (d.decode.n && d.prefill.n) {
        pf_stream = aux_stream_;
        TM_CUDA_CHECK(cudaEventRecord(qkv_event_, stream));
        TM_CUDA_CHECK(cudaStreamWaitEvent(aux_stream_, qkv_event_));
    }

    if (d.prefill.n && !is_warm_up_) {
        const int offset = d.decode.n;
        // We are executing prefill & decoding kernels concurrently, but only have 1 workspace
        // disable split kv for prefill for now
        auto params = CreateParams(offset, d.prefill, use_dflash_grouped_paged_q8 ? 8 : 1, pf_stream);
        if constexpr (sizeof(T) == 2) {
            if (!p.frozen_kv) {
                invokeProcessKV_v2_(params);
                TM_CUDA_CHECK(cudaGetLastError());
            }

            if (!p.kv_only) {
                if (use_dflash_grouped_paged_q8) {
                    int device{};
                    TM_CUDA_CHECK(cudaGetDevice(&device));
                    if (ClaimGroupedQ8Parity(device, p.layer_id)) {
                        Tensor reference{attn.shape(), attn.dtype(), attn.device()};
                        auto   reference_params = params;
                        reference_params.out    = (T*)reference.raw_data();
                        dispatchPagedAttention(reference_params);
                        dispatchGroupedPagedAttention(params);
                        CheckGroupedQ8Parity(reference, attn, p.layer_id, d.prefill.k_max, pf_stream);
                    }
                    else {
                        dispatchGroupedPagedAttention(params);
                    }
                }
                else if (use_dflash_paged_q8) {
                    dispatchPagedAttention(params);
                }
                else {
                    invokeFlattenKV_v2_(params, d.prefill.k_sum);
                    TM_CUDA_CHECK(cudaGetLastError());
                    if (p.trace_fn && p.trace_flattened_kv) {
                        p.trace_fn(p.trace_context, p.trace_flattened_kv, tmp_kv);
                    }
                    dispatchAttention(params);
                }
                TM_CUDA_CHECK(cudaGetLastError());
            }
        }
    }

    if (d.decode.n && !is_warm_up_) {
        auto params = CreateParams(0, d.decode, kMaxKVSplits, dc_stream);
        if constexpr (sizeof(T) == 2) {
            dispatchDecoding<T>(params);
            TM_CUDA_CHECK(cudaGetLastError());
        }
    }

    if (d.decode.n && d.prefill.n) {
        TM_CUDA_CHECK(cudaEventRecord(aux_event_, aux_stream_));
        TM_CUDA_CHECK(cudaStreamWaitEvent(stream, aux_event_));
    }

    if (is_warm_up_) {
        rng_.set_stream(stream);
        rng_.GenerateUniform(attn.data<T>(), attn.size(), .02f, -.01f);
    }

    return attn;
}

Tensor UnifiedAttentionLayer::forward_mla(const Tensor& hidden_state, const WeightType& w)
{
    TM_FUNCTION_SCOPE();

    const int tp_size           = w.tp_size;
    const int local_head_num    = w.head_num / tp_size;
    const int local_kv_head_num = w.kv_head_num / tp_size;
    const int size_per_head     = w.head_dim;

    const auto token_num = hidden_state.shape(0);
    const auto dtype     = hidden_state.dtype();

    const int q_lora_rank  = w.q_a_proj->output_dim;
    const int kv_lora_rank = w.kv_a_layernorm->weight.size();
    const int qk_rope_dim  = w.kv_a_proj->output_dim - kv_lora_rank;

    Tensor q;

    const auto stream = core::Context::stream().handle();

    if (w.q_proj && w.q_proj->weight) {
        TM_SCOPE_CALL(linear_.Forward(hidden_state, *w.q_proj, q));
    }
    else {
        Tensor q_a;
        TM_SCOPE_CALL(linear_.Forward(hidden_state, *w.q_a_proj, q_a));

        invokeRMSNorm(
            q_a, q_a, w.q_a_layernorm->weight, w.q_a_layernorm->norm_eps_, w.q_a_layernorm->zero_centered_, stream);
        TM_CUDA_CHECK(cudaGetLastError());

        TM_SCOPE_CALL(linear_.Forward(q_a, *w.q_b_proj, q));
    }

    Tensor kv_a_k_pe;
    TM_SCOPE_CALL(linear_.Forward(hidden_state, *w.kv_a_proj, kv_a_k_pe));

    auto kv_a = kv_a_k_pe.slice({0, 0}, {-1, kv_lora_rank});
    invokeRMSNorm(
        kv_a, kv_a, w.kv_a_layernorm->weight, w.kv_a_layernorm->norm_eps_, w.kv_a_layernorm->zero_centered_, stream);
    TM_CUDA_CHECK(cudaGetLastError());

    const int local_q_kv_head_num = local_head_num + 1 * local_kv_head_num;

    Tensor qkv{{token_num, local_q_kv_head_num, size_per_head}, dtype, hidden_state.device()};
    MLACopyQKV(dtype,
               qkv.raw_data(),
               q.raw_data(),
               kv_a_k_pe.raw_data(),
               token_num,
               local_head_num,
               kv_lora_rank,
               qk_rope_dim,
               stream);
    TM_CUDA_CHECK(cudaGetLastError());

    return qkv;
}

void UnifiedAttentionLayer::qk_norm(Tensor& qkv, const WeightType& weights)
{
    if (!(weights.q_norm || weights.k_norm)) {
        return;
    }

    TM_CHECK(weights.q_norm && weights.k_norm);

    const int tp_size           = weights.tp_size;
    const int local_head_num    = weights.head_num / tp_size;
    const int local_kv_head_num = weights.kv_head_num / tp_size;
    const int size_per_head     = weights.head_dim;

    const auto stream = core::Context::stream().handle();

    const auto token_num = qkv.shape(0);

    auto qkv3 = qkv.view({token_num, -1, size_per_head});

    invokeQkRMSNorm(qkv3,
                    weights.q_norm->weight,
                    weights.k_norm->weight,
                    local_head_num,
                    local_kv_head_num,
                    weights.q_norm->norm_eps_,
                    weights.q_norm->zero_centered_,
                    stream);
    TM_CUDA_CHECK(cudaGetLastError());
}

}  // namespace turbomind
