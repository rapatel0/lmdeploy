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
// https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/attention_layers/GptContextAttentionLayer.h

#pragma once

#include <cuda_runtime.h>
#include <set>
#include <unordered_map>
#include <vector>

#include "src/turbomind/core/core.h"
#include "src/turbomind/engine/batch.h"
#include "src/turbomind/engine/cache_registry.h"
#include "src/turbomind/kernels/attention/cp_utils.h"
#include "src/turbomind/kernels/gemm/test/test_utils.h"
#include "src/turbomind/models/attention_weight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/context.h"
#include "src/turbomind/models/llama/llama_params.h"
#include "src/turbomind/models/llama/llama_rope.h"

namespace turbomind {

struct AttentionData;

class UnifiedAttentionLayer {
public:
    using WeightType = AttentionWeight;

    static constexpr int kMaxKVSplits        = 128;
    static constexpr int kMaxWorkspaceTokens = 4096;

    using TraceFn = void (*)(const void*, const char*, const Tensor&);

    struct ForwardParam {
        int               phase;
        Tensor            input;
        Tensor            output;
        const WeightType* weights;
        int               layer_id;
        float             output_input_scale{1.f};
        bool              kv_only{false};
        bool              use_dflash_workspace{false};
        bool              frozen_kv{false};
        Tensor            q_replay{};
        Tensor            k_replay{};
        const void*       trace_context{};
        TraceFn           trace_fn{};
        const char*       trace_qkv_projection{};
        const char*       trace_qkv_pre{};
        const char*       trace_qkv_post{};
        const char*       trace_flattened_kv{};
        const char*       trace_attention{};
    };

    ~UnifiedAttentionLayer();

    UnifiedAttentionLayer(std::vector<AttentionWeight*> weights,
                          CacheRegistry&                registry,
                          const EngineParam&            engine,
                          const Context&                context,
                          int                           phases);

    void Run(BatchOp op, int phase, TensorMap& env);

    void Forward(ForwardParam p);

    /// Overwrite logically dead target-cache entries from a rejected DFlash
    /// verifier suffix. This diagnostic never touches committed positions or
    /// the five dedicated DFlash draft slots.
    void PoisonVerifierSuffix(int phase,
                              const int* begin_positions,
                              const int* end_positions,
                              int        row_count,
                              int        target_attention_count);

    /// Validate the DFlash draft key span against a post-rollback committed
    /// frontier. The optional rebuild refreshes only shape metadata; the
    /// existing phase-owned pointers remain valid.
    void ValidateDFlashDraftMetadata(int        phase,
                                     const int* committed_seq_lens,
                                     int        batch_size,
                                     int        block_size,
                                     bool       rebuild,
                                     bool       assert_exact);

    /// Check the live device cu_k_len spans after the proposal block advances
    /// them and before draft attention reads them.
    void AssertDFlashDraftKeySpans(int phase, int batch_size);

private:
    void Setup(int phase, TensorMap& env);

    Tensor forward_mla(const Tensor& hidden_state, const WeightType& weights);

    /// TODO: dropping the `T` here requires deep refactor of attention dispatch
    template<class T>
    Tensor core_attention(Tensor& qkv, const ForwardParam& p, const WeightType& weights);

    void qk_norm(Tensor& qkv, const WeightType& weights);

private:
    const int              quant_policy_;
    const core::RopeConfig rope_;
    const EngineParam      engine_param_;
    const Context&         context_;
    int&                   is_warm_up_;

    LlamaLinear& linear_;
    const int    arch_{};

    cudaStream_t aux_stream_;
    cudaEvent_t  qkv_event_;
    cudaEvent_t  aux_event_;

    RNG rng_;

    RopeKernelParam                                         rope_param_{};
    std::unordered_map<const WeightType*, RopeKernelParam>  rope_params_;

    std::vector<std::shared_ptr<AttentionData>> data_;
    std::vector<AttentionWeight*>               weights_;

    size_t prefix_cache_offset_{};

    // Host staging for Setup's uploads lives on AttentionData, one set per
    // phase. It was here, shared across phases, and the draft's Setup refilled
    // it while the target's stream-ordered copy was still queued -- the target
    // then attended through the draft's block pointers. See AttentionData.

    /// Phases whose first Setup has already been logged. Diagnostic only.
    std::set<int>  setup_logged_;

    /// Bounded counter for the KV kernel argument dump. Diagnostic only.

    ///////////////////////////////////////////////////////
    /// temp runtime buffers (allocated in constructor)
    Tensor_<float> partial_O_;
    Tensor_<float> partial_ML_;
    Tensor_<int>   split_cnt_;

    Buffer_<int>   mrope_default_buf_;

    CpPostContext cp_fn_ctx_;  // context parallel
};

}  // namespace turbomind
