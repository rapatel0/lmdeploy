/*
 * Copyright (c) OpenMMLab. All rights reserved.
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

// Modified from https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/FfnLayer.h

#include "src/turbomind/models/llama/LlamaFfnLayer.h"

#include <cstdint>
#include <cstdlib>

#include "src/turbomind/core/logger.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/activation.h"
#include "src/turbomind/models/llama/llama_utils.h"
#include "src/turbomind/utils/anomaly_handler.h"

namespace turbomind {

void LlamaFfnLayer::forward(ForwardParam param)
{
    TM_FUNCTION_SCOPE();

    NvtxScope scope("ffn");

    const auto& mlp = *param.weights;

    const int token_num  = param.input.shape(0);
    const int inter_size = mlp.inter_size;
    const int layer_id   = param.layer_id;

    const auto stream = core::Context::stream().handle();

    auto* fused     = mlp.w1w3.get();
    bool  use_fused = fused && fused->weight;

    Workspace* target_workspace = nullptr;
    if (param.use_target_workspace && param.phase >= 0 && token_num == 8) {
        auto& workspace = target_workspaces_[param.phase];
        if (use_fused) {
            const int mix_dim = fused->epilogue == gemm::Epilogue::kGatedSilu ? fused->output_dim / 2 :
                                                                                 fused->output_dim;
            if (!workspace.mix) {
                workspace.mix = {{8, mix_dim}, fused->output_dtype(), kDEVICE};
                if (fused->output_dtype() == kFloat8_e4m3) {
                    constexpr int group_size = 128;
                    constexpr int alignment  = 16 / sizeof(float);
                    const int     scale_dim  = cdiv(mix_dim, group_size);
                    const int     aligned_m  = round_up(8, alignment);
                    workspace.inter_scales =
                        Tensor_<float>{{{scale_dim, 8}, {aligned_m, 1}}, kDEVICE};
                }
            }
        }
        else if (!workspace.gating) {
            workspace.gating = {{8, inter_size}, param.input.dtype(), kDEVICE};
            workspace.inter  = {{8, inter_size}, param.input.dtype(), kDEVICE};
        }
        target_workspace = &workspace;
    }

    Tensor gating = target_workspace && target_workspace->gating ?
                        target_workspace->gating.slice(0, token_num) :
                        Tensor{};
    Tensor inter = target_workspace && target_workspace->inter ? target_workspace->inter.slice(0, token_num) :
                                                                 Tensor{};
    Tensor inter_scales = target_workspace && target_workspace->inter_scales ?
                              target_workspace->inter_scales :
                              Tensor{};
    Tensor unused_scales;

    if (use_fused) {
        Tensor mix = target_workspace ? target_workspace->mix.slice(0, token_num) : Tensor{};
        if (mlp.is_fused_silu && fused->output_dtype() == kFloat8_e4m3) {
            TM_SCOPE_CALL(linear_.Forward(param.input, unused_scales, *fused, mix, inter_scales));
            gating = mix;  // FP8 fused output is already half-N (inter_size)
        }
        else {
            TM_SCOPE_CALL(linear_.Forward(param.input, *fused, mix));
            gating = mix.slice({0, 0}, {(int)token_num, inter_size});
            if (!mlp.is_fused_silu) {
                inter = mix.slice({0, inter_size}, {(ssize_t)token_num, inter_size});
            }
        }
    }
    else {
        TM_SCOPE_CALL(linear_.Forward(param.input, *mlp.w1, gating));
        TM_DEBUG_TENSOR(gating, Concat("w1", layer_id), 3);

        TM_SCOPE_CALL(linear_.Forward(param.input, *mlp.w3, inter));
        TM_DEBUG_TENSOR(inter, Concat("w3", layer_id), 3);
    }

    // When using the fused kernel (w1w3 + fused silu), activation is already applied.
    // Otherwise (separate w1/w3 or non-fused), apply activation explicitly.
    if (!use_fused || !mlp.is_fused_silu) {
        // gate' = silu(gate) * up
        Activation(gating, inter, mlp.act_type, stream);
        TM_CUDA_CHECK(cudaGetLastError());
        TM_DEBUG_TENSOR(gating, Concat("act", layer_id), 3);
    }

    if (param.trace_activation) {
        TM_CHECK_EQ(layer_id, 0);
        TM_CHECK_EQ(param.trace_activation->dtype(), gating.dtype());
        TM_CHECK_EQ(param.trace_activation->shape(0), 1);
        TM_CHECK_EQ(param.trace_activation->shape(1), gating.shape(1));
        TM_CUDA_CHECK(cudaMemcpyAsync(param.trace_activation->raw_data(),
                                      (const char*)gating.raw_data()
                                          + (gating.shape(0) - 1) * byte_size(gating.dtype(), gating.stride(0)),
                                      byte_size(gating.dtype(), gating.shape(1)),
                                      cudaMemcpyDeviceToDevice,
                                      stream));
    }

    static const bool trace_workspace = [] {
        const char* value = std::getenv("TM_DFLASH_WORKSPACE_TRACE");
        return value && value[0] == '1';
    }();
    if (target_workspace && trace_workspace && !target_workspace->traced) {
        target_workspace->traced = true;
        TM_LOG_INFO("[DFlash2] target FFN workspace phase={} mix={} gating={} inter={} scales={}",
                    param.phase,
                    (uintptr_t)target_workspace->mix.data_or((void*)nullptr),
                    (uintptr_t)target_workspace->gating.data_or((void*)nullptr),
                    (uintptr_t)target_workspace->inter.data_or((void*)nullptr),
                    (uintptr_t)target_workspace->inter_scales.data_or((void*)nullptr));
    }

    {  // w2(x)
        NvtxScope scope("w2");
        if (inter_scales) {
            TM_SCOPE_CALL(linear_.Forward(gating, inter_scales, *mlp.w2, param.output, unused_scales));
        }
        else {
            TM_SCOPE_CALL(linear_.Forward(gating, *mlp.w2, param.output));
        }
    }
}

}  // namespace turbomind
