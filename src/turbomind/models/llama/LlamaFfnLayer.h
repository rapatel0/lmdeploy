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

// Modified from https://github.com/NVIDIA/FasterTransformer/blob/main/src/fastertransformer/layers/FfnLayer.cc

#pragma once

#include "src/turbomind/core/core.h"
#include "src/turbomind/models/ffn_weight.h"
#include "src/turbomind/models/llama/LlamaLinear.h"
#include "src/turbomind/models/llama/context.h"

#include <unordered_map>

namespace turbomind {

class LlamaFfnLayer {
public:
    LlamaFfnLayer(const Context& ctx): linear_(*ctx.linear) {}

    struct ForwardParam {
        Tensor           input;
        Tensor           output;
        const FfnWeight* weights;
        int              layer_id;
        int              phase{-1};
        bool             use_target_workspace{false};
    };

    void forward(ForwardParam param);

private:
    struct Workspace {
        Tensor mix;
        Tensor gating;
        Tensor inter;
        Tensor inter_scales;
        bool   traced{};
    };

    LlamaLinear&                       linear_;
    std::unordered_map<int, Workspace> target_workspaces_;
};

}  // namespace turbomind
