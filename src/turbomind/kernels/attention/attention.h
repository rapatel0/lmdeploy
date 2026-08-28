// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "attention_params.h"

namespace turbomind {

constexpr int MAX_CTA_S = 64;

template<typename T>
void dispatchAttention(const AttentionParams<T>& params);

template<typename T>
void dispatchPagedAttention(const AttentionParams<T>& params);

}  // namespace turbomind
