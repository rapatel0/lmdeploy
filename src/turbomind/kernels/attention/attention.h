// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "attention_params.h"

#include <cstddef>

namespace turbomind {

constexpr int MAX_CTA_S = 64;

template<typename T>
void dispatchAttention(const AttentionParams<T>& params);

template<typename T>
void dispatchPagedAttention(const AttentionParams<T>& params);

template<typename T>
void dispatchGroupedPagedAttention(const AttentionParams<T>& params);

void dispatchDFlashTileLangAttention(const AttentionParams<half>& params,
                                      int                          context_len,
                                      std::size_t                  workspace_elements);

}  // namespace turbomind
