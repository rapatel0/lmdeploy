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

void prepareDFlashTileLangAttention();

void dispatchDFlashTileLangAttention(const AttentionParams<half>& params,
                                      int                          context_len,
                                      half*                        packed_workspace,
                                      std::size_t                  packed_workspace_elements,
                                      int*                         metadata_workspace,
                                      bool                         graph_replay_safe);

}  // namespace turbomind
