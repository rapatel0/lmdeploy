// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/block_iterator.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_884.h"
#include "src/turbomind/kernels/attention/linear_iterator.h"
#include "src/turbomind/kernels/attention/mainloop_sm70.h"
#include "src/turbomind/kernels/attention/registrar.h"

namespace turbomind::attention {

constexpr int kHeadDim = 128;
constexpr int kCTA_Q   = 64;
constexpr int kCTA_S   = 64;
constexpr int kWARP_Q  = 8;
constexpr int kStages  = 2;

template<class T, bool Causal = true>
using KT = AttentionUniversal<
    arch::Sm70,
    Mainloop<arch::Sm70, Impl<MMA_884, T, T, 1, kCTA_Q, kCTA_S, 1, kWARP_Q, kCTA_S, kHeadDim, kStages>>,
    LinearIteratorFactory<T, kCTA_S, kHeadDim>,
    AttentionCtaMap,
    Causal>;

template<class T>
using PagedKT = AttentionUniversal<
    arch::Sm70,
    Mainloop<arch::Sm70, Impl<MMA_884, T, T, 1, kCTA_Q, kCTA_S, 1, kWARP_Q, kCTA_S, kHeadDim, kStages>>,
    GetBlockIterFactory<T, T, kCTA_S, kHeadDim>,
    AttentionCtaMap,
    true>;

namespace {
Registrar reg([](Collector& c) {
    c.add<KT<half>>();
    c.add<KT<half, false>>();
    c.add<PagedKT<half>>();
});
}

}  // namespace turbomind::attention
