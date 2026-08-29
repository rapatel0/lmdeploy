// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/block_iterator.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_884.h"
#include "src/turbomind/kernels/attention/impl_884_grouped.h"
#include "src/turbomind/kernels/attention/linear_iterator.h"
#include "src/turbomind/kernels/attention/mainloop_sm70.h"
#include "src/turbomind/kernels/attention/registrar.h"

namespace turbomind::attention {

constexpr int kHeadDim = 256;
constexpr int kCTA_Q   = 64;
constexpr int kCTA_S   = 64;
constexpr int kWARP_Q  = 16;
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

// DFlash target verification has eight query positions and local 6Q:1KV GQA.
// Two CTAs cover the heads as 4+2. Each CTA retains all eight query positions,
// so split-K does not duplicate K/V loads across query positions.
template<class T>
using GroupedPagedQ8KT = AttentionUniversal<
    arch::Sm70,
    Mainloop<arch::Sm70, Impl<MMA_884_GROUPED, T, T, 4, 8, 64, 1, 16, 64, kHeadDim, kStages>>,
    GetBlockIterFactory<T, T, 64, kHeadDim>,
    AttentionCtaMap,
    true>;

namespace {
Registrar reg([](Collector& c) {
    c.add<KT<half>>();
    c.add<KT<half, false>>();
    c.add<PagedKT<half>>();
    c.add<GroupedPagedQ8KT<half>>();
});
}

}  // namespace turbomind::attention
