// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_884.h"
#include "src/turbomind/kernels/attention/linear_iterator.h"
#include "src/turbomind/kernels/attention/mainloop_sm70.h"
#include "src/turbomind/kernels/attention/registrar.h"

namespace turbomind::attention {

constexpr int kHeadDim = 128;
constexpr int kWARP_Q  = 16;
constexpr int kStages  = 2;

template<class T, int CTA_Q, int CTA_S, bool Causal = true>
using KT = AttentionUniversal<
    arch::Sm70,
    Mainloop<arch::Sm70, Impl<MMA_884, T, T, 1, CTA_Q, CTA_S, 1, kWARP_Q, CTA_S, kHeadDim, kStages>>,
    LinearIteratorFactory<T, CTA_S, kHeadDim>,
    AttentionCtaMap,
    Causal>;

namespace {
Registrar reg([](Collector& c) {
    // DFlash2 submits eight query rows at a time. Keep the legacy q64 tile for
    // ordinary prefill, and add q16/q32 SM70 variants so runtime A/B can avoid
    // spending four warps on a half-warp worth of useful query rows.
    c.add<KT<half, 64, 64>>();
    c.add<KT<half, 64, 64, false>>();
    c.add<KT<half, 32, 64>>();
    c.add<KT<half, 32, 64, false>>();
    c.add<KT<half, 16, 64>>();
    c.add<KT<half, 16, 64, false>>();
    c.add<KT<half, 32, 32>>();
    c.add<KT<half, 32, 32, false>>();
    c.add<KT<half, 16, 32>>();
    c.add<KT<half, 16, 32, false>>();
});
}

}  // namespace turbomind::attention
