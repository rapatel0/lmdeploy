// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/attention/attention_universal.h"
#include "src/turbomind/kernels/attention/cta_map.h"
#include "src/turbomind/kernels/attention/impl.h"
#include "src/turbomind/kernels/attention/impl_884.h"
#include "src/turbomind/kernels/attention/linear_iterator.h"
#include "src/turbomind/kernels/attention/mainloop_sm70.h"
#include "src/turbomind/kernels/attention/registrar.h"

namespace turbomind::attention {

constexpr int kHeadDim = 256;
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
