// Copyright (c) OpenMMLab. All rights reserved.

#include "attention.h"
#include "src/turbomind/core/data_type.h"
#include "src/turbomind/core/scope.h"
#include "src/turbomind/kernels/attention/registry.h"
#include "src/turbomind/utils/cuda_utils.h"

namespace turbomind {

template<class T>
void dispatchAttentionImpl(const AttentionParams<T>& params, attention::AttnDesc::Mode mode)
{
    using namespace attention;

    auto&    reg = Registry::instance();
    AttnDesc desc{};
    desc.mode      = mode;
    desc.head_dim  = params.size_per_head;
    desc.data_type = data_type_v<T>;
    desc.causal    = params.causal;

    auto* kernel = reg.Find(desc);

    TM_CHECK(kernel) << "No attention kernel found: " + to_string(desc);

    TM_SCOPE_CALL(kernel->Launch(&params, reg.sm_count()));
}

template<class T>
void dispatchAttention(const AttentionParams<T>& params)
{
    dispatchAttentionImpl(params, attention::AttnDesc::kPrefill);
}

template<class T>
void dispatchPagedAttention(const AttentionParams<T>& params)
{
    dispatchAttentionImpl(params, attention::AttnDesc::kPagedPrefill);
}

template<class T>
void dispatchGroupedPagedAttention(const AttentionParams<T>& params)
{
    dispatchAttentionImpl(params, attention::AttnDesc::kGroupedPagedPrefill);
}

template void dispatchAttention(const AttentionParams<half>& params);
template void dispatchPagedAttention(const AttentionParams<half>& params);
template void dispatchGroupedPagedAttention(const AttentionParams<half>& params);
#if ENABLE_BF16
template void dispatchAttention(const AttentionParams<nv_bfloat16>& params);
template void dispatchPagedAttention(const AttentionParams<nv_bfloat16>& params);
template void dispatchGroupedPagedAttention(const AttentionParams<nv_bfloat16>& params);
#endif

}  // namespace turbomind
