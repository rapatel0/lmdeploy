// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "src/turbomind/kernels/attention/impl_884.h"

namespace turbomind::attention {

/// Run SM70 MMA_884 with [query, GQA head] flattened into its M dimension.
///
/// AttentionUniversal exposes logical CTA_Q and CTA_H dimensions. Its Q
/// prologue stores rows as `query * CTA_H + head`. The inherited MMA consumes
/// that flattened layout. Callback wrappers restore logical indices before
/// masking, score tracing, split storage, and final output storage.
template<class T_, int CTA_H_, int CTA_Q_, int CTA_S_, int WARP_H_, int WARP_Q_, int WARP_S_, int HeadDim>
struct Impl<MMA_884_GROUPED, T_, T_, CTA_H_, CTA_Q_, CTA_S_, WARP_H_, WARP_Q_, WARP_S_, HeadDim>:
    Impl<MMA_884, T_, T_, 1, CTA_H_ * CTA_Q_, CTA_S_, 1, WARP_Q_, WARP_S_, HeadDim> {
private:
    using Base = Impl<MMA_884, T_, T_, 1, CTA_H_ * CTA_Q_, CTA_S_, 1, WARP_Q_, WARP_S_, HeadDim>;

    __device__ static constexpr int LogicalHead(int flat_m)
    {
        return flat_m % CTA_H_;
    }

    __device__ static constexpr int LogicalQuery(int flat_m)
    {
        return flat_m / CTA_H_;
    }

public:
    static_assert(CTA_H_ > 1 && CTA_Q_ > 1, "grouped MMA requires query and head grouping");
    static_assert((CTA_H_ * CTA_Q_) % WARP_Q_ == 0, "flattened M must contain complete MMA warps");

    static constexpr int CTA_H = CTA_H_;
    static constexpr int CTA_Q = CTA_Q_;
    static constexpr int CTA_S = CTA_S_;

    template<class Fragment, class Func>
    __device__ static void ForeachS(Fragment& scores, Func&& func)
    {
        Base::ForeachS(scores, [&](int, int flat_m, int si, int ri, auto& score) {
            ((Func &&)func)(LogicalHead(flat_m), LogicalQuery(flat_m), si, ri, score);
        });
    }

    template<class Func>
    __device__ static void ForeachML(typename Base::FragM& maxima, typename Base::FragL& sums, Func&& func)
    {
        Base::ForeachML(maxima, sums, [&](int, int flat_m, int ri, float& maximum, float& sum) {
            ((Func &&)func)(LogicalHead(flat_m), LogicalQuery(flat_m), ri, maximum, sum);
        });
    }

    template<bool is_norm, class Func>
    __device__ static void StoreO(typename Base::FragO&         output,
                                  typename Base::FragL&         sums,
                                  typename Base::SharedStorage& storage,
                                  Func&&                        func)
    {
        Base::template StoreO<is_norm>(output, sums, storage, [&](int, int flat_m, int di, const auto& value) {
            ((Func &&)func)(LogicalHead(flat_m), LogicalQuery(flat_m), di, value);
        });
    }
};

}  // namespace turbomind::attention
