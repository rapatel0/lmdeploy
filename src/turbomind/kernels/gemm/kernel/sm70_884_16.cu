// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gemm/types.h"

#include <atomic>
#include <cstdio>

namespace turbomind::gemm {

using namespace sm70_s884;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

namespace {
Registrar reg([](Collector& c, int /*arch*/) {
    if constexpr (1) {
        // clang-format off
        using C = Config_F16<kColMajor, 0>;
        c.add<C::Type<256, 128,  16, 4, 2, 1, D, D, 2,   0 , 1, 1, 128, 128>>();
        c.add<C::Type<128, 256,  16, 2, 4, 1, D, D, 2,   0 , 1, 1, 128, 128>>();
        c.add<C::Type<128, 256,  16, 2, 4, 1, D, D, 2,   0 , 1, 1, 128, 128>>();
        c.add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 1,  64, 128>>();
        c.add<C::Type< 96,  64,  32, 2, 2, 1, D, D, 2, true, 1, 1>>();
        c.add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 1>>();
        c.add<C::Type< 64,  64,  64, 2, 2, 1, D, S, 2, true, 1, 1>>();
        c.add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 1>>();
        c.add<C::Type< 16, 128,  64, 1, 4, 1, D, S, 2, true, 1, 1>>();
        c.add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 1>>();
        c.add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 1>>();
        // clang-format on
    }
    if (Sm70Fp16FlatGdnMode() == 2) {
        using F = Config_F16_Flat<kColMajor>;
        // clang-format off
        c.add<F::Type<8,  64, 32, 1, 2, 1, D, S, 2, true, 1, 1>>();
        c.add<F::Type<8,  64, 64, 1, 2, 1, D, S, 2, true, 1, 1>>();
        c.add<F::Type<8, 128, 32, 1, 4, 1, D, S, 2, true, 1, 1>>();
        c.add<F::Type<8, 128, 64, 1, 4, 1, D, S, 2, true, 1, 1>>();
        c.add<F::Type<8, 256, 32, 1, 8, 1, D, S, 2, true, 1, 1>>();
        c.add<F::Type<8, 256, 64, 1, 8, 1, D, S, 2, true, 1, 1>>();
        // clang-format on
        int device = -1;
        const auto status = cudaGetDevice(&device);
        if (status != cudaSuccess || device < 0 || device >= 16) {
            std::fprintf(stderr,
                         "SM70_FP16_FLAT_GDN_REGISTRATION_ERROR status=%d device=%d\n",
                         static_cast<int>(status),
                         device);
            std::fflush(stderr);
        }
        else {
            static std::atomic<bool> logged[16]{};
            if (!logged[device].exchange(true, std::memory_order_relaxed)) {
                std::fprintf(stderr, "SM70_FP16_FLAT_GDN_REGISTERED device=%d candidates=6\n", device);
                std::fflush(stderr);
            }
        }
    }
});
}  // namespace

}  // namespace turbomind::gemm
