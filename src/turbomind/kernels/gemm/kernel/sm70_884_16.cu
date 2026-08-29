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
    const bool register_flat_gdn  = Sm70Fp16FlatGdnMode() == 2;
    const bool register_flat_head = Sm70Fp16FlatHeadMode() == 2;
    if (register_flat_gdn || register_flat_head) {
        using F = Config_F16_Flat<kColMajor>;
        // The GDN sweep selected this exact 8x64x64 kernel. Reuse the same
        // unpacked SM70 catalog entry for an explicitly transposed output head;
        // one registration serves both logical KxN shapes.
        c.add<F::Type<8, 64, 64, 1, 2, 1, D, S, 2, true, 1, 1>>();
        int device = -1;
        const auto status = cudaGetDevice(&device);
        if (status != cudaSuccess || device < 0 || device >= 16) {
            std::fprintf(stderr,
                         "SM70_FP16_FLAT_REGISTRATION_ERROR status=%d device=%d\n",
                         static_cast<int>(status),
                         device);
            std::fflush(stderr);
        }
        else {
            static std::atomic<bool> logged_gdn[16]{};
            static std::atomic<bool> logged_head[16]{};
            if (register_flat_gdn && !logged_gdn[device].exchange(true, std::memory_order_relaxed)) {
                std::fprintf(stderr, "SM70_FP16_FLAT_GDN_REGISTERED device=%d candidates=1\n", device);
                std::fflush(stderr);
            }
            if (register_flat_head && !logged_head[device].exchange(true, std::memory_order_relaxed)) {
                std::fprintf(stderr, "SM70_FP16_FLAT_HEAD_REGISTERED device=%d candidates=1\n", device);
                std::fflush(stderr);
            }
        }
    }
});
}  // namespace

}  // namespace turbomind::gemm
