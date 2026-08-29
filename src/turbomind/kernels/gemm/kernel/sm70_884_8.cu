// Copyright (c) OpenMMLab. All rights reserved.

#include "src/turbomind/kernels/gemm/arch/config_sm70_s884.h"
#include "src/turbomind/kernels/gemm/registrar.h"
#include "src/turbomind/kernels/gemm/types.h"

#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdlib>

namespace turbomind::gemm {

using namespace sm70_s884;
using namespace cache_policy;
using S = cache_policy::Stream;
using D = cache_policy::Default;

namespace {
Registrar reg([](Collector& c, int /*arch*/) {
    if constexpr (1) {
        // clang-format off
        using C = Config_E4M3<kColMajor, 0>;
        c.add<C::Type<128, 128,  16, 2, 2, 1, D, D, 2, true, 1, 128,  64, 128>>();
        c.add<C::Type< 64, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128,  32, 128>>();
        c.add<C::Type< 32, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        c.add<C::Type< 16, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
        c.add<C::Type<  8, 128,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();

        // These M=8 variants are opt-in because every additional kernel enters
        // the process-start tuning catalog. N=128/256 and K=32/64/128 reuse
        // packed low-bit catalog maps. N=64 uses the audited two-warp SM70
        // map; scale grouping remains on K, so it does not constrain CTA_N.
        const char* candidates = std::getenv("TM_SM70_FP8_M8_TILE_CANDIDATES");
        if (candidates && candidates[0] == '1') {
            c.add<C::Type<  8,  64,  32, 1, 2, 1, D, S, 2, true, 1, 128>>();
            c.add<C::Type<  8,  64,  64, 1, 2, 1, D, S, 2, true, 1, 128>>();
            c.add<C::Type<  8,  64, 128, 1, 2, 1, D, S, 2, true, 1, 128>>();
            c.add<C::Type<  8, 128,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
            c.add<C::Type<  8, 128, 128, 1, 4, 1, D, S, 2, true, 1, 128>>();
            c.add<C::Type<  8, 256,  32, 1, 4, 1, D, S, 2, true, 1, 128>>();
            c.add<C::Type<  8, 256,  64, 1, 4, 1, D, S, 2, true, 1, 128>>();
            int device = -1;
            cudaGetDevice(&device);
            std::fprintf(stderr,
                         "SM70_FP8_M8_TILE_CANDIDATES_REGISTERED "
                         "device=%d tiles=8x64x32,8x64x64,8x64x128,8x128x32,"
                         "8x128x128,8x256x32,8x256x64\n",
                         device);
        }
        // clang-format on
    }
});
}  // namespace

}  // namespace turbomind::gemm
