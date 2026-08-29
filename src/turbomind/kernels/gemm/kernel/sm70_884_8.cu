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
        // Keep one descriptor for the M=8 kernel. Registering both variants
        // would give the dispatch cache indistinguishable candidates.
        const char* reuse_scale = std::getenv("TM_SM70_FP8_M8_REUSE_SCALE");
        if (reuse_scale && reuse_scale[0] == '1') {
            c.add<C::Type<8, 128, 64, 1, 4, 1, D, S, 2, true, 1, 128, -1, -1, true>>();
            int device = -1;
            cudaGetDevice(&device);
            std::fprintf(stderr, "SM70_FP8_M8_REUSE_SCALE_REGISTERED device=%d\n", device);
        }
        else {
            c.add<C::Type<8, 128, 64, 1, 4, 1, D, S, 2, true, 1, 128>>();
        }
        // clang-format on
    }
});
}  // namespace

}  // namespace turbomind::gemm
