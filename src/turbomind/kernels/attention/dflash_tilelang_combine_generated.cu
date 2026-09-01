#include <tl_templates/cuda/gemm.h>
#include <tl_templates/cuda/copy.h>
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void main_kernel(half_t* __restrict__ Output, const float* __restrict__ Partial_LSE, const half_t* __restrict__ Partial_O, const int* __restrict__ cache_seqlens, const int* __restrict__ query_start_loc);
extern "C" __global__ void __launch_bounds__(128, 1) main_kernel(half_t* __restrict__ Output, const float* __restrict__ Partial_LSE, const half_t* __restrict__ Partial_O, const int* __restrict__ cache_seqlens, const int* __restrict__ query_start_loc) {
  extern __shared__ __align__(1024) float lse[];
  float lse_max[1];
  float lse_sum[1];
  float acc_o[1];
  int active_splits = min(40, max(1, ((cache_seqlens[0] + 127) >> 7)));
  int q_start = query_start_loc[0];
  int q_len = (query_start_loc[1] - q_start);
  if (((int)blockIdx.x) < q_len) {
    if (((int)threadIdx.x) < 40) {
      float condval;
      if ((((int)threadIdx.x) < active_splits)) {
        condval = Partial_LSE[(((((int)threadIdx.x) * 128) + (((int)blockIdx.x) * 8)) + ((int)blockIdx.y))];
      } else {
        condval = -0x1p+30f/*-1.073742e+09*/;
      }
      lse[((int)threadIdx.x)] = condval;
    }
    lse_max[0] = -0x1p+30f/*-1.073742e+09*/;
  }
  __syncthreads();
  if (((int)blockIdx.x) < q_len) {
    for (int split_i = 0; split_i < 40; ++split_i) {
      lse_max[0] = max(lse_max[0], lse[split_i]);
    }
    lse_sum[0] = 0x0p+0f/*0.000000e+00*/;
    for (int split_i_1 = 0; split_i_1 < 40; ++split_i_1) {
      if (split_i_1 < active_splits) {
        lse_sum[0] = (lse_sum[0] + exp2f((lse[split_i_1] - lse_max[0])));
      }
    }
    acc_o[0] = 0x0p+0f/*0.000000e+00*/;
    for (int split_i_2 = 0; split_i_2 < 40; ++split_i_2) {
      if (split_i_2 < active_splits) {
        float weight = (exp2f((lse[split_i_2] - lse_max[0])) / lse_sum[0]);
        acc_o[0] = (acc_o[0] + (weight * ((float)Partial_O[((((split_i_2 * 16384) + (((int)blockIdx.x) * 1024)) + (((int)blockIdx.y) * 128)) + ((int)threadIdx.x))])));
      }
    }
    if (0 <= (q_start + ((int)blockIdx.x))) {
      if ((q_start + ((int)blockIdx.x)) < 16) {
        Output[((((((int64_t)q_start) * (int64_t)1024) + (((int64_t)((int)blockIdx.x)) * (int64_t)1024)) + (((int64_t)((int)blockIdx.y)) * (int64_t)128)) + ((int64_t)((int)threadIdx.x)))] = ((half_t)acc_o[0]);
      }
    }
  }
}
