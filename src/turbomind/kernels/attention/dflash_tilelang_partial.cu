// Generated from SGLang's Apache-2.0 TileLang SM70 DFlash verifier.
// Specialization: B1, Q8, H8/HKV2, D128, FP16 KV, noncausal, 40 split slots.
#include "tilelang_compat/tilelang_sm70_compat.h"
#include <math_constants.h>

extern "C" __global__ void dflash_tilelang_partial_kernel(const half_t* __restrict__ K_cache, float* __restrict__ Partial_LSE, half_t* __restrict__ Partial_O, const half_t* __restrict__ Q, const half_t* __restrict__ V_cache, const int* __restrict__ block_table, const int* __restrict__ cache_seqlens, const int* __restrict__ query_start_loc, int nt, float sm_scale);
extern "C" __global__ void __launch_bounds__(256, 1) dflash_tilelang_partial_kernel(const half_t* __restrict__ K_cache, float* __restrict__ Partial_LSE, half_t* __restrict__ Partial_O, const half_t* __restrict__ Q, const half_t* __restrict__ V_cache, const int* __restrict__ block_table, const int* __restrict__ cache_seqlens, const int* __restrict__ query_start_loc, int nt, float sm_scale) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float acc_o[32];
  float m_i[2];
  float l_i[2];
  float acc_s[16];
  half_t A_local[4];
  half_t B_local[8];
  float m_prev[2];
  float m_i_clear[2];
  float scale[2];
  float row_sum[2];
  half_t P_shared_local_cast[2];
  half_t acc_s_cast[64];
  half_t B_local_1[16];
  half_t Partial_O_local_cast_1[2];
  int total_kv = cache_seqlens[0];
  int q_start = query_start_loc[0];
  int q_len = (query_start_loc[1] - q_start);
  float scale_log2 = (sm_scale * 0x1.71547652b82fep+0f/*1.442695e+00*/);
  if ((((int)blockIdx.y) < 1) || (((int)blockIdx.y) < ((total_kv + 127) >> 7))) {
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
      half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
      *(uint2*)(((half_t*)buf_dyn_shmem) + ((((((((int)threadIdx.x) & 31) * 256) + ((i >> 1) * 64)) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + ((((((((int)threadIdx.x) & 7) >> 2) + ((i & 3) >> 1)) + (i & 1)) & 1) * 8)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 4))) = make_uint2(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
    }
    __syncthreads();
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; ++i_1) {
      if ((((i_1 * 2) + (((int)threadIdx.x) >> 7)) < q_len) & (bool)1) {
        half_t broadcast_var_1 = half_t(0x0p+0f/*0.000000e+00*/);
        uint2 condval;
        if (((0 <= (((i_1 * 2) + (((int)threadIdx.x) >> 7)) + q_start)) && ((((i_1 * 2) + (((int)threadIdx.x) >> 7)) + q_start) < nt))) {
          condval = *(uint2*)(Q + (((((((int64_t)i_1) * (int64_t)2048) + ((((int64_t)((int)threadIdx.x)) >> (int64_t)7) * (int64_t)1024)) + (((int64_t)q_start) * (int64_t)1024)) + (((int64_t)((int)blockIdx.x)) * (int64_t)512)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)127) * (int64_t)4)));
        } else {
          condval = make_uint2(__pack_half2(broadcast_var_1, broadcast_var_1), __pack_half2(broadcast_var_1, broadcast_var_1));
        }
        *(uint2*)(((half_t*)buf_dyn_shmem) + ((((((((int)threadIdx.x) & 31) * 256) + ((i_1 >> 1) * 64)) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + ((((((((int)threadIdx.x) & 7) >> 2) + ((i_1 & 3) >> 1)) + (i_1 & 1)) & 1) * 8)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 4))) = condval;
      }
    }
    #pragma unroll
    for (int i_2 = 0; i_2 < 8; ++i_2) {
      float broadcast_var_2 = 0x0p+0f/*0.000000e+00*/;
      *(float4*)(acc_o + (i_2 * 4)) = make_float4(broadcast_var_2, broadcast_var_2, broadcast_var_2, broadcast_var_2);
    }
    float broadcast_var_3 = -CUDART_INF_F;
    *(float2*)(m_i + 0) = make_float2(broadcast_var_3, broadcast_var_3);
    float broadcast_var_4 = 0x0p+0f/*0.000000e+00*/;
    *(float2*)(l_i + 0) = make_float2(broadcast_var_4, broadcast_var_4);
    for (int tile_i = 0; tile_i < (((min(((((int)blockIdx.y) + 1) * ((((total_kv - 1) / min(40, max(1, ((total_kv + 127) >> 7)))) + (((total_kv - 1) % min(40, max(1, ((total_kv + 127) >> 7)))) >> 31)) + 1)), total_kv) + 63) - (((int)blockIdx.y) * ((((total_kv - 1) / min(40, max(1, ((total_kv + 127) >> 7)))) + (((total_kv - 1) % min(40, max(1, ((total_kv + 127) >> 7)))) >> 31)) + 1))) >> 6); ++tile_i) {
      __syncthreads();
      #pragma unroll
      for (int i_3 = 0; i_3 < 8; ++i_3) {
        half_t broadcast_var_5 = half_t(0x0p+0f/*0.000000e+00*/);
        *(uint2*)(((half_t*)buf_dyn_shmem) + (((((((((int)threadIdx.x) & 31) * 256) + ((i_3 >> 1) * 64)) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + ((((((((int)threadIdx.x) & 7) >> 2) + ((i_3 & 3) >> 1)) + (i_3 & 1)) & 1) * 8)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 4)) + 8192)) = make_uint2(__pack_half2(broadcast_var_5, broadcast_var_5), __pack_half2(broadcast_var_5, broadcast_var_5));
      }
      __syncthreads();
      #pragma unroll
      for (int i_4 = 0; i_4 < 8; ++i_4) {
        if ((((((tile_i * 64) + (i_4 * 8)) + (((int)threadIdx.x) >> 5)) + (((int)blockIdx.y) * ((((total_kv - 1) / min(40, max(1, ((total_kv + 127) >> 7)))) + (((total_kv - 1) % min(40, max(1, ((total_kv + 127) >> 7)))) >> 31)) + 1))) < ((((int)blockIdx.y) + 1) * ((((total_kv - 1) / min(40, max(1, ((total_kv + 127) >> 7)))) + (((total_kv - 1) % min(40, max(1, ((total_kv + 127) >> 7)))) >> 31)) + 1))) && (((((tile_i * 64) + (i_4 * 8)) + (((int)threadIdx.x) >> 5)) + (((int)blockIdx.y) * ((((total_kv - 1) / min(40, max(1, ((total_kv + 127) >> 7)))) + (((total_kv - 1) % min(40, max(1, ((total_kv + 127) >> 7)))) >> 31)) + 1))) < total_kv)) {
          int condval_1;
          if ((((((((((int)threadIdx.x) >> 5) + (((int)blockIdx.y) * ((((total_kv - 1) / min(40, max(1, ((total_kv + 127) >> 7)))) + (((total_kv - 1) % min(40, max(1, ((total_kv + 127) >> 7)))) >> 31)) + 1))) >> 3) + i_4) >> 3) + tile_i) < 256)) {
            condval_1 = block_table[((((int64_t)tile_i) * (int64_t)4) + (((((((int64_t)((int)threadIdx.x)) >> (int64_t)5) + (((int64_t)((int)blockIdx.y)) * ((((((int64_t)total_kv) - (int64_t)1) / min((int64_t)40, max((int64_t)1, ((((int64_t)total_kv) + (int64_t)127) >> (int64_t)7)))) + (((((int64_t)total_kv) - (int64_t)1) % min((int64_t)40, max((int64_t)1, ((((int64_t)total_kv) + (int64_t)127) >> (int64_t)7)))) >> (int64_t)63)) + (int64_t)1))) >> (int64_t)3) + ((int64_t)i_4)) >> (int64_t)1))];
          } else {
            condval_1 = 0;
          }
          int physical_page = condval_1;
          half_t broadcast_var_6 = half_t(0x0p+0f/*0.000000e+00*/);
          uint2 condval_2;
          if (((0 <= physical_page) && (physical_page < 1024))) {
            condval_2 = *(uint2*)(K_cache + ((((((int64_t)physical_page) * (int64_t)4096) + (((((((int64_t)i_4) * (int64_t)8) + (((int64_t)((int)threadIdx.x)) >> (int64_t)5)) + (((int64_t)((int)blockIdx.y)) * ((((((int64_t)total_kv) - (int64_t)1) / min((int64_t)40, max((int64_t)1, ((((int64_t)total_kv) + (int64_t)127) >> (int64_t)7)))) + (((((int64_t)total_kv) - (int64_t)1) % min((int64_t)40, max((int64_t)1, ((((int64_t)total_kv) + (int64_t)127) >> (int64_t)7)))) >> (int64_t)63)) + (int64_t)1))) & (int64_t)15) * (int64_t)256)) + (((int64_t)((int)blockIdx.x)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)31) * (int64_t)4)));
          } else {
            condval_2 = make_uint2(__pack_half2(broadcast_var_6, broadcast_var_6), __pack_half2(broadcast_var_6, broadcast_var_6));
          }
          *(uint2*)(((half_t*)buf_dyn_shmem) + (((((((((int)threadIdx.x) & 31) * 256) + ((i_4 >> 1) * 64)) + (((((int)threadIdx.x) & 127) >> 5) * 16)) + ((((((((int)threadIdx.x) & 7) >> 2) + ((i_4 & 3) >> 1)) + (i_4 & 1)) & 1) * 8)) + ((((((int)threadIdx.x) >> 7) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 4)) + 8192)) = condval_2;
        }
      }
      #pragma unroll
      for (int i_5 = 0; i_5 < 16; ++i_5) {
        float condval_3;
        if ((((((((((int)threadIdx.x) & 127) >> 5) * 4) + (((((int)threadIdx.x) & 7) >> 2) * 2)) + ((((int)threadIdx.x) & 31) >> 4)) < q_len) & (((((((((tile_i * 64) + ((((int)threadIdx.x) >> 7) * 32)) + ((i_5 >> 2) * 8)) + (((((int)threadIdx.x) & 15) >> 3) * 4)) + (((((int)threadIdx.x) & 3) >> 1) * 2)) + (((int)blockIdx.y) * ((((total_kv - 1) / min(40, max(1, ((total_kv + 127) >> 7)))) + (((total_kv - 1) % min(40, max(1, ((total_kv + 127) >> 7)))) >> 31)) + 1))) + (i_5 & 1)) < ((((int)blockIdx.y) + 1) * ((((total_kv - 1) / min(40, max(1, ((total_kv + 127) >> 7)))) + (((total_kv - 1) % min(40, max(1, ((total_kv + 127) >> 7)))) >> 31)) + 1))) && ((((((((tile_i * 64) + ((((int)threadIdx.x) >> 7) * 32)) + ((i_5 >> 2) * 8)) + (((((int)threadIdx.x) & 15) >> 3) * 4)) + (((((int)threadIdx.x) & 3) >> 1) * 2)) + (((int)blockIdx.y) * ((((total_kv - 1) / min(40, max(1, ((total_kv + 127) >> 7)))) + (((total_kv - 1) % min(40, max(1, ((total_kv + 127) >> 7)))) >> 31)) + 1))) + (i_5 & 1)) < total_kv)))) {
          condval_3 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_3 = -CUDART_INF_F;
        }
        acc_s[i_5] = condval_3;
      }
      __syncthreads();
      for (int ki = 0; ki < 32; ++ki) {
        *(uint2*)(A_local + 0) = *(uint2*)(((half_t*)buf_dyn_shmem) + (((((ki * 256) + (((((int)threadIdx.x) & 127) >> 5) * 64)) + ((((int)threadIdx.x) & 3) * 16)) + ((((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 7) >> 2)) + ((ki & 7) >> 2)) & 1) * 8)) + (((((((int)threadIdx.x) & 31) >> 4) + ((ki & 3) >> 1)) & 1) * 4)));
        for (int i_6 = 0; i_6 < 2; ++i_6) {
          *(uint2*)(B_local + (i_6 * 4)) = *(uint2*)(((half_t*)buf_dyn_shmem) + (((((((ki * 256) + ((((int)threadIdx.x) >> 7) * 128)) + (i_6 * 64)) + ((((int)threadIdx.x) & 3) * 16)) + ((((((((int)threadIdx.x) & 31) >> 4) + ((ki & 7) >> 2)) + i_6) & 1) * 8)) + (((((((int)threadIdx.x) & 15) >> 3) + ((ki & 3) >> 1)) & 1) * 4)) + 8192));
        }
        for (int j = 0; j < 2; ++j) {
          tl::mma_sync_sm70<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 16, 4, false, true>(reinterpret_cast<float*>(acc_s + (j * 8)), reinterpret_cast<const unsigned*>(A_local + 0), reinterpret_cast<const unsigned*>(B_local + (j * 4)));
        }
      }
      *(float2*)(m_prev + 0) = *(float2*)(m_i + 0);
      __syncthreads();
      #pragma unroll
      for (int i_7 = 0; i_7 < 2; ++i_7) {
        m_i_clear[i_7] = -CUDART_INF_F;
        #pragma unroll
        for (int rv = 0; rv < 8; ++rv) {
          m_i_clear[i_7] = max(m_i_clear[i_7], acc_s[((((rv & 3) * 4) + (i_7 * 2)) + (rv >> 2))]);
        }
        m_i_clear[i_7] = tl::AllReduce<tl::MaxOp, 256, 128, 0>::run(m_i_clear[i_7], (&(((float*)buf_dyn_shmem)[14592])));
        m_i_clear[i_7] = tl::AllReduce<tl::MaxOp, 16, 8, 0>::run(m_i_clear[i_7]);
        m_i_clear[i_7] = tl::AllReduce<tl::MaxOp, 4, 2, 0>::run(m_i_clear[i_7]);
        m_i[i_7] = max(m_i[i_7], m_i_clear[i_7]);
      }
      #pragma unroll
      for (int i_8 = 0; i_8 < 2; ++i_8) {
        float condval_4;
        if ((m_i[i_8] == -CUDART_INF_F)) {
          condval_4 = 0x0p+0f/*0.000000e+00*/;
        } else {
          condval_4 = m_i[i_8];
        }
        m_i[i_8] = condval_4;
        m_i[i_8] = max(m_i[i_8], m_prev[i_8]);
        scale[i_8] = exp2f(((m_prev[i_8] - m_i[i_8]) * scale_log2));
        l_i[i_8] = (l_i[i_8] * scale[i_8]);
      }
      #pragma unroll
      for (int i_9 = 0; i_9 < 32; ++i_9) {
        acc_o[i_9] = (acc_o[i_9] * scale[((i_9 & 3) >> 1)]);
      }
      #pragma unroll
      for (int i_10 = 0; i_10 < 16; ++i_10) {
        acc_s[i_10] = exp2f(((acc_s[i_10] - m_i[((i_10 & 3) >> 1)]) * scale_log2));
      }
      __syncthreads();
      #pragma unroll
      for (int i_11 = 0; i_11 < 2; ++i_11) {
        row_sum[i_11] = 0x0p+0f/*0.000000e+00*/;
        #pragma unroll
        for (int rv_1 = 0; rv_1 < 8; ++rv_1) {
          row_sum[i_11] = (row_sum[i_11] + acc_s[((((rv_1 & 3) * 4) + (i_11 * 2)) + (rv_1 >> 2))]);
        }
        row_sum[i_11] = tl::AllReduce<tl::SumOp, 256, 128, 0>::run(row_sum[i_11], (&(((float*)buf_dyn_shmem)[14336])));
        row_sum[i_11] = tl::AllReduce<tl::SumOp, 16, 8, 0>::run(row_sum[i_11]);
        row_sum[i_11] = tl::AllReduce<tl::SumOp, 4, 2, 0>::run(row_sum[i_11]);
      }
      #pragma unroll
      for (int i_12 = 0; i_12 < 2; ++i_12) {
        l_i[i_12] = (l_i[i_12] + row_sum[i_12]);
      }
      __syncthreads();
      #pragma unroll
      for (int i_13 = 0; i_13 < 4; ++i_13) {
        half_t broadcast_var_7 = half_t(0x0p+0f/*0.000000e+00*/);
        *(uint4*)(((half_t*)buf_dyn_shmem) + (((((((i_13 * 2048) + ((((int)threadIdx.x) >> 6) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + (((((int)threadIdx.x) & 15) >> 2) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)) = make_uint4(__pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7), __pack_half2(broadcast_var_7, broadcast_var_7));
      }
      __syncthreads();
      #pragma unroll
      for (int i_14 = 0; i_14 < 4; ++i_14) {
        if ((((((tile_i * 64) + (i_14 * 16)) + (((int)threadIdx.x) >> 4)) + (((int)blockIdx.y) * ((((total_kv - 1) / min(40, max(1, ((total_kv + 127) >> 7)))) + (((total_kv - 1) % min(40, max(1, ((total_kv + 127) >> 7)))) >> 31)) + 1))) < ((((int)blockIdx.y) + 1) * ((((total_kv - 1) / min(40, max(1, ((total_kv + 127) >> 7)))) + (((total_kv - 1) % min(40, max(1, ((total_kv + 127) >> 7)))) >> 31)) + 1))) && (((((tile_i * 64) + (i_14 * 16)) + (((int)threadIdx.x) >> 4)) + (((int)blockIdx.y) * ((((total_kv - 1) / min(40, max(1, ((total_kv + 127) >> 7)))) + (((total_kv - 1) % min(40, max(1, ((total_kv + 127) >> 7)))) >> 31)) + 1))) < total_kv)) {
          int condval_5;
          if ((((((((((int)threadIdx.x) >> 4) + (((int)blockIdx.y) * ((((total_kv - 1) / min(40, max(1, ((total_kv + 127) >> 7)))) + (((total_kv - 1) % min(40, max(1, ((total_kv + 127) >> 7)))) >> 31)) + 1))) >> 4) + i_14) >> 2) + tile_i) < 256)) {
            condval_5 = block_table[(((((int64_t)tile_i) * (int64_t)4) + (((((int64_t)((int)threadIdx.x)) >> (int64_t)4) + (((int64_t)((int)blockIdx.y)) * ((((((int64_t)total_kv) - (int64_t)1) / min((int64_t)40, max((int64_t)1, ((((int64_t)total_kv) + (int64_t)127) >> (int64_t)7)))) + (((((int64_t)total_kv) - (int64_t)1) % min((int64_t)40, max((int64_t)1, ((((int64_t)total_kv) + (int64_t)127) >> (int64_t)7)))) >> (int64_t)63)) + (int64_t)1))) >> (int64_t)4)) + ((int64_t)i_14))];
          } else {
            condval_5 = 0;
          }
          int physical_page_1 = condval_5;
          half_t broadcast_var_8 = half_t(0x0p+0f/*0.000000e+00*/);
          uint4 condval_6;
          if (((0 <= physical_page_1) && (physical_page_1 < 1024))) {
            condval_6 = *(uint4*)(V_cache + ((((((int64_t)physical_page_1) * (int64_t)4096) + ((((((int64_t)((int)threadIdx.x)) >> (int64_t)4) + (((int64_t)((int)blockIdx.y)) * ((((((int64_t)total_kv) - (int64_t)1) / min((int64_t)40, max((int64_t)1, ((((int64_t)total_kv) + (int64_t)127) >> (int64_t)7)))) + (((((int64_t)total_kv) - (int64_t)1) % min((int64_t)40, max((int64_t)1, ((((int64_t)total_kv) + (int64_t)127) >> (int64_t)7)))) >> (int64_t)63)) + (int64_t)1))) & (int64_t)15) * (int64_t)256)) + (((int64_t)((int)blockIdx.x)) * (int64_t)128)) + ((((int64_t)((int)threadIdx.x)) & (int64_t)15) * (int64_t)8)));
          } else {
            condval_6 = make_uint4(__pack_half2(broadcast_var_8, broadcast_var_8), __pack_half2(broadcast_var_8, broadcast_var_8), __pack_half2(broadcast_var_8, broadcast_var_8), __pack_half2(broadcast_var_8, broadcast_var_8));
          }
          *(uint4*)(((half_t*)buf_dyn_shmem) + (((((((i_14 * 2048) + ((((int)threadIdx.x) >> 6) * 512)) + ((((int)threadIdx.x) & 3) * 128)) + (((((int)threadIdx.x) & 15) >> 2) * 32)) + (((((((int)threadIdx.x) & 63) >> 5) + ((((int)threadIdx.x) & 3) >> 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + 16384)) = condval_6;
        }
      }
      #pragma unroll
      for (int i_15 = 0; i_15 < 8; ++i_15) {
        uint1 __1;
        float2 v_ = *(float2*)(acc_s + (i_15 * 2));
        ((half2*)(&__1))[0] = __float22half2_rn(((float2*)(&v_))[0]);
        *(uint1*)(P_shared_local_cast + 0) = __1;
        *(uint1*)(((half_t*)buf_dyn_shmem) + ((((((((((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 7) >> 2) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((i_15 & 1) * 128)) + ((((int)threadIdx.x) & 1) * 64)) + ((((int)threadIdx.x) >> 7) * 32)) + ((i_15 >> 1) * 8)) + (((((int)threadIdx.x) & 15) >> 3) * 4)) + (((((int)threadIdx.x) & 3) >> 1) * 2)) + 24576)) = *(uint1*)(P_shared_local_cast + 0);
      }
      __syncthreads();
      #pragma unroll
      for (int i_16 = 0; i_16 < 8; ++i_16) {
        *(uint4*)(acc_s_cast + (i_16 * 8)) = *(uint4*)(((half_t*)buf_dyn_shmem) + ((((((((((int)threadIdx.x) & 127) >> 5) * 1024) + (((((int)threadIdx.x) & 7) >> 2) * 512)) + (((((int)threadIdx.x) & 31) >> 4) * 256)) + ((((int)threadIdx.x) & 3) * 64)) + (i_16 * 8)) + 24576));
      }
      for (int ki_1 = 0; ki_1 < 16; ++ki_1) {
        for (int i_17 = 0; i_17 < 4; ++i_17) {
          *(uint2*)(B_local_1 + (i_17 * 4)) = *(uint2*)(((half_t*)buf_dyn_shmem) + (((((((((ki_1 * 512) + ((i_17 & 1) * 256)) + (((((int)threadIdx.x) & 31) >> 4) * 128)) + ((((int)threadIdx.x) >> 7) * 64)) + ((i_17 >> 1) * 32)) + (((((((int)threadIdx.x) & 3) >> 1) + (i_17 & 1)) & 1) * 16)) + (((((((int)threadIdx.x) & 31) >> 4) + (((int)threadIdx.x) & 1)) & 1) * 8)) + (((((int)threadIdx.x) & 15) >> 3) * 4)) + 16384));
        }
        for (int j_1 = 0; j_1 < 4; ++j_1) {
          tl::mma_sync_sm70<tl::DataType::kFloat16, tl::DataType::kFloat16, tl::DataType::kFloat32, 16, 16, 4, false, false>(reinterpret_cast<float*>(acc_o + (j_1 * 8)), reinterpret_cast<const unsigned*>(acc_s_cast + (ki_1 * 4)), reinterpret_cast<const unsigned*>(B_local_1 + (j_1 * 4)));
        }
      }
    }
    #pragma unroll
    for (int i_18 = 0; i_18 < 16; ++i_18) {
      if (((((((((int)threadIdx.x) & 127) >> 5) * 4) + (((((int)threadIdx.x) & 7) >> 2) * 2)) + ((((int)threadIdx.x) & 31) >> 4)) < q_len) & (bool)1) {
        uint1 __2;
        float2 __3;
          float2 v__1 = *(float2*)(acc_o + (i_18 * 2));
          float condval_7;
          if ((l_i[(i_18 & 1)] == 0x0p+0f/*0.000000e+00*/)) {
            condval_7 = 0x1p+0f/*1.000000e+00*/;
          } else {
            condval_7 = l_i[(i_18 & 1)];
          }
          float2 v__2 = make_float2(condval_7, condval_7);
          __3.x = (v__1.x/v__2.x);
          __3.y = (v__1.y/v__2.y);
        ((half2*)(&__2))[0] = __float22half2_rn(((float2*)(&__3))[0]);
        *(uint1*)(Partial_O_local_cast_1 + 0) = __2;
        *(uint1*)(Partial_O + (((((((((((((int)blockIdx.y) * 16384) + (((((int)threadIdx.x) & 127) >> 5) * 4096)) + (((((int)threadIdx.x) & 7) >> 2) * 2048)) + (((((int)threadIdx.x) & 31) >> 4) * 1024)) + (((int)blockIdx.x) * 512)) + ((i_18 & 1) * 256)) + ((((int)threadIdx.x) & 1) * 128)) + ((((int)threadIdx.x) >> 7) * 64)) + ((i_18 >> 1) * 8)) + (((((int)threadIdx.x) & 15) >> 3) * 4)) + (((((int)threadIdx.x) & 3) >> 1) * 2))) = *(uint1*)(Partial_O_local_cast_1 + 0);
      }
    }
    if ((((((((int)threadIdx.x) & 3) >> 1) * 4) + (((((int)threadIdx.x) & 15) >> 3) * 2)) + (((int)threadIdx.x) >> 7)) == 0) {
      #pragma unroll
      for (int i_19 = 0; i_19 < 2; ++i_19) {
        if (((((((((int)threadIdx.x) & 127) >> 5) * 4) + (((((int)threadIdx.x) & 7) >> 2) * 2)) + ((((int)threadIdx.x) & 31) >> 4)) < q_len) & (bool)1) {
          float condval_8;
          if ((l_i[i_19] == 0x0p+0f/*0.000000e+00*/)) {
            condval_8 = -0x1p+30f/*-1.073742e+09*/;
          } else {
            condval_8 = (log2f(l_i[i_19]) + (m_i[i_19] * scale_log2));
          }
          Partial_LSE[(((((((((int)blockIdx.y) * 128) + (((((int)threadIdx.x) & 127) >> 5) * 32)) + (((((int)threadIdx.x) & 7) >> 2) * 16)) + (((((int)threadIdx.x) & 31) >> 4) * 8)) + (((int)blockIdx.x) * 4)) + (i_19 * 2)) + (((int)threadIdx.x) & 1))] = condval_8;
        }
      }
    }
  }
}
