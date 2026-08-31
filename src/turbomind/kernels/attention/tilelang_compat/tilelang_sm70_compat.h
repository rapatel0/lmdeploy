// Minimal Apache-2.0 TileLang compatibility for the generated SM70 verifier.
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/fast_math.h>
#include <cutlass/numeric_types.h>

#include <cstdint>
#include <type_traits>

using cutlass::half_t;
using uchar = unsigned char;

#define TL_DEVICE __forceinline__ __device__

TL_DEVICE unsigned __pack_half2(const half_t x, const half_t y)
{
    union {
        half_t h[2];
        unsigned u;
    } value{{x, y}};
    return value.u;
}

namespace tl {

enum class DataType : int {
    kFloat16,
    kFloat32,
};

struct SumOp {
    template<typename T>
    TL_DEVICE T operator()(const T& x, const T& y)
    {
        return x + y;
    }
};

struct MaxOp {
    template<typename T>
    TL_DEVICE T operator()(const T& x, const T& y)
    {
        return cutlass::fast_max(x, y);
    }
};

template<class Reducer, int threads, int scale, int thread_offset = 0>
struct AllReduce {
    static_assert(threads % scale == 0);

    template<typename T>
    static TL_DEVICE T run(T x, T* red_buf = nullptr)
    {
        if constexpr (threads == scale) {
            return x;
        }
        else {
            constexpr int offset = threads / 2;
            if constexpr (offset >= 32) {
                __syncthreads();
                red_buf[threadIdx.x - thread_offset] = x;
                __syncthreads();
                x = Reducer{}(x, red_buf[(threadIdx.x - thread_offset) ^ offset]);
            }
            else {
                x = Reducer{}(x, __shfl_xor_sync(uint32_t(-1), x, offset));
            }
            if constexpr (offset == scale) {
                return x;
            }
            else {
                return AllReduce<Reducer, offset, scale, thread_offset>::run(x, red_buf);
            }
        }
    }
};

template<DataType AType,
         DataType BType,
         DataType CType,
         int M,
         int N,
         int K,
         bool TransA,
         bool TransB>
TL_DEVICE void mma_sync_sm70(float* c, const unsigned* a, const unsigned* b)
{
    static_assert(AType == DataType::kFloat16);
    static_assert(BType == DataType::kFloat16);
    static_assert(CType == DataType::kFloat32);
    static_assert(M == 16 && N == 16 && K == 4);
    static_assert(!TransA);
    if constexpr (TransB) {
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                     "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]),
                       "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                     : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
    }
    else {
        asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                     "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]),
                       "+f"(c[4]), "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
                     : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]));
    }
}

}  // namespace tl
