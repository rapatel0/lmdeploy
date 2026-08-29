// Standalone SM70 FP16 vocabulary-head driver for layout/native parity.
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/gemm.h"

#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace tb = turbomind;
namespace gemm = turbomind::gemm;
namespace {
constexpr int K = 5120;
constexpr int N = 62080;
constexpr int MaxM = 8;

void Check(cudaError_t ec, const char* what)
{
    if (ec != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(ec));
        std::exit(2);
    }
}

__device__ uint16_t ValueBits(size_t i, size_t j)
{
    constexpr uint16_t values[12] = {
        0x0000, 0x1800, 0x1c00, 0x2000, 0x2400, 0x2800, 0x9800, 0x9c00, 0xa000, 0xa400, 0xa800, 0x2c00};
    return values[(i * 17 + j * 13 + (i ^ j)) % 12];
}

__global__ void FillA(uint16_t* a)
{
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size_t(MaxM) * K;
         i += size_t(gridDim.x) * blockDim.x) {
        a[i] = ValueBits(i / K, i % K);
    }
}

__global__ void FillB(uint16_t* b, bool transposed)
{
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < size_t(K) * N;
         i += size_t(gridDim.x) * blockDim.x) {
        const size_t k = i / N;
        const size_t n = i % N;
        b[transposed ? n * K + k : i] = ValueBits(k, n);
    }
}

struct Buffers {
    half* a{};
    half* b{};
    half* d{};
};

void Allocate(Buffers& x, bool transposed)
{
    Check(cudaMalloc(&x.a, size_t(MaxM) * K * sizeof(half)), "cudaMalloc A");
    Check(cudaMalloc(&x.b, size_t(K) * N * sizeof(half)), "cudaMalloc B");
    Check(cudaMalloc(&x.d, size_t(MaxM) * N * sizeof(half)), "cudaMalloc D");
    FillA<<<256, 256>>>((uint16_t*)x.a);
    FillB<<<4096, 256>>>((uint16_t*)x.b, transposed);
    Check(cudaGetLastError(), "fill kernels");
}

int Run(gemm::Gemm& runner, const gemm::Workspace& workspace, Buffers& x, int m, bool transposed, cudaStream_t stream)
{
    gemm::Operation op{};
    op.dispatch = gemm::DispatchPolicy::kMeasure;
    gemm::MatrixLayout a{tb::kHalf, gemm::kRowMajor, m, K, K};
    gemm::MatrixLayout b{tb::kHalf, transposed ? gemm::kColMajor : gemm::kRowMajor, K, N, transposed ? K : N};
    gemm::MatrixLayout d{tb::kHalf, gemm::kRowMajor, m, N, N};
    gemm::MatrixLayout empty{};
    return runner.Run(op, 1.f, x.a, a, nullptr, empty, x.b, b, nullptr, empty, 0.f, x.d, d, x.d, d, nullptr,
                      empty, workspace, stream);
}

void Dump(const Buffers& x, int m, const char* dir)
{
    std::vector<uint16_t> output(size_t(m) * N);
    Check(cudaMemcpy(output.data(), x.d, output.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "cudaMemcpy D");
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/m%d.bin", dir, m);
    std::FILE* file = std::fopen(path, "wb");
    if (!file || std::fwrite(output.data(), sizeof(uint16_t), output.size(), file) != output.size()) {
        std::fprintf(stderr, "failed to write %s\n", path);
        std::exit(4);
    }
    std::fclose(file);
}
}  // namespace

int main()
{
    Check(cudaSetDevice(0), "cudaSetDevice");
    cudaStream_t stream{};
    Check(cudaStreamCreate(&stream), "cudaStreamCreate");
    gemm::Workspace workspace{};
    workspace.barriers_size = gemm::Gemm::kBarriersSize;
    workspace.partials_size = gemm::Gemm::kPartialsSize;
    workspace.tensormaps_size = 1 << 20;
    Check(cudaMalloc(&workspace.barriers, workspace.barriers_size), "cudaMalloc barriers");
    Check(cudaMalloc(&workspace.partials, workspace.partials_size), "cudaMalloc partials");
    Check(cudaMalloc(&workspace.tensormaps, workspace.tensormaps_size), "cudaMalloc tensormaps");
    Check(cudaMalloc(&workspace.flags, sizeof(int)), "cudaMalloc flags");
    Check(cudaMemset(workspace.barriers, 0, workspace.barriers_size), "cudaMemset barriers");

    const int mode = gemm::Sm70Fp16FlatHeadMode();
    const bool transposed = mode != 0;
    Buffers buffers;
    Allocate(buffers, transposed);
    gemm::Gemm runner;
    for (int m : {1, 7, 8}) {
        if (Run(runner, workspace, buffers, m, transposed, stream) != 0) return 3;
    }
    Check(cudaStreamSynchronize(stream), "warmup sync");
    if (const char* dir = std::getenv("TM_FP16_HEAD_DUMP_DIR")) {
        for (int m : {1, 7, 8}) {
            if (Run(runner, workspace, buffers, m, transposed, stream) != 0) return 3;
            Check(cudaStreamSynchronize(stream), "dump sync");
            Dump(buffers, m, dir);
        }
    }
    std::printf("SM70_FP16_FLAT_HEAD_MICRO_PASS mode=%d transposed=%d\n", mode, int(transposed));
    return 0;
}
