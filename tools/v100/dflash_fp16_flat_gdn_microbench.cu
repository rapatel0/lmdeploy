// Standalone SM70 FP16 GDN projection driver for layout/native parity and NCU.
#include "src/turbomind/kernels/gemm/convert.h"
#include "src/turbomind/kernels/gemm/gemm.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace tb = turbomind;
namespace gemm = turbomind::gemm;

namespace {

constexpr int K = 5120;
constexpr int N = 4120;
constexpr int MaxM = 8;

void Check(cudaError_t ec, const char* what)
{
    if (ec != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(ec));
        std::exit(2);
    }
}

uint16_t ValueBits(size_t i, size_t j)
{
    constexpr std::array<uint16_t, 12> values{
        0x0000, 0x1800, 0x1c00, 0x2000, 0x2400, 0x2800, 0x9800, 0x9c00, 0xa000, 0xa400, 0xa800, 0x2c00};
    return values[(i * 17 + j * 13 + (i ^ j)) % values.size()];
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

    std::vector<uint16_t> a(size_t(MaxM) * K);
    std::vector<uint16_t> b(size_t(K) * N);
    for (int m = 0; m < MaxM; ++m) {
        for (int k = 0; k < K; ++k) {
            a[size_t(m) * K + k] = ValueBits(m, k);
        }
    }
    for (int k = 0; k < K; ++k) {
        for (int n = 0; n < N; ++n) {
            const size_t physical = transposed ? size_t(n) * K + k : size_t(k) * N + n;
            b[physical] = ValueBits(k, n);
        }
    }
    Check(cudaMemcpy(x.a, a.data(), a.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "cudaMemcpy A");
    Check(cudaMemcpy(x.b, b.data(), b.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "cudaMemcpy B");
    Check(cudaMemset(x.d, 0, size_t(MaxM) * N * sizeof(half)), "cudaMemset D");
}

int Run(gemm::Gemm& runner, const gemm::Workspace& workspace, Buffers& x, int m, bool transposed, cudaStream_t stream)
{
    gemm::Operation op{};
    op.dispatch = gemm::DispatchPolicy::kMeasure;
    gemm::MatrixLayout a{tb::kHalf, gemm::kRowMajor, m, K, K};
    gemm::MatrixLayout b{tb::kHalf, transposed ? gemm::kColMajor : gemm::kRowMajor, K, N, transposed ? K : N};
    gemm::MatrixLayout d{tb::kHalf, gemm::kRowMajor, m, N, N};
    gemm::MatrixLayout empty{};
    return runner.Run(op,
                      1.f,
                      x.a,
                      a,
                      nullptr,
                      empty,
                      x.b,
                      b,
                      nullptr,
                      empty,
                      0.f,
                      x.d,
                      d,
                      x.d,
                      d,
                      nullptr,
                      empty,
                      workspace,
                      stream);
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
    workspace.barriers_size   = gemm::Gemm::kBarriersSize;
    workspace.partials_size   = gemm::Gemm::kPartialsSize;
    workspace.tensormaps_size = 1 << 20;
    Check(cudaMalloc(&workspace.barriers, workspace.barriers_size), "cudaMalloc barriers");
    Check(cudaMalloc(&workspace.partials, workspace.partials_size), "cudaMalloc partials");
    Check(cudaMalloc(&workspace.tensormaps, workspace.tensormaps_size), "cudaMalloc tensormaps");
    Check(cudaMalloc(&workspace.flags, sizeof(int)), "cudaMalloc flags");
    Check(cudaMemset(workspace.barriers, 0, workspace.barriers_size), "cudaMemset barriers");

    const int mode = gemm::Sm70Fp16FlatGdnMode();
    const bool transposed = mode != 0;
    Buffers buffers;
    Allocate(buffers, transposed);
    gemm::Gemm runner;
    for (int m : {1, 7, 8}) {
        if (Run(runner, workspace, buffers, m, transposed, stream) != 0) {
            std::fprintf(stderr, "SM70_FP16_FLAT_GDN_MICRO_FAIL mode=%d m=%d\n", mode, m);
            return 3;
        }
    }
    Check(cudaStreamSynchronize(stream), "warmup sync");
    Check(cudaProfilerStart(), "cudaProfilerStart");
    for (int repeat = 0; repeat < 20; ++repeat) {
        for (int m : {1, 7, 8}) {
            if (Run(runner, workspace, buffers, m, transposed, stream) != 0) {
                return 3;
            }
        }
    }
    Check(cudaStreamSynchronize(stream), "profile sync");
    Check(cudaProfilerStop(), "cudaProfilerStop");
    if (const char* dir = std::getenv("TM_FP16_GDN_DUMP_DIR")) {
        for (int m : {1, 7, 8}) {
            Run(runner, workspace, buffers, m, transposed, stream);
            Check(cudaStreamSynchronize(stream), "dump sync");
            Dump(buffers, m, dir);
        }
    }
    std::printf("SM70_FP16_FLAT_GDN_MICRO_PASS mode=%d transposed=%d\n", mode, int(transposed));
    return 0;
}
