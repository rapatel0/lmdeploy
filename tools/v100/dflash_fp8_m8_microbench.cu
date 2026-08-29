// Standalone single-GPU driver for profiling TurboMind's SM70 block-FP8 M=8 GEMM.
#include "src/turbomind/kernels/gemm/gemm.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdio>
#include <cstdlib>

namespace tb = turbomind;
namespace gemm = turbomind::gemm;

namespace {

void Check(cudaError_t ec, const char* what)
{
    if (ec != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(ec));
        std::exit(2);
    }
}

struct Buffers {
    void* a{};
    void* b{};
    void* v{};
    void* d{};
};

struct Problem {
    const char* name;
    int n;
    int k;
    gemm::Epilogue epilogue;
    Buffers buffers;
};

void Allocate(Problem& p, int m)
{
    const size_t a_count = static_cast<size_t>(m) * p.k;
    const size_t b_count = static_cast<size_t>(p.k) * p.n;
    const size_t v_count = static_cast<size_t>(p.k / 128) * p.n;
    const size_t d_count = static_cast<size_t>(m) * p.n;

    Check(cudaMalloc(&p.buffers.a, a_count * sizeof(half)), "cudaMalloc A");
    Check(cudaMalloc(&p.buffers.b, b_count), "cudaMalloc B");
    Check(cudaMalloc(&p.buffers.v, v_count * sizeof(uint16_t)), "cudaMalloc V");
    Check(cudaMalloc(&p.buffers.d, d_count * sizeof(half)), "cudaMalloc D");

    constexpr std::array<uint16_t, 10> finite_f16{
        0x0000, 0x2000, 0x2400, 0x2800, 0x2c00, 0xa000, 0xa400, 0xa800, 0xac00, 0x3000};
    auto* a = static_cast<uint16_t*>(std::malloc(a_count * sizeof(uint16_t)));
    if (!a) {
        std::fprintf(stderr, "host allocation failed for A\n");
        std::exit(2);
    }
    for (size_t i = 0; i < a_count; ++i) {
        a[i] = finite_f16[(i * 17 + i / 31) % finite_f16.size()];
    }
    constexpr std::array<char, 12> finite_e4m3{
        0x00, 0x18, 0x20, 0x28, 0x30, 0x38, 0x40, static_cast<char>(0x98), static_cast<char>(0xa0),
        static_cast<char>(0xa8), static_cast<char>(0xb0), static_cast<char>(0xb8)};
    auto* b = static_cast<char*>(std::malloc(b_count));
    auto* v = static_cast<uint16_t*>(std::malloc(v_count * sizeof(uint16_t)));
    if (!b || !v) {
        std::fprintf(stderr, "host allocation failed for B/V\n");
        std::exit(2);
    }
    for (size_t i = 0; i < b_count; ++i) {
        b[i] = finite_e4m3[(i * 13 + i / 97) % finite_e4m3.size()];
    }
    constexpr std::array<uint16_t, 6> finite_scales{0x2000, 0x2200, 0x2400, 0x2600, 0x2800, 0x2a00};
    for (size_t i = 0; i < v_count; ++i) {
        v[i] = finite_scales[(i * 7 + i / 19) % finite_scales.size()];
    }

    Check(cudaMemcpy(p.buffers.a, a, a_count * sizeof(uint16_t), cudaMemcpyHostToDevice), "cudaMemcpy A");
    Check(cudaMemcpy(p.buffers.b, b, b_count, cudaMemcpyHostToDevice), "cudaMemcpy B");
    Check(cudaMemcpy(p.buffers.v, v, v_count * sizeof(uint16_t), cudaMemcpyHostToDevice), "cudaMemcpy V");
    Check(cudaMemset(p.buffers.d, 0, d_count * sizeof(half)), "cudaMemset D");
    std::free(a);
    std::free(b);
    std::free(v);
}

void Dump(const Problem& p, int m, const char* dir)
{
    const size_t count = static_cast<size_t>(m) * p.n;
    auto* output = static_cast<uint16_t*>(std::malloc(count * sizeof(uint16_t)));
    if (!output) {
        std::fprintf(stderr, "host allocation failed for output\n");
        std::exit(4);
    }
    Check(cudaMemcpy(output, p.buffers.d, count * sizeof(uint16_t), cudaMemcpyDeviceToHost), "cudaMemcpy D");
    char path[4096];
    if (std::snprintf(path, sizeof(path), "%s/%s.bin", dir, p.name) < 0) {
        std::fprintf(stderr, "failed to format output path\n");
        std::exit(4);
    }
    std::FILE* file = std::fopen(path, "wb");
    if (!file || std::fwrite(output, sizeof(uint16_t), count, file) != count) {
        std::fprintf(stderr, "failed to write %s\n", path);
        std::exit(4);
    }
    std::fclose(file);
    std::free(output);
}

void Free(Problem& p)
{
    cudaFree(p.buffers.a);
    cudaFree(p.buffers.b);
    cudaFree(p.buffers.v);
    cudaFree(p.buffers.d);
}

int Run(gemm::Gemm& runner, const gemm::Workspace& workspace, Problem& p, int m, cudaStream_t stream)
{
    gemm::Operation op{};
    // The production path autotunes kernel launch parameters per problem. The
    // first call measures; subsequent calls reuse the exact cached launch.
    op.dispatch  = gemm::DispatchPolicy::kMeasure;
    op.epilogue  = p.epilogue;
    op.quant_a   = {gemm::QuantType::kNone, 0};
    op.quant_b   = {gemm::QuantType::kK, 128};
    op.batch_dim = 0;

    gemm::MatrixLayout a{tb::kHalf, gemm::kRowMajor, m, p.k, p.k};
    gemm::MatrixLayout u{};
    gemm::MatrixLayout b{tb::kFloat8_e4m3,
                         gemm::kColMajor,
                         p.k,
                         p.n,
                         p.k,
                         static_cast<gemm::Pack>(gemm::HMMA_884 | gemm::OPERAND_B | 1)};
    gemm::MatrixLayout v{tb::kUint16,
                         gemm::kColMajor,
                         p.k / 128,
                         p.n,
                         p.k / 128,
                         static_cast<gemm::Pack>(gemm::HMMA_884 | gemm::OPERAND_V | 1)};
    gemm::MatrixLayout d{tb::kHalf, gemm::kRowMajor, m, p.n, p.n};
    gemm::MatrixLayout w{};

    return runner.Run(op,
                      1.f,
                      p.buffers.a,
                      a,
                      nullptr,
                      u,
                      p.buffers.b,
                      b,
                      p.buffers.v,
                      v,
                      0.f,
                      p.buffers.d,
                      d,
                      p.buffers.d,
                      d,
                      nullptr,
                      w,
                      workspace,
                      stream);
}

}  // namespace

int main()
{
    constexpr int m = 8;
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

    std::array<Problem, 3> problems{{
        {"gate_up", 8704, 5120, gemm::Epilogue::kGatedSilu, {}},
        {"down", 5120, 4352, gemm::Epilogue::kNone, {}},
        {"out", 5120, 1536, gemm::Epilogue::kNone, {}},
    }};
    for (auto& problem : problems) {
        Allocate(problem, m);
    }

    gemm::Gemm runner;
    for (auto& problem : problems) {
        if (Run(runner, workspace, problem, m, stream) != 0) {
            std::fprintf(stderr, "warmup failed for %s\n", problem.name);
            return 3;
        }
    }
    Check(cudaStreamSynchronize(stream), "warmup sync");

    Check(cudaProfilerStart(), "cudaProfilerStart");
    for (auto& problem : problems) {
        if (Run(runner, workspace, problem, m, stream) != 0) {
            std::fprintf(stderr, "profile launch failed for %s\n", problem.name);
            return 3;
        }
    }
    Check(cudaStreamSynchronize(stream), "profile sync");
    Check(cudaProfilerStop(), "cudaProfilerStop");
    if (const char* dump_dir = std::getenv("TM_FP8_M8_DUMP_DIR")) {
        for (const auto& problem : problems) {
            Dump(problem, m, dump_dir);
        }
    }
    std::puts("DFLASH_FP8_M8_MICROBENCH_COMPLETE");

    for (auto& problem : problems) {
        Free(problem);
    }
    cudaFree(workspace.barriers);
    cudaFree(workspace.partials);
    cudaFree(workspace.tensormaps);
    cudaFree(workspace.flags);
    cudaStreamDestroy(stream);
    return 0;
}
