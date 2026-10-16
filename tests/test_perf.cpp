#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <vector>

#include "arena.hpp"
#include "autograd.hpp"
#include "base.hpp"
#include "device.hpp"
#include "fused.hpp"
#include "matmul_ladder.hpp"
#include "prng.hpp"

#ifdef RL_USE_CUDA
#include "cuda/cuda_backend.hpp"
#endif

TEST_CASE("Performance Ladder: Numerical equivalence across all CPU rungs", "[perf][matmul][cpu]")
{
    // Test on odd/prime dimensions to rigorously check SIMD and tiling boundary conditions
    constexpr u32 M = 67;
    constexpr u32 K = 73;
    constexpr u32 N = 97;

    Prng rng(42, 99);
    std::vector<f32> A(M * K);
    std::vector<f32> B(K * N);
    for (f32& v : A) { v = rng.NextFloat() * 2.0f - 1.0f; }
    for (f32& v : B) { v = rng.NextFloat() * 2.0f - 1.0f; }

    std::vector<f32> C_ref(M * N, 0.0f);
    matmul::Rung1_Naive(C_ref.data(), A.data(), B.data(), M, K, N);

    SECTION("Rung 2: Reordered matches Rung 1 Naive")
    {
        std::vector<f32> C_test(M * N, 0.0f);
        matmul::Rung2_Reordered(C_test.data(), A.data(), B.data(), M, K, N);
        for (u64 i = 0; i < C_ref.size(); ++i) {
            CHECK(std::fabs(C_test[i] - C_ref[i]) < 1e-4f);
        }
    }

    SECTION("Rung 3: Tiled matches Rung 1 Naive")
    {
        std::vector<f32> C_test(M * N, 0.0f);
        matmul::Rung3_Tiled(C_test.data(), A.data(), B.data(), M, K, N);
        for (u64 i = 0; i < C_ref.size(); ++i) {
            CHECK(std::fabs(C_test[i] - C_ref[i]) < 1e-4f);
        }
    }

    SECTION("Rung 4: AVX2 matches Rung 1 Naive")
    {
        std::vector<f32> C_test(M * N, 0.0f);
        matmul::Rung4_AVX2(C_test.data(), A.data(), B.data(), M, K, N);
        for (u64 i = 0; i < C_ref.size(); ++i) {
            CHECK(std::fabs(C_test[i] - C_ref[i]) < 1e-4f);
        }
    }

    SECTION("Rung 5: Parallel matches Rung 1 Naive")
    {
        std::vector<f32> C_test(M * N, 0.0f);
        matmul::Rung5_Parallel(C_test.data(), A.data(), B.data(), M, K, N);
        for (u64 i = 0; i < C_ref.size(); ++i) {
            CHECK(std::fabs(C_test[i] - C_ref[i]) < 1e-4f);
        }
    }

    SECTION("Auto-dispatcher OptimizedCPU matches Rung 1 Naive")
    {
        std::vector<f32> C_test(M * N, 0.0f);
        matmul::OptimizedCPU(C_test.data(), A.data(), B.data(), M, K, N);
        for (u64 i = 0; i < C_ref.size(); ++i) {
            CHECK(std::fabs(C_test[i] - C_ref[i]) < 1e-4f);
        }
    }
}

#ifdef RL_USE_CUDA
TEST_CASE("CUDA GPU Backend: Numerical parity with CPU reference", "[perf][cuda][gpu]")
{
    if (!HasCudaDevice()) {
        SKIP("CUDA GPU device not available on this environment.");
    }

    constexpr u32 M = 64;
    constexpr u32 K = 128;
    constexpr u32 N = 96;

    Prng rng(111, 222);
    std::vector<f32> A(M * K);
    std::vector<f32> B(K * N);
    std::vector<f32> C_ref(M * N, 0.0f);

    for (f32& v : A) { v = rng.NextFloat() * 2.0f - 1.0f; }
    for (f32& v : B) { v = rng.NextFloat() * 2.0f - 1.0f; }
    matmul::Rung1_Naive(C_ref.data(), A.data(), B.data(), M, K, N);

    GpuBuffer d_A(A.size() * sizeof(f32));
    GpuBuffer d_B(B.size() * sizeof(f32));
    GpuBuffer d_C(C_ref.size() * sizeof(f32));

    d_A.CopyFromHost(A.data(), A.size() * sizeof(f32));
    d_B.CopyFromHost(B.data(), B.size() * sizeof(f32));

    SECTION("CUDA Tiled MatMul matches CPU")
    {
        cuda::MatMul(d_C.AsFloat(), d_A.AsFloat(), d_B.AsFloat(), M, K, N);
        cuda::Synchronize();

        std::vector<f32> C_gpu(M * N, 0.0f);
        d_C.CopyToHost(C_gpu.data(), C_gpu.size() * sizeof(f32));

        for (u64 i = 0; i < C_ref.size(); ++i) {
            CHECK(std::fabs(C_gpu[i] - C_ref[i]) < 1e-4f);
        }
    }

    SECTION("CUDA Elementwise Relu and Tanh")
    {
        std::vector<f32> host_out(A.size(), 0.0f);
        cuda::Relu(d_A.AsFloat(), d_A.AsFloat(), A.size());
        cuda::Synchronize();
        d_A.CopyToHost(host_out.data(), host_out.size() * sizeof(f32));

        for (u64 i = 0; i < A.size(); ++i) {
            const f32 expected = std::max(0.0f, A[i]);
            CHECK(std::fabs(host_out[i] - expected) < 1e-5f);
        }
    }
}
#endif

TEST_CASE("Fused Operator: FusedLinearBiasRelu matches unfused pipeline", "[perf][fused]")
{
    constexpr u32 B = 16;
    constexpr u32 InDim = 32;
    constexpr u32 OutDim = 24;

    Prng rng(77, 88);
    std::vector<f32> x(B * InDim);
    std::vector<f32> w(InDim * OutDim);
    std::vector<f32> bias(OutDim);

    for (f32& v : x)    { v = rng.NextFloat() * 2.0f - 1.0f; }
    for (f32& v : w)    { v = rng.NextFloat() * 2.0f - 1.0f; }
    for (f32& v : bias) { v = rng.NextFloat() * 0.5f; }

    // 1. Unfused reference: MatMul + Add bias + ReLU
    std::vector<f32> unfused_out(B * OutDim, 0.0f);
    matmul::Rung1_Naive(unfused_out.data(), x.data(), w.data(), B, InDim, OutDim);
    for (u32 i = 0; i < B; ++i) {
        for (u32 j = 0; j < OutDim; ++j) {
            const f32 z = unfused_out[i * OutDim + j] + bias[j];
            unfused_out[i * OutDim + j] = std::max(0.0f, z);
        }
    }

    // 2. Fused CPU forward
    std::vector<f32> fused_out(B * OutDim, 0.0f);
    ops::FusedLinearBiasReluForward(fused_out.data(), x.data(), w.data(), bias.data(),
                                   B, InDim, OutDim, Device::CPU);

    for (u64 i = 0; i < unfused_out.size(); ++i) {
        CHECK(std::fabs(fused_out[i] - unfused_out[i]) < 1e-4f);
    }

#ifdef RL_USE_CUDA
    if (HasCudaDevice()) {
        GpuBuffer d_x(x.size() * sizeof(f32));
        GpuBuffer d_w(w.size() * sizeof(f32));
        GpuBuffer d_b(bias.size() * sizeof(f32));
        GpuBuffer d_out(fused_out.size() * sizeof(f32));

        d_x.CopyFromHost(x.data(), x.size() * sizeof(f32));
        d_w.CopyFromHost(w.data(), w.size() * sizeof(f32));
        d_b.CopyFromHost(bias.data(), bias.size() * sizeof(f32));

        cuda::FusedLinearBiasRelu(d_out.AsFloat(), d_x.AsFloat(), d_w.AsFloat(), d_b.AsFloat(),
                                 B, InDim, OutDim);
        cuda::Synchronize();

        std::vector<f32> gpu_out(B * OutDim, 0.0f);
        d_out.CopyToHost(gpu_out.data(), gpu_out.size() * sizeof(f32));

        for (u64 i = 0; i < unfused_out.size(); ++i) {
            CHECK(std::fabs(gpu_out[i] - unfused_out[i]) < 1e-4f);
        }
    }
#endif
}
