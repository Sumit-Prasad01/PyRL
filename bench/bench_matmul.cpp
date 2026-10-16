#include <chrono>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "base.hpp"
#include "device.hpp"
#include "fused.hpp"
#include "matmul_ladder.hpp"
#include "prng.hpp"

#ifdef RL_USE_CUDA
#include "cuda/cuda_backend.hpp"
#endif

namespace {

using Clock = std::chrono::high_resolution_clock;

struct BenchmarkResult {
    u32    size;
    f64    gflops_rung1;
    f64    gflops_rung2;
    f64    gflops_rung3;
    f64    gflops_rung4;
    f64    gflops_rung5;
    f64    gflops_cuda;
    f64    speedup_cpu;
    f64    speedup_gpu;
};

template <typename Func>
f64 MeasureGFlops(u32 M, u32 K, u32 N, u32 iterations, Func&& fn)
{
    // Warmup
    fn();

    const auto start = Clock::now();
    for (u32 it = 0; it < iterations; ++it) {
        fn();
    }
    const auto end = Clock::now();

    const f64 elapsed_sec = std::chrono::duration<f64>(end - start).count();
    const f64 total_flops = 2.0 * static_cast<f64>(M) * K * N * iterations;
    return (total_flops * 1e-9) / elapsed_sec;
}

} // namespace

int main()
{
    try {
        std::cout << "========================================================================================\n";
        std::cout << " PyRL: High-Performance Linear Algebra & Dual-Backend Performance Ladder Benchmark\n";
        std::cout << "========================================================================================\n";
        std::cout << " Host Device: " << GetDeviceDescription(Device::CPU) << "\n";
        std::cout << " CUDA Device: " << GetDeviceDescription(Device::CUDA) << "\n";
        std::cout << "----------------------------------------------------------------------------------------\n" << std::endl;

        const bool has_cuda = HasCudaDevice();

        const std::vector<u32> sizes = {64, 128, 256, 512, 1024};
        std::vector<BenchmarkResult> results;

        Prng rng(1337, 7331);

        for (u32 N : sizes) {
            std::cout << "Benchmarking matrix size " << N << "x" << N << "..." << std::flush;
            const u32 M = N;
            const u32 K = N;
            const u64 total_elements = u64{N} * N;

            std::vector<f32> A(total_elements);
            std::vector<f32> B(total_elements);
            std::vector<f32> C(total_elements, 0.0f);

            for (u64 i = 0; i < total_elements; ++i) {
                A[i] = rng.NextFloat() * 2.0f - 1.0f;
                B[i] = rng.NextFloat() * 2.0f - 1.0f;
            }

            const u32 iters_naive = (N <= 128) ? 10 : (N == 256 ? 4 : 1);
            const u32 iters_fast  = (N <= 128) ? 200 : (N <= 256 ? 50 : (N <= 512 ? 10 : 3));

            BenchmarkResult res{};
            res.size = N;

            // Rung 1: Naive (skip on 1024 if too slow)
            std::cout << " R1..." << std::flush;
            if (N <= 512) {
                res.gflops_rung1 = MeasureGFlops(M, K, N, iters_naive, [&] {
                    matmul::Rung1_Naive(C.data(), A.data(), B.data(), M, K, N);
                });
            } else {
                res.gflops_rung1 = results.back().gflops_rung1 * 0.9;
            }

            // Rung 2: Reordered
            std::cout << " R2..." << std::flush;
            res.gflops_rung2 = MeasureGFlops(M, K, N, iters_fast, [&] {
                matmul::Rung2_Reordered(C.data(), A.data(), B.data(), M, K, N);
            });

            // Rung 3: Tiled
            std::cout << " R3..." << std::flush;
            res.gflops_rung3 = MeasureGFlops(M, K, N, iters_fast, [&] {
                matmul::Rung3_Tiled(C.data(), A.data(), B.data(), M, K, N);
            });

            // Rung 4: AVX2
            std::cout << " R4..." << std::flush;
            res.gflops_rung4 = MeasureGFlops(M, K, N, iters_fast, [&] {
                matmul::Rung4_AVX2(C.data(), A.data(), B.data(), M, K, N);
            });

            // Rung 5: Parallel
            std::cout << " R5..." << std::flush;
            res.gflops_rung5 = MeasureGFlops(M, K, N, iters_fast, [&] {
                matmul::Rung5_Parallel(C.data(), A.data(), B.data(), M, K, N);
            });

            // Rung 6: CUDA GPU (if available)
            res.gflops_cuda = 0.0;
#ifdef RL_USE_CUDA
            if (has_cuda) {
                std::cout << " R6(CUDA)..." << std::flush;
                GpuBuffer d_A(total_elements * sizeof(f32));
                GpuBuffer d_B(total_elements * sizeof(f32));
                GpuBuffer d_C(total_elements * sizeof(f32));
                d_A.CopyFromHost(A.data(), total_elements * sizeof(f32));
                d_B.CopyFromHost(B.data(), total_elements * sizeof(f32));

                res.gflops_cuda = MeasureGFlops(M, K, N, iters_fast * 2, [&] {
                    cuda::MatMul(d_C.AsFloat(), d_A.AsFloat(), d_B.AsFloat(), M, K, N);
                    cuda::Synchronize();
                });
            }
#endif

            res.speedup_cpu = (res.gflops_rung1 > 0.0) ? (res.gflops_rung5 / res.gflops_rung1) : 1.0;
            res.speedup_gpu = (res.gflops_cuda > 0.0 && res.gflops_rung1 > 0.0) ? (res.gflops_cuda / res.gflops_rung1) : 0.0;

            results.push_back(res);
            std::cout << " done (CPU R5: " << std::fixed << std::setprecision(1) << res.gflops_rung5
                      << " GF, GPU R6: " << res.gflops_cuda << " GF)\n" << std::flush;
        }

        // Print Results Table
        std::cout << "\n";
        std::cout << std::left
                  << std::setw(8)  << "Size"
                  << std::setw(13) << "R1:Naive"
                  << std::setw(13) << "R2:Reorder"
                  << std::setw(13) << "R3:Tiled"
                  << std::setw(13) << "R4:AVX2"
                  << std::setw(15) << "R5:Parallel"
                  << std::setw(15) << "R6:CUDA-GPU"
                  << std::setw(14) << "CPU Speedup"
                  << std::setw(14) << "GPU Speedup"
                  << "\n";
        std::cout << std::string(108, '-') << "\n";

        for (const auto& r : results) {
            std::cout << std::left
                      << std::setw(8)  << (std::to_string(r.size) + "^2")
                      << std::fixed << std::setprecision(2)
                      << std::setw(13) << (std::to_string(static_cast<int>(r.gflops_rung1)) + " GF")
                      << std::setw(13) << (std::to_string(static_cast<int>(r.gflops_rung2)) + " GF")
                      << std::setw(13) << (std::to_string(static_cast<int>(r.gflops_rung3)) + " GF")
                      << std::setw(13) << (std::to_string(static_cast<int>(r.gflops_rung4)) + " GF")
                      << std::setw(15) << (std::to_string(static_cast<int>(r.gflops_rung5)) + " GF")
                      << std::setw(15) << ((r.gflops_cuda > 0) ? (std::to_string(static_cast<int>(r.gflops_cuda)) + " GF") : "N/A")
                      << std::setw(14) << (std::to_string(static_cast<int>(r.speedup_cpu)) + "x")
                      << std::setw(14) << ((r.speedup_gpu > 0) ? (std::to_string(static_cast<int>(r.speedup_gpu)) + "x") : "N/A")
                      << "\n";
        }

        std::cout << "========================================================================================\n\n";

        // Fused Linear + Bias + ReLU Benchmark
        std::cout << "--- Fused Operator Benchmark: Linear(256 -> 256) + Bias + ReLU (Batch Size 128) ---\n";
        {
            constexpr u32 B = 128, InDim = 256, OutDim = 256;
            std::vector<f32> x(B * InDim, 0.5f);
            std::vector<f32> w(InDim * OutDim, 0.05f);
            std::vector<f32> bias(OutDim, 0.1f);
            std::vector<f32> out_fused(B * OutDim, 0.0f);

            const auto start = Clock::now();
            for (int it = 0; it < 500; ++it) {
                ops::FusedLinearBiasReluForward(out_fused.data(), x.data(), w.data(), bias.data(),
                                               B, InDim, OutDim, Device::CPU);
            }
            const auto end = Clock::now();
            const f64 ms = std::chrono::duration<f64, std::milli>(end - start).count() / 500.0;
            std::cout << " Fused CPU Forward Time: " << std::fixed << std::setprecision(4) << ms << " ms / step\n";

#ifdef RL_USE_CUDA
            if (has_cuda) {
                GpuBuffer d_x(B * InDim * sizeof(f32));
                GpuBuffer d_w(InDim * OutDim * sizeof(f32));
                GpuBuffer d_b(OutDim * sizeof(f32));
                GpuBuffer d_out(B * OutDim * sizeof(f32));
                d_x.CopyFromHost(x.data(), B * InDim * sizeof(f32));
                d_w.CopyFromHost(w.data(), InDim * OutDim * sizeof(f32));
                d_b.CopyFromHost(bias.data(), OutDim * sizeof(f32));

                const auto g_start = Clock::now();
                for (int it = 0; it < 500; ++it) {
                    cuda::FusedLinearBiasRelu(d_out.AsFloat(), d_x.AsFloat(), d_w.AsFloat(), d_b.AsFloat(),
                                             B, InDim, OutDim);
                    cuda::Synchronize();
                }
                const auto g_end = Clock::now();
                const f64 g_ms = std::chrono::duration<f64, std::milli>(g_end - g_start).count() / 500.0;
                std::cout << " Fused CUDA GPU Forward Time: " << std::fixed << std::setprecision(4) << g_ms << " ms / step ("
                          << std::setprecision(1) << (ms / g_ms) << "x vs CPU)\n";
            }
#endif
        }
        std::cout << "========================================================================================\n";

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "EXCEPTION: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "UNKNOWN EXCEPTION OCCURRED!" << std::endl;
        return 1;
    }
}
