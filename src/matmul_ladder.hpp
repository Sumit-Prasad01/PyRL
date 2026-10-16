#pragma once

#include "base.hpp"
#include "thread_pool.hpp"

namespace matmul {

// Rung 1: Naive triple loop (i, j, k) - Baseline
void Rung1_Naive(f32* C, const f32* A, const f32* B, u32 M, u32 K, u32 N) noexcept;

// Rung 2: Loop reordered (i, k, j) - Contiguous memory streaming
void Rung2_Reordered(f32* C, const f32* A, const f32* B, u32 M, u32 K, u32 N) noexcept;

// Rung 3: Cache blocking / tiling for L1/L2 cache residency
void Rung3_Tiled(f32* C, const f32* A, const f32* B, u32 M, u32 K, u32 N) noexcept;

// Rung 4: AVX2 + FMA SIMD vectorized micro-kernel
void Rung4_AVX2(f32* C, const f32* A, const f32* B, u32 M, u32 K, u32 N) noexcept;

// Rung 5: Multi-threaded parallel-for across CPU cores with AVX2 micro-kernel
void Rung5_Parallel(f32* C, const f32* A, const f32* B, u32 M, u32 K, u32 N,
                    ThreadPool* pool = nullptr) noexcept;

// Auto-dispatcher: chooses the highest-throughput CPU rung based on matrix dimensions
void OptimizedCPU(f32* C, const f32* A, const f32* B, u32 M, u32 K, u32 N) noexcept;

} // namespace matmul
