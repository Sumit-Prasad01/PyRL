#include "matmul_ladder.hpp"

#include <algorithm>
#include <cstring>

#if defined(__AVX2__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#define RL_HAS_AVX2 1
#endif

namespace matmul {

void Rung1_Naive(f32* C, const f32* A, const f32* B, u32 M, u32 K, u32 N) noexcept
{
    if (M == 0 || K == 0 || N == 0) { return; }

    for (u32 i = 0; i < M; ++i) {
        for (u32 j = 0; j < N; ++j) {
            f32 sum = 0.0f;
            for (u32 k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

void Rung2_Reordered(f32* C, const f32* A, const f32* B, u32 M, u32 K, u32 N) noexcept
{
    if (M == 0 || K == 0 || N == 0) { return; }

    std::memset(C, 0, u64{M} * N * sizeof(f32));

    for (u32 i = 0; i < M; ++i) {
        f32* const c_row = C + i * N;
        for (u32 k = 0; k < K; ++k) {
            const f32 a_ik = A[i * K + k];
            const f32* const b_row = B + k * N;
            for (u32 j = 0; j < N; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

void Rung3_Tiled(f32* C, const f32* A, const f32* B, u32 M, u32 K, u32 N) noexcept
{
    if (M == 0 || K == 0 || N == 0) { return; }

    std::memset(C, 0, u64{M} * N * sizeof(f32));

    // Cache blocking factors targeted for L1/L2 data cache
    constexpr u32 MC = 64;
    constexpr u32 KC = 64;
    constexpr u32 NC = 64;

    for (u32 ii = 0; ii < M; ii += MC) {
        const u32 i_end = std::min(ii + MC, M);
        for (u32 kk = 0; kk < K; kk += KC) {
            const u32 k_end = std::min(kk + KC, K);
            for (u32 jj = 0; jj < N; jj += NC) {
                const u32 j_end = std::min(jj + NC, N);

                for (u32 i = ii; i < i_end; ++i) {
                    f32* const c_row = C + i * N;
                    for (u32 k = kk; k < k_end; ++k) {
                        const f32 a_ik = A[i * K + k];
                        const f32* const b_row = B + k * N;
                        for (u32 j = jj; j < j_end; ++j) {
                            c_row[j] += a_ik * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

void Rung4_AVX2(f32* C, const f32* A, const f32* B, u32 M, u32 K, u32 N) noexcept
{
    if (M == 0 || K == 0 || N == 0) { return; }

    std::memset(C, 0, u64{M} * N * sizeof(f32));

#if defined(RL_HAS_AVX2)
    constexpr u32 MC = 64;
    constexpr u32 KC = 64;

    for (u32 ii = 0; ii < M; ii += MC) {
        const u32 i_end = std::min(ii + MC, M);
        for (u32 kk = 0; kk < K; kk += KC) {
            const u32 k_end = std::min(kk + KC, K);

            for (u32 i = ii; i < i_end; ++i) {
                f32* const c_row = C + i * N;
                for (u32 k = kk; k < k_end; ++k) {
                    const f32 a_val = A[i * K + k];
                    const __m256 a_broadcast = _mm256_set1_ps(a_val);
                    const f32* const b_row = B + k * N;

                    u32 j = 0;
                    // Unroll 2x8 floats per vector iteration
                    for (; j + 15 < N; j += 16) {
                        __m256 c0 = _mm256_loadu_ps(c_row + j);
                        __m256 c1 = _mm256_loadu_ps(c_row + j + 8);
                        const __m256 b0 = _mm256_loadu_ps(b_row + j);
                        const __m256 b1 = _mm256_loadu_ps(b_row + j + 8);

                        c0 = _mm256_fmadd_ps(a_broadcast, b0, c0);
                        c1 = _mm256_fmadd_ps(a_broadcast, b1, c1);

                        _mm256_storeu_ps(c_row + j, c0);
                        _mm256_storeu_ps(c_row + j + 8, c1);
                    }

                    // Single 8-float vector loop
                    for (; j + 7 < N; j += 8) {
                        __m256 c0 = _mm256_loadu_ps(c_row + j);
                        const __m256 b0 = _mm256_loadu_ps(b_row + j);
                        c0 = _mm256_fmadd_ps(a_broadcast, b0, c0);
                        _mm256_storeu_ps(c_row + j, c0);
                    }

                    // Scalar tail
                    for (; j < N; ++j) {
                        c_row[j] += a_val * b_row[j];
                    }
                }
            }
        }
    }
#else
    Rung3_Tiled(C, A, B, M, K, N);
#endif
}

void Rung5_Parallel(f32* C, const f32* A, const f32* B, u32 M, u32 K, u32 N,
                    ThreadPool* pool) noexcept
{
    if (M == 0 || K == 0 || N == 0) { return; }

    ThreadPool& tp = (pool != nullptr) ? *pool : ThreadPool::Global();

    // Parallelize over M (rows of C)
    constexpr u32 kRowBlock = 32;

    tp.ParallelFor(0, M, [&](u32 row_start, u32 row_end) {
        if (row_start >= row_end) { return; }
        // Zero only the rows assigned to this thread
        const u64 chunk_bytes = u64{row_end - row_start} * N * sizeof(f32);
        std::memset(C + u64{row_start} * N, 0, chunk_bytes);

#if defined(RL_HAS_AVX2)
        for (u32 ii = row_start; ii < row_end; ii += kRowBlock) {
            const u32 i_end = std::min(ii + kRowBlock, row_end);

            for (u32 kk = 0; kk < K; kk += 64) {
                const u32 k_end = std::min(kk + 64, K);

                for (u32 i = ii; i < i_end; ++i) {
                    f32* const c_row = C + i * N;
                    for (u32 k = kk; k < k_end; ++k) {
                        const f32 a_val = A[i * K + k];
                        const __m256 a_broadcast = _mm256_set1_ps(a_val);
                        const f32* const b_row = B + k * N;

                        u32 j = 0;
                        for (; j + 15 < N; j += 16) {
                            __m256 c0 = _mm256_loadu_ps(c_row + j);
                            __m256 c1 = _mm256_loadu_ps(c_row + j + 8);
                            const __m256 b0 = _mm256_loadu_ps(b_row + j);
                            const __m256 b1 = _mm256_loadu_ps(b_row + j + 8);

                            c0 = _mm256_fmadd_ps(a_broadcast, b0, c0);
                            c1 = _mm256_fmadd_ps(a_broadcast, b1, c1);

                            _mm256_storeu_ps(c_row + j, c0);
                            _mm256_storeu_ps(c_row + j + 8, c1);
                        }

                        for (; j + 7 < N; j += 8) {
                            __m256 c0 = _mm256_loadu_ps(c_row + j);
                            const __m256 b0 = _mm256_loadu_ps(b_row + j);
                            c0 = _mm256_fmadd_ps(a_broadcast, b0, c0);
                            _mm256_storeu_ps(c_row + j, c0);
                        }

                        for (; j < N; ++j) {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
#else
        for (u32 i = row_start; i < row_end; ++i) {
            f32* const c_row = C + i * N;
            for (u32 k = 0; k < K; ++k) {
                const f32 a_ik = A[i * K + k];
                const f32* const b_row = B + k * N;
                for (u32 j = 0; j < N; ++j) {
                    c_row[j] += a_ik * b_row[j];
                }
            }
        }
#endif
    });
}

void OptimizedCPU(f32* C, const f32* A, const f32* B, u32 M, u32 K, u32 N) noexcept
{
    // For small matrices (e.g. RL policy rollout on a single observation),
    // thread dispatch overhead exceeds computation time. Use AVX2 single-threaded.
    // For batch sizes / matrix sizes with >= 65,536 FLOPs, multithread.
    const u64 total_ops = u64{M} * K * N;
    if (total_ops >= 64ULL * 64 * 64 && M >= 8) {
        Rung5_Parallel(C, A, B, M, K, N);
    } else {
        Rung4_AVX2(C, A, B, M, K, N);
    }
}

} // namespace matmul
