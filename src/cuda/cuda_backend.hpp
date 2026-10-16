#pragma once

#include "base.hpp"

#if defined(RL_USE_CUDA) || defined(__CUDACC__) || defined(__CUDA__)
#include <cuda_runtime.h>
#else
using cudaStream_t = void*;
#endif

namespace cuda {

// Initializes the CUDA GPU runtime and prints device information.
bool Init() noexcept;

// Rung 6: Tiled shared-memory matrix multiplication: C = A * B
// A is [M, K], B is [K, N], C is [M, N]
void MatMul(f32* d_C, const f32* d_A, const f32* d_B, u32 M, u32 K, u32 N, cudaStream_t stream = nullptr);

// Fused Linear + Bias + ReLU: out = max(0, x * w + bias)
// x is [B, in_dim], w is [in_dim, out_dim], bias is [1, out_dim], out is [B, out_dim]
void FusedLinearBiasRelu(f32* d_out, const f32* d_x, const f32* d_w, const f32* d_b,
                        u32 B, u32 in_dim, u32 out_dim, cudaStream_t stream = nullptr);

// Elementwise activations
void Relu(f32* d_out, const f32* d_in, u64 size, cudaStream_t stream = nullptr);
void Tanh(f32* d_out, const f32* d_in, u64 size, cudaStream_t stream = nullptr);
void Sigmoid(f32* d_out, const f32* d_in, u64 size, cudaStream_t stream = nullptr);

// Elementwise binary arithmetic
void Add(f32* d_C, const f32* d_A, const f32* d_B, u64 size, cudaStream_t stream = nullptr);
void Sub(f32* d_C, const f32* d_A, const f32* d_B, u64 size, cudaStream_t stream = nullptr);
void Mul(f32* d_C, const f32* d_A, const f32* d_B, u64 size, cudaStream_t stream = nullptr);

// Device synchronization
void Synchronize(cudaStream_t stream = nullptr);

} // namespace cuda
