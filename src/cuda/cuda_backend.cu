#include "cuda_backend.hpp"

#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

namespace cuda {

namespace {

constexpr int kTileDim = 16;

__global__ void MatMulTiledKernel(float* __restrict__ C,
                                  const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  int M, int K, int N)
{
    __shared__ float sA[kTileDim][kTileDim];
    __shared__ float sB[kTileDim][kTileDim];

    const int row = blockIdx.y * kTileDim + threadIdx.y;
    const int col = blockIdx.x * kTileDim + threadIdx.x;

    float sum = 0.0f;
    const int num_tiles = (K + kTileDim - 1) / kTileDim;

    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * kTileDim + threadIdx.x;
        const int b_row = t * kTileDim + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTileDim; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

__global__ void FusedLinearBiasReluKernel(float* __restrict__ out,
                                         const float* __restrict__ x,
                                         const float* __restrict__ w,
                                         const float* __restrict__ bias,
                                         int B, int in_dim, int out_dim)
{
    __shared__ float sA[kTileDim][kTileDim];
    __shared__ float sB[kTileDim][kTileDim];

    const int row = blockIdx.y * kTileDim + threadIdx.y;
    const int col = blockIdx.x * kTileDim + threadIdx.x;

    float sum = 0.0f;
    const int num_tiles = (in_dim + kTileDim - 1) / kTileDim;

    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * kTileDim + threadIdx.x;
        const int b_row = t * kTileDim + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < B && a_col < in_dim) ? x[row * in_dim + a_col] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (b_row < in_dim && col < out_dim) ? w[b_row * out_dim + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTileDim; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < B && col < out_dim) {
        const float b_val = (bias != nullptr) ? bias[col] : 0.0f;
        const float res   = sum + b_val;
        out[row * out_dim + col] = fmaxf(0.0f, res);
    }
}

__global__ void ReluKernel(float* out, const float* in, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = fmaxf(0.0f, in[idx]);
    }
}

__global__ void TanhKernel(float* out, const float* in, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = tanhf(in[idx]);
    }
}

__global__ void SigmoidKernel(float* out, const float* in, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = 1.0f / (1.0f + expf(-in[idx]));
    }
}

__global__ void AddKernel(float* C, const float* A, const float* B, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

__global__ void SubKernel(float* C, const float* A, const float* B, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] - B[idx];
    }
}

__global__ void MulKernel(float* C, const float* A, const float* B, int size)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

} // namespace

bool Init() noexcept
{
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        return false;
    }
    return (cudaSetDevice(0) == cudaSuccess);
}

void MatMul(f32* d_C, const f32* d_A, const f32* d_B, u32 M, u32 K, u32 N, cudaStream_t stream)
{
    if (M == 0 || K == 0 || N == 0) { return; }

    const dim3 block(kTileDim, kTileDim);
    const dim3 grid((N + kTileDim - 1) / kTileDim, (M + kTileDim - 1) / kTileDim);

    MatMulTiledKernel<<<grid, block, 0, stream>>>(d_C, d_A, d_B, static_cast<int>(M),
                                                  static_cast<int>(K), static_cast<int>(N));
}

void FusedLinearBiasRelu(f32* d_out, const f32* d_x, const f32* d_w, const f32* d_b,
                        u32 B, u32 in_dim, u32 out_dim, cudaStream_t stream)
{
    if (B == 0 || in_dim == 0 || out_dim == 0) { return; }

    const dim3 block(kTileDim, kTileDim);
    const dim3 grid((out_dim + kTileDim - 1) / kTileDim, (B + kTileDim - 1) / kTileDim);

    FusedLinearBiasReluKernel<<<grid, block, 0, stream>>>(d_out, d_x, d_w, d_b,
                                                          static_cast<int>(B),
                                                          static_cast<int>(in_dim),
                                                          static_cast<int>(out_dim));
}

void Relu(f32* d_out, const f32* d_in, u64 size, cudaStream_t stream)
{
    if (size == 0) { return; }
    constexpr int kBlock = 256;
    const int grid = static_cast<int>((size + kBlock - 1) / kBlock);
    ReluKernel<<<grid, kBlock, 0, stream>>>(d_out, d_in, static_cast<int>(size));
}

void Tanh(f32* d_out, const f32* d_in, u64 size, cudaStream_t stream)
{
    if (size == 0) { return; }
    constexpr int kBlock = 256;
    const int grid = static_cast<int>((size + kBlock - 1) / kBlock);
    TanhKernel<<<grid, kBlock, 0, stream>>>(d_out, d_in, static_cast<int>(size));
}

void Sigmoid(f32* d_out, const f32* d_in, u64 size, cudaStream_t stream)
{
    if (size == 0) { return; }
    constexpr int kBlock = 256;
    const int grid = static_cast<int>((size + kBlock - 1) / kBlock);
    SigmoidKernel<<<grid, kBlock, 0, stream>>>(d_out, d_in, static_cast<int>(size));
}

void Add(f32* d_C, const f32* d_A, const f32* d_B, u64 size, cudaStream_t stream)
{
    if (size == 0) { return; }
    constexpr int kBlock = 256;
    const int grid = static_cast<int>((size + kBlock - 1) / kBlock);
    AddKernel<<<grid, kBlock, 0, stream>>>(d_C, d_A, d_B, static_cast<int>(size));
}

void Sub(f32* d_C, const f32* d_A, const f32* d_B, u64 size, cudaStream_t stream)
{
    if (size == 0) { return; }
    constexpr int kBlock = 256;
    const int grid = static_cast<int>((size + kBlock - 1) / kBlock);
    SubKernel<<<grid, kBlock, 0, stream>>>(d_C, d_A, d_B, static_cast<int>(size));
}

void Mul(f32* d_C, const f32* d_A, const f32* d_B, u64 size, cudaStream_t stream)
{
    if (size == 0) { return; }
    constexpr int kBlock = 256;
    const int grid = static_cast<int>((size + kBlock - 1) / kBlock);
    MulKernel<<<grid, kBlock, 0, stream>>>(d_C, d_A, d_B, static_cast<int>(size));
}

void Synchronize(cudaStream_t stream)
{
    if (stream != nullptr) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }
}

} // namespace cuda
