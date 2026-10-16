#include "device.hpp"

#include <stdexcept>
#include <thread>
#include <utility>

#ifdef RL_USE_CUDA
#include <cuda_runtime.h>
#endif

bool HasCudaDevice() noexcept
{
#ifdef RL_USE_CUDA
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    return (err == cudaSuccess && count > 0);
#else
    return false;
#endif
}

std::string GetDeviceDescription(Device dev)
{
    if (dev == Device::CPU) {
        return "CPU (x86_64, AVX2 + FMA, " + std::to_string(GetCpuCoreCount()) + " threads)";
    }
#ifdef RL_USE_CUDA
    if (dev == Device::CUDA && HasCudaDevice()) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
            return std::string(prop.name) + " (Compute " +
                   std::to_string(prop.major) + "." + std::to_string(prop.minor) + ", " +
                   std::to_string(prop.totalGlobalMem / (1024 * 1024)) + " MB VRAM)";
        }
        return "NVIDIA CUDA GPU (Active)";
    }
#endif
    return "CUDA GPU (Not available)";
}

u32 GetCpuCoreCount() noexcept
{
    const u32 count = std::thread::hardware_concurrency();
    return (count > 0) ? count : 4;
}

// ---------------------------------------------------------------- GpuBuffer

GpuBuffer::GpuBuffer(std::size_t bytes)
{
    Allocate(bytes);
}

GpuBuffer::~GpuBuffer()
{
    Free();
}

GpuBuffer::GpuBuffer(GpuBuffer&& other) noexcept
    : ptr_(std::exchange(other.ptr_, nullptr)),
      size_bytes_(std::exchange(other.size_bytes_, 0))
{
}

GpuBuffer& GpuBuffer::operator=(GpuBuffer&& other) noexcept
{
    if (this != &other) {
        Free();
        ptr_        = std::exchange(other.ptr_, nullptr);
        size_bytes_ = std::exchange(other.size_bytes_, 0);
    }
    return *this;
}

void GpuBuffer::Allocate(std::size_t bytes)
{
    Free();
    if (bytes == 0) { return; }

#ifdef RL_USE_CUDA
    const cudaError_t err = cudaMalloc(&ptr_, bytes);
    if (err != cudaSuccess || ptr_ == nullptr) {
        throw std::runtime_error(std::string("GpuBuffer: cudaMalloc failed: ") + cudaGetErrorString(err));
    }
    size_bytes_ = bytes;
#else
    throw std::runtime_error("GpuBuffer: CUDA support was not compiled in this build.");
#endif
}

void GpuBuffer::Free() noexcept
{
    if (ptr_ != nullptr) {
#ifdef RL_USE_CUDA
        cudaFree(ptr_);
#endif
        ptr_ = nullptr;
    }
    size_bytes_ = 0;
}

void GpuBuffer::CopyFromHost(const void* host_src, std::size_t bytes)
{
    if (bytes == 0) { return; }
    if (ptr_ == nullptr || bytes > size_bytes_) {
        throw std::runtime_error("GpuBuffer: buffer not allocated or size too small for CopyFromHost.");
    }
#ifdef RL_USE_CUDA
    const cudaError_t err = cudaMemcpy(ptr_, host_src, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("GpuBuffer: cudaMemcpy HostToDevice failed: ") + cudaGetErrorString(err));
    }
#else
    (void)host_src;
    throw std::runtime_error("GpuBuffer: CUDA not compiled in.");
#endif
}

void GpuBuffer::CopyToHost(void* host_dst, std::size_t bytes) const
{
    if (bytes == 0) { return; }
    if (ptr_ == nullptr || bytes > size_bytes_) {
        throw std::runtime_error("GpuBuffer: buffer not allocated or size too small for CopyToHost.");
    }
#ifdef RL_USE_CUDA
    const cudaError_t err = cudaMemcpy(host_dst, ptr_, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("GpuBuffer: cudaMemcpy DeviceToHost failed: ") + cudaGetErrorString(err));
    }
#else
    (void)host_dst;
    throw std::runtime_error("GpuBuffer: CUDA not compiled in.");
#endif
}

void GpuBuffer::CopyFromDevice(const void* dev_src, std::size_t bytes)
{
    if (bytes == 0) { return; }
    if (ptr_ == nullptr || bytes > size_bytes_) {
        throw std::runtime_error("GpuBuffer: buffer not allocated or size too small for CopyFromDevice.");
    }
#ifdef RL_USE_CUDA
    const cudaError_t err = cudaMemcpy(ptr_, dev_src, bytes, cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("GpuBuffer: cudaMemcpy DeviceToDevice failed: ") + cudaGetErrorString(err));
    }
#else
    (void)dev_src;
    throw std::runtime_error("GpuBuffer: CUDA not compiled in.");
#endif
}

void GpuBuffer::Zero() noexcept
{
    if (ptr_ != nullptr && size_bytes_ > 0) {
#ifdef RL_USE_CUDA
        cudaMemset(ptr_, 0, size_bytes_);
#endif
    }
}
