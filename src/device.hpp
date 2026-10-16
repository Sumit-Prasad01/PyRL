#pragma once

#include <cstddef>
#include <string>
#include <string_view>

#include "base.hpp"

enum class Device : u8 {
    CPU  = 0,
    CUDA = 1,
};

[[nodiscard]] constexpr std::string_view DeviceName(Device dev) noexcept
{
    switch (dev) {
    case Device::CPU:  return "CPU";
    case Device::CUDA: return "CUDA";
    default:           return "Unknown";
    }
}

// Queries if an NVIDIA CUDA GPU device is available and functional.
[[nodiscard]] bool HasCudaDevice() noexcept;

// Returns device description (e.g. "NVIDIA GeForce RTX 3050 Laptop GPU" or "CPU (AVX2)")
[[nodiscard]] std::string GetDeviceDescription(Device dev);

// Returns the number of physical CPU cores.
[[nodiscard]] u32 GetCpuCoreCount() noexcept;

// RAII Device Buffer for GPU memory management
class GpuBuffer {
public:
    GpuBuffer() noexcept : ptr_(nullptr), size_bytes_(0) {}
    explicit GpuBuffer(std::size_t bytes);
    ~GpuBuffer();

    GpuBuffer(const GpuBuffer&) = delete;
    GpuBuffer& operator=(const GpuBuffer&) = delete;

    GpuBuffer(GpuBuffer&& other) noexcept;
    GpuBuffer& operator=(GpuBuffer&& other) noexcept;

    void Allocate(std::size_t bytes);
    void Free() noexcept;

    void CopyFromHost(const void* host_src, std::size_t bytes);
    void CopyToHost(void* host_dst, std::size_t bytes) const;
    void CopyFromDevice(const void* dev_src, std::size_t bytes);
    void Zero() noexcept;

    [[nodiscard]] void* Data() noexcept { return ptr_; }
    [[nodiscard]] const void* Data() const noexcept { return ptr_; }
    [[nodiscard]] f32* AsFloat() noexcept { return static_cast<f32*>(ptr_); }
    [[nodiscard]] const f32* AsFloat() const noexcept { return static_cast<const f32*>(ptr_); }
    [[nodiscard]] std::size_t SizeBytes() const noexcept { return size_bytes_; }
    [[nodiscard]] bool IsAllocated() const noexcept { return ptr_ != nullptr; }

private:
    void*       ptr_        = nullptr;
    std::size_t size_bytes_ = 0;
};
