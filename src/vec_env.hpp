#pragma once

#include <functional>
#include <memory>
#include <optional>
#include <vector>

#include "base.hpp"
#include "device.hpp"
#include "environment.hpp"
#include "thread_pool.hpp"

namespace rl {

struct StepResultBatch {
    std::vector<f32>  rewards;
    std::vector<u8>   terminated; // 1 if terminated, 0 otherwise
    std::vector<u8>   truncated;  // 1 if truncated, 0 otherwise
    std::vector<u8>   dones;      // 1 if (terminated || truncated)
    std::vector<std::optional<std::vector<f32>>> terminal_obs;
    std::vector<f32>  ep_returns;
    std::vector<u32>  ep_lengths;
};

class VecEnv {
public:
    // Constructs VecEnv by cloning prototype_env num_envs times with decorrelated seeds
    VecEnv(const Env& prototype_env, u32 num_envs, u64 base_seed = 42,
           bool use_threads = true);

    // Constructs VecEnv from a factory lambda: std::function<std::unique_ptr<Env>(u32 env_id, u64 seed)>
    VecEnv(std::function<std::unique_ptr<Env>(u32 env_id, u64 seed)> factory,
           u32 num_envs, u64 base_seed = 42, bool use_threads = true);

    ~VecEnv() = default;

    VecEnv(const VecEnv&) = delete;
    VecEnv& operator=(const VecEnv&) = delete;
    VecEnv(VecEnv&&) noexcept = default;
    VecEnv& operator=(VecEnv&&) noexcept = default;

    // Resets all environments and populates the batch observation buffer
    void Reset(std::optional<u64> base_seed = std::nullopt);

    // Steps all environments with a flat actions buffer:
    // For discrete actions: size is [num_envs]
    // For continuous actions: size is [num_envs * action_dim]
    StepResultBatch Step(const std::vector<f32>& actions);

    // Transfers host batch observation buffer to GPU device memory if requested
    void SyncObsToDevice(Device dev);

    // Returns pointer to observation buffer on requested device (Host memory or CUDA GpuBuffer)
    [[nodiscard]] f32* GetObsPtr(Device dev) noexcept;
    [[nodiscard]] const f32* GetObsPtr(Device dev) const noexcept;

    [[nodiscard]] u32 NumEnvs() const noexcept { return num_envs_; }
    [[nodiscard]] u32 ObsDim() const noexcept { return obs_dim_; }
    [[nodiscard]] u32 ActionDim() const noexcept { return action_dim_; }
    [[nodiscard]] bool IsDiscreteAction() const noexcept { return is_discrete_action_; }
    [[nodiscard]] const std::vector<f32>& HostBatchObs() const noexcept { return batch_obs_; }
    [[nodiscard]] GpuBuffer& GpuObsBuffer() noexcept { return gpu_obs_; }

    [[nodiscard]] Env& GetEnv(u32 index) { return *envs_[index]; }
    [[nodiscard]] const Env& GetEnv(u32 index) const { return *envs_[index]; }

private:
    void InitializeBuffers();

    u32                              num_envs_           = 1;
    u32                              obs_dim_            = 1;
    u32                              action_dim_         = 1;
    bool                             is_discrete_action_ = true;
    bool                             use_threads_        = true;
    u64                              base_seed_          = 42;

    std::vector<std::unique_ptr<Env>> envs_;
    std::vector<f32>                 batch_obs_; // Flat shape: [num_envs, obs_dim]
    GpuBuffer                        gpu_obs_;   // Dual-backend GPU buffer
};

} // namespace rl
