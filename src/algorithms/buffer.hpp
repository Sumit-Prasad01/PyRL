#pragma once

#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <vector>

#include "base.hpp"
#include "device.hpp"
#include "prng.hpp"

namespace rl {

// ------------------------------------------------------------- RolloutBuffer
// Fixed-size trajectory buffer for on-policy algorithms (REINFORCE, A2C, PPO)
struct Minibatch {
    std::vector<f32> obs;        // [batch_size * obs_dim]
    std::vector<f32> actions;    // [batch_size * act_dim]
    std::vector<f32> log_probs;  // [batch_size]
    std::vector<f32> advantages; // [batch_size]
    std::vector<f32> returns;    // [batch_size]
    std::vector<f32> values;     // [batch_size]
};

class RolloutBuffer {
public:
    RolloutBuffer(u32 num_steps, u32 num_envs, u32 obs_dim, u32 act_dim,
                  bool is_discrete = true);

    void Reset();

    // Inserts a single environment transition across all parallel environments
    void Insert(const f32* obs, const f32* actions, const f32* rewards,
                const u8* dones, const f32* values, const f32* log_probs);

    // Computes Generalized Advantage Estimation (GAE) and Monte Carlo targets
    void ComputeReturnsAndGAE(const f32* last_values, const u8* last_dones,
                              f32 gamma = 0.99f, f32 gae_lambda = 0.95f);

    // Normalizes advantages globally or per-minibatch
    void NormalizeAdvantages() noexcept;

    // Generates randomized minibatches for PPO training epochs
    [[nodiscard]] std::vector<Minibatch> GenerateMinibatches(u32 minibatch_size, Prng& rng) const;

    // Dual-Backend device support: transfers full rollout buffer to GPU memory
    void SyncToDevice(Device dev);

    [[nodiscard]] u32 NumSteps() const noexcept { return num_steps_; }
    [[nodiscard]] u32 NumEnvs() const noexcept { return num_envs_; }
    [[nodiscard]] u32 TotalSamples() const noexcept { return num_steps_ * num_envs_; }
    [[nodiscard]] u32 ObsDim() const noexcept { return obs_dim_; }
    [[nodiscard]] u32 ActDim() const noexcept { return act_dim_; }
    [[nodiscard]] u32 StepIndex() const noexcept { return step_idx_; }
    [[nodiscard]] bool IsFull() const noexcept { return step_idx_ == num_steps_; }

    [[nodiscard]] const std::vector<f32>& Obs() const noexcept { return obs_; }
    [[nodiscard]] const std::vector<f32>& Actions() const noexcept { return actions_; }
    [[nodiscard]] const std::vector<f32>& Returns() const noexcept { return returns_; }
    [[nodiscard]] const std::vector<f32>& Advantages() const noexcept { return advantages_; }
    [[nodiscard]] const std::vector<f32>& LogProbs() const noexcept { return log_probs_; }
    [[nodiscard]] const std::vector<f32>& Values() const noexcept { return values_; }
    [[nodiscard]] GpuBuffer& GpuObsBuffer() noexcept { return gpu_obs_; }

private:
    u32 num_steps_   = 0;
    u32 num_envs_    = 0;
    u32 obs_dim_     = 0;
    u32 act_dim_     = 0;
    bool is_discrete_ = true;
    u32 step_idx_    = 0;

    std::vector<f32> obs_;        // [num_steps * num_envs * obs_dim]
    std::vector<f32> actions_;    // [num_steps * num_envs * act_dim]
    std::vector<f32> rewards_;    // [num_steps * num_envs]
    std::vector<u8>  dones_;      // [num_steps * num_envs]
    std::vector<f32> values_;     // [num_steps * num_envs]
    std::vector<f32> log_probs_;  // [num_steps * num_envs]
    std::vector<f32> advantages_; // [num_steps * num_envs]
    std::vector<f32> returns_;    // [num_steps * num_envs]

    GpuBuffer gpu_obs_; // Dual backend GPU buffer
};

// -------------------------------------------------------------- ReplayBuffer
// Ring buffer for off-policy algorithms (DQN, SAC)
struct TransitionBatch {
    std::vector<f32> obs;      // [batch_size * obs_dim]
    std::vector<f32> actions;  // [batch_size * act_dim]
    std::vector<f32> rewards;  // [batch_size]
    std::vector<f32> next_obs; // [batch_size * obs_dim]
    std::vector<f32> dones;    // [batch_size]
    std::vector<f32> weights;  // [batch_size] (for prioritized replay)
    std::vector<u64> indices;  // [batch_size] (for priority updating)
};

class ReplayBuffer {
public:
    ReplayBuffer(u32 capacity, u32 obs_dim, u32 act_dim, bool is_discrete = true);

    void Push(const std::vector<f32>& obs, const std::vector<f32>& action,
              f32 reward, const std::vector<f32>& next_obs, bool done);

    [[nodiscard]] TransitionBatch Sample(u32 batch_size, Prng& rng) const;

    [[nodiscard]] u32 Size() const noexcept { return size_; }
    [[nodiscard]] u32 Capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool IsReady(u32 batch_size) const noexcept { return size_ >= batch_size; }

private:
    u32 capacity_    = 10000;
    u32 obs_dim_     = 1;
    u32 act_dim_     = 1;
    bool is_discrete_ = true;
    u32 pos_         = 0;
    u32 size_        = 0;

    std::vector<f32> obs_;
    std::vector<f32> actions_;
    std::vector<f32> rewards_;
    std::vector<f32> next_obs_;
    std::vector<f32> dones_;
};

// -------------------------------------------------- PrioritizedReplayBuffer
// SumTree-based proportional prioritized experience replay
class PrioritizedReplayBuffer {
public:
    PrioritizedReplayBuffer(u32 capacity, u32 obs_dim, u32 act_dim,
                            f32 alpha = 0.6f, f32 beta = 0.4f, f32 beta_anneal = 0.0001f);

    void Push(const std::vector<f32>& obs, const std::vector<f32>& action,
              f32 reward, const std::vector<f32>& next_obs, bool done);

    [[nodiscard]] TransitionBatch Sample(u32 batch_size, Prng& rng);

    void UpdatePriorities(const std::vector<u64>& indices, const std::vector<f32>& td_errors);

    [[nodiscard]] u32 Size() const noexcept { return size_; }
    [[nodiscard]] u32 Capacity() const noexcept { return capacity_; }

private:
    void TreeUpdate(u64 tree_idx, f32 priority);
    [[nodiscard]] u64 TreeGet(f32 s) const;

    u32 capacity_;
    u32 obs_dim_;
    u32 act_dim_;
    f32 alpha_ = 0.6f;
    f32 beta_  = 0.4f;
    f32 beta_anneal_ = 0.0001f;
    f32 max_priority_ = 1.0f;

    u32 pos_  = 0;
    u32 size_ = 0;

    std::vector<f32> sum_tree_; // Size: 2 * capacity
    std::vector<f32> obs_;
    std::vector<f32> actions_;
    std::vector<f32> rewards_;
    std::vector<f32> next_obs_;
    std::vector<f32> dones_;
};

} // namespace rl
