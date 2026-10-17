#pragma once

#include <memory>
#include <vector>

#include "arena.hpp"
#include "autograd.hpp"
#include "buffer.hpp"
#include "optimizer.hpp"
#include "policy.hpp"

namespace rl {

struct DQNConfig {
    f32  lr                  = 2.5e-4f;
    f32  gamma               = 0.99f;
    u32  batch_size          = 64;
    u32  buffer_size         = 10000;
    u32  target_update_freq  = 500;
    f32  tau                 = 1.0f; // 1.0 for hard periodic copy, < 1.0 for Polyak soft update
    f32  eps_start           = 1.0f;
    f32  eps_end             = 0.05f;
    u32  eps_decay_steps     = 10000;
    bool double_dqn          = true;
    bool dueling             = true;
    bool prioritized         = false;
    u32  hidden_dim          = 64;
};

struct DQNStats {
    f32 loss       = 0.0f;
    f32 mean_q     = 0.0f;
    f32 epsilon    = 1.0f;
    u64 train_step = 0;
};

class DQN {
public:
    DQN(Arena& arena, ModelState& model, u32 obs_dim, u32 act_dim,
        DQNConfig config = {}, Prng* rng = nullptr);

    // Selects an action using epsilon-greedy exploration
    [[nodiscard]] u32 Act(const std::vector<f32>& obs, Prng& rng);

    // Records transition and performs a gradient descent update step if buffer is ready
    DQNStats TrainStep(const std::vector<f32>& obs, u32 action, f32 reward,
                       const std::vector<f32>& next_obs, bool done, Prng& rng);

    [[nodiscard]] DuelingQNetwork& OnlineNet() noexcept { return *online_net_; }
    [[nodiscard]] DuelingQNetwork& TargetNet() noexcept { return *target_net_; }
    [[nodiscard]] const DQNConfig& Config() const noexcept { return config_; }

private:
    DQNConfig                        config_;
    u32                              obs_dim_ = 1;
    u32                              act_dim_ = 1;
    f32                              epsilon_ = 1.0f;
    u64                              step_count_ = 0;

    std::unique_ptr<DuelingQNetwork> online_net_;
    std::unique_ptr<DuelingQNetwork> target_net_;
    std::unique_ptr<optimizer::Adam> opt_;

    std::unique_ptr<ReplayBuffer>            uniform_buffer_;
    std::unique_ptr<PrioritizedReplayBuffer> per_buffer_;
};

} // namespace rl
