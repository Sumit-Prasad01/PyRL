#pragma once

#include <memory>
#include <vector>

#include "arena.hpp"
#include "autograd.hpp"
#include "buffer.hpp"
#include "nn.hpp"
#include "optimizer.hpp"
#include "prng.hpp"

namespace rl {

struct SACConfig {
    f32 policy_lr    = 3e-4f;
    f32 q_lr         = 1e-3f;
    f32 alpha_lr     = 3e-4f;
    f32 gamma        = 0.99f;
    f32 tau          = 0.005f;
    u32 batch_size   = 64;
    u32 buffer_size  = 10000;
    f32 init_alpha   = 0.2f;
    bool auto_alpha  = true;
    u32 hidden_dim   = 64;
};

struct SACStats {
    f32 actor_loss  = 0.0f;
    f32 critic_loss = 0.0f;
    f32 alpha       = 0.2f;
    u64 train_step  = 0;
};

class ContinuousActor {
public:
    ContinuousActor(Arena& arena, ModelState& model, u32 obs_dim, u32 act_dim,
                    u32 hidden_dim = 64, Prng* rng = nullptr);

    // Returns reparameterized squashed action and log probability
    void Sample(const f32* obs, u32 batch_size, Prng& rng,
                std::vector<f32>& out_actions, std::vector<f32>& out_log_probs);

    // Returns deterministic mean action: tanh(mu)
    void ActDeterministic(const f32* obs, u32 batch_size, std::vector<f32>& out_actions);

    [[nodiscard]] std::vector<Var*> Parameters() const;

private:
    u32 obs_dim_;
    u32 act_dim_;
    u32 hidden_dim_;

    std::unique_ptr<nn::Linear> fc1_;
    std::unique_ptr<nn::Linear> fc2_;
    std::unique_ptr<nn::Linear> mu_head_;
    std::unique_ptr<nn::Linear> log_std_head_;
};

class ContinuousCritic {
public:
    ContinuousCritic(Arena& arena, ModelState& model, u32 obs_dim, u32 act_dim,
                     u32 hidden_dim = 64, Prng* rng = nullptr);

    // Predicts Q(s, a): input is concat(obs, action) of size [batch_size, obs_dim + act_dim]
    void Forward(const f32* obs_action, u32 batch_size, std::vector<f32>& out_q);

    void CopyWeightsFrom(const ContinuousCritic& src);
    void SoftUpdateFrom(const ContinuousCritic& src, f32 tau);

    [[nodiscard]] std::vector<Var*> Parameters() const;

private:
    u32 input_dim_;
    u32 hidden_dim_;

    std::unique_ptr<nn::Linear> fc1_;
    std::unique_ptr<nn::Linear> fc2_;
    std::unique_ptr<nn::Linear> out_head_;
};

class SAC {
public:
    SAC(Arena& arena, ModelState& model, u32 obs_dim, u32 act_dim,
        SACConfig config = {}, Prng* rng = nullptr);

    [[nodiscard]] std::vector<f32> Act(const std::vector<f32>& obs, Prng& rng,
                                       bool deterministic = false);

    SACStats TrainStep(const std::vector<f32>& obs, const std::vector<f32>& action,
                       f32 reward, const std::vector<f32>& next_obs, bool done, Prng& rng);

    [[nodiscard]] const SACConfig& Config() const noexcept { return config_; }

private:
    SACConfig config_;
    u32 obs_dim_;
    u32 act_dim_;
    u64 step_count_ = 0;
    f32 log_alpha_  = -1.6094f; // log(0.2)
    f32 target_entropy_ = -1.0f;

    std::unique_ptr<ContinuousActor>  actor_;
    std::unique_ptr<ContinuousCritic> q1_;
    std::unique_ptr<ContinuousCritic> q2_;
    std::unique_ptr<ContinuousCritic> q1_target_;
    std::unique_ptr<ContinuousCritic> q2_target_;

    std::unique_ptr<optimizer::Adam> actor_opt_;
    std::unique_ptr<optimizer::Adam> q_opt_;

    std::unique_ptr<ReplayBuffer> buffer_;
};

} // namespace rl
