#pragma once

#include <memory>
#include <vector>

#include "arena.hpp"
#include "autograd.hpp"
#include "buffer.hpp"
#include "device.hpp"
#include "optimizer.hpp"
#include "policy.hpp"
#include "vec_env.hpp"

namespace rl {

struct PPOConfig {
    f32    lr               = 2.5e-4f;
    f32    gamma            = 0.99f;
    f32    gae_lambda       = 0.95f;
    u32    num_steps        = 128;
    u32    num_epochs       = 4;
    u32    num_minibatches  = 4;
    f32    clip_coef        = 0.2f;
    bool   clip_vloss       = true;
    f32    ent_coef         = 0.01f;
    f32    vf_coef          = 0.5f;
    f32    max_grad_norm    = 0.5f;
    f32    target_kl        = 0.015f;
    bool   norm_adv         = true;
    bool   anneal_lr        = true;
    u32    hidden_dim       = 64;
    Device device           = Device::CPU;
};

struct PPOStats {
    f32 policy_loss   = 0.0f;
    f32 value_loss    = 0.0f;
    f32 entropy       = 0.0f;
    f32 approx_kl     = 0.0f;
    f32 clip_fraction = 0.0f;
    f32 grad_norm     = 0.0f;
};

class PPO {
public:
    PPO(Arena& arena, ModelState& model, u32 obs_dim, u32 act_dim,
        PPOConfig config = {}, Prng* rng = nullptr);

    // Executes one full PPO rollout collection and multiple minibatch training epochs
    PPOStats TrainIteration(VecEnv& vec_env, Prng& rng, f32 lr_scale = 1.0f);

    [[nodiscard]] ActorCritic& Policy() noexcept { return *ac_; }
    [[nodiscard]] const PPOConfig& Config() const noexcept { return config_; }

private:
    PPOConfig                    config_;
    std::unique_ptr<ActorCritic> ac_;
    std::unique_ptr<optimizer::Adam> opt_;
};

} // namespace rl
