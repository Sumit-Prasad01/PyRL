#pragma once

#include <memory>
#include <vector>

#include "arena.hpp"
#include "autograd.hpp"
#include "buffer.hpp"
#include "optimizer.hpp"
#include "policy.hpp"
#include "vec_env.hpp"

namespace rl {

struct A2CConfig {
    f32 lr            = 7e-4f;
    f32 gamma         = 0.99f;
    f32 gae_lambda    = 0.95f;
    u32 n_steps       = 5;
    f32 ent_coef      = 0.01f;
    f32 vf_coef       = 0.5f;
    f32 max_grad_norm = 0.5f;
    u32 hidden_dim    = 64;
};

struct A2CStats {
    f32 policy_loss = 0.0f;
    f32 value_loss  = 0.0f;
    f32 entropy     = 0.0f;
    f32 grad_norm   = 0.0f;
};

class A2C {
public:
    A2C(Arena& arena, ModelState& model, u32 obs_dim, u32 act_dim,
        A2CConfig config = {}, Prng* rng = nullptr);

    // Collects n-step rollout from VecEnv and executes one synchronous update step
    A2CStats TrainStep(VecEnv& vec_env, Prng& rng);

    [[nodiscard]] ActorCritic& Policy() noexcept { return *ac_; }

private:
    A2CConfig                    config_;
    std::unique_ptr<ActorCritic> ac_;
    std::unique_ptr<optimizer::Adam> opt_;
};

} // namespace rl
