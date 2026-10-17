#pragma once

#include <vector>

#include "arena.hpp"
#include "autograd.hpp"
#include "environment.hpp"
#include "optimizer.hpp"
#include "policy.hpp"

namespace rl {

struct REINFORCEConfig {
    f32  lr         = 1e-3f;
    f32  gamma      = 0.99f;
    f32  value_coef = 0.5f;
    u32  hidden_dim = 64;
};

struct REINFORCEStats {
    f32 policy_loss = 0.0f;
    f32 value_loss  = 0.0f;
    f32 episode_return = 0.0f;
    u32 episode_length = 0;
};

class REINFORCE {
public:
    REINFORCE(Arena& arena, ModelState& model, u32 obs_dim, u32 act_dim,
              REINFORCEConfig config = {}, Prng* rng = nullptr);

    // Executes one full episode and performs a policy gradient + baseline update
    REINFORCEStats TrainEpisode(Env& env, Prng& rng);

    [[nodiscard]] ActorCritic& Policy() noexcept { return *ac_; }

private:
    REINFORCEConfig             config_;
    std::unique_ptr<ActorCritic> ac_;
    std::unique_ptr<optimizer::Adam> opt_;
};

} // namespace rl
