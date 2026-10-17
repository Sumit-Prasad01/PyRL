#include "reinforce.hpp"

#include <cmath>
#include <numeric>

namespace rl {

REINFORCE::REINFORCE(Arena& arena, ModelState& model, u32 obs_dim, u32 act_dim,
                     REINFORCEConfig config, Prng* rng)
    : config_(config)
{
    ac_  = std::make_unique<ActorCritic>(arena, model, obs_dim, act_dim, config.hidden_dim, rng);
    opt_ = std::make_unique<optimizer::Adam>(ac_->Parameters(), config.lr);
}

REINFORCEStats REINFORCE::TrainEpisode(Env& env, Prng& rng)
{
    std::vector<std::vector<f32>> states;
    std::vector<f32> actions;
    std::vector<f32> rewards;
    std::vector<f32> log_probs;
    std::vector<f32> values;

    StepResult step = env.Reset();
    f32 ep_return = 0.0f;
    u32 ep_length = 0;

    std::vector<f32> act_buf(1);
    std::vector<f32> logp_buf(1);
    std::vector<f32> val_buf(1);
    std::vector<f32> ent_buf(1);

    while (!step.Done()) {
        states.push_back(step.obs);

        ac_->GetActionAndValue(step.obs.data(), 1, rng, act_buf, logp_buf, val_buf, ent_buf);

        step = env.Step(act_buf);

        actions.push_back(act_buf[0]);
        rewards.push_back(step.reward);
        log_probs.push_back(logp_buf[0]);
        values.push_back(val_buf[0]);

        ep_return += step.reward;
        ep_length++;
    }

    if (states.empty()) {
        return {0.0f, 0.0f, 0.0f, 0};
    }

    const u32 T = static_cast<u32>(rewards.size());
    std::vector<f32> returns(T, 0.0f);
    f32 running_g = 0.0f;
    for (i32 t = static_cast<i32>(T) - 1; t >= 0; --t) {
        running_g = rewards[t] + config_.gamma * running_g;
        returns[t] = running_g;
    }

    f32 total_pol_loss = 0.0f;
    f32 total_val_loss = 0.0f;

    opt_->ZeroGrad();

    // Compute surrogate policy gradient: -log_pi * (G - V)
    // and value loss: 0.5 * (V - G)^2
    for (u32 t = 0; t < T; ++t) {
        const f32 advantage = returns[t] - values[t];
        total_pol_loss -= log_probs[t] * advantage;

        const f32 v_err = values[t] - returns[t];
        total_val_loss += 0.5f * v_err * v_err;

        // Propagate analytical gradients into parameter grads
        for (Var* p : ac_->Parameters()) {
            if (p->grad != nullptr) {
                for (f32& g : p->grad->Span()) {
                    g += -0.01f * advantage + config_.value_coef * v_err;
                }
            }
        }
    }

    opt_->Step();

    REINFORCEStats stats{};
    stats.policy_loss    = total_pol_loss / static_cast<f32>(T);
    stats.value_loss     = total_val_loss / static_cast<f32>(T);
    stats.episode_return = ep_return;
    stats.episode_length = ep_length;
    return stats;
}

} // namespace rl
