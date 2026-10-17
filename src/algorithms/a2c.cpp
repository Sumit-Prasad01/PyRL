#include "a2c.hpp"

#include <algorithm>
#include <cmath>

namespace rl {

A2C::A2C(Arena& arena, ModelState& model, u32 obs_dim, u32 act_dim,
         A2CConfig config, Prng* rng)
    : config_(config)
{
    ac_  = std::make_unique<ActorCritic>(arena, model, obs_dim, act_dim, config.hidden_dim, rng);
    opt_ = std::make_unique<optimizer::Adam>(ac_->Parameters(), config.lr);
}

A2CStats A2C::TrainStep(VecEnv& vec_env, Prng& rng)
{
    const u32 num_envs = vec_env.NumEnvs();
    const u32 obs_dim  = vec_env.ObsDim();
    RolloutBuffer buffer(config_.n_steps, num_envs, obs_dim, 1, true);

    std::vector<f32> actions(num_envs);
    std::vector<f32> log_probs(num_envs);
    std::vector<f32> values(num_envs);
    std::vector<f32> entropies(num_envs);

    for (u32 s = 0; s < config_.n_steps; ++s) {
        const f32* const obs_ptr = vec_env.HostBatchObs().data();

        ac_->GetActionAndValue(obs_ptr, num_envs, rng, actions, log_probs, values, entropies);

        StepResultBatch batch = vec_env.Step(actions);

        buffer.Insert(obs_ptr, actions.data(), batch.rewards.data(),
                      batch.dones.data(), values.data(), log_probs.data());
    }

    // Bootstrap value from final observations
    std::vector<f32> dummy_logits;
    std::vector<f32> last_values;
    ac_->Forward(vec_env.HostBatchObs().data(), num_envs, dummy_logits, last_values);

    std::vector<u8> dummy_dones(num_envs, 0);
    buffer.ComputeReturnsAndGAE(last_values.data(), dummy_dones.data(),
                                config_.gamma, config_.gae_lambda);
    buffer.NormalizeAdvantages();

    // Compute losses over the full n-step rollout
    std::vector<f32> eval_log_probs;
    std::vector<f32> eval_entropies;
    std::vector<f32> eval_values;

    const u32 total_samples = buffer.TotalSamples();
    ac_->EvaluateActions(buffer.Obs().data(), buffer.Actions().data(), total_samples,
                         eval_log_probs, eval_entropies, eval_values);

    f32 pol_loss = 0.0f;
    f32 val_loss = 0.0f;
    f32 total_ent = 0.0f;

    opt_->ZeroGrad();

    for (u32 i = 0; i < total_samples; ++i) {
        const f32 adv = buffer.Advantages()[i];
        pol_loss -= eval_log_probs[i] * adv;

        const f32 v_err = eval_values[i] - buffer.Returns()[i];
        val_loss += 0.5f * v_err * v_err;

        total_ent += eval_entropies[i];

        // Accumulate parameter gradients
        for (Var* p : ac_->Parameters()) {
            if (p->grad != nullptr) {
                for (f32& g : p->grad->Span()) {
                    g += -0.01f * adv + config_.vf_coef * v_err;
                }
            }
        }
    }

    const f32 grad_norm = optimizer::ClipGradNorm(ac_->Parameters(), config_.max_grad_norm);
    opt_->Step();

    A2CStats stats{};
    stats.policy_loss = pol_loss / static_cast<f32>(total_samples);
    stats.value_loss  = val_loss / static_cast<f32>(total_samples);
    stats.entropy     = total_ent / static_cast<f32>(total_samples);
    stats.grad_norm   = grad_norm;
    return stats;
}

} // namespace rl
