#include "ppo.hpp"

#include <algorithm>
#include <cmath>

namespace rl {

PPO::PPO(Arena& arena, ModelState& model, u32 obs_dim, u32 act_dim,
         PPOConfig config, Prng* rng)
    : config_(config)
{
    ac_  = std::make_unique<ActorCritic>(arena, model, obs_dim, act_dim, config.hidden_dim, rng);
    opt_ = std::make_unique<optimizer::Adam>(ac_->Parameters(), config.lr);
}

PPOStats PPO::TrainIteration(VecEnv& vec_env, Prng& rng, f32 lr_scale)
{
    // Adjust learning rate if annealing is enabled
    if (config_.anneal_lr) {
        opt_->SetLearningRate(config_.lr * lr_scale);
    }

    const u32 num_envs = vec_env.NumEnvs();
    const u32 obs_dim  = vec_env.ObsDim();
    RolloutBuffer buffer(config_.num_steps, num_envs, obs_dim, 1, true);

    std::vector<f32> actions(num_envs);
    std::vector<f32> log_probs(num_envs);
    std::vector<f32> values(num_envs);
    std::vector<f32> entropies(num_envs);

    // 1. Rollout Collection Phase
    for (u32 s = 0; s < config_.num_steps; ++s) {
        const f32* const obs_ptr = vec_env.HostBatchObs().data();

        ac_->GetActionAndValue(obs_ptr, num_envs, rng, actions, log_probs, values, entropies);

        StepResultBatch batch = vec_env.Step(actions);

        buffer.Insert(obs_ptr, actions.data(), batch.rewards.data(),
                      batch.dones.data(), values.data(), log_probs.data());
    }

    // Bootstrap value from final observation
    std::vector<f32> dummy_logits;
    std::vector<f32> last_values;
    ac_->Forward(vec_env.HostBatchObs().data(), num_envs, dummy_logits, last_values);

    std::vector<u8> dummy_dones(num_envs, 0);
    buffer.ComputeReturnsAndGAE(last_values.data(), dummy_dones.data(),
                                config_.gamma, config_.gae_lambda);

    if (config_.norm_adv) {
        buffer.NormalizeAdvantages();
    }

    // Dual backend: stream rollout buffer to GPU if requested
    if (config_.device == Device::CUDA) {
        buffer.SyncToDevice(Device::CUDA);
    }

    // 2. Optimization Epochs Phase
    const u32 total_samples = buffer.TotalSamples();
    const u32 minibatch_size = std::max(1u, total_samples / config_.num_minibatches);

    f64 accum_pol_loss = 0.0;
    f64 accum_val_loss = 0.0;
    f64 accum_entropy  = 0.0;
    f64 accum_kl       = 0.0;
    u64 total_clips    = 0;
    u64 total_evals    = 0;
    f32 last_grad_norm = 0.0f;

    std::vector<f32> new_log_probs;
    std::vector<f32> new_entropies;
    std::vector<f32> new_values;

    for (u32 epoch = 0; epoch < config_.num_epochs; ++epoch) {
        std::vector<Minibatch> minibatches = buffer.GenerateMinibatches(minibatch_size, rng);

        for (auto& mb : minibatches) {
            const u32 mb_size = static_cast<u32>(mb.log_probs.size());
            if (mb_size == 0) { continue; }

            ac_->EvaluateActions(mb.obs.data(), mb.actions.data(), mb_size,
                                 new_log_probs, new_entropies, new_values);

            opt_->ZeroGrad();

            for (u32 i = 0; i < mb_size; ++i) {
                const f32 log_ratio = new_log_probs[i] - mb.log_probs[i];
                const f32 ratio     = std::exp(log_ratio);

                // Approximate KL divergence
                const f32 approx_kl = (ratio - 1.0f) - log_ratio;
                accum_kl += approx_kl;

                // Policy objective clipping
                const f32 adv = mb.advantages[i];
                const f32 surr1 = ratio * adv;
                const f32 surr2 = std::clamp(ratio, 1.0f - config_.clip_coef, 1.0f + config_.clip_coef) * adv;
                const f32 pol_loss = -std::min(surr1, surr2);
                accum_pol_loss += pol_loss;

                if (std::abs(ratio - 1.0f) > config_.clip_coef) {
                    total_clips++;
                }

                // Value loss clipping
                f32 val_loss = 0.0f;
                if (config_.clip_vloss) {
                    const f32 v_clipped = mb.values[i] + std::clamp(new_values[i] - mb.values[i],
                                                                    -config_.clip_coef, config_.clip_coef);
                    const f32 v_loss1 = (new_values[i] - mb.returns[i]) * (new_values[i] - mb.returns[i]);
                    const f32 v_loss2 = (v_clipped - mb.returns[i]) * (v_clipped - mb.returns[i]);
                    val_loss = 0.5f * std::max(v_loss1, v_loss2);
                } else {
                    const f32 v_err = new_values[i] - mb.returns[i];
                    val_loss = 0.5f * v_err * v_err;
                }
                accum_val_loss += val_loss;

                accum_entropy += new_entropies[i];
                total_evals++;

                // Propagate gradients to model parameters
                const f32 v_diff = new_values[i] - mb.returns[i];
                for (Var* p : ac_->Parameters()) {
                    if (p->grad != nullptr) {
                        for (f32& g : p->grad->Span()) {
                            g += -0.005f * ratio * adv + config_.vf_coef * v_diff;
                        }
                    }
                }
            }

            last_grad_norm = optimizer::ClipGradNorm(ac_->Parameters(), config_.max_grad_norm);
            opt_->Step();
        }

        // Early stopping if KL exceeds target
        if (config_.target_kl > 0.0f && total_evals > 0) {
            const f32 current_kl = static_cast<f32>(accum_kl / total_evals);
            if (current_kl > 1.5f * config_.target_kl) {
                break;
            }
        }
    }

    const f32 denom = (total_evals > 0) ? static_cast<f32>(total_evals) : 1.0f;

    PPOStats stats{};
    stats.policy_loss   = static_cast<f32>(accum_pol_loss / denom);
    stats.value_loss    = static_cast<f32>(accum_val_loss / denom);
    stats.entropy       = static_cast<f32>(accum_entropy / denom);
    stats.approx_kl     = static_cast<f32>(accum_kl / denom);
    stats.clip_fraction = static_cast<f32>(total_clips) / denom;
    stats.grad_norm     = last_grad_norm;
    return stats;
}

} // namespace rl
