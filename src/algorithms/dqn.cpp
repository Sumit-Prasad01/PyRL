#include "dqn.hpp"

#include <algorithm>
#include <cmath>

namespace rl {

DQN::DQN(Arena& arena, ModelState& model, u32 obs_dim, u32 act_dim,
         DQNConfig config, Prng* rng)
    : config_(config),
      obs_dim_(obs_dim),
      act_dim_(act_dim),
      epsilon_(config.eps_start)
{
    online_net_ = std::make_unique<DuelingQNetwork>(arena, model, obs_dim, act_dim,
                                                    config.hidden_dim, config.dueling, rng);
    target_net_ = std::make_unique<DuelingQNetwork>(arena, model, obs_dim, act_dim,
                                                    config.hidden_dim, config.dueling, rng);
    target_net_->CopyWeightsFrom(*online_net_);

    opt_ = std::make_unique<optimizer::Adam>(online_net_->Parameters(), config.lr);

    if (config.prioritized) {
        per_buffer_ = std::make_unique<PrioritizedReplayBuffer>(config.buffer_size, obs_dim, 1);
    } else {
        uniform_buffer_ = std::make_unique<ReplayBuffer>(config.buffer_size, obs_dim, 1, true);
    }
}

u32 DQN::Act(const std::vector<f32>& obs, Prng& rng)
{
    if (rng.NextFloat() < epsilon_) {
        return rng.NextBelow(act_dim_);
    }

    std::vector<u32> greedy_acts;
    online_net_->ActGreedy(obs.data(), 1, greedy_acts);
    return greedy_acts[0];
}

DQNStats DQN::TrainStep(const std::vector<f32>& obs, u32 action, f32 reward,
                        const std::vector<f32>& next_obs, bool done, Prng& rng)
{
    step_count_++;

    // Anneal epsilon linearly
    const f32 decay_frac = std::min(1.0f, static_cast<f32>(step_count_) / config_.eps_decay_steps);
    epsilon_ = config_.eps_start + decay_frac * (config_.eps_end - config_.eps_start);

    const std::vector<f32> act_vec = {static_cast<f32>(action)};

    if (config_.prioritized) {
        per_buffer_->Push(obs, act_vec, reward, next_obs, done);
    } else {
        uniform_buffer_->Push(obs, act_vec, reward, next_obs, done);
    }

    const u32 cur_size = config_.prioritized ? per_buffer_->Size() : uniform_buffer_->Size();
    if (cur_size < config_.batch_size) {
        return {0.0f, 0.0f, epsilon_, step_count_};
    }

    TransitionBatch batch = config_.prioritized ? per_buffer_->Sample(config_.batch_size, rng)
                                               : uniform_buffer_->Sample(config_.batch_size, rng);

    // Predict online Q-values on current states
    std::vector<f32> q_pred;
    online_net_->Forward(batch.obs.data(), config_.batch_size, q_pred);

    // Evaluate target values on next states
    std::vector<f32> q_next_target;
    target_net_->Forward(batch.next_obs.data(), config_.batch_size, q_next_target);

    std::vector<u32> next_best_actions;
    if (config_.double_dqn) {
        // Double DQN: select best actions with online network
        online_net_->ActGreedy(batch.next_obs.data(), config_.batch_size, next_best_actions);
    }

    f32 total_loss = 0.0f;
    f32 sum_q = 0.0f;
    std::vector<f32> td_errors(config_.batch_size);

    opt_->ZeroGrad();

    for (u32 b = 0; b < config_.batch_size; ++b) {
        const u32 act = static_cast<u32>(batch.actions[b]);
        const f32 cur_q = q_pred[b * act_dim_ + act];
        sum_q += cur_q;

        f32 max_next_q = 0.0f;
        if (config_.double_dqn) {
            const u32 best_a = next_best_actions[b];
            max_next_q = q_next_target[b * act_dim_ + best_a];
        } else {
            max_next_q = q_next_target[b * act_dim_ + 0];
            for (u32 a = 1; a < act_dim_; ++a) {
                if (q_next_target[b * act_dim_ + a] > max_next_q) {
                    max_next_q = q_next_target[b * act_dim_ + a];
                }
            }
        }

        const f32 non_terminal = 1.0f - batch.dones[b];
        const f32 target_y = batch.rewards[b] + config_.gamma * max_next_q * non_terminal;

        const f32 td_err = cur_q - target_y;
        td_errors[b] = td_err;

        // Smooth L1 / Huber loss
        f32 loss_sample = 0.0f;
        f32 grad_sample = 0.0f;
        if (std::abs(td_err) < 1.0f) {
            loss_sample = 0.5f * td_err * td_err;
            grad_sample = td_err;
        } else {
            loss_sample = std::abs(td_err) - 0.5f;
            grad_sample = (td_err > 0.0f) ? 1.0f : -1.0f;
        }

        const f32 weight = batch.weights[b];
        total_loss += weight * loss_sample;

        // Accumulate gradients into online parameters
        for (Var* p : online_net_->Parameters()) {
            if (p->grad != nullptr) {
                for (f32& g : p->grad->Span()) {
                    g += 0.01f * weight * grad_sample;
                }
            }
        }
    }

    opt_->Step();

    if (config_.prioritized) {
        per_buffer_->UpdatePriorities(batch.indices, td_errors);
    }

    // Target network update: Polyak soft update or periodic hard update
    if (config_.tau < 1.0f) {
        target_net_->SoftUpdateFrom(*online_net_, config_.tau);
    } else if (step_count_ % config_.target_update_freq == 0) {
        target_net_->CopyWeightsFrom(*online_net_);
    }

    DQNStats stats{};
    stats.loss       = total_loss / static_cast<f32>(config_.batch_size);
    stats.mean_q     = sum_q / static_cast<f32>(config_.batch_size);
    stats.epsilon    = epsilon_;
    stats.train_step = step_count_;
    return stats;
}

} // namespace rl
