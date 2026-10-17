#include "sac.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace rl {

namespace {
constexpr f32 kSqrt2 = 1.414213562373095f;
constexpr f32 kPi    = 3.141592653589793f;
}

// ----------------------------------------------------------- ContinuousActor

ContinuousActor::ContinuousActor(Arena& arena, ModelState& model, u32 obs_dim, u32 act_dim,
                                 u32 hidden_dim, Prng* rng)
    : obs_dim_(obs_dim),
      act_dim_(act_dim),
      hidden_dim_(hidden_dim)
{
    fc1_ = std::make_unique<nn::Linear>(arena, model, obs_dim, hidden_dim, true, true, rng, kSqrt2);
    fc2_ = std::make_unique<nn::Linear>(arena, model, hidden_dim, hidden_dim, true, true, rng, kSqrt2);

    mu_head_      = std::make_unique<nn::Linear>(arena, model, hidden_dim, act_dim, true, true, rng, 0.01f);
    log_std_head_ = std::make_unique<nn::Linear>(arena, model, hidden_dim, act_dim, true, true, rng, 0.01f);
}

void ContinuousActor::Sample(const f32* obs, u32 batch_size, Prng& rng,
                             std::vector<f32>& out_actions, std::vector<f32>& out_log_probs)
{
    out_actions.resize(static_cast<u64>(batch_size) * act_dim_);
    out_log_probs.resize(batch_size);

    std::vector<f32> h1(hidden_dim_);
    std::vector<f32> h2(hidden_dim_);
    std::vector<f32> mu(act_dim_);
    std::vector<f32> log_std(act_dim_);

    const Mat& w1 = *fc1_->Weight()->val;
    const Mat& b1 = *fc1_->Bias()->val;
    const Mat& w2 = *fc2_->Weight()->val;
    const Mat& b2 = *fc2_->Bias()->val;
    const Mat& w_mu = *mu_head_->Weight()->val;
    const Mat& b_mu = *mu_head_->Bias()->val;
    const Mat& w_std = *log_std_head_->Weight()->val;
    const Mat& b_std = *log_std_head_->Bias()->val;

    std::normal_distribution<f32> std_norm(0.0f, 1.0f);

    for (u32 b = 0; b < batch_size; ++b) {
        const f32* const x = obs + static_cast<u64>(b) * obs_dim_;

        // h1 = ReLU(w1 * x + b1)
        for (u32 j = 0; j < hidden_dim_; ++j) {
            f32 sum = b1.data[j];
            for (u32 i = 0; i < obs_dim_; ++i) {
                sum += w1.data[j * obs_dim_ + i] * x[i];
            }
            h1[j] = std::max(0.0f, sum);
        }

        // h2 = ReLU(w2 * h1 + b2)
        for (u32 j = 0; j < hidden_dim_; ++j) {
            f32 sum = b2.data[j];
            for (u32 i = 0; i < hidden_dim_; ++i) {
                sum += w2.data[j * hidden_dim_ + i] * h1[i];
            }
            h2[j] = std::max(0.0f, sum);
        }

        // mu and log_std
        for (u32 j = 0; j < act_dim_; ++j) {
            f32 sum_m = b_mu.data[j];
            f32 sum_s = b_std.data[j];
            for (u32 i = 0; i < hidden_dim_; ++i) {
                sum_m += w_mu.data[j * hidden_dim_ + i] * h2[i];
                sum_s += w_std.data[j * hidden_dim_ + i] * h2[i];
            }
            mu[j] = sum_m;
            log_std[j] = std::clamp(sum_s, -20.0f, 2.0f);
        }

        f32 log_prob_sum = 0.0f;
        f32* const act_b = out_actions.data() + static_cast<u64>(b) * act_dim_;

        for (u32 j = 0; j < act_dim_; ++j) {
            const f32 std_val = std::exp(log_std[j]);
            const f32 eps = std_norm(rng);
            const f32 u = mu[j] + std_val * eps;
            const f32 a = std::tanh(u);

            act_b[j] = a;

            // log N(u; mu, std)
            const f32 diff = (u - mu[j]) / std_val;
            const f32 log_norm = -0.5f * (diff * diff + 2.0f * log_std[j] + std::log(2.0f * kPi));
            // Jacobian adjustment for tanh: log(1 - a^2 + 1e-6)
            const f32 jacob = std::log(std::max(1e-6f, 1.0f - a * a));
            log_prob_sum += log_norm - jacob;
        }

        out_log_probs[b] = log_prob_sum;
    }
}

void ContinuousActor::ActDeterministic(const f32* obs, u32 batch_size, std::vector<f32>& out_actions)
{
    out_actions.resize(static_cast<u64>(batch_size) * act_dim_);

    std::vector<f32> h1(hidden_dim_);
    std::vector<f32> h2(hidden_dim_);

    const Mat& w1 = *fc1_->Weight()->val;
    const Mat& b1 = *fc1_->Bias()->val;
    const Mat& w2 = *fc2_->Weight()->val;
    const Mat& b2 = *fc2_->Bias()->val;
    const Mat& w_mu = *mu_head_->Weight()->val;
    const Mat& b_mu = *mu_head_->Bias()->val;

    for (u32 b = 0; b < batch_size; ++b) {
        const f32* const x = obs + static_cast<u64>(b) * obs_dim_;

        for (u32 j = 0; j < hidden_dim_; ++j) {
            f32 sum = b1.data[j];
            for (u32 i = 0; i < obs_dim_; ++i) {
                sum += w1.data[j * obs_dim_ + i] * x[i];
            }
            h1[j] = std::max(0.0f, sum);
        }

        for (u32 j = 0; j < hidden_dim_; ++j) {
            f32 sum = b2.data[j];
            for (u32 i = 0; i < hidden_dim_; ++i) {
                sum += w2.data[j * hidden_dim_ + i] * h1[i];
            }
            h2[j] = std::max(0.0f, sum);
        }

        f32* const act_b = out_actions.data() + static_cast<u64>(b) * act_dim_;
        for (u32 j = 0; j < act_dim_; ++j) {
            f32 sum_m = b_mu.data[j];
            for (u32 i = 0; i < hidden_dim_; ++i) {
                sum_m += w_mu.data[j * hidden_dim_ + i] * h2[i];
            }
            act_b[j] = std::tanh(sum_m);
        }
    }
}

std::vector<Var*> ContinuousActor::Parameters() const
{
    std::vector<Var*> params;
    for (Var* p : fc1_->Parameters())          { params.push_back(p); }
    for (Var* p : fc2_->Parameters())          { params.push_back(p); }
    for (Var* p : mu_head_->Parameters())      { params.push_back(p); }
    for (Var* p : log_std_head_->Parameters()) { params.push_back(p); }
    return params;
}

// ---------------------------------------------------------- ContinuousCritic

ContinuousCritic::ContinuousCritic(Arena& arena, ModelState& model, u32 obs_dim, u32 act_dim,
                                   u32 hidden_dim, Prng* rng)
    : input_dim_(obs_dim + act_dim),
      hidden_dim_(hidden_dim)
{
    fc1_ = std::make_unique<nn::Linear>(arena, model, input_dim_, hidden_dim, true, true, rng, kSqrt2);
    fc2_ = std::make_unique<nn::Linear>(arena, model, hidden_dim, hidden_dim, true, true, rng, kSqrt2);
    out_head_ = std::make_unique<nn::Linear>(arena, model, hidden_dim, 1, true, true, rng, 1.0f);
}

void ContinuousCritic::Forward(const f32* obs_action, u32 batch_size, std::vector<f32>& out_q)
{
    out_q.resize(batch_size);

    std::vector<f32> h1(hidden_dim_);
    std::vector<f32> h2(hidden_dim_);

    const Mat& w1 = *fc1_->Weight()->val;
    const Mat& b1 = *fc1_->Bias()->val;
    const Mat& w2 = *fc2_->Weight()->val;
    const Mat& b2 = *fc2_->Bias()->val;
    const Mat& w_out = *out_head_->Weight()->val;
    const Mat& b_out = *out_head_->Bias()->val;

    for (u32 b = 0; b < batch_size; ++b) {
        const f32* const x = obs_action + static_cast<u64>(b) * input_dim_;

        for (u32 j = 0; j < hidden_dim_; ++j) {
            f32 sum = b1.data[j];
            for (u32 i = 0; i < input_dim_; ++i) {
                sum += w1.data[j * input_dim_ + i] * x[i];
            }
            h1[j] = std::max(0.0f, sum);
        }

        for (u32 j = 0; j < hidden_dim_; ++j) {
            f32 sum = b2.data[j];
            for (u32 i = 0; i < hidden_dim_; ++i) {
                sum += w2.data[j * hidden_dim_ + i] * h1[i];
            }
            h2[j] = std::max(0.0f, sum);
        }

        f32 q_val = b_out.data[0];
        for (u32 i = 0; i < hidden_dim_; ++i) {
            q_val += w_out.data[i] * h2[i];
        }
        out_q[b] = q_val;
    }
}

void ContinuousCritic::CopyWeightsFrom(const ContinuousCritic& src)
{
    const auto my_params = Parameters();
    const auto src_params = src.Parameters();
    for (std::size_t i = 0; i < my_params.size(); ++i) {
        my_params[i]->val->CopyFrom(*src_params[i]->val);
    }
}

void ContinuousCritic::SoftUpdateFrom(const ContinuousCritic& src, f32 tau)
{
    const auto my_params = Parameters();
    const auto src_params = src.Parameters();
    for (std::size_t p = 0; p < my_params.size(); ++p) {
        Mat& target = *my_params[p]->val;
        const Mat& source = *src_params[p]->val;
        for (u64 i = 0; i < target.Size(); ++i) {
            target.data[i] = tau * source.data[i] + (1.0f - tau) * target.data[i];
        }
    }
}

std::vector<Var*> ContinuousCritic::Parameters() const
{
    std::vector<Var*> params;
    for (Var* p : fc1_->Parameters())      { params.push_back(p); }
    for (Var* p : fc2_->Parameters())      { params.push_back(p); }
    for (Var* p : out_head_->Parameters()) { params.push_back(p); }
    return params;
}

// ----------------------------------------------------------------------- SAC

SAC::SAC(Arena& arena, ModelState& model, u32 obs_dim, u32 act_dim,
         SACConfig config, Prng* rng)
    : config_(config),
      obs_dim_(obs_dim),
      act_dim_(act_dim),
      log_alpha_(std::log(std::max(1e-4f, config.init_alpha))),
      target_entropy_(-static_cast<f32>(act_dim))
{
    actor_     = std::make_unique<ContinuousActor>(arena, model, obs_dim, act_dim, config.hidden_dim, rng);
    q1_        = std::make_unique<ContinuousCritic>(arena, model, obs_dim, act_dim, config.hidden_dim, rng);
    q2_        = std::make_unique<ContinuousCritic>(arena, model, obs_dim, act_dim, config.hidden_dim, rng);
    q1_target_ = std::make_unique<ContinuousCritic>(arena, model, obs_dim, act_dim, config.hidden_dim, rng);
    q2_target_ = std::make_unique<ContinuousCritic>(arena, model, obs_dim, act_dim, config.hidden_dim, rng);

    q1_target_->CopyWeightsFrom(*q1_);
    q2_target_->CopyWeightsFrom(*q2_);

    actor_opt_ = std::make_unique<optimizer::Adam>(actor_->Parameters(), config.policy_lr);

    std::vector<Var*> q_params;
    for (Var* p : q1_->Parameters()) { q_params.push_back(p); }
    for (Var* p : q2_->Parameters()) { q_params.push_back(p); }
    q_opt_ = std::make_unique<optimizer::Adam>(q_params, config.q_lr);

    buffer_ = std::make_unique<ReplayBuffer>(config.buffer_size, obs_dim, act_dim, false);
}

std::vector<f32> SAC::Act(const std::vector<f32>& obs, Prng& rng, bool deterministic)
{
    std::vector<f32> actions(act_dim_);
    if (deterministic) {
        actor_->ActDeterministic(obs.data(), 1, actions);
    } else {
        std::vector<f32> log_probs(1);
        actor_->Sample(obs.data(), 1, rng, actions, log_probs);
    }
    return actions;
}

SACStats SAC::TrainStep(const std::vector<f32>& obs, const std::vector<f32>& action,
                       f32 reward, const std::vector<f32>& next_obs, bool done, Prng& rng)
{
    step_count_++;
    buffer_->Push(obs, action, reward, next_obs, done);

    if (buffer_->Size() < config_.batch_size) {
        return {0.0f, 0.0f, std::exp(log_alpha_), step_count_};
    }

    TransitionBatch batch = buffer_->Sample(config_.batch_size, rng);

    // 1. Sample next actions and log_probs from actor
    std::vector<f32> next_actions;
    std::vector<f32> next_log_probs;
    actor_->Sample(batch.next_obs.data(), config_.batch_size, rng, next_actions, next_log_probs);

    // Concatenate [next_obs, next_actions]
    const u32 sa_dim = obs_dim_ + act_dim_;
    std::vector<f32> next_sa(static_cast<u64>(config_.batch_size) * sa_dim);
    for (u32 b = 0; b < config_.batch_size; ++b) {
        std::memcpy(next_sa.data() + b * sa_dim,
                    batch.next_obs.data() + b * obs_dim_, obs_dim_ * sizeof(f32));
        std::memcpy(next_sa.data() + b * sa_dim + obs_dim_,
                    next_actions.data() + b * act_dim_, act_dim_ * sizeof(f32));
    }

    // Evaluate target critics
    std::vector<f32> q1_next;
    std::vector<f32> q2_next;
    q1_target_->Forward(next_sa.data(), config_.batch_size, q1_next);
    q2_target_->Forward(next_sa.data(), config_.batch_size, q2_next);

    const f32 alpha = std::exp(log_alpha_);
    std::vector<f32> target_y(config_.batch_size);

    for (u32 b = 0; b < config_.batch_size; ++b) {
        const f32 min_next_q = std::min(q1_next[b], q2_next[b]);
        const f32 non_terminal = 1.0f - batch.dones[b];
        target_y[b] = batch.rewards[b] + config_.gamma * non_terminal * (min_next_q - alpha * next_log_probs[b]);
    }

    // Concatenate current [obs, actions]
    std::vector<f32> cur_sa(static_cast<u64>(config_.batch_size) * sa_dim);
    for (u32 b = 0; b < config_.batch_size; ++b) {
        std::memcpy(cur_sa.data() + b * sa_dim,
                    batch.obs.data() + b * obs_dim_, obs_dim_ * sizeof(f32));
        std::memcpy(cur_sa.data() + b * sa_dim + obs_dim_,
                    batch.actions.data() + b * act_dim_, act_dim_ * sizeof(f32));
    }

    // Forward current critics
    std::vector<f32> q1_pred;
    std::vector<f32> q2_pred;
    q1_->Forward(cur_sa.data(), config_.batch_size, q1_pred);
    q2_->Forward(cur_sa.data(), config_.batch_size, q2_pred);

    f32 critic_loss = 0.0f;
    q_opt_->ZeroGrad();

    for (u32 b = 0; b < config_.batch_size; ++b) {
        const f32 err1 = q1_pred[b] - target_y[b];
        const f32 err2 = q2_pred[b] - target_y[b];
        critic_loss += 0.5f * (err1 * err1 + err2 * err2);

        for (Var* p : q1_->Parameters()) {
            if (p->grad != nullptr) {
                for (f32& g : p->grad->Span()) { g += 0.01f * err1; }
            }
        }
        for (Var* p : q2_->Parameters()) {
            if (p->grad != nullptr) {
                for (f32& g : p->grad->Span()) { g += 0.01f * err2; }
            }
        }
    }
    q_opt_->Step();

    // 2. Policy Update
    actor_opt_->ZeroGrad();
    std::vector<f32> cur_sampled_actions;
    std::vector<f32> cur_log_probs;
    actor_->Sample(batch.obs.data(), config_.batch_size, rng, cur_sampled_actions, cur_log_probs);

    f32 actor_loss = 0.0f;
    for (u32 b = 0; b < config_.batch_size; ++b) {
        // Surrogate objective: alpha * log_prob - Q(s, a)
        const f32 q_eval = 0.5f * (q1_pred[b] + q2_pred[b]);
        const f32 pol_err = alpha * cur_log_probs[b] - q_eval;
        actor_loss += pol_err;

        for (Var* p : actor_->Parameters()) {
            if (p->grad != nullptr) {
                for (f32& g : p->grad->Span()) { g += 0.01f * pol_err; }
            }
        }
    }
    actor_opt_->Step();

    // 3. Automated Entropy Temperature Adjustment
    if (config_.auto_alpha) {
        f32 alpha_loss_grad = 0.0f;
        for (u32 b = 0; b < config_.batch_size; ++b) {
            alpha_loss_grad += -(cur_log_probs[b] + target_entropy_);
        }
        alpha_loss_grad /= static_cast<f32>(config_.batch_size);
        log_alpha_ -= config_.alpha_lr * alpha_loss_grad;
        log_alpha_ = std::clamp(log_alpha_, -10.0f, 2.0f);
    }

    // 4. Soft Target Updates
    q1_target_->SoftUpdateFrom(*q1_, config_.tau);
    q2_target_->SoftUpdateFrom(*q2_, config_.tau);

    SACStats stats{};
    stats.actor_loss  = actor_loss / static_cast<f32>(config_.batch_size);
    stats.critic_loss = critic_loss / static_cast<f32>(config_.batch_size);
    stats.alpha       = std::exp(log_alpha_);
    stats.train_step  = step_count_;
    return stats;
}

} // namespace rl
