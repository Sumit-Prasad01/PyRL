#include "policy.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>

namespace rl {

namespace {
constexpr f32 kSqrt2 = 1.414213562373095f;

void SoftmaxAndLogSoftmax(const f32* logits, u32 K, std::vector<f32>& probs, std::vector<f32>& log_probs)
{
    probs.resize(K);
    log_probs.resize(K);

    f32 max_val = logits[0];
    for (u32 i = 1; i < K; ++i) {
        if (logits[i] > max_val) { max_val = logits[i]; }
    }

    f32 sum_exp = 0.0f;
    for (u32 i = 0; i < K; ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum_exp += probs[i];
    }

    const f32 inv_sum = 1.0f / sum_exp;
    const f32 log_sum = std::log(sum_exp);

    for (u32 i = 0; i < K; ++i) {
        probs[i] *= inv_sum;
        log_probs[i] = (logits[i] - max_val) - log_sum;
    }
}

} // namespace

// --------------------------------------------------------------- ActorCritic

ActorCritic::ActorCritic(Arena& arena, ModelState& model, u32 obs_dim, u32 act_dim,
                         u32 hidden_dim, Prng* rng)
    : obs_dim_(obs_dim),
      act_dim_(act_dim),
      hidden_dim_(hidden_dim)
{
    fc1_ = std::make_unique<nn::Linear>(arena, model, obs_dim, hidden_dim, true, true, rng, kSqrt2);
    fc2_ = std::make_unique<nn::Linear>(arena, model, hidden_dim, hidden_dim, true, true, rng, kSqrt2);

    // Actor head has gain 0.01 (clean standard to start policy near uniform)
    actor_head_ = std::make_unique<nn::Linear>(arena, model, hidden_dim, act_dim, true, true, rng, 0.01f);
    // Critic head has gain 1.0
    critic_head_ = std::make_unique<nn::Linear>(arena, model, hidden_dim, 1, true, true, rng, 1.0f);
}

void ActorCritic::Forward(const f32* obs, u32 batch_size,
                          std::vector<f32>& out_logits, std::vector<f32>& out_values)
{
    out_logits.resize(static_cast<u64>(batch_size) * act_dim_);
    out_values.resize(batch_size);

    std::vector<f32> h1(hidden_dim_);
    std::vector<f32> h2(hidden_dim_);

    const Mat& w1 = *fc1_->Weight()->val;
    const Mat& b1 = *fc1_->Bias()->val;
    const Mat& w2 = *fc2_->Weight()->val;
    const Mat& b2 = *fc2_->Bias()->val;
    const Mat& w_pi = *actor_head_->Weight()->val;
    const Mat& b_pi = *actor_head_->Bias()->val;
    const Mat& w_v  = *critic_head_->Weight()->val;
    const Mat& b_v  = *critic_head_->Bias()->val;

    for (u32 b = 0; b < batch_size; ++b) {
        const f32* const x = obs + static_cast<u64>(b) * obs_dim_;

        // Layer 1: h1 = ReLU(w1 * x + b1)
        for (u32 j = 0; j < hidden_dim_; ++j) {
            f32 sum = b1.data[j];
            for (u32 i = 0; i < obs_dim_; ++i) {
                sum += w1.data[j * obs_dim_ + i] * x[i];
            }
            h1[j] = std::max(0.0f, sum);
        }

        // Layer 2: h2 = ReLU(w2 * h1 + b2)
        for (u32 j = 0; j < hidden_dim_; ++j) {
            f32 sum = b2.data[j];
            for (u32 i = 0; i < hidden_dim_; ++i) {
                sum += w2.data[j * hidden_dim_ + i] * h1[i];
            }
            h2[j] = std::max(0.0f, sum);
        }

        // Actor Head: logits = w_pi * h2 + b_pi
        f32* const logits_b = out_logits.data() + static_cast<u64>(b) * act_dim_;
        for (u32 j = 0; j < act_dim_; ++j) {
            f32 sum = b_pi.data[j];
            for (u32 i = 0; i < hidden_dim_; ++i) {
                sum += w_pi.data[j * hidden_dim_ + i] * h2[i];
            }
            logits_b[j] = sum;
        }

        // Critic Head: value = w_v * h2 + b_v
        f32 v_sum = b_v.data[0];
        for (u32 i = 0; i < hidden_dim_; ++i) {
            v_sum += w_v.data[i] * h2[i];
        }
        out_values[b] = v_sum;
    }
}

void ActorCritic::GetActionAndValue(const f32* obs, u32 batch_size, Prng& rng,
                                   std::vector<f32>& out_actions,
                                   std::vector<f32>& out_log_probs,
                                   std::vector<f32>& out_values,
                                   std::vector<f32>& out_entropies)
{
    std::vector<f32> logits;
    Forward(obs, batch_size, logits, out_values);

    out_actions.resize(batch_size);
    out_log_probs.resize(batch_size);
    out_entropies.resize(batch_size);

    std::vector<f32> probs;
    std::vector<f32> log_probs;

    for (u32 b = 0; b < batch_size; ++b) {
        const f32* const logit_b = logits.data() + static_cast<u64>(b) * act_dim_;
        SoftmaxAndLogSoftmax(logit_b, act_dim_, probs, log_probs);

        // Entropy: H = - sum p * log(p)
        f32 entropy = 0.0f;
        for (u32 k = 0; k < act_dim_; ++k) {
            entropy -= probs[k] * log_probs[k];
        }
        out_entropies[b] = entropy;

        // Categorical sampling via CDF
        const f32 r = rng.NextFloat();
        f32 cdf = 0.0f;
        u32 selected_action = act_dim_ - 1;
        for (u32 k = 0; k < act_dim_; ++k) {
            cdf += probs[k];
            if (r <= cdf) {
                selected_action = k;
                break;
            }
        }

        out_actions[b]   = static_cast<f32>(selected_action);
        out_log_probs[b] = log_probs[selected_action];
    }
}

void ActorCritic::EvaluateActions(const f32* obs, const f32* actions, u32 batch_size,
                                 std::vector<f32>& out_log_probs,
                                 std::vector<f32>& out_entropies,
                                 std::vector<f32>& out_values)
{
    std::vector<f32> logits;
    Forward(obs, batch_size, logits, out_values);

    out_log_probs.resize(batch_size);
    out_entropies.resize(batch_size);

    std::vector<f32> probs;
    std::vector<f32> log_probs;

    for (u32 b = 0; b < batch_size; ++b) {
        const f32* const logit_b = logits.data() + static_cast<u64>(b) * act_dim_;
        SoftmaxAndLogSoftmax(logit_b, act_dim_, probs, log_probs);

        f32 entropy = 0.0f;
        for (u32 k = 0; k < act_dim_; ++k) {
            entropy -= probs[k] * log_probs[k];
        }
        out_entropies[b] = entropy;

        const u32 act = static_cast<u32>(std::clamp<i64>(static_cast<i64>(actions[b]), 0, act_dim_ - 1));
        out_log_probs[b] = log_probs[act];
    }
}

std::vector<Var*> ActorCritic::Parameters() const
{
    std::vector<Var*> params;
    for (Var* p : fc1_->Parameters())        { params.push_back(p); }
    for (Var* p : fc2_->Parameters())        { params.push_back(p); }
    for (Var* p : actor_head_->Parameters()) { params.push_back(p); }
    for (Var* p : critic_head_->Parameters()) { params.push_back(p); }
    return params;
}

// ----------------------------------------------------------- DuelingQNetwork

DuelingQNetwork::DuelingQNetwork(Arena& arena, ModelState& model, u32 obs_dim, u32 act_dim,
                                 u32 hidden_dim, bool dueling, Prng* rng)
    : obs_dim_(obs_dim),
      act_dim_(act_dim),
      hidden_dim_(hidden_dim),
      dueling_(dueling)
{
    fc1_ = std::make_unique<nn::Linear>(arena, model, obs_dim, hidden_dim, true, true, rng, kSqrt2);
    fc2_ = std::make_unique<nn::Linear>(arena, model, hidden_dim, hidden_dim, true, true, rng, kSqrt2);

    adv_head_ = std::make_unique<nn::Linear>(arena, model, hidden_dim, act_dim, true, true, rng, 1.0f);
    if (dueling) {
        val_head_ = std::make_unique<nn::Linear>(arena, model, hidden_dim, 1, true, true, rng, 1.0f);
    }
}

void DuelingQNetwork::Forward(const f32* obs, u32 batch_size, std::vector<f32>& out_q_values)
{
    out_q_values.resize(static_cast<u64>(batch_size) * act_dim_);

    std::vector<f32> h1(hidden_dim_);
    std::vector<f32> h2(hidden_dim_);
    std::vector<f32> adv(act_dim_);

    const Mat& w1 = *fc1_->Weight()->val;
    const Mat& b1 = *fc1_->Bias()->val;
    const Mat& w2 = *fc2_->Weight()->val;
    const Mat& b2 = *fc2_->Bias()->val;
    const Mat& w_adv = *adv_head_->Weight()->val;
    const Mat& b_adv = *adv_head_->Bias()->val;

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

        // Advantage: adv = w_adv * h2 + b_adv
        f32 sum_adv = 0.0f;
        for (u32 j = 0; j < act_dim_; ++j) {
            f32 s = b_adv.data[j];
            for (u32 i = 0; i < hidden_dim_; ++i) {
                s += w_adv.data[j * hidden_dim_ + i] * h2[i];
            }
            adv[j] = s;
            sum_adv += s;
        }

        f32* const q_out = out_q_values.data() + static_cast<u64>(b) * act_dim_;

        if (dueling_ && val_head_) {
            const Mat& w_val = *val_head_->Weight()->val;
            const Mat& b_val = *val_head_->Bias()->val;
            f32 v = b_val.data[0];
            for (u32 i = 0; i < hidden_dim_; ++i) {
                v += w_val.data[i] * h2[i];
            }
            const f32 mean_adv = sum_adv / static_cast<f32>(act_dim_);
            for (u32 j = 0; j < act_dim_; ++j) {
                q_out[j] = v + (adv[j] - mean_adv);
            }
        } else {
            for (u32 j = 0; j < act_dim_; ++j) {
                q_out[j] = adv[j];
            }
        }
    }
}

void DuelingQNetwork::ActGreedy(const f32* obs, u32 batch_size, std::vector<u32>& out_actions)
{
    std::vector<f32> q_values;
    Forward(obs, batch_size, q_values);

    out_actions.resize(batch_size);
    for (u32 b = 0; b < batch_size; ++b) {
        const f32* const q_b = q_values.data() + static_cast<u64>(b) * act_dim_;
        u32 best_act = 0;
        f32 max_q = q_b[0];
        for (u32 j = 1; j < act_dim_; ++j) {
            if (q_b[j] > max_q) {
                max_q = q_b[j];
                best_act = j;
            }
        }
        out_actions[b] = best_act;
    }
}

void DuelingQNetwork::CopyWeightsFrom(const DuelingQNetwork& src)
{
    const auto my_params = Parameters();
    const auto src_params = src.Parameters();
    for (std::size_t i = 0; i < my_params.size(); ++i) {
        my_params[i]->val->CopyFrom(*src_params[i]->val);
    }
}

void DuelingQNetwork::SoftUpdateFrom(const DuelingQNetwork& src, f32 tau)
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

std::vector<Var*> DuelingQNetwork::Parameters() const
{
    std::vector<Var*> params;
    for (Var* p : fc1_->Parameters())      { params.push_back(p); }
    for (Var* p : fc2_->Parameters())      { params.push_back(p); }
    for (Var* p : adv_head_->Parameters()) { params.push_back(p); }
    if (dueling_ && val_head_) {
        for (Var* p : val_head_->Parameters()) { params.push_back(p); }
    }
    return params;
}

} // namespace rl
