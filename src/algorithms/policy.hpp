#pragma once

#include <memory>
#include <vector>

#include "arena.hpp"
#include "autograd.hpp"
#include "base.hpp"
#include "device.hpp"
#include "mat.hpp"
#include "nn.hpp"
#include "prng.hpp"

namespace rl {

// Discrete Actor-Critic Network (PPO, A2C, REINFORCE)
class ActorCritic {
public:
    ActorCritic(Arena& arena, ModelState& model, u32 obs_dim, u32 act_dim,
                u32 hidden_dim = 64, Prng* rng = nullptr);

    // Evaluates a batch of observations:
    // obs is [batch_size, obs_dim]
    // out_logits: [batch_size, act_dim]
    // out_values: [batch_size, 1]
    void Forward(const f32* obs, u32 batch_size,
                 std::vector<f32>& out_logits, std::vector<f32>& out_values);

    // Samples actions and computes log_prob, entropy, and value for rollout
    void GetActionAndValue(const f32* obs, u32 batch_size, Prng& rng,
                           std::vector<f32>& out_actions,
                           std::vector<f32>& out_log_probs,
                           std::vector<f32>& out_values,
                           std::vector<f32>& out_entropies);

    // Evaluates specific actions taken during rollout (for PPO/A2C update)
    void EvaluateActions(const f32* obs, const f32* actions, u32 batch_size,
                         std::vector<f32>& out_log_probs,
                         std::vector<f32>& out_entropies,
                         std::vector<f32>& out_values);

    [[nodiscard]] std::vector<Var*> Parameters() const;

    [[nodiscard]] u32 ObsDim() const noexcept { return obs_dim_; }
    [[nodiscard]] u32 ActDim() const noexcept { return act_dim_; }
    [[nodiscard]] u32 HiddenDim() const noexcept { return hidden_dim_; }

private:
    u32 obs_dim_    = 1;
    u32 act_dim_    = 1;
    u32 hidden_dim_ = 64;

    std::unique_ptr<nn::Linear> fc1_;
    std::unique_ptr<nn::Linear> fc2_;
    std::unique_ptr<nn::Linear> actor_head_;
    std::unique_ptr<nn::Linear> critic_head_;
};

// Dueling Q-Network (for DQN, Double DQN, and Dueling DQN)
class DuelingQNetwork {
public:
    DuelingQNetwork(Arena& arena, ModelState& model, u32 obs_dim, u32 act_dim,
                    u32 hidden_dim = 64, bool dueling = true, Prng* rng = nullptr);

    // Predicts Q-values: [batch_size, act_dim]
    void Forward(const f32* obs, u32 batch_size, std::vector<f32>& out_q_values);

    // Selects greedy actions: argmax_a Q(s, a)
    void ActGreedy(const f32* obs, u32 batch_size, std::vector<u32>& out_actions);

    // Copies weights from another network (for target network hard update)
    void CopyWeightsFrom(const DuelingQNetwork& src);

    // Polyak soft update: this = tau * src + (1 - tau) * this
    void SoftUpdateFrom(const DuelingQNetwork& src, f32 tau);

    [[nodiscard]] std::vector<Var*> Parameters() const;

    [[nodiscard]] bool IsDueling() const noexcept { return dueling_; }

private:
    u32  obs_dim_    = 1;
    u32  act_dim_    = 1;
    u32  hidden_dim_ = 64;
    bool dueling_    = true;

    std::unique_ptr<nn::Linear> fc1_;
    std::unique_ptr<nn::Linear> fc2_;
    std::unique_ptr<nn::Linear> val_head_; // [hidden, 1]
    std::unique_ptr<nn::Linear> adv_head_; // [hidden, act_dim]
};

} // namespace rl
