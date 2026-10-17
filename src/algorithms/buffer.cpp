#include "buffer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace rl {

// ------------------------------------------------------------- RolloutBuffer

RolloutBuffer::RolloutBuffer(u32 num_steps, u32 num_envs, u32 obs_dim, u32 act_dim,
                             bool is_discrete)
    : num_steps_(num_steps),
      num_envs_(num_envs),
      obs_dim_(obs_dim),
      act_dim_(act_dim),
      is_discrete_(is_discrete)
{
    const u64 total = static_cast<u64>(num_steps) * num_envs;
    obs_.assign(total * obs_dim, 0.0f);
    actions_.assign(total * act_dim, 0.0f);
    rewards_.assign(total, 0.0f);
    dones_.assign(total, 0);
    values_.assign(total, 0.0f);
    log_probs_.assign(total, 0.0f);
    advantages_.assign(total, 0.0f);
    returns_.assign(total, 0.0f);
}

void RolloutBuffer::Reset()
{
    step_idx_ = 0;
}

void RolloutBuffer::Insert(const f32* obs, const f32* actions, const f32* rewards,
                           const u8* dones, const f32* values, const f32* log_probs)
{
    if (step_idx_ >= num_steps_) {
        throw std::runtime_error("RolloutBuffer::Insert: buffer is already full.");
    }

    const u64 offset_samples = static_cast<u64>(step_idx_) * num_envs_;
    std::memcpy(obs_.data() + offset_samples * obs_dim_, obs,
                num_envs_ * obs_dim_ * sizeof(f32));
    std::memcpy(actions_.data() + offset_samples * act_dim_, actions,
                num_envs_ * act_dim_ * sizeof(f32));
    std::memcpy(rewards_.data() + offset_samples, rewards,
                num_envs_ * sizeof(f32));
    std::memcpy(dones_.data() + offset_samples, dones,
                num_envs_ * sizeof(u8));
    std::memcpy(values_.data() + offset_samples, values,
                num_envs_ * sizeof(f32));
    std::memcpy(log_probs_.data() + offset_samples, log_probs,
                num_envs_ * sizeof(f32));

    step_idx_++;
}

void RolloutBuffer::ComputeReturnsAndGAE(const f32* last_values, const u8* last_dones,
                                        f32 gamma, f32 gae_lambda)
{
    for (u32 env = 0; env < num_envs_; ++env) {
        f32 gae = 0.0f;
        for (i32 t = static_cast<i32>(num_steps_) - 1; t >= 0; --t) {
            const u64 cur_idx = static_cast<u64>(t) * num_envs_ + env;

            f32 next_val = 0.0f;
            u8  next_done = 0;

            if (t == static_cast<i32>(num_steps_) - 1) {
                next_val  = last_values[env];
                next_done = last_dones[env];
            } else {
                const u64 next_idx = static_cast<u64>(t + 1) * num_envs_ + env;
                next_val  = values_[next_idx];
                next_done = dones_[next_idx];
            }

            const f32 non_terminal = 1.0f - static_cast<f32>(next_done);
            const f32 delta = rewards_[cur_idx] + gamma * next_val * non_terminal - values_[cur_idx];
            gae = delta + gamma * gae_lambda * non_terminal * gae;

            advantages_[cur_idx] = gae;
            returns_[cur_idx]    = gae + values_[cur_idx];
        }
    }
}

void RolloutBuffer::NormalizeAdvantages() noexcept
{
    const u64 total = TotalSamples();
    if (total <= 1) { return; }

    f64 sum = 0.0;
    for (u64 i = 0; i < total; ++i) { sum += advantages_[i]; }
    const f32 mean = static_cast<f32>(sum / total);

    f64 sq_diff = 0.0;
    for (u64 i = 0; i < total; ++i) {
        const f32 d = advantages_[i] - mean;
        sq_diff += d * d;
    }
    const f32 std_dev = std::sqrt(static_cast<f32>(sq_diff / total)) + 1e-8f;

    for (u64 i = 0; i < total; ++i) {
        advantages_[i] = (advantages_[i] - mean) / std_dev;
    }
}

std::vector<Minibatch> RolloutBuffer::GenerateMinibatches(u32 minibatch_size, Prng& rng) const
{
    const u32 total = TotalSamples();
    std::vector<u32> indices(total);
    std::iota(indices.begin(), indices.end(), 0u);

    // Knuth-Fisher-Yates shuffle
    for (u32 i = total - 1; i > 0; --i) {
        const u32 j = rng.NextBelow(i + 1);
        std::swap(indices[i], indices[j]);
    }

    const u32 num_minibatches = (total + minibatch_size - 1) / minibatch_size;
    std::vector<Minibatch> batches;
    batches.reserve(num_minibatches);

    for (u32 b = 0; b < num_minibatches; ++b) {
        const u32 start = b * minibatch_size;
        const u32 end   = std::min(start + minibatch_size, total);
        const u32 cur_size = end - start;

        Minibatch mb{};
        mb.obs.resize(cur_size * obs_dim_);
        mb.actions.resize(cur_size * act_dim_);
        mb.log_probs.resize(cur_size);
        mb.advantages.resize(cur_size);
        mb.returns.resize(cur_size);
        mb.values.resize(cur_size);

        for (u32 i = 0; i < cur_size; ++i) {
            const u32 idx = indices[start + i];
            std::memcpy(mb.obs.data() + i * obs_dim_,
                        obs_.data() + idx * obs_dim_,
                        obs_dim_ * sizeof(f32));
            std::memcpy(mb.actions.data() + i * act_dim_,
                        actions_.data() + idx * act_dim_,
                        act_dim_ * sizeof(f32));
            mb.log_probs[i]  = log_probs_[idx];
            mb.advantages[i] = advantages_[idx];
            mb.returns[i]    = returns_[idx];
            mb.values[i]     = values_[idx];
        }

        batches.push_back(std::move(mb));
    }

    return batches;
}

void RolloutBuffer::SyncToDevice(Device dev)
{
    if (dev == Device::CUDA) {
#ifdef RL_USE_CUDA
        if (HasCudaDevice()) {
            const std::size_t bytes = obs_.size() * sizeof(f32);
            if (!gpu_obs_.IsAllocated() || gpu_obs_.SizeBytes() < bytes) {
                gpu_obs_.Allocate(bytes);
            }
            gpu_obs_.CopyFromHost(obs_.data(), bytes);
        }
#endif
    }
}

// -------------------------------------------------------------- ReplayBuffer

ReplayBuffer::ReplayBuffer(u32 capacity, u32 obs_dim, u32 act_dim, bool is_discrete)
    : capacity_(capacity),
      obs_dim_(obs_dim),
      act_dim_(act_dim),
      is_discrete_(is_discrete)
{
    obs_.assign(static_cast<u64>(capacity) * obs_dim, 0.0f);
    actions_.assign(static_cast<u64>(capacity) * act_dim, 0.0f);
    rewards_.assign(capacity, 0.0f);
    next_obs_.assign(static_cast<u64>(capacity) * obs_dim, 0.0f);
    dones_.assign(capacity, 0.0f);
}

void ReplayBuffer::Push(const std::vector<f32>& obs, const std::vector<f32>& action,
                        f32 reward, const std::vector<f32>& next_obs, bool done)
{
    const u64 idx = pos_;
    std::memcpy(obs_.data() + idx * obs_dim_, obs.data(), obs_dim_ * sizeof(f32));
    std::memcpy(actions_.data() + idx * act_dim_, action.data(), act_dim_ * sizeof(f32));
    rewards_[idx] = reward;
    std::memcpy(next_obs_.data() + idx * obs_dim_, next_obs.data(), obs_dim_ * sizeof(f32));
    dones_[idx]   = done ? 1.0f : 0.0f;

    pos_ = (pos_ + 1) % capacity_;
    if (size_ < capacity_) { size_++; }
}

TransitionBatch ReplayBuffer::Sample(u32 batch_size, Prng& rng) const
{
    if (size_ == 0) {
        throw std::runtime_error("ReplayBuffer::Sample: cannot sample from empty buffer.");
    }

    TransitionBatch batch{};
    batch.obs.resize(batch_size * obs_dim_);
    batch.actions.resize(batch_size * act_dim_);
    batch.rewards.resize(batch_size);
    batch.next_obs.resize(batch_size * obs_dim_);
    batch.dones.resize(batch_size);
    batch.weights.assign(batch_size, 1.0f);
    batch.indices.resize(batch_size);

    for (u32 i = 0; i < batch_size; ++i) {
        const u32 idx = rng.NextBelow(size_);
        batch.indices[i] = idx;
        std::memcpy(batch.obs.data() + i * obs_dim_,
                    obs_.data() + idx * obs_dim_, obs_dim_ * sizeof(f32));
        std::memcpy(batch.actions.data() + i * act_dim_,
                    actions_.data() + idx * act_dim_, act_dim_ * sizeof(f32));
        batch.rewards[i] = rewards_[idx];
        std::memcpy(batch.next_obs.data() + i * obs_dim_,
                    next_obs_.data() + idx * obs_dim_, obs_dim_ * sizeof(f32));
        batch.dones[i]   = dones_[idx];
    }

    return batch;
}

// -------------------------------------------------- PrioritizedReplayBuffer

PrioritizedReplayBuffer::PrioritizedReplayBuffer(u32 capacity, u32 obs_dim, u32 act_dim,
                                                 f32 alpha, f32 beta, f32 beta_anneal)
    : capacity_(capacity),
      obs_dim_(obs_dim),
      act_dim_(act_dim),
      alpha_(alpha),
      beta_(beta),
      beta_anneal_(beta_anneal)
{
    sum_tree_.assign(2ULL * capacity, 0.0f);
    obs_.assign(static_cast<u64>(capacity) * obs_dim, 0.0f);
    actions_.assign(static_cast<u64>(capacity) * act_dim, 0.0f);
    rewards_.assign(capacity, 0.0f);
    next_obs_.assign(static_cast<u64>(capacity) * obs_dim, 0.0f);
    dones_.assign(capacity, 0.0f);
}

void PrioritizedReplayBuffer::TreeUpdate(u64 tree_idx, f32 priority)
{
    const f32 change = priority - sum_tree_[tree_idx];
    sum_tree_[tree_idx] = priority;

    u64 idx = tree_idx;
    while (idx > 0) {
        idx = (idx - 1) / 2;
        sum_tree_[idx] += change;
    }
}

u64 PrioritizedReplayBuffer::TreeGet(f32 s) const
{
    u64 parent = 0;
    while (true) {
        const u64 left_child  = 2 * parent + 1;
        const u64 right_child = left_child + 1;

        if (left_child >= sum_tree_.size()) {
            return parent;
        }

        if (s <= sum_tree_[left_child]) {
            parent = left_child;
        } else {
            s -= sum_tree_[left_child];
            parent = right_child;
        }
    }
}

void PrioritizedReplayBuffer::Push(const std::vector<f32>& obs, const std::vector<f32>& action,
                                  f32 reward, const std::vector<f32>& next_obs, bool done)
{
    const u64 tree_idx = pos_ + capacity_ - 1;

    std::memcpy(obs_.data() + pos_ * obs_dim_, obs.data(), obs_dim_ * sizeof(f32));
    std::memcpy(actions_.data() + pos_ * act_dim_, action.data(), act_dim_ * sizeof(f32));
    rewards_[pos_] = reward;
    std::memcpy(next_obs_.data() + pos_ * obs_dim_, next_obs.data(), obs_dim_ * sizeof(f32));
    dones_[pos_]   = done ? 1.0f : 0.0f;

    TreeUpdate(tree_idx, max_priority_);

    pos_ = (pos_ + 1) % capacity_;
    if (size_ < capacity_) { size_++; }
}

TransitionBatch PrioritizedReplayBuffer::Sample(u32 batch_size, Prng& rng)
{
    if (size_ == 0) {
        throw std::runtime_error("PrioritizedReplayBuffer::Sample: cannot sample from empty buffer.");
    }

    TransitionBatch batch{};
    batch.obs.resize(batch_size * obs_dim_);
    batch.actions.resize(batch_size * act_dim_);
    batch.rewards.resize(batch_size);
    batch.next_obs.resize(batch_size * obs_dim_);
    batch.dones.resize(batch_size);
    batch.weights.resize(batch_size);
    batch.indices.resize(batch_size);

    beta_ = std::min(1.0f, beta_ + beta_anneal_);

    const f32 total_p = sum_tree_[0];
    const f32 segment = total_p / static_cast<f32>(batch_size);

    f32 max_w = 0.0f;

    for (u32 i = 0; i < batch_size; ++i) {
        const f32 a = segment * i;
        const f32 b = segment * (i + 1);
        const f32 s = a + rng.NextFloat() * (b - a);

        const u64 tree_idx = TreeGet(s);
        const u64 data_idx = (tree_idx >= capacity_ - 1) ? (tree_idx - (capacity_ - 1)) : 0;
        const u64 clamped_data_idx = std::min<u64>(data_idx, size_ - 1);

        batch.indices[i] = tree_idx;
        std::memcpy(batch.obs.data() + i * obs_dim_,
                    obs_.data() + clamped_data_idx * obs_dim_, obs_dim_ * sizeof(f32));
        std::memcpy(batch.actions.data() + i * act_dim_,
                    actions_.data() + clamped_data_idx * act_dim_, act_dim_ * sizeof(f32));
        batch.rewards[i] = rewards_[clamped_data_idx];
        std::memcpy(batch.next_obs.data() + i * obs_dim_,
                    next_obs_.data() + clamped_data_idx * obs_dim_, obs_dim_ * sizeof(f32));
        batch.dones[i]   = dones_[clamped_data_idx];

        const f32 p = sum_tree_[tree_idx] / total_p;
        const f32 w = std::pow(static_cast<f32>(size_) * p, -beta_);
        batch.weights[i] = w;
        if (w > max_w) { max_w = w; }
    }

    // Normalize importance sampling weights
    if (max_w > 0.0f) {
        for (f32& w : batch.weights) {
            w /= max_w;
        }
    }

    return batch;
}

void PrioritizedReplayBuffer::UpdatePriorities(const std::vector<u64>& indices,
                                              const std::vector<f32>& td_errors)
{
    for (std::size_t i = 0; i < indices.size(); ++i) {
        const f32 p = std::pow(std::abs(td_errors[i]) + 1e-5f, alpha_);
        TreeUpdate(indices[i], p);
        if (p > max_priority_) {
            max_priority_ = p;
        }
    }
}

} // namespace rl
