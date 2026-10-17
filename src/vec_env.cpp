#include "vec_env.hpp"

#include <cstring>
#include <stdexcept>

namespace rl {

VecEnv::VecEnv(const Env& prototype_env, u32 num_envs, u64 base_seed, bool use_threads)
    : num_envs_(num_envs),
      obs_dim_(prototype_env.ObsDim()),
      action_dim_(prototype_env.ActionDim()),
      is_discrete_action_(prototype_env.IsDiscreteAction()),
      use_threads_(use_threads),
      base_seed_(base_seed)
{
    if (num_envs == 0) {
        throw std::invalid_argument("VecEnv: num_envs must be >= 1.");
    }
    envs_.reserve(num_envs);
    for (u32 i = 0; i < num_envs; ++i) {
        auto env = prototype_env.Clone();
        env->Reset(base_seed + static_cast<u64>(i) * 1000);
        envs_.push_back(std::move(env));
    }
    InitializeBuffers();
}

VecEnv::VecEnv(std::function<std::unique_ptr<Env>(u32 env_id, u64 seed)> factory,
               u32 num_envs, u64 base_seed, bool use_threads)
    : num_envs_(num_envs),
      use_threads_(use_threads),
      base_seed_(base_seed)
{
    if (num_envs == 0) {
        throw std::invalid_argument("VecEnv: num_envs must be >= 1.");
    }
    envs_.reserve(num_envs);
    for (u32 i = 0; i < num_envs; ++i) {
        const u64 s = base_seed + static_cast<u64>(i) * 1000;
        auto env = factory(i, s);
        if (!env) {
            throw std::invalid_argument("VecEnv: factory returned null env.");
        }
        env->Reset(s);
        envs_.push_back(std::move(env));
    }
    obs_dim_ = envs_.front()->ObsDim();
    action_dim_ = envs_.front()->ActionDim();
    is_discrete_action_ = envs_.front()->IsDiscreteAction();
    InitializeBuffers();
}

void VecEnv::InitializeBuffers()
{
    batch_obs_.assign(static_cast<u64>(num_envs_) * obs_dim_, 0.0f);
    // Fill initial batch observations
    for (u32 i = 0; i < num_envs_; ++i) {
        StepResult res = envs_[i]->Reset();
        std::memcpy(batch_obs_.data() + static_cast<u64>(i) * obs_dim_,
                    res.obs.data(), obs_dim_ * sizeof(f32));
    }

#ifdef RL_USE_CUDA
    if (HasCudaDevice()) {
        gpu_obs_.Allocate(batch_obs_.size() * sizeof(f32));
    }
#endif
}

void VecEnv::Reset(std::optional<u64> base_seed)
{
    if (base_seed.has_value()) {
        base_seed_ = base_seed.value();
    }

    auto reset_worker = [&](u32 start, u32 end) {
        for (u32 i = start; i < end; ++i) {
            const u64 s = base_seed_ + static_cast<u64>(i) * 1000;
            StepResult res = envs_[i]->Reset(s);
            std::memcpy(batch_obs_.data() + static_cast<u64>(i) * obs_dim_,
                        res.obs.data(), obs_dim_ * sizeof(f32));
        }
    };

    if (use_threads_ && num_envs_ > 1) {
        ThreadPool::Global().ParallelFor(0, num_envs_, reset_worker);
    } else {
        reset_worker(0, num_envs_);
    }
}

StepResultBatch VecEnv::Step(const std::vector<f32>& actions)
{
    const u32 act_stride = is_discrete_action_ ? 1 : action_dim_;
    const u64 expected_acts = static_cast<u64>(num_envs_) * act_stride;
    if (actions.size() != expected_acts) {
        throw std::invalid_argument("VecEnv::Step: actions size mismatch. Expected " +
                                    std::to_string(expected_acts) + ", got " +
                                    std::to_string(actions.size()));
    }

    StepResultBatch batch{};
    batch.rewards.resize(num_envs_, 0.0f);
    batch.terminated.resize(num_envs_, 0);
    batch.truncated.resize(num_envs_, 0);
    batch.dones.resize(num_envs_, 0);
    batch.terminal_obs.resize(num_envs_);
    batch.ep_returns.resize(num_envs_, 0.0f);
    batch.ep_lengths.resize(num_envs_, 0);

    auto step_worker = [&](u32 start, u32 end) {
        std::vector<f32> act_single(act_stride);
        for (u32 i = start; i < end; ++i) {
            for (u32 a = 0; a < act_stride; ++a) {
                act_single[a] = actions[i * act_stride + a];
            }

            StepResult res = envs_[i]->Step(act_single);
            batch.rewards[i]    = res.reward;
            batch.terminated[i] = res.terminated ? 1 : 0;
            batch.truncated[i]  = res.truncated ? 1 : 0;
            batch.dones[i]      = res.Done() ? 1 : 0;
            batch.ep_returns[i] = res.ep_return;
            batch.ep_lengths[i] = res.ep_length;

            if (res.Done()) {
                batch.terminal_obs[i] = res.obs;
                // Auto-reset on done
                StepResult reset_res = envs_[i]->Reset();
                std::memcpy(batch_obs_.data() + static_cast<u64>(i) * obs_dim_,
                            reset_res.obs.data(), obs_dim_ * sizeof(f32));
            } else {
                std::memcpy(batch_obs_.data() + static_cast<u64>(i) * obs_dim_,
                            res.obs.data(), obs_dim_ * sizeof(f32));
            }
        }
    };

    if (use_threads_ && num_envs_ > 1) {
        ThreadPool::Global().ParallelFor(0, num_envs_, step_worker);
    } else {
        step_worker(0, num_envs_);
    }

    return batch;
}

void VecEnv::SyncObsToDevice(Device dev)
{
    if (dev == Device::CUDA) {
#ifdef RL_USE_CUDA
        if (HasCudaDevice()) {
            if (!gpu_obs_.IsAllocated()) {
                gpu_obs_.Allocate(batch_obs_.size() * sizeof(f32));
            }
            gpu_obs_.CopyFromHost(batch_obs_.data(), batch_obs_.size() * sizeof(f32));
        }
#endif
    }
}

f32* VecEnv::GetObsPtr(Device dev) noexcept
{
    if (dev == Device::CUDA && gpu_obs_.IsAllocated()) {
        return gpu_obs_.AsFloat();
    }
    return batch_obs_.data();
}

const f32* VecEnv::GetObsPtr(Device dev) const noexcept
{
    if (dev == Device::CUDA && gpu_obs_.IsAllocated()) {
        return gpu_obs_.AsFloat();
    }
    return batch_obs_.data();
}

} // namespace rl
