#include "bandit.hpp"

#include <algorithm>

namespace rl {

BanditEnv::BanditEnv(u32 num_arms, u64 seed)
    : rng_(seed, seed ^ 0x9e3779b97f4a7c15ULL),
      means_(num_arms, 0.0f),
      noise_std_(1.0f),
      obs_space_(1, 0.0f, 1.0f),
      action_space_(num_arms),
      current_seed_(seed)
{
    std::normal_distribution<f32> dist(0.0f, 1.0f);
    for (f32& m : means_) {
        m = dist(rng_);
    }
}

BanditEnv::BanditEnv(std::vector<f32> true_means, f32 noise_std, u64 seed)
    : rng_(seed, seed ^ 0x9e3779b97f4a7c15ULL),
      means_(std::move(true_means)),
      noise_std_(noise_std),
      obs_space_(1, 0.0f, 1.0f),
      action_space_(static_cast<u32>(means_.size())),
      current_seed_(seed)
{
}

StepResult BanditEnv::Reset(std::optional<u64> seed)
{
    if (seed.has_value()) {
        current_seed_ = seed.value();
        rng_.Seed(current_seed_, current_seed_ ^ 0x9e3779b97f4a7c15ULL);
    }
    StepResult res{};
    res.obs = {1.0f}; // Constant context
    res.reward = 0.0f;
    res.terminated = false;
    res.truncated = false;
    res.ep_return = 0.0f;
    res.ep_length = 0;
    return res;
}

StepResult BanditEnv::Step(const std::vector<f32>& action)
{
    u32 arm = 0;
    if (!action.empty()) {
        arm = static_cast<u32>(std::clamp<i64>(static_cast<i64>(action[0]), 0, static_cast<i64>(means_.size() - 1)));
    }

    std::normal_distribution<f32> dist(0.0f, 1.0f);
    const f32 reward = means_[arm] + dist(rng_) * noise_std_;

    StepResult res{};
    res.obs = {1.0f};
    res.reward = reward;
    res.terminated = true; // 1-step episode
    res.truncated = false;
    res.ep_return = reward;
    res.ep_length = 1;
    return res;
}

u32 BanditEnv::OptimalArm() const noexcept
{
    if (means_.empty()) { return 0; }
    auto it = std::max_element(means_.begin(), means_.end());
    return static_cast<u32>(std::distance(means_.begin(), it));
}

std::unique_ptr<Env> BanditEnv::Clone() const
{
    return std::make_unique<BanditEnv>(means_, noise_std_, current_seed_);
}

} // namespace rl
