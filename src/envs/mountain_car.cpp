#include "mountain_car.hpp"

#include <cmath>
#include <algorithm>

namespace rl {

MountainCarEnv::MountainCarEnv(u64 seed)
    : seed_(seed),
      rng_(seed, seed ^ 0x112233445566ULL),
      obs_space_(2, -1.2f, 0.6f),
      action_space_(3)
{
}

StepResult MountainCarEnv::Reset(std::optional<u64> seed)
{
    if (seed.has_value()) {
        seed_ = seed.value();
        rng_.Seed(seed_, seed_ ^ 0x112233445566ULL);
    }

    // Uniformly sample position in [-0.6, -0.4], velocity = 0
    pos_ = rng_.NextFloat() * 0.2f - 0.6f;
    vel_ = 0.0f;

    ep_return_ = 0.0f;
    ep_length_ = 0;

    StepResult res{};
    res.obs = {pos_, vel_};
    res.reward = 0.0f;
    res.terminated = false;
    res.truncated = false;
    res.ep_return = 0.0f;
    res.ep_length = 0;
    return res;
}

StepResult MountainCarEnv::Step(const std::vector<f32>& action)
{
    // Action: 0=push left, 1=no push, 2=push right
    const i32 act = (!action.empty()) ? std::clamp<i32>(static_cast<i32>(action[0]), 0, 2) : 1;
    const f32 force = static_cast<f32>(act - 1); // {-1, 0, 1}

    vel_ += force * 0.001f - std::cos(3.0f * pos_) * 0.0025f;
    vel_ = std::clamp(vel_, -kMaxVelocity, kMaxVelocity);

    pos_ += vel_;
    pos_ = std::clamp(pos_, kMinPosition, kMaxPosition);

    if (pos_ <= kMinPosition && vel_ < 0.0f) {
        vel_ = 0.0f;
    }

    const bool terminated = (pos_ >= kGoalPosition);
    const f32 reward = -1.0f;

    ep_return_ += reward;
    ep_length_++;

    StepResult res{};
    res.obs = {pos_, vel_};
    res.reward = reward;
    res.terminated = terminated;
    res.truncated = false;
    res.ep_return = ep_return_;
    res.ep_length = ep_length_;
    return res;
}

std::unique_ptr<Env> MountainCarEnv::Clone() const
{
    auto c = std::make_unique<MountainCarEnv>(seed_);
    c->pos_ = pos_;
    c->vel_ = vel_;
    c->ep_return_ = ep_return_;
    c->ep_length_ = ep_length_;
    return c;
}

} // namespace rl
