#include "pendulum.hpp"

#include <cmath>
#include <algorithm>

namespace rl {

namespace {
constexpr f32 kPi = 3.14159265358979323846f;
}

f32 PendulumEnv::AngleNormalize(f32 x) noexcept
{
    // Return angle in [-pi, pi]
    return std::fmod(x + kPi, 2.0f * kPi) - ((x + kPi < 0.0f) ? (2.0f * kPi - kPi) : kPi);
}

PendulumEnv::PendulumEnv(u64 seed)
    : seed_(seed),
      rng_(seed, seed ^ 0x998877665544ULL),
      obs_space_({1, 1, 8}, -8.0f, 8.0f),
      action_space_(1, -kMaxTorque, kMaxTorque)
{
}

StepResult PendulumEnv::Reset(std::optional<u64> seed)
{
    if (seed.has_value()) {
        seed_ = seed.value();
        rng_.Seed(seed_, seed_ ^ 0x998877665544ULL);
    }

    // theta in [-pi, pi], theta_dot in [-1.0, 1.0]
    theta_     = rng_.NextFloat() * 2.0f * kPi - kPi;
    theta_dot_ = rng_.NextFloat() * 2.0f - 1.0f;

    ep_return_ = 0.0f;
    ep_length_ = 0;

    StepResult res{};
    res.obs = {std::cos(theta_), std::sin(theta_), theta_dot_};
    res.reward = 0.0f;
    res.terminated = false;
    res.truncated = false;
    res.ep_return = 0.0f;
    res.ep_length = 0;
    return res;
}

StepResult PendulumEnv::Step(const std::vector<f32>& action)
{
    const f32 u = (!action.empty()) ? std::clamp(action[0], -kMaxTorque, kMaxTorque) : 0.0f;

    const f32 norm_th = AngleNormalize(theta_);
    const f32 cost = norm_th * norm_th + 0.1f * theta_dot_ * theta_dot_ + 0.001f * u * u;
    const f32 reward = -cost;

    const f32 alpha = (3.0f * kGravity / (2.0f * kLength)) * std::sin(theta_) +
                      (3.0f / (kMass * kLength * kLength)) * u;

    theta_dot_ += alpha * kDt;
    theta_dot_ = std::clamp(theta_dot_, -kMaxSpeed, kMaxSpeed);
    theta_     += theta_dot_ * kDt;

    ep_return_ += reward;
    ep_length_++;

    StepResult res{};
    res.obs = {std::cos(theta_), std::sin(theta_), theta_dot_};
    res.reward = reward;
    res.terminated = false; // Does not terminate naturally; use TimeLimit wrapper
    res.truncated = false;
    res.ep_return = ep_return_;
    res.ep_length = ep_length_;
    return res;
}

std::unique_ptr<Env> PendulumEnv::Clone() const
{
    auto c = std::make_unique<PendulumEnv>(seed_);
    c->theta_ = theta_;
    c->theta_dot_ = theta_dot_;
    c->ep_return_ = ep_return_;
    c->ep_length_ = ep_length_;
    return c;
}

} // namespace rl
