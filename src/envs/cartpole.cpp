#include "cartpole.hpp"

#include <cmath>
#include <algorithm>

namespace rl {

CartPoleEnv::CartPoleEnv(u64 seed)
    : seed_(seed),
      rng_(seed, seed ^ 0xabcdef012345ULL),
      obs_space_(4, -10.0f, 10.0f),
      action_space_(2)
{
}

StepResult CartPoleEnv::Reset(std::optional<u64> seed)
{
    if (seed.has_value()) {
        seed_ = seed.value();
        rng_.Seed(seed_, seed_ ^ 0xabcdef012345ULL);
    }

    // Uniformly sample initial state in [-0.05, 0.05]
    x_         = rng_.NextFloat() * 0.1f - 0.05f;
    x_dot_     = rng_.NextFloat() * 0.1f - 0.05f;
    theta_     = rng_.NextFloat() * 0.1f - 0.05f;
    theta_dot_ = rng_.NextFloat() * 0.1f - 0.05f;

    ep_return_ = 0.0f;
    ep_length_ = 0;

    StepResult res{};
    res.obs = {x_, x_dot_, theta_, theta_dot_};
    res.reward = 0.0f;
    res.terminated = false;
    res.truncated = false;
    res.ep_return = 0.0f;
    res.ep_length = 0;
    return res;
}

StepResult CartPoleEnv::Step(const std::vector<f32>& action)
{
    const u32 act = (!action.empty() && action[0] > 0.5f) ? 1 : 0;
    const f32 force = (act == 1) ? kForceMag : -kForceMag;

    const f32 costheta = std::cos(theta_);
    const f32 sintheta = std::sin(theta_);

    const f32 temp = (force + kPoleMassLength * theta_dot_ * theta_dot_ * sintheta) / kTotalMass;
    const f32 thetaacc = (kGravity * sintheta - costheta * temp) /
                         (kLength * (4.0f / 3.0f - (kMassPole * costheta * costheta) / kTotalMass));
    const f32 xacc = temp - (kPoleMassLength * thetaacc * costheta) / kTotalMass;

    // Semi-implicit Euler integration
    x_         += kTau * x_dot_;
    x_dot_     += kTau * xacc;
    theta_     += kTau * theta_dot_;
    theta_dot_ += kTau * thetaacc;

    ep_length_++;
    const bool terminated = (x_ < -kXThreshold || x_ > kXThreshold ||
                             theta_ < -kThetaThreshold || theta_ > kThetaThreshold);
    const f32 reward = 1.0f;
    ep_return_ += reward;

    StepResult res{};
    res.obs = {x_, x_dot_, theta_, theta_dot_};
    res.reward = reward;
    res.terminated = terminated;
    res.truncated = false;
    res.ep_return = ep_return_;
    res.ep_length = ep_length_;
    return res;
}

std::unique_ptr<Env> CartPoleEnv::Clone() const
{
    auto c = std::make_unique<CartPoleEnv>(seed_);
    c->x_ = x_;
    c->x_dot_ = x_dot_;
    c->theta_ = theta_;
    c->theta_dot_ = theta_dot_;
    c->ep_return_ = ep_return_;
    c->ep_length_ = ep_length_;
    return c;
}

} // namespace rl
