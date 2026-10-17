#pragma once

#include <vector>
#include "environment.hpp"
#include "prng.hpp"

namespace rl {

class PendulumEnv final : public Env {
public:
    static constexpr f32 kGravity   = 9.81f;
    static constexpr f32 kMass      = 1.0f;
    static constexpr f32 kLength    = 1.0f;
    static constexpr f32 kMaxSpeed  = 8.0f;
    static constexpr f32 kMaxTorque = 2.0f;
    static constexpr f32 kDt        = 0.05f;

    explicit PendulumEnv(u64 seed = 42);

    StepResult Reset(std::optional<u64> seed = std::nullopt) override;
    StepResult Step(const std::vector<f32>& action) override;

    [[nodiscard]] const Space& ObservationSpace() const override { return obs_space_; }
    [[nodiscard]] const Space& ActionSpace() const override { return action_space_; }
    [[nodiscard]] u32 ObsDim() const noexcept override { return 3; }
    [[nodiscard]] u32 ActionDim() const noexcept override { return 1; }
    [[nodiscard]] bool IsDiscreteAction() const noexcept override { return false; }
    [[nodiscard]] std::unique_ptr<Env> Clone() const override;

    [[nodiscard]] f32 Theta() const noexcept { return theta_; }
    [[nodiscard]] f32 ThetaDot() const noexcept { return theta_dot_; }

    [[nodiscard]] static f32 AngleNormalize(f32 x) noexcept;

private:
    u64      seed_ = 42;
    Prng     rng_;
    f32      theta_     = 0.0f;
    f32      theta_dot_ = 0.0f;

    f32      ep_return_ = 0.0f;
    u32      ep_length_ = 0;

    BoxSpace obs_space_;
    BoxSpace action_space_;
};

} // namespace rl
