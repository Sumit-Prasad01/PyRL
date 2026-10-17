#pragma once

#include <vector>
#include "environment.hpp"
#include "prng.hpp"

namespace rl {

class CartPoleEnv final : public Env {
public:
    static constexpr f32 kGravity         = 9.8f;
    static constexpr f32 kMassCart        = 1.0f;
    static constexpr f32 kMassPole        = 0.1f;
    static constexpr f32 kTotalMass       = kMassCart + kMassPole;
    static constexpr f32 kLength          = 0.5f; // Half-length of the pole
    static constexpr f32 kPoleMassLength  = kMassPole * kLength;
    static constexpr f32 kForceMag        = 10.0f;
    static constexpr f32 kTau             = 0.02f; // Seconds between state updates
    static constexpr f32 kThetaThreshold  = 12.0f * 3.141592653589793f / 180.0f; // 12 degrees in rad
    static constexpr f32 kXThreshold      = 2.4f;

    explicit CartPoleEnv(u64 seed = 42);

    StepResult Reset(std::optional<u64> seed = std::nullopt) override;
    StepResult Step(const std::vector<f32>& action) override;

    [[nodiscard]] const Space& ObservationSpace() const override { return obs_space_; }
    [[nodiscard]] const Space& ActionSpace() const override { return action_space_; }
    [[nodiscard]] u32 ObsDim() const noexcept override { return 4; }
    [[nodiscard]] u32 ActionDim() const noexcept override { return 2; }
    [[nodiscard]] bool IsDiscreteAction() const noexcept override { return true; }
    [[nodiscard]] std::unique_ptr<Env> Clone() const override;

    [[nodiscard]] f32 X() const noexcept { return x_; }
    [[nodiscard]] f32 XDot() const noexcept { return x_dot_; }
    [[nodiscard]] f32 Theta() const noexcept { return theta_; }
    [[nodiscard]] f32 ThetaDot() const noexcept { return theta_dot_; }

private:
    u64           seed_ = 42;
    Prng          rng_;
    f32           x_         = 0.0f;
    f32           x_dot_     = 0.0f;
    f32           theta_     = 0.0f;
    f32           theta_dot_ = 0.0f;

    f32           ep_return_ = 0.0f;
    u32           ep_length_ = 0;

    BoxSpace      obs_space_;
    DiscreteSpace action_space_;
};

} // namespace rl
