#pragma once

#include <vector>
#include "environment.hpp"
#include "prng.hpp"

namespace rl {

class MountainCarEnv final : public Env {
public:
    static constexpr f32 kMinPosition = -1.2f;
    static constexpr f32 kMaxPosition = 0.6f;
    static constexpr f32 kMaxVelocity = 0.07f;
    static constexpr f32 kGoalPosition = 0.5f;

    explicit MountainCarEnv(u64 seed = 42);

    StepResult Reset(std::optional<u64> seed = std::nullopt) override;
    StepResult Step(const std::vector<f32>& action) override;

    [[nodiscard]] const Space& ObservationSpace() const override { return obs_space_; }
    [[nodiscard]] const Space& ActionSpace() const override { return action_space_; }
    [[nodiscard]] u32 ObsDim() const noexcept override { return 2; }
    [[nodiscard]] u32 ActionDim() const noexcept override { return 3; }
    [[nodiscard]] bool IsDiscreteAction() const noexcept override { return true; }
    [[nodiscard]] std::unique_ptr<Env> Clone() const override;

    [[nodiscard]] f32 Position() const noexcept { return pos_; }
    [[nodiscard]] f32 Velocity() const noexcept { return vel_; }

private:
    u64           seed_ = 42;
    Prng          rng_;
    f32           pos_ = -0.5f;
    f32           vel_ = 0.0f;

    f32           ep_return_ = 0.0f;
    u32           ep_length_ = 0;

    BoxSpace      obs_space_;
    DiscreteSpace action_space_;
};

} // namespace rl
