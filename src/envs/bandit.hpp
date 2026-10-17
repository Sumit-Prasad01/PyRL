#pragma once

#include <vector>
#include "environment.hpp"
#include "prng.hpp"

namespace rl {

class BanditEnv final : public Env {
public:
    explicit BanditEnv(u32 num_arms = 10, u64 seed = 42);
    BanditEnv(std::vector<f32> true_means, f32 noise_std = 1.0f, u64 seed = 42);

    StepResult Reset(std::optional<u64> seed = std::nullopt) override;
    StepResult Step(const std::vector<f32>& action) override;

    [[nodiscard]] const Space& ObservationSpace() const override { return obs_space_; }
    [[nodiscard]] const Space& ActionSpace() const override { return action_space_; }
    [[nodiscard]] u32 ObsDim() const noexcept override { return 1; }
    [[nodiscard]] u32 ActionDim() const noexcept override { return static_cast<u32>(means_.size()); }
    [[nodiscard]] bool IsDiscreteAction() const noexcept override { return true; }
    [[nodiscard]] std::unique_ptr<Env> Clone() const override;

    [[nodiscard]] u32 OptimalArm() const noexcept;
    [[nodiscard]] const std::vector<f32>& Means() const noexcept { return means_; }

private:
    Prng             rng_;
    std::vector<f32> means_;
    f32              noise_std_ = 1.0f;
    BoxSpace         obs_space_;
    DiscreteSpace    action_space_;
    u64              current_seed_ = 42;
};

} // namespace rl
