#pragma once

#include <vector>
#include "environment.hpp"
#include "prng.hpp"

namespace rl {

struct GridPos {
    i32 x = 0;
    i32 y = 0;

    friend constexpr bool operator==(const GridPos& a, const GridPos& b) noexcept
    {
        return a.x == b.x && a.y == b.y;
    }
};

class GridWorldEnv final : public Env {
public:
    explicit GridWorldEnv(u32 width = 4, u32 height = 4, f32 slip_prob = 0.0f, u64 seed = 42);

    StepResult Reset(std::optional<u64> seed = std::nullopt) override;
    StepResult Step(const std::vector<f32>& action) override;

    [[nodiscard]] const Space& ObservationSpace() const override { return obs_space_; }
    [[nodiscard]] const Space& ActionSpace() const override { return action_space_; }
    [[nodiscard]] u32 ObsDim() const noexcept override { return 2; }
    [[nodiscard]] u32 ActionDim() const noexcept override { return 4; }
    [[nodiscard]] bool IsDiscreteAction() const noexcept override { return true; }
    [[nodiscard]] std::unique_ptr<Env> Clone() const override;

    void SetObstacles(std::vector<GridPos> walls) { walls_ = std::move(walls); }
    void SetTraps(std::vector<GridPos> traps) { traps_ = std::move(traps); }
    void SetGoal(GridPos goal) { goal_ = goal; }

private:
    [[nodiscard]] bool IsWall(GridPos p) const noexcept;
    [[nodiscard]] bool IsTrap(GridPos p) const noexcept;
    [[nodiscard]] GridPos NextPos(GridPos cur, u32 action) const noexcept;

    u32                  width_  = 4;
    u32                  height_ = 4;
    f32                  slip_prob_ = 0.0f;
    u64                  seed_ = 42;
    Prng                 rng_;
    GridPos              pos_{0, 0};
    GridPos              goal_{3, 3};
    std::vector<GridPos> walls_;
    std::vector<GridPos> traps_;

    BoxSpace             obs_space_;
    DiscreteSpace        action_space_;

    f32                  ep_return_ = 0.0f;
    u32                  ep_length_ = 0;
};

} // namespace rl
