#pragma once

#include <deque>
#include <vector>
#include "environment.hpp"
#include "prng.hpp"

namespace rl {

struct SnakePoint {
    i32 x = 0;
    i32 y = 0;

    friend constexpr bool operator==(const SnakePoint& a, const SnakePoint& b) noexcept
    {
        return a.x == b.x && a.y == b.y;
    }
};

enum class SnakeObsMode : u8 {
    FeatureVector = 0, // 12-dim compact feature vector
    GridTensor    = 1, // W * H spatial grid
};

class SnakeEnvV2 final : public Env {
public:
    static constexpr f32 kFoodReward   = 10.0f;
    static constexpr f32 kDeathPenalty = -10.0f;
    static constexpr f32 kStepPenalty  = -0.01f;

    explicit SnakeEnvV2(u32 width = 10, u32 height = 10, u32 num_foods = 1,
                        SnakeObsMode obs_mode = SnakeObsMode::FeatureVector, u64 seed = 42);

    StepResult Reset(std::optional<u64> seed = std::nullopt) override;
    StepResult Step(const std::vector<f32>& action) override;

    [[nodiscard]] const Space& ObservationSpace() const override { return obs_space_; }
    [[nodiscard]] const Space& ActionSpace() const override { return action_space_; }
    [[nodiscard]] u32 ObsDim() const noexcept override;
    [[nodiscard]] u32 ActionDim() const noexcept override { return 4; }
    [[nodiscard]] bool IsDiscreteAction() const noexcept override { return true; }
    [[nodiscard]] std::unique_ptr<Env> Clone() const override;

    [[nodiscard]] u32 Score() const noexcept { return score_; }
    [[nodiscard]] u32 SnakeLength() const noexcept { return static_cast<u32>(body_.size()); }
    [[nodiscard]] const std::deque<SnakePoint>& Body() const noexcept { return body_; }
    [[nodiscard]] const std::vector<SnakePoint>& Foods() const noexcept { return foods_; }

private:
    [[nodiscard]] bool IsCellOccupiedByBody(SnakePoint pt) const noexcept;
    [[nodiscard]] bool IsDeadly(SnakePoint pt) const noexcept;
    void SpawnFood();
    [[nodiscard]] std::vector<f32> BuildObs() const;

    u32                    width_     = 10;
    u32                    height_    = 10;
    u32                    num_foods_ = 1;
    SnakeObsMode           obs_mode_  = SnakeObsMode::FeatureVector;
    u64                    seed_      = 42;
    Prng                   rng_;

    std::deque<SnakePoint> body_;
    std::vector<SnakePoint> foods_;
    u32                    dir_ = 1; // 0=Left, 1=Right, 2=Up, 3=Down
    u32                    score_ = 0;
    f32                    ep_return_ = 0.0f;
    u32                    ep_length_ = 0;

    BoxSpace               obs_space_;
    DiscreteSpace          action_space_;
};

} // namespace rl
