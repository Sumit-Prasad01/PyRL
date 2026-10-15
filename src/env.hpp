#pragma once

#include "base.hpp"
#include "mat.hpp"
#include "prng.hpp"

enum class Action : u8 {
    Left  = 0,
    Right = 1,
    Up    = 2,
    Down  = 3,
    None  = 4,
};

inline constexpr u32 kNumActions    = 5;
inline constexpr u32 kNumDirections = 4;

struct Point {
    i32 x = 0;
    i32 y = 0;

    friend constexpr bool operator==(const Point&, const Point&) noexcept = default;
};

class SnakeEnv {
public: 
    static constexpr f32 kFoodReward  = 20.0f;
    static constexpr f32 kWallPenalty = -10.0f;
    static constexpr u64 kDefaultSeed = 42;

    struct StepResult {
        f32 reward;
        bool done;
    };

    explicit SnakeEnv(u32 grid_size, u64 seed = kDefaultSeed);

    void Reset();

    [[nodiscard]] StepResult Step(Action action);

    void EncodeState(Mat& out, Point snake, Point food, Action pov) const noexcept;

    [[nodiscard]] bool IsGameOver() const noexcept { return OutOfBounds(); }

    [[nodiscard]] u32   InputSize() const noexcept  { return 2 * grid_size_ + kNumDirections; }
    [[nodiscard]] Point Snake() const noexcept      { return snake_; }
    [[nodiscard]] Point Food() const noexcept       { return food_; }
    [[nodiscard]] Action Pov() const noexcept       { return pov_; }
    [[nodiscard]] f32   Score() const noexcept      { return score_; }
    [[nodiscard]] u32   FoodsEaten() const noexcept { return foods_eaten_; }
    [[nodiscard]] u64   Steps() const noexcept      { return steps_; }
    [[nodiscard]] u32   Rows() const noexcept       { return rows_; }
    [[nodiscard]] u32   Cols() const noexcept       { return cols_; }
    [[nodiscard]] u32   GridSize() const noexcept   { return grid_size_; }

private:
    [[nodiscard]] bool OutOfBounds() const noexcept;
    [[nodiscard]] Point RandomPoint(); 
    void PlaceFood();
    void Move(Action action) noexcept;

    Prng    rng_;
    Point   snake_{};
    Point   food_{};
    Action  pov_         = Action::Right;
    f32     score_       = 0.0f;
    u32     foods_eaten_ = 0;
    u64     steps_       = 0;
    u32     rows_        = 0;
    u32     cols_        = 0;
    u32     grid_size_   = 0;
};
