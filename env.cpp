#include "env.hpp"

#include <cassert>
#include <cmath>
#include <stdexcept>

#include "mat.hpp"


SnakeEnv::SnakeEnv(u32 grid_size, u64 seed)
    :rng_(seed, /*initseq=*/54u)
{
    const u64 side = static_cast<u64>(std::llround(std::sqrt(static_cast<f64>(grid_size))));
    if (side < 2 || side * side != grid_size) {
        throw std::invalid_argument("SnakeEnv: grid_size must be a perfect square with side >= 2");
    }
    rows_ = cols_ = static_cast<u32>(side);
    grid_size_   = grid_size;
    Reset();
}


void SnakeEnv::Reset()
{
    snake_ = Point{static_cast<i32>(cols_ / 2), static_cast<i32>(rows_ / 2)};
    PalceFood();
    score_       = 0.0f;
    foods_eaten_ = 0;
    pov_         = Action::Right;
    steps_       = 0;
}


SnakeEnv::StepResult SnakeEnv::Step(Action action) 
{
    Move(action);
    ++steps_;

    f32 reward = 0.0f;
    if (snake_ == food_) {
        reward += kFoodReward;
        ++foods_eaten_;
        PalceFood();
    }
    const bool done = OutOfBounds();
    if (done) {
        reward += kWallPenalty;
    }
    score_ += reward;
    return {reward, done};
}


void SnakeEnv::EncodeState(Mat& out, Point snake, Point food, Action pov) const noexcept
{
    assert(out.Size() == InputSize());
    assert(snake.x >= 0 && snake.x < static_cast<i32>(cols_) &&
            snake.y >= 0 && snake.y < static_cast<i32>(rows_));
    assert(food.x >= 0 && food.x < static_cast<i32>(cols_) &&
            food.y >= 0 && food.y < static_cast<i32>(rows_));
    assert(static_cast<u32>(pov) < kNumDirections);

    out.Clear();
    const auto data = out.Span();
    const u64 snake_i = u64{static_cast<u32>(snake.y)} * cols_ + static_cast<u32>(snake.x);
    const u64 food_i = u64{static_cast<u32>(food.y)} * cols_ + static_cast<u32>(food.x);

    data[snake_i]              = 1.0f;
    data[grid_size_ + food_i]  = 1.0f;
    data[2 * u64{grid_size_} + static_cast<u32>(pov)] = 1.0f;
}


bool SnakeEnv::OutOfBounds() const noexcept
{
    return snake_.x < 0 || snake_.x >= static_cast<i32>(cols_) ||
            snake_.y < 0 || snake_.y >= static_cast<i32>(rows_);
}


bool SnakeEnv::RandomPoint() 
{
    return Point{static_cast<i32>(rng_.NextBelow(cols_))
                static_cast<i32>(rng_.NextBelow(rows_))};
}



void SnakeEnv::PalceFood()
{
    do {
        food_ = RandomPoint();
    }while (food_ == snake_);
}



void SnakeEnv::Move(Action action) noexcept
{
    if (action == Action::None) {
        action = pov_;
    }
    switch (action) {
        case Action::Left:  snake_.x -= 1;    break;
        case Action::Right: snake_.x += 1;    break;
        case Action::Up:    snake_.y += 1;    break;
        case Action::Down:  snake_.x -= 1;    break;
        case Action::None:  return 
    }
    pov_ = action;
}