#include "gridworld.hpp"

#include <algorithm>

namespace rl {

GridWorldEnv::GridWorldEnv(u32 width, u32 height, f32 slip_prob, u64 seed)
    : width_(width),
      height_(height),
      slip_prob_(slip_prob),
      seed_(seed),
      rng_(seed, seed ^ 0x85437892345ULL),
      pos_{0, 0},
      goal_{static_cast<i32>(width - 1), static_cast<i32>(height - 1)},
      obs_space_(2, 0.0f, 1.0f),
      action_space_(4)
{
}

StepResult GridWorldEnv::Reset(std::optional<u64> seed)
{
    if (seed.has_value()) {
        seed_ = seed.value();
        rng_.Seed(seed_, seed_ ^ 0x85437892345ULL);
    }
    pos_ = {0, 0};
    ep_return_ = 0.0f;
    ep_length_ = 0;

    StepResult res{};
    res.obs = {
        static_cast<f32>(pos_.x) / std::max(1.0f, static_cast<f32>(width_ - 1)),
        static_cast<f32>(pos_.y) / std::max(1.0f, static_cast<f32>(height_ - 1))
    };
    res.reward = 0.0f;
    res.terminated = false;
    res.truncated = false;
    res.ep_return = 0.0f;
    res.ep_length = 0;
    return res;
}

StepResult GridWorldEnv::Step(const std::vector<f32>& action)
{
    u32 act = 0;
    if (!action.empty()) {
        act = static_cast<u32>(std::clamp<i64>(static_cast<i64>(action[0]), 0, 3));
    }

    // Stochastic transition slip
    if (slip_prob_ > 0.0f && rng_.NextFloat() < slip_prob_) {
        // Pick an orthogonal action (if horizontal, slip vertically; if vertical, slip horizontally)
        if (act == 0 || act == 1) {
            act = (rng_.NextFloat() < 0.5f) ? 2 : 3;
        } else {
            act = (rng_.NextFloat() < 0.5f) ? 0 : 1;
        }
    }

    const GridPos next = NextPos(pos_, act);
    if (!IsWall(next)) {
        pos_ = next;
    }

    f32 reward = -0.01f; // Step cost
    bool terminated = false;

    if (pos_ == goal_) {
        reward += 1.0f;
        terminated = true;
    } else if (IsTrap(pos_)) {
        reward -= 1.0f;
        terminated = true;
    }

    ep_return_ += reward;
    ep_length_++;

    StepResult res{};
    res.obs = {
        static_cast<f32>(pos_.x) / std::max(1.0f, static_cast<f32>(width_ - 1)),
        static_cast<f32>(pos_.y) / std::max(1.0f, static_cast<f32>(height_ - 1))
    };
    res.reward = reward;
    res.terminated = terminated;
    res.truncated = false;
    res.ep_return = ep_return_;
    res.ep_length = ep_length_;
    return res;
}

bool GridWorldEnv::IsWall(GridPos p) const noexcept
{
    for (const auto& w : walls_) {
        if (w == p) { return true; }
    }
    return false;
}

bool GridWorldEnv::IsTrap(GridPos p) const noexcept
{
    for (const auto& t : traps_) {
        if (t == p) { return true; }
    }
    return false;
}

GridPos GridWorldEnv::NextPos(GridPos cur, u32 action) const noexcept
{
    GridPos next = cur;
    switch (action) {
    case 0: next.x = std::max(0, cur.x - 1); break;                         // Left
    case 1: next.x = std::min(static_cast<i32>(width_ - 1), cur.x + 1); break;  // Right
    case 2: next.y = std::max(0, cur.y - 1); break;                         // Up
    case 3: next.y = std::min(static_cast<i32>(height_ - 1), cur.y + 1); break; // Down
    default: break;
    }
    return next;
}

std::unique_ptr<Env> GridWorldEnv::Clone() const
{
    auto c = std::make_unique<GridWorldEnv>(width_, height_, slip_prob_, seed_);
    c->walls_ = walls_;
    c->traps_ = traps_;
    c->goal_ = goal_;
    return c;
}

} // namespace rl
