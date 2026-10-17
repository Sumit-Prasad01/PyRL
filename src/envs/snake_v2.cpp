#include "snake_v2.hpp"

#include <algorithm>
#include <cmath>

namespace rl {

SnakeEnvV2::SnakeEnvV2(u32 width, u32 height, u32 num_foods, SnakeObsMode obs_mode, u64 seed)
    : width_(width),
      height_(height),
      num_foods_(num_foods),
      obs_mode_(obs_mode),
      seed_(seed),
      rng_(seed, seed ^ 0x334455667788ULL),
      obs_space_((obs_mode == SnakeObsMode::FeatureVector) ? 12 : (width * height), -1.0f, 1.0f),
      action_space_(4)
{
}

u32 SnakeEnvV2::ObsDim() const noexcept
{
    return (obs_mode_ == SnakeObsMode::FeatureVector) ? 12 : (width_ * height_);
}

StepResult SnakeEnvV2::Reset(std::optional<u64> seed)
{
    if (seed.has_value()) {
        seed_ = seed.value();
        rng_.Seed(seed_, seed_ ^ 0x334455667788ULL);
    }

    body_.clear();
    foods_.clear();
    score_ = 0;
    ep_return_ = 0.0f;
    ep_length_ = 0;
    dir_ = 1; // Facing Right

    const i32 start_x = static_cast<i32>(width_ / 2);
    const i32 start_y = static_cast<i32>(height_ / 2);

    // Initial 3 segments: head at start_x, body trailing to left
    body_.push_back({start_x, start_y});
    body_.push_back({start_x - 1, start_y});
    body_.push_back({start_x - 2, start_y});

    for (u32 f = 0; f < num_foods_; ++f) {
        SpawnFood();
    }

    StepResult res{};
    res.obs = BuildObs();
    res.reward = 0.0f;
    res.terminated = false;
    res.truncated = false;
    res.ep_return = 0.0f;
    res.ep_length = 0;
    return res;
}

StepResult SnakeEnvV2::Step(const std::vector<f32>& action)
{
    u32 act = dir_;
    if (!action.empty()) {
        act = static_cast<u32>(std::clamp<i64>(static_cast<i64>(action[0]), 0, 3));
    }

    // Prevent immediate 180-degree turnaround into neck
    const bool is_opposite = (dir_ == 0 && act == 1) || (dir_ == 1 && act == 0) ||
                             (dir_ == 2 && act == 3) || (dir_ == 3 && act == 2);
    if (!is_opposite) {
        dir_ = act;
    }

    SnakePoint next_head = body_.front();
    switch (dir_) {
    case 0: next_head.x -= 1; break; // Left
    case 1: next_head.x += 1; break; // Right
    case 2: next_head.y -= 1; break; // Up
    case 3: next_head.y += 1; break; // Down
    default: break;
    }

    ep_length_++;

    // Check boundary or self-collision
    if (IsDeadly(next_head)) {
        ep_return_ += kDeathPenalty;
        StepResult res{};
        res.obs = BuildObs();
        res.reward = kDeathPenalty;
        res.terminated = true;
        res.truncated = false;
        res.ep_return = ep_return_;
        res.ep_length = ep_length_;
        return res;
    }

    body_.push_front(next_head);

    // Check if food eaten
    auto food_it = std::find(foods_.begin(), foods_.end(), next_head);
    f32 reward = kStepPenalty;

    if (food_it != foods_.end()) {
        foods_.erase(food_it);
        score_++;
        reward += kFoodReward;
        SpawnFood();
        // Do not pop tail, snake grows!
    } else {
        // Pop tail
        body_.pop_back();
    }

    ep_return_ += reward;

    StepResult res{};
    res.obs = BuildObs();
    res.reward = reward;
    res.terminated = false;
    res.truncated = false;
    res.ep_return = ep_return_;
    res.ep_length = ep_length_;
    return res;
}

bool SnakeEnvV2::IsCellOccupiedByBody(SnakePoint pt) const noexcept
{
    for (const auto& segment : body_) {
        if (segment == pt) { return true; }
    }
    return false;
}

bool SnakeEnvV2::IsDeadly(SnakePoint pt) const noexcept
{
    // Out of bounds
    if (pt.x < 0 || pt.x >= static_cast<i32>(width_) ||
        pt.y < 0 || pt.y >= static_cast<i32>(height_)) {
        return true;
    }

    // Self-collision (skip tail if snake will vacate it this step)
    for (std::size_t i = 0; i < body_.size() - 1; ++i) {
        if (body_[i] == pt) {
            return true;
        }
    }
    return false;
}

void SnakeEnvV2::SpawnFood()
{
    const u32 max_attempts = 100;
    for (u32 it = 0; it < max_attempts; ++it) {
        const SnakePoint cand{
            static_cast<i32>(rng_.NextBelow(width_)),
            static_cast<i32>(rng_.NextBelow(height_))
        };
        if (!IsCellOccupiedByBody(cand)) {
            bool on_existing_food = false;
            for (const auto& f : foods_) {
                if (f == cand) { on_existing_food = true; break; }
            }
            if (!on_existing_food) {
                foods_.push_back(cand);
                return;
            }
        }
    }
    // Fallback: search grid sequentially for open cell
    for (u32 y = 0; y < height_; ++y) {
        for (u32 x = 0; x < width_; ++x) {
            const SnakePoint cand{static_cast<i32>(x), static_cast<i32>(y)};
            if (!IsCellOccupiedByBody(cand)) {
                foods_.push_back(cand);
                return;
            }
        }
    }
}

std::vector<f32> SnakeEnvV2::BuildObs() const
{
    if (obs_mode_ == SnakeObsMode::GridTensor) {
        std::vector<f32> grid(width_ * height_, 0.0f);
        for (const auto& f : foods_) {
            if (f.x >= 0 && f.x < static_cast<i32>(width_) && f.y >= 0 && f.y < static_cast<i32>(height_)) {
                grid[f.y * width_ + f.x] = -1.0f;
            }
        }
        for (std::size_t i = 1; i < body_.size(); ++i) {
            const auto& b = body_[i];
            if (b.x >= 0 && b.x < static_cast<i32>(width_) && b.y >= 0 && b.y < static_cast<i32>(height_)) {
                grid[b.y * width_ + b.x] = 0.5f;
            }
        }
        if (!body_.empty()) {
            const auto& h = body_.front();
            if (h.x >= 0 && h.x < static_cast<i32>(width_) && h.y >= 0 && h.y < static_cast<i32>(height_)) {
                grid[h.y * width_ + h.x] = 1.0f;
            }
        }
        return grid;
    }

    // FeatureVector: 12 features
    // 0: Danger Straight
    // 1: Danger Left
    // 2: Danger Right
    // 3..6: Direction One-Hot (Left, Right, Up, Down)
    // 7..10: Food Direction (Left, Right, Up, Down)
    // 11: Food Distance (normalized)
    std::vector<f32> feats(12, 0.0f);
    const SnakePoint head = body_.empty() ? SnakePoint{0, 0} : body_.front();

    SnakePoint pt_straight = head;
    SnakePoint pt_left     = head;
    SnakePoint pt_right    = head;

    switch (dir_) {
    case 0: // Left
        pt_straight = {head.x - 1, head.y};
        pt_left     = {head.x, head.y + 1}; // Down
        pt_right    = {head.x, head.y - 1}; // Up
        break;
    case 1: // Right
        pt_straight = {head.x + 1, head.y};
        pt_left     = {head.x, head.y - 1}; // Up
        pt_right    = {head.x, head.y + 1}; // Down
        break;
    case 2: // Up
        pt_straight = {head.x, head.y - 1};
        pt_left     = {head.x - 1, head.y}; // Left
        pt_right    = {head.x + 1, head.y}; // Right
        break;
    case 3: // Down
        pt_straight = {head.x, head.y + 1};
        pt_left     = {head.x + 1, head.y}; // Right
        pt_right    = {head.x - 1, head.y}; // Left
        break;
    default: break;
    }

    feats[0] = IsDeadly(pt_straight) ? 1.0f : 0.0f;
    feats[1] = IsDeadly(pt_left)     ? 1.0f : 0.0f;
    feats[2] = IsDeadly(pt_right)    ? 1.0f : 0.0f;

    feats[3 + dir_] = 1.0f;

    if (!foods_.empty()) {
        const SnakePoint f = foods_.front();
        feats[7]  = (f.x < head.x) ? 1.0f : 0.0f; // Food Left
        feats[8]  = (f.x > head.x) ? 1.0f : 0.0f; // Food Right
        feats[9]  = (f.y < head.y) ? 1.0f : 0.0f; // Food Up
        feats[10] = (f.y > head.y) ? 1.0f : 0.0f; // Food Down

        const f32 dist = static_cast<f32>(std::abs(f.x - head.x) + std::abs(f.y - head.y));
        const f32 max_dist = static_cast<f32>(width_ + height_);
        feats[11] = dist / max_dist;
    }

    return feats;
}

std::unique_ptr<Env> SnakeEnvV2::Clone() const
{
    auto c = std::make_unique<SnakeEnvV2>(width_, height_, num_foods_, obs_mode_, seed_);
    c->body_ = body_;
    c->foods_ = foods_;
    c->dir_ = dir_;
    c->score_ = score_;
    c->ep_return_ = ep_return_;
    c->ep_length_ = ep_length_;
    return c;
}

} // namespace rl
