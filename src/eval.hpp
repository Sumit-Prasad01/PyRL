#pragma once

#include <functional>
#include <vector>

#include "base.hpp"
#include "environment.hpp"

namespace rl {

struct EvalStats {
    f32 mean_return  = 0.0f;
    f32 std_return   = 0.0f;
    f32 min_return   = 0.0f;
    f32 max_return   = 0.0f;
    f32 mean_length  = 0.0f;
    f32 success_rate = 0.0f;

    std::vector<f32> returns;
    std::vector<u32> lengths;
};

// Evaluates a policy function deterministically on a test environment across N independent episodes
EvalStats EvaluatePolicy(Env& env,
                         const std::function<std::vector<f32>(const std::vector<f32>& obs)>& policy_fn,
                         u32 num_episodes = 10,
                         u64 base_seed = 10000,
                         u32 max_steps = 1000);

} // namespace rl
