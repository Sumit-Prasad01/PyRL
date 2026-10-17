#include "eval.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace rl {

EvalStats EvaluatePolicy(Env& env,
                         const std::function<std::vector<f32>(const std::vector<f32>& obs)>& policy_fn,
                         u32 num_episodes,
                         u64 base_seed,
                         u32 max_steps)
{
    EvalStats stats{};
    if (num_episodes == 0) { return stats; }

    stats.returns.reserve(num_episodes);
    stats.lengths.reserve(num_episodes);

    u32 successes = 0;

    for (u32 ep = 0; ep < num_episodes; ++ep) {
        const u64 ep_seed = base_seed + static_cast<u64>(ep) * 100;
        StepResult res = env.Reset(ep_seed);

        f32 ep_return = 0.0f;
        u32 ep_length = 0;

        while (!res.Done() && ep_length < max_steps) {
            const std::vector<f32> act = policy_fn(res.obs);
            res = env.Step(act);
            ep_return += res.reward;
            ep_length++;
        }

        stats.returns.push_back(ep_return);
        stats.lengths.push_back(ep_length);

        // Success condition: reached non-negative return or completed horizon
        if (ep_return > 0.0f || (res.truncated && !res.terminated)) {
            successes++;
        }
    }

    const f32 n = static_cast<f32>(num_episodes);
    const f32 sum_ret = std::accumulate(stats.returns.begin(), stats.returns.end(), 0.0f);
    const f32 sum_len = static_cast<f32>(std::accumulate(stats.lengths.begin(), stats.lengths.end(), 0ULL));

    stats.mean_return = sum_ret / n;
    stats.mean_length = sum_len / n;
    stats.min_return  = *std::min_element(stats.returns.begin(), stats.returns.end());
    stats.max_return  = *std::max_element(stats.returns.begin(), stats.returns.end());
    stats.success_rate = static_cast<f32>(successes) / n;

    f32 sq_diff_sum = 0.0f;
    for (f32 r : stats.returns) {
        const f32 diff = r - stats.mean_return;
        sq_diff_sum += diff * diff;
    }
    stats.std_return = std::sqrt(sq_diff_sum / n);

    return stats;
}

} // namespace rl
