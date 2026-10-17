#pragma once

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "base.hpp"
#include "spaces.hpp"

namespace rl {

struct StepResult {
    std::vector<f32> obs;
    f32  reward     = 0.0f;
    bool terminated = false; // Task goal reached or agent failed (absorbing state)
    bool truncated  = false; // Time limit or horizon reached
    f32  ep_return  = 0.0f; // Running cumulative return for current episode
    u32  ep_length  = 0;    // Running step count for current episode

    [[nodiscard]] bool Done() const noexcept { return terminated || truncated; }
};

class Env {
public:
    virtual ~Env() = default;

    // Resets the environment to initial state, optionally reseeding the PRNG.
    virtual StepResult Reset(std::optional<u64> seed = std::nullopt) = 0;

    // Advances the environment by one step given a continuous action vector or discrete action.
    virtual StepResult Step(const std::vector<f32>& action) = 0;

    // Convenience method for discrete action environments
    StepResult Step(u32 discrete_action)
    {
        return Step(std::vector<f32>{static_cast<f32>(discrete_action)});
    }

    [[nodiscard]] virtual const Space& ObservationSpace() const = 0;
    [[nodiscard]] virtual const Space& ActionSpace() const = 0;

    [[nodiscard]] virtual u32 ObsDim() const noexcept = 0;
    [[nodiscard]] virtual u32 ActionDim() const noexcept = 0;
    [[nodiscard]] virtual bool IsDiscreteAction() const noexcept = 0;

    // Creates an identical independent deep copy of the environment (useful for VecEnv workers)
    [[nodiscard]] virtual std::unique_ptr<Env> Clone() const = 0;
};

} // namespace rl
