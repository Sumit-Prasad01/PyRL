#pragma once

#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>
#include <optional>
#include <vector>

#include "environment.hpp"

namespace rl {

// Base wrapper class forwarding calls to wrapped environment
class EnvWrapper : public Env {
public:
    explicit EnvWrapper(std::unique_ptr<Env> env) : env_(std::move(env))
    {
        if (!env_) {
            throw std::invalid_argument("EnvWrapper: wrapped env cannot be null.");
        }
    }

    StepResult Reset(std::optional<u64> seed = std::nullopt) override
    {
        return env_->Reset(seed);
    }

    StepResult Step(const std::vector<f32>& action) override
    {
        return env_->Step(action);
    }

    [[nodiscard]] const Space& ObservationSpace() const override
    {
        return env_->ObservationSpace();
    }

    [[nodiscard]] const Space& ActionSpace() const override
    {
        return env_->ActionSpace();
    }

    [[nodiscard]] u32 ObsDim() const noexcept override
    {
        return env_->ObsDim();
    }

    [[nodiscard]] u32 ActionDim() const noexcept override
    {
        return env_->ActionDim();
    }

    [[nodiscard]] bool IsDiscreteAction() const noexcept override
    {
        return env_->IsDiscreteAction();
    }

    [[nodiscard]] Env& Inner() noexcept { return *env_; }
    [[nodiscard]] const Env& Inner() const noexcept { return *env_; }

protected:
    std::unique_ptr<Env> env_;
};

// ---------------------------------------------------------------- TimeLimit
class TimeLimit final : public EnvWrapper {
public:
    TimeLimit(std::unique_ptr<Env> env, u32 max_episode_steps)
        : EnvWrapper(std::move(env)), max_steps_(max_episode_steps)
    {
    }

    StepResult Reset(std::optional<u64> seed = std::nullopt) override
    {
        current_step_ = 0;
        return env_->Reset(seed);
    }

    StepResult Step(const std::vector<f32>& action) override
    {
        StepResult res = env_->Step(action);
        current_step_++;
        if (current_step_ >= max_steps_ && !res.terminated) {
            res.truncated = true;
        }
        return res;
    }

    [[nodiscard]] std::unique_ptr<Env> Clone() const override
    {
        return std::make_unique<TimeLimit>(env_->Clone(), max_steps_);
    }

private:
    u32 max_steps_     = 1000;
    u32 current_step_  = 0;
};

// ----------------------------------------------------------- RunningMeanStd
class RunningMeanStd {
public:
    explicit RunningMeanStd(u32 dim, f32 epsilon = 1e-4f)
        : mean(dim, 0.0f), var(dim, 1.0f), count(epsilon)
    {
    }

    void Update(const std::vector<f32>& x)
    {
        const f32 batch_count = 1.0f;
        const f32 tot_count = count + batch_count;

        for (u32 i = 0; i < mean.size(); ++i) {
            const f32 delta = x[i] - mean[i];
            const f32 new_mean = mean[i] + delta * (batch_count / tot_count);
            const f32 m_a = var[i] * count;
            const f32 m_b = 0.0f; // single sample variance is 0
            const f32 m2 = m_a + m_b + delta * delta * (count * batch_count / tot_count);
            mean[i] = new_mean;
            var[i] = std::max(1e-4f, m2 / tot_count);
        }
        count = tot_count;
    }

    std::vector<f32> mean;
    std::vector<f32> var;
    f32 count = 1e-4f;
};

// ---------------------------------------------------- NormalizeObservation
class NormalizeObservation final : public EnvWrapper {
public:
    NormalizeObservation(std::unique_ptr<Env> env, f32 epsilon = 1e-8f, f32 clip = 10.0f)
        : EnvWrapper(std::move(env)),
          rms_(env_->ObsDim(), 1e-4f),
          epsilon_(epsilon),
          clip_(clip)
    {
    }

    StepResult Reset(std::optional<u64> seed = std::nullopt) override
    {
        StepResult res = env_->Reset(seed);
        FilterObs(res.obs);
        return res;
    }

    StepResult Step(const std::vector<f32>& action) override
    {
        StepResult res = env_->Step(action);
        FilterObs(res.obs);
        return res;
    }

    [[nodiscard]] std::unique_ptr<Env> Clone() const override
    {
        auto clone = std::make_unique<NormalizeObservation>(env_->Clone(), epsilon_, clip_);
        clone->rms_ = rms_;
        return clone;
    }

    [[nodiscard]] const RunningMeanStd& RMS() const noexcept { return rms_; }

private:
    void FilterObs(std::vector<f32>& obs)
    {
        rms_.Update(obs);
        for (u32 i = 0; i < obs.size(); ++i) {
            const f32 std_dev = std::sqrt(rms_.var[i] + epsilon_);
            obs[i] = std::clamp((obs[i] - rms_.mean[i]) / std_dev, -clip_, clip_);
        }
    }

    RunningMeanStd rms_;
    f32            epsilon_ = 1e-8f;
    f32            clip_    = 10.0f;
};

// --------------------------------------------------------- NormalizeReward
class NormalizeReward final : public EnvWrapper {
public:
    NormalizeReward(std::unique_ptr<Env> env, f32 gamma = 0.99f, f32 epsilon = 1e-8f, f32 clip = 10.0f)
        : EnvWrapper(std::move(env)),
          gamma_(gamma),
          epsilon_(epsilon),
          clip_(clip),
          r_rms_(1, 1e-4f)
    {
    }

    StepResult Reset(std::optional<u64> seed = std::nullopt) override
    {
        r_return_ = 0.0f;
        return env_->Reset(seed);
    }

    StepResult Step(const std::vector<f32>& action) override
    {
        StepResult res = env_->Step(action);
        r_return_ = r_return_ * gamma_ + res.reward;
        std::vector<f32> r_vec = {r_return_};
        r_rms_.Update(r_vec);

        const f32 std_dev = std::sqrt(r_rms_.var[0] + epsilon_);
        res.reward = std::clamp(res.reward / std_dev, -clip_, clip_);

        if (res.Done()) {
            r_return_ = 0.0f;
        }
        return res;
    }

    [[nodiscard]] std::unique_ptr<Env> Clone() const override
    {
        auto clone = std::make_unique<NormalizeReward>(env_->Clone(), gamma_, epsilon_, clip_);
        clone->r_rms_ = r_rms_;
        return clone;
    }

private:
    f32            gamma_    = 0.99f;
    f32            epsilon_  = 1e-8f;
    f32            clip_     = 10.0f;
    f32            r_return_ = 0.0f;
    RunningMeanStd r_rms_;
};

// -------------------------------------------------------------- FrameStack
class FrameStack final : public EnvWrapper {
public:
    FrameStack(std::unique_ptr<Env> env, u32 num_frames)
        : EnvWrapper(std::move(env)),
          k_(num_frames),
          stacked_space_(env_->ObsDim() * num_frames, -1e6f, 1e6f)
    {
        if (k_ == 0) {
            throw std::invalid_argument("FrameStack: num_frames must be >= 1.");
        }
    }

    StepResult Reset(std::optional<u64> seed = std::nullopt) override
    {
        frames_.clear();
        StepResult res = env_->Reset(seed);
        for (u32 i = 0; i < k_; ++i) {
            frames_.push_back(res.obs);
        }
        res.obs = GetStackedObs();
        return res;
    }

    StepResult Step(const std::vector<f32>& action) override
    {
        StepResult res = env_->Step(action);
        frames_.pop_front();
        frames_.push_back(res.obs);
        res.obs = GetStackedObs();
        return res;
    }

    [[nodiscard]] const Space& ObservationSpace() const override
    {
        return stacked_space_;
    }

    [[nodiscard]] u32 ObsDim() const noexcept override
    {
        return env_->ObsDim() * k_;
    }

    [[nodiscard]] std::unique_ptr<Env> Clone() const override
    {
        return std::make_unique<FrameStack>(env_->Clone(), k_);
    }

private:
    [[nodiscard]] std::vector<f32> GetStackedObs() const
    {
        std::vector<f32> stacked;
        stacked.reserve(ObsDim());
        for (const auto& frame : frames_) {
            stacked.insert(stacked.end(), frame.begin(), frame.end());
        }
        return stacked;
    }

    u32                          k_;
    BoxSpace                     stacked_space_;
    std::deque<std::vector<f32>> frames_;
};

// ------------------------------------------------------------ ActionRepeat
class ActionRepeat final : public EnvWrapper {
public:
    ActionRepeat(std::unique_ptr<Env> env, u32 repeat)
        : EnvWrapper(std::move(env)), repeat_(repeat)
    {
        if (repeat_ == 0) {
            throw std::invalid_argument("ActionRepeat: repeat must be >= 1.");
        }
    }

    StepResult Step(const std::vector<f32>& action) override
    {
        f32 total_reward = 0.0f;
        StepResult res{};
        for (u32 r = 0; r < repeat_; ++r) {
            res = env_->Step(action);
            total_reward += res.reward;
            if (res.Done()) {
                break;
            }
        }
        res.reward = total_reward;
        return res;
    }

    [[nodiscard]] std::unique_ptr<Env> Clone() const override
    {
        return std::make_unique<ActionRepeat>(env_->Clone(), repeat_);
    }

private:
    u32 repeat_ = 4;
};

} // namespace rl
