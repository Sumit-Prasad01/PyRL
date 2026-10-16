#pragma once

#include <cmath>
#include <numbers>
#include <span>
#include <vector>

#include "autograd.hpp"
#include "base.hpp"

namespace optim {

// Base interface for all optimizers.
enum class OptimizerType : u32 {
    Unknown = 0,
    SGD     = 1,
    Adam    = 2,
    AdamW   = 3,
    RMSProp = 4,
};

class Optimizer {
public:
    explicit Optimizer(std::vector<Var*> params, f32 lr)
        : params_(std::move(params)), lr_(lr) {}

    virtual ~Optimizer() = default;

    virtual void Step() = 0;
    [[nodiscard]] virtual OptimizerType Type() const noexcept = 0;

    void ZeroGrad() noexcept
    {
        for (Var* p : params_) {
            if (p != nullptr && p->grad != nullptr) {
                p->grad->Clear();
            }
        }
    }

    [[nodiscard]] f32 GetLr() const noexcept { return lr_; }
    void SetLr(f32 lr) noexcept { lr_ = lr; }

    [[nodiscard]] std::span<Var* const> Params() const noexcept { return params_; }

protected:
    std::vector<Var*> params_;
    f32 lr_;
};

// Stochastic Gradient Descent with momentum, Nesterov, and weight decay.
struct SGDConfig {
    f32  lr           = 0.01f;
    f32  momentum     = 0.0f;
    f32  dampening    = 0.0f;
    f32  weight_decay = 0.0f;
    bool nesterov     = false;
};

class SGD : public Optimizer {
public:
    SGD(std::vector<Var*> params, SGDConfig cfg);
    void Step() override;
    [[nodiscard]] OptimizerType Type() const noexcept override { return OptimizerType::SGD; }

    [[nodiscard]] const SGDConfig& Config() const noexcept { return cfg_; }
    [[nodiscard]] const std::vector<std::vector<f32>>& Velocity() const noexcept { return velocity_; }
    std::vector<std::vector<f32>>& Velocity() noexcept { return velocity_; }

private:
    SGDConfig cfg_;
    std::vector<std::vector<f32>> velocity_;
};

// RMSProp optimizer with smoothing constant and optional centered variance.
struct RMSPropConfig {
    f32  lr           = 0.01f;
    f32  alpha        = 0.99f;
    f32  eps          = 1e-8f;
    f32  weight_decay = 0.0f;
    f32  momentum     = 0.0f;
    bool centered     = false;
};

class RMSProp : public Optimizer {
public:
    RMSProp(std::vector<Var*> params, RMSPropConfig cfg);
    void Step() override;
    [[nodiscard]] OptimizerType Type() const noexcept override { return OptimizerType::RMSProp; }

    [[nodiscard]] const RMSPropConfig& Config() const noexcept { return cfg_; }
    [[nodiscard]] const std::vector<std::vector<f32>>& SquareAvg() const noexcept { return square_avg_; }
    std::vector<std::vector<f32>>& SquareAvg() noexcept { return square_avg_; }
    [[nodiscard]] const std::vector<std::vector<f32>>& MomBuffer() const noexcept { return mom_buffer_; }
    std::vector<std::vector<f32>>& MomBuffer() noexcept { return mom_buffer_; }
    [[nodiscard]] const std::vector<std::vector<f32>>& GradAvg() const noexcept { return grad_avg_; }
    std::vector<std::vector<f32>>& GradAvg() noexcept { return grad_avg_; }

private:
    RMSPropConfig cfg_;
    std::vector<std::vector<f32>> square_avg_;
    std::vector<std::vector<f32>> mom_buffer_;
    std::vector<std::vector<f32>> grad_avg_;
};

// Adam optimizer (Kingma & Ba, 2014)
struct AdamConfig {
    f32 lr           = 1e-3f;
    f32 beta1        = 0.9f;
    f32 beta2        = 0.999f;
    f32 eps          = 1e-8f;
    f32 weight_decay = 0.0f;
};

class Adam : public Optimizer {
public:
    Adam(std::vector<Var*> params, AdamConfig cfg);
    void Step() override;
    [[nodiscard]] OptimizerType Type() const noexcept override { return OptimizerType::Adam; }
    [[nodiscard]] const AdamConfig& Config() const noexcept { return cfg_; }

    [[nodiscard]] u64 StepCount() const noexcept { return step_count_; }
    void SetStepCount(u64 count) noexcept { step_count_ = count; }

    [[nodiscard]] const std::vector<std::vector<f32>>& M() const noexcept { return m_; }
    [[nodiscard]] const std::vector<std::vector<f32>>& V() const noexcept { return v_; }
    std::vector<std::vector<f32>>& M() noexcept { return m_; }
    std::vector<std::vector<f32>>& V() noexcept { return v_; }

private:
    AdamConfig cfg_;
    u64 step_count_ = 0;
    std::vector<std::vector<f32>> m_;
    std::vector<std::vector<f32>> v_;
};

// AdamW optimizer with decoupled weight decay (Loshchilov & Hutter, 2017)
class AdamW : public Optimizer {
public:
    AdamW(std::vector<Var*> params, AdamConfig cfg);
    void Step() override;
    [[nodiscard]] OptimizerType Type() const noexcept override { return OptimizerType::AdamW; }
    [[nodiscard]] const AdamConfig& Config() const noexcept { return cfg_; }

    [[nodiscard]] u64 StepCount() const noexcept { return step_count_; }
    void SetStepCount(u64 count) noexcept { step_count_ = count; }

    [[nodiscard]] const std::vector<std::vector<f32>>& M() const noexcept { return m_; }
    [[nodiscard]] const std::vector<std::vector<f32>>& V() const noexcept { return v_; }
    std::vector<std::vector<f32>>& M() noexcept { return m_; }
    std::vector<std::vector<f32>>& V() noexcept { return v_; }

private:
    AdamConfig cfg_;
    u64 step_count_ = 0;
    std::vector<std::vector<f32>> m_;
    std::vector<std::vector<f32>> v_;
};

// Gradient clipping
f32  ClipGradNorm(std::span<Var* const> params, f32 max_norm);
void ClipGradValue(std::span<Var* const> params, f32 clip_value);

// Learning-rate schedules
class LinearAnnealLR {
public:
    LinearAnnealLR(Optimizer& opt, f32 init_lr, f32 end_lr, u64 total_steps)
        : opt_(opt), init_lr_(init_lr), end_lr_(end_lr), total_steps_(total_steps) {}

    void Step(u64 current_step)
    {
        if (total_steps_ == 0) {
            opt_.SetLr(end_lr_);
            return;
        }
        const f32 progress = std::min(1.0f, static_cast<f32>(current_step) / static_cast<f32>(total_steps_));
        const f32 lr = init_lr_ + progress * (end_lr_ - init_lr_);
        opt_.SetLr(lr);
    }

private:
    Optimizer& opt_;
    f32        init_lr_;
    f32        end_lr_;
    u64        total_steps_;
};

class CosineAnnealLR {
public:
    CosineAnnealLR(Optimizer& opt, f32 max_lr, f32 min_lr, u64 total_steps)
        : opt_(opt), max_lr_(max_lr), min_lr_(min_lr), total_steps_(total_steps) {}

    void Step(u64 current_step)
    {
        if (total_steps_ == 0) {
            opt_.SetLr(min_lr_);
            return;
        }
        constexpr f32 kPi = 3.14159265358979323846f;
        const f32 progress = std::min(1.0f, static_cast<f32>(current_step) / static_cast<f32>(total_steps_));
        const f32 cos_val  = std::cos(kPi * progress);
        const f32 lr       = min_lr_ + 0.5f * (max_lr_ - min_lr_) * (1.0f + cos_val);
        opt_.SetLr(lr);
    }

private:
    Optimizer& opt_;
    f32        max_lr_;
    f32        min_lr_;
    u64        total_steps_;
};

} // namespace optim
