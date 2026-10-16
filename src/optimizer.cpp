#include "optimizer.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace optim {

// ---------------------------------------------------------------- SGD

SGD::SGD(std::vector<Var*> params, SGDConfig cfg)
    : Optimizer(std::move(params), cfg.lr), cfg_(cfg)
{
    velocity_.resize(params_.size());
    for (size_t i = 0; i < params_.size(); ++i) {
        if (params_[i] != nullptr && params_[i]->val != nullptr) {
            velocity_[i].resize(params_[i]->val->Size(), 0.0f);
        }
    }
}

void SGD::Step()
{
    for (size_t p_idx = 0; p_idx < params_.size(); ++p_idx) {
        Var* var = params_[p_idx];
        if (var == nullptr || var->val == nullptr || var->grad == nullptr) {
            continue;
        }

        const u64 size = var->val->Size();
        f32* val = var->val->data;
        f32* grad = var->grad->data;
        f32* v = velocity_[p_idx].data();

        for (u64 i = 0; i < size; ++i) {
            f32 d_p = grad[i];

            if (cfg_.weight_decay != 0.0f) {
                d_p += cfg_.weight_decay * val[i];
            }

            if (cfg_.momentum != 0.0f) {
                v[i] = cfg_.momentum * v[i] + (1.0f - cfg_.dampening) * d_p;
                if (cfg_.nesterov) {
                    d_p += cfg_.momentum * v[i];
                } else {
                    d_p = v[i];
                }
            }

            val[i] -= lr_ * d_p;
        }
    }
}

// ---------------------------------------------------------------- RMSProp

RMSProp::RMSProp(std::vector<Var*> params, RMSPropConfig cfg)
    : Optimizer(std::move(params), cfg.lr), cfg_(cfg)
{
    square_avg_.resize(params_.size());
    if (cfg_.momentum > 0.0f) {
        mom_buffer_.resize(params_.size());
    }
    if (cfg_.centered) {
        grad_avg_.resize(params_.size());
    }

    for (size_t i = 0; i < params_.size(); ++i) {
        if (params_[i] != nullptr && params_[i]->val != nullptr) {
            const u64 size = params_[i]->val->Size();
            square_avg_[i].resize(size, 0.0f);
            if (cfg_.momentum > 0.0f) {
                mom_buffer_[i].resize(size, 0.0f);
            }
            if (cfg_.centered) {
                grad_avg_[i].resize(size, 0.0f);
            }
        }
    }
}

void RMSProp::Step()
{
    for (size_t p_idx = 0; p_idx < params_.size(); ++p_idx) {
        Var* var = params_[p_idx];
        if (var == nullptr || var->val == nullptr || var->grad == nullptr) {
            continue;
        }

        const u64 size = var->val->Size();
        f32* val = var->val->data;
        f32* grad = var->grad->data;
        f32* sq_avg = square_avg_[p_idx].data();

        for (u64 i = 0; i < size; ++i) {
            f32 d_p = grad[i];
            if (cfg_.weight_decay != 0.0f) {
                d_p += cfg_.weight_decay * val[i];
            }

            sq_avg[i] = cfg_.alpha * sq_avg[i] + (1.0f - cfg_.alpha) * d_p * d_p;

            f32 avg = sq_avg[i];
            if (cfg_.centered) {
                f32* g_avg = grad_avg_[p_idx].data();
                g_avg[i] = cfg_.alpha * g_avg[i] + (1.0f - cfg_.alpha) * d_p;
                avg = avg - g_avg[i] * g_avg[i];
            }

            f32 denom = std::sqrt(std::max(0.0f, avg)) + cfg_.eps;
            f32 step = d_p / denom;

            if (cfg_.momentum > 0.0f) {
                f32* buf = mom_buffer_[p_idx].data();
                buf[i] = cfg_.momentum * buf[i] + step;
                val[i] -= lr_ * buf[i];
            } else {
                val[i] -= lr_ * step;
            }
        }
    }
}

// ---------------------------------------------------------------- Adam

Adam::Adam(std::vector<Var*> params, AdamConfig cfg)
    : Optimizer(std::move(params), cfg.lr), cfg_(cfg)
{
    m_.resize(params_.size());
    v_.resize(params_.size());
    for (size_t i = 0; i < params_.size(); ++i) {
        if (params_[i] != nullptr && params_[i]->val != nullptr) {
            const u64 size = params_[i]->val->Size();
            m_[i].resize(size, 0.0f);
            v_[i].resize(size, 0.0f);
        }
    }
}

void Adam::Step()
{
    step_count_++;
    const f32 bias_correction1 = 1.0f - std::pow(cfg_.beta1, static_cast<f32>(step_count_));
    const f32 bias_correction2 = 1.0f - std::pow(cfg_.beta2, static_cast<f32>(step_count_));

    for (size_t p_idx = 0; p_idx < params_.size(); ++p_idx) {
        Var* var = params_[p_idx];
        if (var == nullptr || var->val == nullptr || var->grad == nullptr) {
            continue;
        }

        const u64 size = var->val->Size();
        f32* val = var->val->data;
        f32* grad = var->grad->data;
        f32* m = m_[p_idx].data();
        f32* v = v_[p_idx].data();

        for (u64 i = 0; i < size; ++i) {
            f32 g = grad[i];
            if (cfg_.weight_decay != 0.0f) {
                g += cfg_.weight_decay * val[i];
            }

            m[i] = cfg_.beta1 * m[i] + (1.0f - cfg_.beta1) * g;
            v[i] = cfg_.beta2 * v[i] + (1.0f - cfg_.beta2) * g * g;

            const f32 m_hat = m[i] / bias_correction1;
            const f32 v_hat = v[i] / bias_correction2;

            val[i] -= lr_ * m_hat / (std::sqrt(v_hat) + cfg_.eps);
        }
    }
}

// ---------------------------------------------------------------- AdamW

AdamW::AdamW(std::vector<Var*> params, AdamConfig cfg)
    : Optimizer(std::move(params), cfg.lr), cfg_(cfg)
{
    m_.resize(params_.size());
    v_.resize(params_.size());
    for (size_t i = 0; i < params_.size(); ++i) {
        if (params_[i] != nullptr && params_[i]->val != nullptr) {
            const u64 size = params_[i]->val->Size();
            m_[i].resize(size, 0.0f);
            v_[i].resize(size, 0.0f);
        }
    }
}

void AdamW::Step()
{
    step_count_++;
    const f32 bias_correction1 = 1.0f - std::pow(cfg_.beta1, static_cast<f32>(step_count_));
    const f32 bias_correction2 = 1.0f - std::pow(cfg_.beta2, static_cast<f32>(step_count_));

    for (size_t p_idx = 0; p_idx < params_.size(); ++p_idx) {
        Var* var = params_[p_idx];
        if (var == nullptr || var->val == nullptr || var->grad == nullptr) {
            continue;
        }

        const u64 size = var->val->Size();
        f32* val = var->val->data;
        f32* grad = var->grad->data;
        f32* m = m_[p_idx].data();
        f32* v = v_[p_idx].data();

        for (u64 i = 0; i < size; ++i) {
            // Decoupled weight decay: theta = theta - lr * weight_decay * theta
            if (cfg_.weight_decay != 0.0f) {
                val[i] -= lr_ * cfg_.weight_decay * val[i];
            }

            const f32 g = grad[i];
            m[i] = cfg_.beta1 * m[i] + (1.0f - cfg_.beta1) * g;
            v[i] = cfg_.beta2 * v[i] + (1.0f - cfg_.beta2) * g * g;

            const f32 m_hat = m[i] / bias_correction1;
            const f32 v_hat = v[i] / bias_correction2;

            val[i] -= lr_ * m_hat / (std::sqrt(v_hat) + cfg_.eps);
        }
    }
}

// ---------------------------------------------------------------- Clipping

f32 ClipGradNorm(std::span<Var* const> params, f32 max_norm)
{
    f64 total_norm_sq = 0.0;
    for (Var* p : params) {
        if (p != nullptr && p->grad != nullptr && p->grad->data != nullptr) {
            for (f32 g : p->grad->Span()) {
                total_norm_sq += static_cast<f64>(g) * g;
            }
        }
    }

    const f32 total_norm = static_cast<f32>(std::sqrt(total_norm_sq));
    if (total_norm > max_norm && total_norm > 0.0f) {
        const f32 scale = max_norm / (total_norm + 1e-6f);
        for (Var* p : params) {
            if (p != nullptr && p->grad != nullptr && p->grad->data != nullptr) {
                p->grad->Scale(scale);
            }
        }
    }
    return total_norm;
}

void ClipGradValue(std::span<Var* const> params, f32 clip_value)
{
    assert(clip_value > 0.0f);
    for (Var* p : params) {
        if (p != nullptr && p->grad != nullptr && p->grad->data != nullptr) {
            for (f32& g : p->grad->Span()) {
                g = std::clamp(g, -clip_value, clip_value);
            }
        }
    }
}

} // namespace optim
