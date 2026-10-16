#include "fused.hpp"
#include "cuda/cuda_backend.hpp"
#include "matmul_ladder.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__AVX2__) || defined(_M_AMD64) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace ops {

void FusedLinearBiasReluForward(f32* out, const f32* x, const f32* w, const f32* bias,
                               u32 B, u32 in_dim, u32 out_dim, Device dev)
{
    if (B == 0 || in_dim == 0 || out_dim == 0) { return; }

#ifdef RL_USE_CUDA
    if (dev == Device::CUDA && HasCudaDevice()) {
        cuda::FusedLinearBiasRelu(out, x, w, bias, B, in_dim, out_dim);
        cuda::Synchronize();
        return;
    }
#endif
    (void)dev;

    // CPU AVX2 implementation
    std::memset(out, 0, u64{B} * out_dim * sizeof(f32));

    // 1. Matrix multiplication: out = x * w
    matmul::OptimizedCPU(out, x, w, B, in_dim, out_dim);

    // 2. Add bias and clamp with ReLU directly in-place
    for (u32 i = 0; i < B; ++i) {
        f32* const row = out + i * out_dim;
        u32 j = 0;

#if defined(__AVX2__) || defined(_M_AMD64) || defined(_M_X64)
        const __m256 v_zero = _mm256_setzero_ps();
        for (; j + 7 < out_dim; j += 8) {
            __m256 val = _mm256_loadu_ps(row + j);
            if (bias != nullptr) {
                const __m256 b_val = _mm256_loadu_ps(bias + j);
                val = _mm256_add_ps(val, b_val);
            }
            val = _mm256_max_ps(val, v_zero);
            _mm256_storeu_ps(row + j, val);
        }
#endif
        for (; j < out_dim; ++j) {
            f32 val = row[j];
            if (bias != nullptr) {
                val += bias[j];
            }
            row[j] = std::max(0.0f, val);
        }
    }
}

void FusedLinearBiasReluBackward(const f32* out_grad, const f32* out_val, const f32* x,
                                const f32* w, f32* x_grad, f32* w_grad, f32* bias_grad,
                                u32 B, u32 in_dim, u32 out_dim)
{
    if (B == 0 || in_dim == 0 || out_dim == 0) { return; }

    std::vector<f32> dz(u64{B} * out_dim);
    for (u64 i = 0; i < dz.size(); ++i) {
        dz[i] = (out_val[i] > 0.0f) ? out_grad[i] : 0.0f;
    }

    if (bias_grad != nullptr) {
        for (u32 j = 0; j < out_dim; ++j) {
            f32 sum = 0.0f;
            for (u32 i = 0; i < B; ++i) {
                sum += dz[i * out_dim + j];
            }
            bias_grad[j] += sum;
        }
    }

    if (w_grad != nullptr) {
        // w_grad [in_dim, out_dim] += x^T [in_dim, B] * dz [B, out_dim]
        for (u32 i = 0; i < in_dim; ++i) {
            for (u32 k = 0; k < B; ++k) {
                const f32 x_ki = x[k * in_dim + i];
                const f32* const dz_row = dz.data() + k * out_dim;
                f32* const w_row = w_grad + i * out_dim;
                for (u32 j = 0; j < out_dim; ++j) {
                    w_row[j] += x_ki * dz_row[j];
                }
            }
        }
    }

    if (x_grad != nullptr) {
        // x_grad [B, in_dim] += dz [B, out_dim] * w^T [out_dim, in_dim]
        for (u32 i = 0; i < B; ++i) {
            const f32* const dz_row = dz.data() + i * out_dim;
            f32* const x_row = x_grad + i * in_dim;
            for (u32 k = 0; k < out_dim; ++k) {
                const f32 dz_val = dz_row[k];
                for (u32 j = 0; j < in_dim; ++j) {
                    x_row[j] += dz_val * w[j * out_dim + k];
                }
            }
        }
    }
}

namespace {

std::optional<VarShape> ShapeFused(const Var* a, const Var* b, const Var* c)
{
    if (a == nullptr || b == nullptr) { return std::nullopt; }
    if (a->val->cols != b->val->rows) { return std::nullopt; }
    if (c != nullptr && c->val->cols != b->val->cols) { return std::nullopt; }
    return VarShape{a->val->rows, b->val->cols};
}

void FusedForward(Var& v)
{
    const Var* x = v.inputs[0];
    const Var* w = v.inputs[1];
    const Var* b = v.inputs[2];
    const f32* b_data = (b != nullptr && b->val != nullptr) ? b->val->data : nullptr;

    FusedLinearBiasReluForward(v.val->data, x->val->data, w->val->data, b_data,
                              x->val->rows, x->val->cols, w->val->cols);
}

void FusedBackward(Var& v)
{
    Var* x = v.inputs[0];
    Var* w = v.inputs[1];
    Var* b = v.inputs[2];

    f32* x_grad = (x->RequiresGrad() && x->grad != nullptr) ? x->grad->data : nullptr;
    f32* w_grad = (w->RequiresGrad() && w->grad != nullptr) ? w->grad->data : nullptr;
    f32* b_grad = (b != nullptr && b->RequiresGrad() && b->grad != nullptr) ? b->grad->data : nullptr;

    FusedLinearBiasReluBackward(v.grad->data, v.val->data, x->val->data, w->val->data,
                               x_grad, w_grad, b_grad,
                               x->val->rows, x->val->cols, w->val->cols);
}

const VarType FusedType = {
    .name = "fused_linear_relu",
    .num_inputs = 3,
    .shape = ShapeFused,
    .forward = FusedForward,
    .backward = FusedBackward,
};

} // namespace

Var* FusedLinearRelu(Arena& arena, ModelState& model, Var* x, Var* w, Var* bias)
{
    if (x == nullptr || w == nullptr) { return nullptr; }
    auto shape = ShapeFused(x, w, bias);
    if (!shape.has_value()) { return nullptr; }

    VarFlags flags = VarFlags::None;
    if (x->RequiresGrad() || w->RequiresGrad() || (bias != nullptr && bias->RequiresGrad())) {
        flags = VarFlags::RequiresGrad;
    }

    Var* out = Var::Create(arena, model, shape->rows, shape->cols, flags);
    if (out == nullptr) { return nullptr; }

    out->type = &FusedType;
    out->inputs[0] = x;
    out->inputs[1] = w;
    out->inputs[2] = bias;
    return out;
}

} // namespace ops

namespace nn {

FusedLinearReLU::FusedLinearReLU(Arena& arena, ModelState& model, u32 in_features,
                                 u32 out_features, bool bias, Prng* rng, f32 gain)
    : arena_(arena), model_(model)
{
    weight_ = Var::Create(arena_, model_, in_features, out_features,
                          VarFlags::Parameter | VarFlags::RequiresGrad);
    assert(weight_ != nullptr);

    if (rng != nullptr) {
        init::Orthogonal(*weight_->val, *rng, gain);
    } else {
        init::KaimingUniform(*weight_->val, prng::Global());
    }

    if (bias) {
        bias_ = Var::Create(arena_, model_, 1, out_features,
                            VarFlags::Parameter | VarFlags::RequiresGrad);
        assert(bias_ != nullptr);
        init::Zeros(*bias_->val);
    }
}

Var* FusedLinearReLU::Forward(Var* input)
{
    assert(input != nullptr);
    return ops::FusedLinearRelu(arena_, model_, input, weight_, bias_);
}

std::vector<Var*> FusedLinearReLU::Parameters() const
{
    std::vector<Var*> p;
    if (weight_ != nullptr) { p.push_back(weight_); }
    if (bias_ != nullptr)   { p.push_back(bias_); }
    return p;
}

} // namespace nn
