#include "nn.hpp"

#include <cassert>
#include <cmath>

namespace nn {

namespace init {

void Orthogonal(Mat& mat, Prng& rng, f32 gain)
{
    const u32 rows = mat.rows;
    const u32 cols = mat.cols;
    if (rows == 0 || cols == 0) { return; }

    std::vector<f32> flat(u64{rows} * cols);
    for (f32& v : flat) {
        v = dist::SampleStandardNormal(rng);
    }

    if (rows >= cols) {
        // Gram-Schmidt on columns
        for (u32 j = 0; j < cols; ++j) {
            for (u32 k = 0; k < j; ++k) {
                f32 dot = 0.0f;
                for (u32 i = 0; i < rows; ++i) {
                    dot += flat[i * cols + j] * flat[i * cols + k];
                }
                for (u32 i = 0; i < rows; ++i) {
                    flat[i * cols + j] -= dot * flat[i * cols + k];
                }
            }

            f32 norm_sq = 0.0f;
            for (u32 i = 0; i < rows; ++i) {
                const f32 val = flat[i * cols + j];
                norm_sq += val * val;
            }
            const f32 norm = std::sqrt(norm_sq);
            const f32 inv_norm = (norm > 1e-8f) ? (1.0f / norm) : 0.0f;
            for (u32 i = 0; i < rows; ++i) {
                flat[i * cols + j] *= inv_norm;
            }
        }
    } else {
        // Gram-Schmidt on rows
        for (u32 i = 0; i < rows; ++i) {
            for (u32 k = 0; k < i; ++k) {
                f32 dot = 0.0f;
                for (u32 j = 0; j < cols; ++j) {
                    dot += flat[i * cols + j] * flat[k * cols + j];
                }
                for (u32 j = 0; j < cols; ++j) {
                    flat[i * cols + j] -= dot * flat[k * cols + j];
                }
            }

            f32 norm_sq = 0.0f;
            for (u32 j = 0; j < cols; ++j) {
                const f32 val = flat[i * cols + j];
                norm_sq += val * val;
            }
            const f32 norm = std::sqrt(norm_sq);
            const f32 inv_norm = (norm > 1e-8f) ? (1.0f / norm) : 0.0f;
            for (u32 j = 0; j < cols; ++j) {
                flat[i * cols + j] *= inv_norm;
            }
        }
    }

    // Copy to mat and apply gain
    for (u64 idx = 0, total = mat.Size(); idx < total; ++idx) {
        mat.data[idx] = flat[idx] * gain;
    }
}

} // namespace init

// ---------------------------------------------------------------- Linear

Linear::Linear(Arena& arena, ModelState& model, u32 in_features, u32 out_features,
               bool bias, bool col_vector_format, Prng* rng, f32 gain)
    : arena_(arena), model_(model), col_vector_format_(col_vector_format)
{
    const u32 w_rows = col_vector_format ? out_features : in_features;
    const u32 w_cols = col_vector_format ? in_features : out_features;

    weight_ = Var::Create(arena_, model_, w_rows, w_cols, VarFlags::Parameter | VarFlags::RequiresGrad);
    assert(weight_ != nullptr);

    if (rng != nullptr) {
        init::Orthogonal(*weight_->val, *rng, gain);
    } else {
        // Default Kaiming Uniform
        init::KaimingUniform(*weight_->val, prng::Global());
    }

    if (bias) {
        const u32 b_rows = col_vector_format ? out_features : 1;
        const u32 b_cols = col_vector_format ? 1 : out_features;
        bias_ = Var::Create(arena_, model_, b_rows, b_cols, VarFlags::Parameter | VarFlags::RequiresGrad);
        assert(bias_ != nullptr);
        init::Zeros(*bias_->val);
    }
}

Var* Linear::Forward(Var* input)
{
    assert(input != nullptr);
    Var* wx = nullptr;
    if (col_vector_format_) {
        // [out_features, in_features] * [in_features, 1]
        wx = ops::MatMul(arena_, model_, weight_, input);
    } else {
        // [B, in_features] * [in_features, out_features]
        wx = ops::MatMul(arena_, model_, input, weight_);
    }

    if (bias_ != nullptr) {
        return ops::Add(arena_, model_, wx, bias_);
    }
    return wx;
}

std::vector<Var*> Linear::Parameters() const
{
    if (bias_ != nullptr) {
        return {weight_, bias_};
    }
    return {weight_};
}

// ---------------------------------------------------------------- LayerNorm

LayerNorm::LayerNorm(Arena& arena, ModelState& model, u32 features, f32 eps)
    : arena_(arena), model_(model), eps_(eps)
{
    gamma_ = Var::Create(arena_, model_, 1, features, VarFlags::Parameter | VarFlags::RequiresGrad);
    beta_  = Var::Create(arena_, model_, 1, features, VarFlags::Parameter | VarFlags::RequiresGrad);
    assert(gamma_ != nullptr && beta_ != nullptr);

    init::Constant(*gamma_->val, 1.0f);
    init::Zeros(*beta_->val);
}

Var* LayerNorm::Forward(Var* input)
{
    return ops::LayerNorm(arena_, model_, input, gamma_, beta_, eps_);
}

std::vector<Var*> LayerNorm::Parameters() const
{
    return {gamma_, beta_};
}

} // namespace nn
