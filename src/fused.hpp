#pragma once

#include "autograd.hpp"
#include "base.hpp"
#include "device.hpp"
#include "nn.hpp"

namespace ops {

// Fused Linear + Bias + ReLU forward execution on CPU or CUDA
void FusedLinearBiasReluForward(f32* out, const f32* x, const f32* w, const f32* bias,
                               u32 B, u32 in_dim, u32 out_dim, Device dev = Device::CPU);

// Fused Linear + Bias + ReLU backward execution (computes gradients for x, w, and bias)
void FusedLinearBiasReluBackward(const f32* out_grad, const f32* out_val, const f32* x,
                                const f32* w, f32* x_grad, f32* w_grad, f32* bias_grad,
                                u32 B, u32 in_dim, u32 out_dim);

// Autograd node for fused Linear + Bias + ReLU
[[nodiscard]] Var* FusedLinearRelu(Arena& arena, ModelState& model, Var* x, Var* w, Var* bias = nullptr);

} // namespace ops

namespace nn {

// High-performance fused Linear + ReLU module avoiding memory roundtrips
class FusedLinearReLU : public Module {
public:
    FusedLinearReLU(Arena& arena, ModelState& model, u32 in_features, u32 out_features,
                    bool bias = true, Prng* rng = nullptr, f32 gain = 1.0f);

    Var* Forward(Var* input) override;
    std::vector<Var*> Parameters() const override;
    std::string Name() const override { return "FusedLinearReLU"; }

    [[nodiscard]] Var* Weight() const noexcept { return weight_; }
    [[nodiscard]] Var* Bias() const noexcept { return bias_; }

private:
    Arena&      arena_;
    ModelState& model_;
    Var*        weight_ = nullptr;
    Var*        bias_   = nullptr;
};

} // namespace nn
