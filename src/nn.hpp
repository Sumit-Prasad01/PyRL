#pragma once

#include <cmath>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "arena.hpp"
#include "autograd.hpp"
#include "dist.hpp"
#include "mat.hpp"
#include "prng.hpp"

namespace nn {

// ---------------------------------------------------------------- Initializers

namespace init {

inline void Constant(Mat& mat, f32 val) noexcept
{
    mat.Fill(val);
}

inline void Zeros(Mat& mat) noexcept
{
    mat.Clear();
}

inline void XavierUniform(Mat& mat, Prng& rng, f32 gain = 1.0f)
{
    const f32 fan_in  = static_cast<f32>(mat.cols);
    const f32 fan_out = static_cast<f32>(mat.rows);
    const f32 limit   = gain * std::sqrt(6.0f / (fan_in + fan_out));
    mat.FillRand(-limit, limit, rng);
}

inline void XavierNormal(Mat& mat, Prng& rng, f32 gain = 1.0f)
{
    const f32 fan_in  = static_cast<f32>(mat.cols);
    const f32 fan_out = static_cast<f32>(mat.rows);
    const f32 std_dev = gain * std::sqrt(2.0f / (fan_in + fan_out));
    for (f32& val : mat.Span()) {
        val = dist::SampleStandardNormal(rng) * std_dev;
    }
}

inline void KaimingUniform(Mat& mat, Prng& rng, f32 a = 0.0f)
{
    const f32 gain   = std::sqrt(2.0f / (1.0f + a * a));
    const f32 fan_in = static_cast<f32>(mat.cols);
    const f32 limit  = gain * std::sqrt(3.0f / fan_in);
    mat.FillRand(-limit, limit, rng);
}

inline void KaimingNormal(Mat& mat, Prng& rng, f32 a = 0.0f)
{
    const f32 gain    = std::sqrt(2.0f / (1.0f + a * a));
    const f32 fan_in  = static_cast<f32>(mat.cols);
    const f32 std_dev = gain / std::sqrt(fan_in);
    for (f32& val : mat.Span()) {
        val = dist::SampleStandardNormal(rng) * std_dev;
    }
}

// Orthogonal initialization (Saxe et al., standard for RL/PPO policy heads).
void Orthogonal(Mat& mat, Prng& rng, f32 gain = 1.0f);

} // namespace init

// ---------------------------------------------------------------- Modules

class Module {
public:
    virtual ~Module() = default;

    virtual Var* Forward(Var* input) = 0;

    virtual std::vector<Var*> Parameters() const { return {}; }

    virtual void Train(bool mode = true) { is_training_ = mode; }
    void Eval() { Train(false); }
    [[nodiscard]] bool IsTraining() const noexcept { return is_training_; }

    virtual std::string Name() const { return "Module"; }

protected:
    bool is_training_ = true;
};

// Fully connected Linear layer: out = weight * x + bias (or x * weight + bias)
class Linear : public Module {
public:
    Linear(Arena& arena, ModelState& model, u32 in_features, u32 out_features,
           bool bias = true, bool col_vector_format = true, Prng* rng = nullptr, f32 gain = 1.0f);

    Var* Forward(Var* input) override;

    std::vector<Var*> Parameters() const override;

    std::string Name() const override { return "Linear"; }

    [[nodiscard]] Var* Weight() const noexcept { return weight_; }
    [[nodiscard]] Var* Bias() const noexcept { return bias_; }

private:
    Arena&      arena_;
    ModelState& model_;
    Var*        weight_ = nullptr;
    Var*        bias_   = nullptr;
    bool        col_vector_format_ = true;
};

// Activation modules
class ReLU : public Module {
public:
    ReLU(Arena& arena, ModelState& model) : arena_(arena), model_(model) {}
    Var* Forward(Var* input) override { return ops::Relu(arena_, model_, input); }
    std::string Name() const override { return "ReLU"; }
private:
    Arena& arena_;
    ModelState& model_;
};

class Tanh : public Module {
public:
    Tanh(Arena& arena, ModelState& model) : arena_(arena), model_(model) {}
    Var* Forward(Var* input) override { return ops::Tanh(arena_, model_, input); }
    std::string Name() const override { return "Tanh"; }
private:
    Arena& arena_;
    ModelState& model_;
};

class Sigmoid : public Module {
public:
    Sigmoid(Arena& arena, ModelState& model) : arena_(arena), model_(model) {}
    Var* Forward(Var* input) override { return ops::Sigmoid(arena_, model_, input); }
    std::string Name() const override { return "Sigmoid"; }
private:
    Arena& arena_;
    ModelState& model_;
};

class GELU : public Module {
public:
    GELU(Arena& arena, ModelState& model) : arena_(arena), model_(model) {}
    Var* Forward(Var* input) override { return ops::Gelu(arena_, model_, input); }
    std::string Name() const override { return "GELU"; }
private:
    Arena& arena_;
    ModelState& model_;
};

class Softmax : public Module {
public:
    Softmax(Arena& arena, ModelState& model) : arena_(arena), model_(model) {}
    Var* Forward(Var* input) override { return ops::Softmax(arena_, model_, input); }
    std::string Name() const override { return "Softmax"; }
private:
    Arena& arena_;
    ModelState& model_;
};

class LogSoftmax : public Module {
public:
    LogSoftmax(Arena& arena, ModelState& model) : arena_(arena), model_(model) {}
    Var* Forward(Var* input) override { return ops::LogSoftmax(arena_, model_, input); }
    std::string Name() const override { return "LogSoftmax"; }
private:
    Arena& arena_;
    ModelState& model_;
};

// LayerNorm module
class LayerNorm : public Module {
public:
    LayerNorm(Arena& arena, ModelState& model, u32 features, f32 eps = 1e-5f);
    Var* Forward(Var* input) override;
    std::vector<Var*> Parameters() const override;
    std::string Name() const override { return "LayerNorm"; }

private:
    Arena&      arena_;
    ModelState& model_;
    Var*        gamma_ = nullptr;
    Var*        beta_  = nullptr;
    f32         eps_;
};

// Sequential container chaining modules in order.
class Sequential : public Module {
public:
    Sequential() = default;

    template <typename... Modules>
    explicit Sequential(std::shared_ptr<Modules>... mods)
    {
        (modules_.push_back(std::move(mods)), ...);
    }

    void Add(std::shared_ptr<Module> mod)
    {
        modules_.push_back(std::move(mod));
    }

    Var* Forward(Var* input) override
    {
        Var* cur = input;
        for (auto& mod : modules_) {
            cur = mod->Forward(cur);
        }
        return cur;
    }

    std::vector<Var*> Parameters() const override
    {
        std::vector<Var*> params;
        for (const auto& mod : modules_) {
            auto sub_params = mod->Parameters();
            params.insert(params.end(), sub_params.begin(), sub_params.end());
        }
        return params;
    }

    void Train(bool mode = true) override
    {
        Module::Train(mode);
        for (auto& mod : modules_) {
            mod->Train(mode);
        }
    }

    std::string Name() const override { return "Sequential"; }

    [[nodiscard]] size_t Size() const noexcept { return modules_.size(); }
    [[nodiscard]] std::shared_ptr<Module> operator[](size_t i) const { return modules_[i]; }

private:
    std::vector<std::shared_ptr<Module>> modules_;
};

} // namespace nn
