#pragma once

#include <optional>
#include <span>
#include <type_traits>

#include "arena.hpp"
#include "base.hpp"
#include "mat.hpp"

enum class VarFlags : u32 {
    None         = 0,
    RequiresGrad = 1u << 0,
    Parameter    = 1u << 1,
};

constexpr VarFlags operator|(VarFlags a, VarFlags b) noexcept
{
    return static_cast<VarFlags>(static_cast<u32>(a) | static_cast<u32>(b));
}

constexpr VarFlags& operator|=(VarFlags& a, VarFlags b) noexcept { return a = a | b; }

constexpr bool HasFlag(VarFlags value, VarFlags flag) noexcept
{
    return (static_cast<u32>(value) & static_cast<u32>(flag)) == static_cast<u32>(flag);
}

struct Var;
struct ModelState;

struct VarShape {
    u32 rows;
    u32 cols;
};

struct VarType {
    using ShapeFn = std::optional<VarShape> (*)(const Var* a, const Var* b, const Var* c);
    using PassFn  = void (*)(Var& var);

    const char* name;
    u32         num_inputs;
    ShapeFn     shape;
    PassFn      forward;
    PassFn      backward;
};

namespace var_type {
extern const VarType Relu;
extern const VarType Softmax;
extern const VarType LogSoftmax;
extern const VarType Add;
extern const VarType Sub;
extern const VarType MatMul;
extern const VarType ReinforceLoss;
extern const VarType LogProbLoss;
extern const VarType CrossEntropy;
extern const VarType Tanh;
extern const VarType Sigmoid;
extern const VarType LeakyRelu;
extern const VarType Gelu;
extern const VarType Exp;
extern const VarType Log;
extern const VarType Mul;
extern const VarType Div;
extern const VarType Clip;
extern const VarType Sum;
extern const VarType Mean;
extern const VarType LayerNorm;
}

struct Var {
    static constexpr u32 kMaxInputs = 3;

    u32            index;
    VarFlags       flags;
    Mat*           val;
    Mat*           grad;
    const VarType* type;
    Var*           inputs[kMaxInputs];
    f32            aux_param;
    f32            aux_param2;
    u32            aux_int;

    [[nodiscard]] static Var* Create(Arena& arena, ModelState& model, u32 rows, u32 cols,
                                     VarFlags flags = VarFlags::None) noexcept;

    [[nodiscard]] bool RequiresGrad() const noexcept { return HasFlag(flags, VarFlags::RequiresGrad); }
    [[nodiscard]] bool IsParameter()  const noexcept { return HasFlag(flags, VarFlags::Parameter); }
    [[nodiscard]] u32  NumInputs()    const noexcept { return type != nullptr ? type->num_inputs : 0; }

    operator const Mat&() const noexcept { return *val; }
    operator Mat&() noexcept             { return *val; }
};

class GradMode {
public:
    [[nodiscard]] static bool IsEnabled() noexcept { return enabled_; }
    static void SetEnabled(bool enabled) noexcept { enabled_ = enabled; }

private:
    static inline thread_local bool enabled_ = true;
};

class NoGradGuard {
public:
    NoGradGuard() noexcept : prev_(GradMode::IsEnabled())
    {
        GradMode::SetEnabled(false);
    }
    ~NoGradGuard() noexcept
    {
        GradMode::SetEnabled(prev_);
    }
    NoGradGuard(const NoGradGuard&)            = delete;
    NoGradGuard& operator=(const NoGradGuard&) = delete;

private:
    bool prev_;
};

struct Graph {
    Var** vars;
    u32   size;

    [[nodiscard]] static Graph Build(Arena& arena, const ModelState& model, Var* out_var);
    [[nodiscard]] bool Empty() const noexcept { return size == 0; }
    [[nodiscard]] std::span<Var* const> Vars() const noexcept { return {vars, size}; }

    void Forward() const;
    [[nodiscard]] bool Backward() const;
    void ZeroGrad() const noexcept;
};

struct ModelState {
    u32 num_vars;

    Var* input;
    Var* output;
    Var* advantage;
    Var* cost;

    Graph forward_graph;
    Graph cost_graph;

    [[nodiscard]] static ModelState* Create(Arena& arena) noexcept;
};

namespace ops {

[[nodiscard]] Var* Relu(Arena& arena, ModelState& model, Var* input);
[[nodiscard]] Var* Softmax(Arena& arena, ModelState& model, Var* input);
[[nodiscard]] Var* LogSoftmax(Arena& arena, ModelState& model, Var* input);
[[nodiscard]] Var* Add(Arena& arena, ModelState& model, Var* a, Var* b);
[[nodiscard]] Var* Sub(Arena& arena, ModelState& model, Var* a, Var* b);
[[nodiscard]] Var* MatMul(Arena& arena, ModelState& model, Var* a, Var* b);
[[nodiscard]] Var* ReinforceLoss(Arena& arena, ModelState& model, Var* probs, Var* rt);
[[nodiscard]] Var* LogProbLoss(Arena& arena, ModelState& model, Var* log_probs, Var* target);
[[nodiscard]] Var* CrossEntropy(Arena& arena, ModelState& model, Var* logits, Var* target);
[[nodiscard]] Var* Tanh(Arena& arena, ModelState& model, Var* input);
[[nodiscard]] Var* Sigmoid(Arena& arena, ModelState& model, Var* input);
[[nodiscard]] Var* LeakyRelu(Arena& arena, ModelState& model, Var* input, f32 alpha = 0.01f);
[[nodiscard]] Var* Gelu(Arena& arena, ModelState& model, Var* input);
[[nodiscard]] Var* Exp(Arena& arena, ModelState& model, Var* input);
[[nodiscard]] Var* Log(Arena& arena, ModelState& model, Var* input, f32 eps = 1e-8f);
[[nodiscard]] Var* Mul(Arena& arena, ModelState& model, Var* a, Var* b);
[[nodiscard]] Var* Div(Arena& arena, ModelState& model, Var* a, Var* b);
[[nodiscard]] Var* Clip(Arena& arena, ModelState& model, Var* input, f32 min_val, f32 max_val);
[[nodiscard]] Var* Sum(Arena& arena, ModelState& model, Var* input, i32 axis = -1);
[[nodiscard]] Var* Mean(Arena& arena, ModelState& model, Var* input, i32 axis = -1);
[[nodiscard]] Var* LayerNorm(Arena& arena, ModelState& model, Var* input, Var* gamma, Var* beta, f32 eps = 1e-5f);

}

static_assert(std::is_trivially_default_constructible_v<Var>);
static_assert(std::is_trivially_default_constructible_v<Graph>);
static_assert(std::is_trivially_default_constructible_v<ModelState>);
static_assert(std::is_trivially_destructible_v<Var>);
static_assert(std::is_trivially_destructible_v<ModelState>);
