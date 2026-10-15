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
    using ShapeFn = std::optional<VarShape> (*)(const Var* a, const Var* b);
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
}

struct Var {
    static constexpr u32 kMaxInputs = 2;

    u32            index;
    VarFlags       flags;
    Mat*           val;
    Mat*           grad;
    const VarType* type;
    Var*           inputs[kMaxInputs];

    [[nodiscard]] static Var* Create(Arena& arena, ModelState& model, u32 rows, u32 cols,
                                     VarFlags flags = VarFlags::None) noexcept;

    [[nodiscard]] bool RequiresGrad() const noexcept { return HasFlag(flags, VarFlags::RequiresGrad); }
    [[nodiscard]] bool IsParameter()  const noexcept { return HasFlag(flags, VarFlags::Parameter); }
    [[nodiscard]] u32  NumInputs()    const noexcept { return type != nullptr ? type->num_inputs : 0; }

    operator const Mat&() const noexcept { return *val; }
    operator Mat&() noexcept             { return *val; }
};

struct Graph {
    Var** vars;
    u32   size;

    [[nodiscard]] static Graph Build(Arena& arena, const ModelState& model, Var* out_var);
    [[nodiscard]] bool Empty() const noexcept { return size == 0; }
    [[nodiscard]] std::span<Var* const> Vars() const noexcept { return {vars, size}; }

    void Forward() const;
    [[nodiscard]] bool Backward() const;
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

}

static_assert(std::is_trivially_default_constructible_v<Var>);
static_assert(std::is_trivially_default_constructible_v<Graph>);
static_assert(std::is_trivially_default_constructible_v<ModelState>);
static_assert(std::is_trivially_destructible_v<Var>);
static_assert(std::is_trivially_destructible_v<ModelState>);
