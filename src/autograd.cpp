#include "autograd.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

namespace {

inline void MustSucceed([[maybe_unused]] bool ok) noexcept
{
    assert(ok && "autograd kernel shape mismatch: graph invariant broken");
}

std::optional<VarShape> ShapeSame(const Var* a, const Var* b)
{
    if (a == nullptr) { return std::nullopt; }
    if (b != nullptr && !a->val->SameShape(*b->val)) { return std::nullopt; }
    return VarShape{a->val->rows, a->val->cols};
}

std::optional<VarShape> ShapeMatMul(const Var* a, const Var* b)
{
    if (a == nullptr || b == nullptr) { return std::nullopt; }
    if (a->val->cols != b->val->rows) { return std::nullopt; }
    return VarShape{a->val->rows, b->val->cols};
}

void ReluForward(Var& v) 
{
    MustSucceed(mat::Relu(*v.val, *v.inputs[0]->val));
}

void ReluBackward(Var& v) 
{
    Var* in = v.inputs[0];
    if (in->RequiresGrad()) {
        MustSucceed(mat::ReluAddGrad(*in->grad, *in->val, *v.grad));
    }
}

void SoftmaxForward(Var& v) 
{
    MustSucceed(mat::Softmax(*v.val, *v.inputs[0]->val));
}

void SoftmaxBackward(Var& v) 
{
    Var* in = v.inputs[0];
    if (in->RequiresGrad()) {
        MustSucceed(mat::SoftmaxAddGrad(*in->grad, *v.val, *v.grad));
    }
}

void LogSoftmaxForward(Var& v) 
{
    MustSucceed(mat::LogSoftmax(*v.val, *v.inputs[0]->val));
}

void LogSoftmaxBackward(Var& v) 
{
    Var* in = v.inputs[0];
    if (in->RequiresGrad()) {
        MustSucceed(mat::LogSoftmaxAddGrad(*in->grad, *v.val, *v.grad));
    }
}

void AddForward(Var& v) 
{
    MustSucceed(mat::Add(*v.val, *v.inputs[0]->val, *v.inputs[1]->val));
}

void AddBackward(Var& v)
{
    Var* a = v.inputs[0];
    Var* b = v.inputs[1];
    if (a->RequiresGrad()) { MustSucceed(mat::Add(*a->grad, *a->grad, *v.grad)); }
    if (b->RequiresGrad()) { MustSucceed(mat::Add(*b->grad, *b->grad, *v.grad)); }
}

void SubForward(Var& v)
{
    MustSucceed(mat::Sub(*v.val, *v.inputs[0]->val, *v.inputs[1]->val));
}

void SubBackward(Var& v) 
{
    Var* a = v.inputs[0];
    Var* b = v.inputs[1];
    if (a->RequiresGrad()) { MustSucceed(mat::Add(*a->grad, *a->grad, *v.grad)); }
    if (b->RequiresGrad()) { MustSucceed(mat::Sub(*b->grad, *b->grad, *v.grad)); }
}

void MatMulForward(Var& v) 
{
    MustSucceed(mat::MatMul(*v.val, *v.inputs[0]->val, *v.inputs[1]->val,
                            /*zero_out=*/true, /*transpose_a=*/false, /*transpose_b=*/false));
}

void MatMulBackward(Var& v) 
{
    Var* a = v.inputs[0];
    Var* b = v.inputs[1];

    if (a->RequiresGrad()) {
        MustSucceed(mat::MatMul(*a->grad, *v.grad, *b->val,
                                /*zero_out=*/false, /*transpose_a=*/false, /*transpose_b=*/true));
    }

    if (b->RequiresGrad()) {
        MustSucceed(mat::MatMul(*b->grad, *a->val, *v.grad,
                                /*zero_out=*/false, /*transpose_a=*/true, /*transpose_b=*/false));
    }
}

void ReinforceForward(Var& v)
{
    MustSucceed(mat::ReinforceLoss(*v.val, *v.inputs[0]->val, *v.inputs[1]->val));
}

void ReinforceBackward(Var& v)
{
    Var* probs = v.inputs[0];
    Var* rt    = v.inputs[1];
    if (probs->RequiresGrad()) {
        MustSucceed(mat::ReinforceAddGrad(*probs->grad, *probs->val, *rt->val, *v.grad));
    }
}

std::optional<VarShape> ShapeScalar(const Var* a, const Var* b)
{
    if (a == nullptr || b == nullptr) { return std::nullopt; }
    return VarShape{1, 1};
}

void LogProbLossForward(Var& v)
{
    MustSucceed(mat::LogProbLoss(*v.val, *v.inputs[0]->val, *v.inputs[1]->val));
}

void LogProbLossBackward(Var& v)
{
    Var* log_probs = v.inputs[0];
    Var* target    = v.inputs[1];
    if (log_probs->RequiresGrad()) {
        MustSucceed(mat::LogProbLossAddGrad(*log_probs->grad, *target->val, *v.grad));
    }
}

void CrossEntropyForward(Var& v)
{
    MustSucceed(mat::CrossEntropy(*v.val, *v.inputs[0]->val, *v.inputs[1]->val));
}

void CrossEntropyBackward(Var& v)
{
    Var* logits = v.inputs[0];
    Var* target = v.inputs[1];
    if (logits->RequiresGrad()) {
        MustSucceed(mat::CrossEntropyAddGrad(*logits->grad, *logits->val, *target->val, *v.grad));
    }
}

// ---- node construction

Var* CreateNode(Arena& arena, ModelState& model, const VarType& type, Var* a, Var* b)
{
    assert(type.shape != nullptr);
    assert(type.num_inputs >= 1 && type.num_inputs <= Var::kMaxInputs);

    Var* const inputs[Var::kMaxInputs] = {a, b};

    VarFlags flags = VarFlags::None;
    for (u32 i = 0; i < type.num_inputs; i++) {
        if (inputs[i] == nullptr) { return nullptr; }
        if (inputs[i]->RequiresGrad()) { flags |= VarFlags::RequiresGrad; }
    }

    const std::optional<VarShape> shape = type.shape(a, b);
    if (!shape) { return nullptr; }

    Var* out = Var::Create(arena, model, shape->rows, shape->cols, flags);
    if (out == nullptr) { return nullptr; }

    out->type = &type;
    for (u32 i = 0; i < type.num_inputs; i++) {
        out->inputs[i] = inputs[i];
    }
    return out;
}

} // namespace

// ---------------------------------------------------------------- var types

namespace var_type {

const VarType Relu = {
    .name = "relu", .num_inputs = 1, .shape = ShapeSame,
    .forward = ReluForward, .backward = ReluBackward,
};

const VarType Softmax = {
    .name = "softmax", .num_inputs = 1, .shape = ShapeSame,
    .forward = SoftmaxForward, .backward = SoftmaxBackward,
};

const VarType LogSoftmax = {
    .name = "log_softmax", .num_inputs = 1, .shape = ShapeSame,
    .forward = LogSoftmaxForward, .backward = LogSoftmaxBackward,
};

const VarType Add = {
    .name = "add", .num_inputs = 2, .shape = ShapeSame,
    .forward = AddForward, .backward = AddBackward,
};

const VarType Sub = {
    .name = "sub", .num_inputs = 2, .shape = ShapeSame,
    .forward = SubForward, .backward = SubBackward,
};

const VarType MatMul = {
    .name = "matmul", .num_inputs = 2, .shape = ShapeMatMul,
    .forward = MatMulForward, .backward = MatMulBackward,
};

const VarType ReinforceLoss = {
    .name = "reinforce_loss", .num_inputs = 2, .shape = ShapeSame,
    .forward = ReinforceForward, .backward = ReinforceBackward,
};

const VarType LogProbLoss = {
    .name = "log_prob_loss", .num_inputs = 2, .shape = ShapeScalar,
    .forward = LogProbLossForward, .backward = LogProbLossBackward,
};

const VarType CrossEntropy = {
    .name = "cross_entropy", .num_inputs = 2, .shape = ShapeScalar,
    .forward = CrossEntropyForward, .backward = CrossEntropyBackward,
};

} // namespace var_type

// ---------------------------------------------------------------- Var

Var* Var::Create(Arena& arena, ModelState& model, u32 rows, u32 cols, VarFlags flags) noexcept
{
    Var* out = arena.PushStruct<Var>();
    if (out == nullptr) { return nullptr; }

    out->val = Mat::Create(arena, rows, cols);
    if (out->val == nullptr) { return nullptr; }

    if (HasFlag(flags, VarFlags::RequiresGrad)) {
        out->grad = Mat::Create(arena, rows, cols);
        if (out->grad == nullptr) { return nullptr; }
    }

    out->index = model.num_vars++;
    out->flags = flags;
    out->type = nullptr;
    return out;
}

// ---------------------------------------------------------------- ops

namespace ops {

Var* Relu(Arena& arena, ModelState& model, Var* input)
{
    return CreateNode(arena, model, var_type::Relu, input, nullptr);
}

Var* Softmax(Arena& arena, ModelState& model, Var* input)
{
    return CreateNode(arena, model, var_type::Softmax, input, nullptr);
}

Var* LogSoftmax(Arena& arena, ModelState& model, Var* input)
{
    return CreateNode(arena, model, var_type::LogSoftmax, input, nullptr);
}

Var* Add(Arena& arena, ModelState& model, Var* a, Var* b)
{
    return CreateNode(arena, model, var_type::Add, a, b);
}

Var* Sub(Arena& arena, ModelState& model, Var* a, Var* b)
{
    return CreateNode(arena, model, var_type::Sub, a, b);
}

Var* MatMul(Arena& arena, ModelState& model, Var* a, Var* b)
{
    return CreateNode(arena, model, var_type::MatMul, a, b);
}

Var* ReinforceLoss(Arena& arena, ModelState& model, Var* probs, Var* rt)
{
    return CreateNode(arena, model, var_type::ReinforceLoss, probs, rt);
}

Var* LogProbLoss(Arena& arena, ModelState& model, Var* log_probs, Var* target)
{
    return CreateNode(arena, model, var_type::LogProbLoss, log_probs, target);
}

Var* CrossEntropy(Arena& arena, ModelState& model, Var* logits, Var* target)
{
    return CreateNode(arena, model, var_type::CrossEntropy, logits, target);
}

} // namespace ops

// ---------------------------------------------------------------- Graph

Graph Graph::Build(Arena& arena, const ModelState& model, Var* out_var)
{
    if (out_var == nullptr || model.num_vars == 0) {
        return Graph{};
    }
    if (out_var->index >= model.num_vars) {
        return Graph{};
    }

    enum class NodeState : u8 { Unvisited = 0, Visiting = 1, Visited = 2 };
    std::vector<NodeState> state(model.num_vars, NodeState::Unvisited);
    std::vector<Var*> order;
    order.reserve(model.num_vars);

    struct DfsFrame {
        Var* var;
        u32  input_idx;
    };

    std::vector<DfsFrame> stack;
    stack.reserve(model.num_vars);

    state[out_var->index] = NodeState::Visiting;
    stack.push_back({out_var, 0});

    while (!stack.empty()) {
        DfsFrame& top = stack.back();
        Var* cur = top.var;
        const u32 num_in = cur->NumInputs();

        if (top.input_idx < num_in) {
            Var* in = cur->inputs[top.input_idx++];
            if (in == nullptr) { continue; }
            assert(in->index < model.num_vars && "Var belongs to a different ModelState");
            if (in->index >= model.num_vars) { continue; }

            if (state[in->index] == NodeState::Unvisited) {
                state[in->index] = NodeState::Visiting;
                stack.push_back({in, 0});
            } else if (state[in->index] == NodeState::Visiting) {
                assert(false && "Cycle detected in computation graph!");
            }
        } else {
            state[cur->index] = NodeState::Visited;
            order.push_back(cur);
            stack.pop_back();
        }
    }

    Var** vars = arena.PushArrayUninit<Var*>(order.size());
    if (vars == nullptr) {
        return Graph{};
    }
    std::ranges::copy(order, vars);

    return Graph{vars, static_cast<u32>(order.size())};
}

void Graph::Forward() const 
{
    for (Var* cur : Vars()) {
        if (cur->type != nullptr && cur->type->forward != nullptr) {
            cur->type->forward(*cur);
        }
    }
}

bool Graph::Backward() const
{
    if (Empty()) { return false; }

    Var* output = vars[size - 1];
    if (!output->RequiresGrad()) { return false; }

    for (Var* cur : Vars()) {
        if (cur->RequiresGrad() && !cur->IsParameter()) {
            cur->grad->Clear();
        }
    }

    output->grad->Fill(1.0f);

    for (u32 i = size; i-- > 0; ) {
        Var* cur = vars[i];
        if (!cur->RequiresGrad()) { continue; }
        if (cur->type != nullptr && cur->type->backward != nullptr) {
            cur->type->backward(*cur);
        }
    }
    return true;
}

// ---------------------------------------------------------------- Model

ModelState* ModelState::Create(Arena& arena) noexcept
{
    return arena.PushStruct<ModelState>();
}
