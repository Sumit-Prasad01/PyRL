#include "autograd.hpp"

#include <algorithm>
#include <cassert>
#include <vector>

namespace {

inline void MustSucceed([[maybe_unused]] bool ok) noexcept
{
    assert(ok && "autograd kernel shape mismatch: graph invariant broken");
}

std::optional<VarShape> ShapeSame(const Var* a, const Var* b, const Var* /*c*/)
{
    if (a == nullptr) { return std::nullopt; }
    if (b != nullptr && !a->val->SameShape(*b->val)) { return std::nullopt; }
    return VarShape{a->val->rows, a->val->cols};
}

std::optional<VarShape> ShapeBroadcast(const Var* a, const Var* b, const Var* /*c*/)
{
    if (a == nullptr || b == nullptr) { return std::nullopt; }
    const u32 ra = a->val->rows;
    const u32 ca = a->val->cols;
    const u32 rb = b->val->rows;
    const u32 cb = b->val->cols;
    if ((ra != rb && ra != 1 && rb != 1) || (ca != cb && ca != 1 && cb != 1)) {
        return std::nullopt;
    }
    return VarShape{std::max(ra, rb), std::max(ca, cb)};
}

std::optional<VarShape> ShapeMatMul(const Var* a, const Var* b, const Var* /*c*/)
{
    if (a == nullptr || b == nullptr) { return std::nullopt; }
    if (a->val->cols != b->val->rows) { return std::nullopt; }
    return VarShape{a->val->rows, b->val->cols};
}

std::optional<VarShape> ShapeScalar(const Var* a, const Var* b, const Var* /*c*/)
{
    if (a == nullptr || b == nullptr) { return std::nullopt; }
    return VarShape{1, 1};
}

std::optional<VarShape> ShapeLayerNorm(const Var* a, const Var* b, const Var* c)
{
    if (a == nullptr || b == nullptr || c == nullptr) { return std::nullopt; }
    if (b->val->Size() != a->val->cols || c->val->Size() != a->val->cols) {
        return std::nullopt;
    }
    return VarShape{a->val->rows, a->val->cols};
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
    Mat dummy_a(a->val->rows, a->val->cols, nullptr);
    Mat dummy_b(b->val->rows, b->val->cols, nullptr);
    MustSucceed(mat::AddAddGrad(a->RequiresGrad() ? *a->grad : dummy_a,
                                b->RequiresGrad() ? *b->grad : dummy_b,
                                *a->val, *b->val, *v.grad));
}

void SubForward(Var& v)
{
    MustSucceed(mat::Sub(*v.val, *v.inputs[0]->val, *v.inputs[1]->val));
}

void SubBackward(Var& v) 
{
    Var* a = v.inputs[0];
    Var* b = v.inputs[1];
    Mat dummy_a(a->val->rows, a->val->cols, nullptr);
    Mat dummy_b(b->val->rows, b->val->cols, nullptr);
    MustSucceed(mat::SubAddGrad(a->RequiresGrad() ? *a->grad : dummy_a,
                                b->RequiresGrad() ? *b->grad : dummy_b,
                                *a->val, *b->val, *v.grad));
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

void TanhForward(Var& v)
{
    MustSucceed(mat::Tanh(*v.val, *v.inputs[0]->val));
}
void TanhBackward(Var& v)
{
    Var* in = v.inputs[0];
    if (in->RequiresGrad()) {
        MustSucceed(mat::TanhAddGrad(*in->grad, *v.val, *v.grad));
    }
}

void SigmoidForward(Var& v)
{
    MustSucceed(mat::Sigmoid(*v.val, *v.inputs[0]->val));
}
void SigmoidBackward(Var& v)
{
    Var* in = v.inputs[0];
    if (in->RequiresGrad()) {
        MustSucceed(mat::SigmoidAddGrad(*in->grad, *v.val, *v.grad));
    }
}

void LeakyReluForward(Var& v)
{
    MustSucceed(mat::LeakyRelu(*v.val, *v.inputs[0]->val, v.aux_param));
}
void LeakyReluBackward(Var& v)
{
    Var* in = v.inputs[0];
    if (in->RequiresGrad()) {
        MustSucceed(mat::LeakyReluAddGrad(*in->grad, *in->val, *v.grad, v.aux_param));
    }
}

void GeluForward(Var& v)
{
    MustSucceed(mat::Gelu(*v.val, *v.inputs[0]->val));
}
void GeluBackward(Var& v)
{
    Var* in = v.inputs[0];
    if (in->RequiresGrad()) {
        MustSucceed(mat::GeluAddGrad(*in->grad, *in->val, *v.grad));
    }
}

void ExpForward(Var& v)
{
    MustSucceed(mat::Exp(*v.val, *v.inputs[0]->val));
}
void ExpBackward(Var& v)
{
    Var* in = v.inputs[0];
    if (in->RequiresGrad()) {
        MustSucceed(mat::ExpAddGrad(*in->grad, *v.val, *v.grad));
    }
}

void LogForward(Var& v)
{
    MustSucceed(mat::Log(*v.val, *v.inputs[0]->val, v.aux_param));
}
void LogBackward(Var& v)
{
    Var* in = v.inputs[0];
    if (in->RequiresGrad()) {
        MustSucceed(mat::LogAddGrad(*in->grad, *in->val, *v.grad, v.aux_param));
    }
}

void MulForward(Var& v)
{
    MustSucceed(mat::Mul(*v.val, *v.inputs[0]->val, *v.inputs[1]->val));
}
void MulBackward(Var& v)
{
    Var* a = v.inputs[0];
    Var* b = v.inputs[1];
    Mat dummy_a(a->val->rows, a->val->cols, nullptr);
    Mat dummy_b(b->val->rows, b->val->cols, nullptr);
    MustSucceed(mat::MulAddGrad(a->RequiresGrad() ? *a->grad : dummy_a,
                                b->RequiresGrad() ? *b->grad : dummy_b,
                                *a->val, *b->val, *v.grad));
}

void DivForward(Var& v)
{
    MustSucceed(mat::Div(*v.val, *v.inputs[0]->val, *v.inputs[1]->val));
}
void DivBackward(Var& v)
{
    Var* a = v.inputs[0];
    Var* b = v.inputs[1];
    Mat dummy_a(a->val->rows, a->val->cols, nullptr);
    Mat dummy_b(b->val->rows, b->val->cols, nullptr);
    MustSucceed(mat::DivAddGrad(a->RequiresGrad() ? *a->grad : dummy_a,
                                b->RequiresGrad() ? *b->grad : dummy_b,
                                *a->val, *b->val, *v.grad));
}

void ClipForward(Var& v)
{
    MustSucceed(mat::Clip(*v.val, *v.inputs[0]->val, v.aux_param, v.aux_param2));
}
void ClipBackward(Var& v)
{
    Var* in = v.inputs[0];
    if (in->RequiresGrad()) {
        MustSucceed(mat::ClipAddGrad(*in->grad, *in->val, *v.grad, v.aux_param, v.aux_param2));
    }
}

void SumForward(Var& v)
{
    MustSucceed(mat::Sum(*v.val, *v.inputs[0]->val, static_cast<i32>(v.aux_int)));
}
void SumBackward(Var& v)
{
    Var* in = v.inputs[0];
    if (in->RequiresGrad()) {
        MustSucceed(mat::SumAddGrad(*in->grad, *v.grad, static_cast<i32>(v.aux_int)));
    }
}

void MeanForward(Var& v)
{
    MustSucceed(mat::Mean(*v.val, *v.inputs[0]->val, static_cast<i32>(v.aux_int)));
}
void MeanBackward(Var& v)
{
    Var* in = v.inputs[0];
    if (in->RequiresGrad()) {
        MustSucceed(mat::MeanAddGrad(*in->grad, *v.grad, static_cast<i32>(v.aux_int)));
    }
}

void LayerNormForward(Var& v)
{
    MustSucceed(mat::LayerNorm(*v.val, *v.inputs[0]->val, *v.inputs[1]->val, *v.inputs[2]->val, v.aux_param));
}
void LayerNormBackward(Var& v)
{
    Var* in = v.inputs[0];
    Var* gamma = v.inputs[1];
    Var* beta = v.inputs[2];
    Mat dummy_in(in->val->rows, in->val->cols, nullptr);
    Mat dummy_g(gamma->val->rows, gamma->val->cols, nullptr);
    Mat dummy_b(beta->val->rows, beta->val->cols, nullptr);
    MustSucceed(mat::LayerNormAddGrad(in->RequiresGrad() ? *in->grad : dummy_in,
                                      gamma->RequiresGrad() ? *gamma->grad : dummy_g,
                                      beta->RequiresGrad() ? *beta->grad : dummy_b,
                                      *in->val, *gamma->val, *v.grad, v.aux_param));
}

// ---- node construction

Var* CreateNode(Arena& arena, ModelState& model, const VarType& type,
                Var* a, Var* b = nullptr, Var* c = nullptr)
{
    assert(type.shape != nullptr);
    assert(type.num_inputs >= 1 && type.num_inputs <= Var::kMaxInputs);

    Var* const inputs[Var::kMaxInputs] = {a, b, c};

    VarFlags flags = VarFlags::None;
    for (u32 i = 0; i < type.num_inputs; i++) {
        if (inputs[i] == nullptr) { return nullptr; }
        if (inputs[i]->RequiresGrad()) { flags |= VarFlags::RequiresGrad; }
    }

    const std::optional<VarShape> shape = type.shape(a, b, c);
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
    .name = "add", .num_inputs = 2, .shape = ShapeBroadcast,
    .forward = AddForward, .backward = AddBackward,
};

const VarType Sub = {
    .name = "sub", .num_inputs = 2, .shape = ShapeBroadcast,
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

const VarType Tanh = {
    .name = "tanh", .num_inputs = 1, .shape = ShapeSame,
    .forward = TanhForward, .backward = TanhBackward,
};

const VarType Sigmoid = {
    .name = "sigmoid", .num_inputs = 1, .shape = ShapeSame,
    .forward = SigmoidForward, .backward = SigmoidBackward,
};

const VarType LeakyRelu = {
    .name = "leaky_relu", .num_inputs = 1, .shape = ShapeSame,
    .forward = LeakyReluForward, .backward = LeakyReluBackward,
};

const VarType Gelu = {
    .name = "gelu", .num_inputs = 1, .shape = ShapeSame,
    .forward = GeluForward, .backward = GeluBackward,
};

const VarType Exp = {
    .name = "exp", .num_inputs = 1, .shape = ShapeSame,
    .forward = ExpForward, .backward = ExpBackward,
};

const VarType Log = {
    .name = "log", .num_inputs = 1, .shape = ShapeSame,
    .forward = LogForward, .backward = LogBackward,
};

const VarType Mul = {
    .name = "mul", .num_inputs = 2, .shape = ShapeBroadcast,
    .forward = MulForward, .backward = MulBackward,
};

const VarType Div = {
    .name = "div", .num_inputs = 2, .shape = ShapeBroadcast,
    .forward = DivForward, .backward = DivBackward,
};

const VarType Clip = {
    .name = "clip", .num_inputs = 1, .shape = ShapeSame,
    .forward = ClipForward, .backward = ClipBackward,
};

const VarType Sum = {
    .name = "sum", .num_inputs = 1, .shape = ShapeSame,
    .forward = SumForward, .backward = SumBackward,
};

const VarType Mean = {
    .name = "mean", .num_inputs = 1, .shape = ShapeSame,
    .forward = MeanForward, .backward = MeanBackward,
};

const VarType LayerNorm = {
    .name = "layer_norm", .num_inputs = 3, .shape = ShapeLayerNorm,
    .forward = LayerNormForward, .backward = LayerNormBackward,
};

} // namespace var_type

// ---------------------------------------------------------------- Var

Var* Var::Create(Arena& arena, ModelState& model, u32 rows, u32 cols, VarFlags flags) noexcept
{
    Var* out = arena.PushStruct<Var>();
    if (out == nullptr) { return nullptr; }

    out->val = Mat::Create(arena, rows, cols);
    if (out->val == nullptr) { return nullptr; }

    if (HasFlag(flags, VarFlags::RequiresGrad) && GradMode::IsEnabled()) {
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
    return CreateNode(arena, model, var_type::Relu, input);
}

Var* Softmax(Arena& arena, ModelState& model, Var* input)
{
    return CreateNode(arena, model, var_type::Softmax, input);
}

Var* LogSoftmax(Arena& arena, ModelState& model, Var* input)
{
    return CreateNode(arena, model, var_type::LogSoftmax, input);
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

Var* Tanh(Arena& arena, ModelState& model, Var* input)
{
    return CreateNode(arena, model, var_type::Tanh, input);
}

Var* Sigmoid(Arena& arena, ModelState& model, Var* input)
{
    return CreateNode(arena, model, var_type::Sigmoid, input);
}

Var* LeakyRelu(Arena& arena, ModelState& model, Var* input, f32 alpha)
{
    Var* v = CreateNode(arena, model, var_type::LeakyRelu, input);
    if (v != nullptr) { v->aux_param = alpha; }
    return v;
}

Var* Gelu(Arena& arena, ModelState& model, Var* input)
{
    return CreateNode(arena, model, var_type::Gelu, input);
}

Var* Exp(Arena& arena, ModelState& model, Var* input)
{
    return CreateNode(arena, model, var_type::Exp, input);
}

Var* Log(Arena& arena, ModelState& model, Var* input, f32 eps)
{
    Var* v = CreateNode(arena, model, var_type::Log, input);
    if (v != nullptr) { v->aux_param = eps; }
    return v;
}

Var* Mul(Arena& arena, ModelState& model, Var* a, Var* b)
{
    return CreateNode(arena, model, var_type::Mul, a, b);
}

Var* Div(Arena& arena, ModelState& model, Var* a, Var* b)
{
    return CreateNode(arena, model, var_type::Div, a, b);
}

Var* Clip(Arena& arena, ModelState& model, Var* input, f32 min_val, f32 max_val)
{
    Var* v = CreateNode(arena, model, var_type::Clip, input);
    if (v != nullptr) {
        v->aux_param  = min_val;
        v->aux_param2 = max_val;
    }
    return v;
}

Var* Sum(Arena& arena, ModelState& model, Var* input, i32 axis)
{
    if (input == nullptr) { return nullptr; }
    VarFlags flags = input->RequiresGrad() ? VarFlags::RequiresGrad : VarFlags::None;
    u32 out_rows = 1;
    u32 out_cols = 1;
    if (axis == 0) {
        out_rows = 1;
        out_cols = input->val->cols;
    } else if (axis == 1) {
        out_rows = input->val->rows;
        out_cols = 1;
    } else {
        out_rows = 1;
        out_cols = 1;
    }
    Var* out = Var::Create(arena, model, out_rows, out_cols, flags);
    if (out == nullptr) { return nullptr; }
    out->type = &var_type::Sum;
    out->inputs[0] = input;
    out->aux_int = static_cast<u32>(axis);
    return out;
}

Var* Mean(Arena& arena, ModelState& model, Var* input, i32 axis)
{
    if (input == nullptr) { return nullptr; }
    VarFlags flags = input->RequiresGrad() ? VarFlags::RequiresGrad : VarFlags::None;
    u32 out_rows = 1;
    u32 out_cols = 1;
    if (axis == 0) {
        out_rows = 1;
        out_cols = input->val->cols;
    } else if (axis == 1) {
        out_rows = input->val->rows;
        out_cols = 1;
    } else {
        out_rows = 1;
        out_cols = 1;
    }
    Var* out = Var::Create(arena, model, out_rows, out_cols, flags);
    if (out == nullptr) { return nullptr; }
    out->type = &var_type::Mean;
    out->inputs[0] = input;
    out->aux_int = static_cast<u32>(axis);
    return out;
}

Var* LayerNorm(Arena& arena, ModelState& model, Var* input, Var* gamma, Var* beta, f32 eps)
{
    Var* v = CreateNode(arena, model, var_type::LayerNorm, input, gamma, beta);
    if (v != nullptr) { v->aux_param = eps; }
    return v;
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

void Graph::ZeroGrad() const noexcept
{
    for (Var* cur : Vars()) {
        if (cur != nullptr && cur->grad != nullptr) {
            cur->grad->Clear();
        }
    }
}

// ---------------------------------------------------------------- Model

ModelState* ModelState::Create(Arena& arena) noexcept
{
    return arena.PushStruct<ModelState>();
}
