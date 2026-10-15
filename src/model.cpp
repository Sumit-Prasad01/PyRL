#include "model.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace {

template <typename T>
T* Require(T* p, const char* what)
{
    if (p == nullptr) {
        throw std::runtime_error(std::string("CreateActorModel: ") + what + 
                                " failed (arena exhausted or shape mismatch)");
    }
    return p;
}

f32 XavierBound(u32 fan_in, u32 fan_out)
{
    return std::sqrt(6.0f / static_cast<f32>(fan_in + fan_out));
}

struct Builder {
    Arena&      arena;
    ModelState& model;

    Var* Param(u32 rows, u32 cols)
    {
        return Require(Var::Create(arena, model, rows, cols,
                                   VarFlags::Parameter | VarFlags::RequiresGrad),
                       "parameter");
    }

    Var* Dense(Var* x, Var* w, Var* b)
    {
        Var* wx = Require(ops::MatMul(arena, model, w, x), "matmul");
        return Require(ops::Add(arena, model, wx, b), "add");
    }

    Var* Relu(Var* x) { return Require(ops::Relu(arena, model, x), "relu"); }
};

} // namespace

ModelState* CreateActorModel(Arena& arena, const ActorConfig& cfg, Prng& rng)
{
    if (cfg.input_size == 0 || cfg.hidden_size == 0 || cfg.num_actions == 0) {
        throw std::invalid_argument("CreateActorModel: all sizes must be > 0");
    }

    ModelState* model = Require(ModelState::Create(arena), "model_state");
    Builder b{arena, *model};

    Var* input = Require(Var::Create(arena, *model, cfg.input_size, 1), "input");
    model->input = input;

    Var* W0 = b.Param(cfg.hidden_size, cfg.input_size);
    Var* b0 = b.Param(cfg.hidden_size, 1);
    Var* W1 = b.Param(cfg.hidden_size, cfg.hidden_size);
    Var* b1 = b.Param(cfg.hidden_size, 1);
    Var* W2 = b.Param(cfg.num_actions, cfg.hidden_size);
    Var* b2 = b.Param(cfg.num_actions, 1);

    for (Var* w : {W0, W1, W2}) {
        const f32 bound = XavierBound(w->val->cols, w->val->rows);
        w->val->FillRand(-bound, bound, rng);
    }

    Var* a0 = b.Relu(b.Dense(input, W0, b0));
    Var* a1 = b.Relu(b.Dense(a0, W1, b1));
    Var* z2 = b.Dense(a1, W2, b2);

    model->output = Require(ops::Softmax(arena, *model, z2), "softmax");

    model->advantage = Require(Var::Create(arena, *model, cfg.num_actions, 1), "advantage");
    model->cost = Require(ops::ReinforceLoss(arena, *model, model->output, model->advantage), "reinforce loss");

    model->forward_graph = Graph::Build(arena, *model, model->output);
    model->cost_graph    = Graph::Build(arena, *model, model->cost);
    if (model->forward_graph.Empty() || model->cost_graph.Empty()) {
        throw std::runtime_error("CreateActorModel: graph construction failed (arena exhausted)");
    }
    return model;
}
