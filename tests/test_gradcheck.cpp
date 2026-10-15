#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "autograd.hpp"
#include "env.hpp"
#include "model.hpp"
#include "test_utils.hpp"

using namespace testutil;

TEST_CASE("Gradcheck: Add and Sub", "[gradcheck]")
{
    GraphFixture f;
    Var* a = f.Leaf(3, 2, true);
    Var* b = f.Leaf(3, 2, true);

    SECTION("Add")
    {
        Var* y = Must(ops::Add(f.arena, *f.model, a, b));
        ExpectGradientsMatch(f.Build(y), {a, b});
    }
    SECTION("Sub")
    {
        Var* y = Must(ops::Sub(f.arena, *f.model, a, b));
        ExpectGradientsMatch(f.Build(y), {a, b});
    }
    SECTION("the same variable as both inputs")
    {
        Var* y = Must(ops::Add(f.arena, *f.model, a, a));   // d/da = 2
        ExpectGradientsMatch(f.Build(y), {a});
    }
}

TEST_CASE("Gradcheck: MatMul", "[gradcheck]")
{
    GraphFixture f;

    SECTION("single product")
    {
        Var* a = f.Leaf(3, 4, true);
        Var* b = f.Leaf(4, 2, true);
        Var* y = Must(ops::MatMul(f.arena, *f.model, a, b));
        ExpectGradientsMatch(f.Build(y), {a, b});
    }

    SECTION("chained products")
    {
        Var* a = f.Leaf(2, 3, true);
        Var* b = f.Leaf(3, 4, true);
        Var* c = f.Leaf(4, 2, true);
        Var* ab  = Must(ops::MatMul(f.arena, *f.model, a, b));
        Var* abc = Must(ops::MatMul(f.arena, *f.model, ab, c));
        ExpectGradientsMatch(f.Build(abc), {a, b, c});
    }

    SECTION("matrix-vector, the shape used by the policy network")
    {
        Var* w = f.Param(5, 3);
        Var* x = f.Leaf(3, 1, true);
        Var* y = Must(ops::MatMul(f.arena, *f.model, w, x));
        ExpectGradientsMatch(f.Build(y), {w, x});
    }
}

TEST_CASE("Gradcheck: Relu", "[gradcheck]")
{
    GraphFixture f;
    Var* x = f.Leaf(4, 3, true);
    FillAwayFromZero(*x->val, f.rng);   // finite differences are invalid at the kink
    Var* y = Must(ops::Relu(f.arena, *f.model, x));
    ExpectGradientsMatch(f.Build(y), {x});
}

TEST_CASE("Gradcheck: Softmax followed by the REINFORCE loss", "[gradcheck]")
{
    // sum(softmax(x)) is constant, so softmax alone has a zero gradient under
    // the sum-loss. Composing with the (non-uniform) REINFORCE loss gives a
    // meaningful test of both ops.
    GraphFixture f;
    Var* logits = f.Leaf(5, 1, true, -2.0f, 2.0f);
    Var* adv    = f.Leaf(5, 1, false);
    SetVal(adv, {0.5f, -1.0f, 2.0f, 0.25f, -0.75f});

    Var* probs = Must(ops::Softmax(f.arena, *f.model, logits));
    Var* cost  = Must(ops::ReinforceLoss(f.arena, *f.model, probs, adv));
    ExpectGradientsMatch(f.Build(cost), {logits});
}

TEST_CASE("Gradcheck: a node consumed by two paths accumulates both gradients", "[gradcheck]")
{
    // s = relu(x) + x   =>   ds/dx = relu'(x) + 1  (x feeds two consumers)
    GraphFixture f;
    Var* x = f.Leaf(6, 1, true);
    FillAwayFromZero(*x->val, f.rng);
    Var* r = Must(ops::Relu(f.arena, *f.model, x));
    Var* s = Must(ops::Add(f.arena, *f.model, r, x));
    ExpectGradientsMatch(f.Build(s), {x});
}

TEST_CASE("Gradcheck: a small MLP policy with REINFORCE loss", "[gradcheck]")
{
    // x(3) -> Linear(5) -> ReLU -> Linear(4) -> Softmax -> REINFORCE loss.
    // ReLU kinks break finite differences, so try seeds until every ReLU input
    // is safely away from zero.
    bool checked = false;
    for (u64 seed = 1; seed <= 200 && !checked; ++seed) {
        GraphFixture f(seed);
        Var* x   = f.Leaf(3, 1, false);
        Var* W1  = f.Param(5, 3);
        Var* b1  = f.Param(5, 1);
        Var* W2  = f.Param(4, 5);
        Var* b2  = f.Param(4, 1);
        Var* adv = f.Leaf(4, 1, false);
        SetVal(adv, {0.3f, -0.2f, 1.1f, -0.7f});

        Var* h    = Must(ops::Relu(f.arena, *f.model,
                         Must(ops::Add(f.arena, *f.model,
                              Must(ops::MatMul(f.arena, *f.model, W1, x)), b1))));
        Var* p    = Must(ops::Softmax(f.arena, *f.model,
                         Must(ops::Add(f.arena, *f.model,
                              Must(ops::MatMul(f.arena, *f.model, W2, h)), b2))));
        Var* cost = Must(ops::ReinforceLoss(f.arena, *f.model, p, adv));

        const Graph g = f.Build(cost);
        g.Forward();
        if (ReluMargin(g) < 0.05f) { continue; }

        INFO("seed " << seed);
        ExpectGradientsMatch(g, {W1, b1, W2, b2});
        checked = true;
    }
    REQUIRE(checked);
}

TEST_CASE("Gradcheck: the full actor model's cost graph", "[gradcheck][model]")
{
    SnakeEnv env(16);
    ActorConfig cfg;
    cfg.input_size  = env.InputSize();
    cfg.hidden_size = 8;
    cfg.num_actions = 5;

    bool checked = false;
    for (u64 seed = 1; seed <= 200 && !checked; ++seed) {
        Arena arena(MiB(2));
        Prng rng(seed, 3);
        ModelState* model = CreateActorModel(arena, cfg, rng);

        env.EncodeState(*model->input->val, Point{1, 2}, Point{3, 0}, Action::Up);
        SetVal(model->advantage, {0.5f, -1.0f, 2.0f, 0.25f, -0.75f});
        model->cost_graph.Forward();
        if (ReluMargin(model->cost_graph) < 0.05f) { continue; }

        std::vector<Var*> params;
        for (Var* v : model->cost_graph.Vars()) {
            if (v->IsParameter()) { params.push_back(v); }
        }
        REQUIRE(params.size() == 6);   // W0 b0 W1 b1 W2 b2

        INFO("seed " << seed);
        ExpectGradientsMatch(model->cost_graph, std::span<Var* const>(params));
        checked = true;
    }
    REQUIRE(checked);
}
