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

TEST_CASE("Gradcheck: LogSoftmax followed by LogProbLoss", "[gradcheck]")
{
    GraphFixture f;
    Var* logits = f.Leaf(5, 1, true, -2.0f, 2.0f);
    Var* target = f.Leaf(2, 1, false);
    SetVal(target, {2.0f, 1.5f}); // action 2, advantage 1.5

    Var* log_probs = Must(ops::LogSoftmax(f.arena, *f.model, logits));
    Var* cost      = Must(ops::LogProbLoss(f.arena, *f.model, log_probs, target));
    ExpectGradientsMatch(f.Build(cost), {logits});
}

TEST_CASE("Gradcheck: CrossEntropy from logits", "[gradcheck]")
{
    GraphFixture f;
    Var* logits = f.Leaf(4, 1, true, -1.5f, 1.5f);
    Var* target = f.Leaf(2, 1, false);
    SetVal(target, {1.0f, 0.8f}); // class 1, weight 0.8

    Var* cost = Must(ops::CrossEntropy(f.arena, *f.model, logits, target));
    ExpectGradientsMatch(f.Build(cost), {logits});
}

TEST_CASE("Gradcheck: LogSoftmax and CrossEntropy are stable with large logits", "[gradcheck]")
{
    GraphFixture f;
    Var* logits = f.Leaf(3, 1, true);
    SetVal(logits, {100.0f, 101.5f, 99.0f});
    Var* target = f.Leaf(2, 1, false);
    SetVal(target, {1.0f, 1.0f});

    Var* cost = Must(ops::CrossEntropy(f.arena, *f.model, logits, target));
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

TEST_CASE("Double precision finite difference gradchecks reach tight tolerance (~1e-7)", "[gradcheck][double]")
{
    Arena arena(MiB(4));
    Prng rng(101, 7);
    constexpr f64 kEps = 1e-7;
    constexpr f64 kTol = 1e-7;

    SECTION("MatMul in f64")
    {
        Mat64* a = Mat64::Create(arena, 3, 4);
        Mat64* b = Mat64::Create(arena, 4, 2);
        a->FillRand(-1.5, 1.5, rng);
        b->FillRand(-1.5, 1.5, rng);

        Mat64* grad_out = Mat64::Create(arena, 3, 2);
        grad_out->FillRand(-1.0, 1.0, rng);

        Mat64* grad_a = Mat64::Create(arena, 3, 4);
        grad_a->Clear();
        REQUIRE(mat::MatMul<f64>(*grad_a, *grad_out, *b, false, false, true));

        Mat64* out_plus  = Mat64::Create(arena, 3, 2);
        Mat64* out_minus = Mat64::Create(arena, 3, 2);

        for (u64 i = 0; i < a->Size(); ++i) {
            const f64 orig = a->data[i];
            a->data[i] = orig + kEps;
            REQUIRE(mat::MatMul<f64>(*out_plus, *a, *b, true, false, false));
            f64 loss_plus = 0.0;
            for (u64 k = 0; k < out_plus->Size(); ++k) { loss_plus += out_plus->data[k] * grad_out->data[k]; }

            a->data[i] = orig - kEps;
            REQUIRE(mat::MatMul<f64>(*out_minus, *a, *b, true, false, false));
            f64 loss_minus = 0.0;
            for (u64 k = 0; k < out_minus->Size(); ++k) { loss_minus += out_minus->data[k] * grad_out->data[k]; }

            a->data[i] = orig;
            const f64 num_grad = (loss_plus - loss_minus) / (2.0 * kEps);
            const f64 analytic = grad_a->data[i];
            CHECK(std::fabs(num_grad - analytic) <= kTol);
        }
    }

    SECTION("Softmax in f64")
    {
        Mat64* z = Mat64::Create(arena, 5, 1);
        z->FillRand(-2.0, 2.0, rng);
        Mat64* grad_out = Mat64::Create(arena, 5, 1);
        grad_out->FillRand(-1.0, 1.0, rng);

        Mat64* s = Mat64::Create(arena, 5, 1);
        REQUIRE(mat::Softmax<f64>(*s, *z));

        Mat64* analytic_grad = Mat64::Create(arena, 5, 1);
        analytic_grad->Clear();
        REQUIRE(mat::SoftmaxAddGrad<f64>(*analytic_grad, *s, *grad_out));

        Mat64* s_plus  = Mat64::Create(arena, 5, 1);
        Mat64* s_minus = Mat64::Create(arena, 5, 1);

        for (u64 i = 0; i < z->Size(); ++i) {
            const f64 orig = z->data[i];
            z->data[i] = orig + kEps;
            REQUIRE(mat::Softmax<f64>(*s_plus, *z));
            f64 loss_plus = 0.0;
            for (u64 k = 0; k < 5; ++k) { loss_plus += s_plus->data[k] * grad_out->data[k]; }

            z->data[i] = orig - kEps;
            REQUIRE(mat::Softmax<f64>(*s_minus, *z));
            f64 loss_minus = 0.0;
            for (u64 k = 0; k < 5; ++k) { loss_minus += s_minus->data[k] * grad_out->data[k]; }

            z->data[i] = orig;
            const f64 num_grad = (loss_plus - loss_minus) / (2.0 * kEps);
            const f64 analytic = analytic_grad->data[i];
            CHECK(std::fabs(num_grad - analytic) <= kTol);
        }
    }

    SECTION("LogSoftmax in f64")
    {
        Mat64* z = Mat64::Create(arena, 4, 1);
        z->FillRand(-3.0, 3.0, rng);
        Mat64* grad_out = Mat64::Create(arena, 4, 1);
        grad_out->FillRand(-1.0, 1.0, rng);

        Mat64* ls = Mat64::Create(arena, 4, 1);
        REQUIRE(mat::LogSoftmax<f64>(*ls, *z));

        Mat64* analytic_grad = Mat64::Create(arena, 4, 1);
        analytic_grad->Clear();
        REQUIRE(mat::LogSoftmaxAddGrad<f64>(*analytic_grad, *ls, *grad_out));

        Mat64* ls_plus  = Mat64::Create(arena, 4, 1);
        Mat64* ls_minus = Mat64::Create(arena, 4, 1);

        for (u64 i = 0; i < z->Size(); ++i) {
            const f64 orig = z->data[i];
            z->data[i] = orig + kEps;
            REQUIRE(mat::LogSoftmax<f64>(*ls_plus, *z));
            f64 loss_plus = 0.0;
            for (u64 k = 0; k < 4; ++k) { loss_plus += ls_plus->data[k] * grad_out->data[k]; }

            z->data[i] = orig - kEps;
            REQUIRE(mat::LogSoftmax<f64>(*ls_minus, *z));
            f64 loss_minus = 0.0;
            for (u64 k = 0; k < 4; ++k) { loss_minus += ls_minus->data[k] * grad_out->data[k]; }

            z->data[i] = orig;
            const f64 num_grad = (loss_plus - loss_minus) / (2.0 * kEps);
            const f64 analytic = analytic_grad->data[i];
            CHECK(std::fabs(num_grad - analytic) <= kTol);
        }
    }

    SECTION("CrossEntropy in f64")
    {
        Mat64* z = Mat64::Create(arena, 4, 1);
        z->FillRand(-2.0, 2.0, rng);
        Mat64* target = Mat64::Create(arena, 2, 1);
        target->data[0] = 2.0; // class 2
        target->data[1] = 1.75; // weight 1.75

        Mat64* grad_out = Mat64::Create(arena, 1, 1);
        grad_out->data[0] = 1.0;

        Mat64* analytic_grad = Mat64::Create(arena, 4, 1);
        analytic_grad->Clear();
        REQUIRE(mat::CrossEntropyAddGrad<f64>(*analytic_grad, *z, *target, *grad_out));

        Mat64* loss_plus  = Mat64::Create(arena, 1, 1);
        Mat64* loss_minus = Mat64::Create(arena, 1, 1);

        for (u64 i = 0; i < z->Size(); ++i) {
            const f64 orig = z->data[i];
            z->data[i] = orig + kEps;
            REQUIRE(mat::CrossEntropy<f64>(*loss_plus, *z, *target));

            z->data[i] = orig - kEps;
            REQUIRE(mat::CrossEntropy<f64>(*loss_minus, *z, *target));

            z->data[i] = orig;
            const f64 num_grad = (loss_plus->data[0] - loss_minus->data[0]) / (2.0 * kEps);
            const f64 analytic = analytic_grad->data[i];
            CHECK(std::fabs(num_grad - analytic) <= kTol);
        }
    }
}

