#include <algorithm>
#include <unordered_map>
#include <unordered_set>

#include <catch2/catch_test_macros.hpp>

#include "autograd.hpp"
#include "test_utils.hpp"

using namespace testutil;

TEST_CASE("Var::Create assigns increasing indices and allocates gradients on demand", "[autograd]")
{
    GraphFixture f;
    Var* a = Must(Var::Create(f.arena, *f.model, 2, 3));
    Var* b = Must(Var::Create(f.arena, *f.model, 2, 3,
                              VarFlags::RequiresGrad | VarFlags::Parameter));

    CHECK(a->index == 0);
    CHECK(b->index == 1);
    CHECK(f.model->num_vars == 2);

    CHECK(a->val->rows == 2);
    CHECK(a->val->cols == 3);
    CHECK_FALSE(a->RequiresGrad());
    CHECK_FALSE(a->IsParameter());
    CHECK(a->grad == nullptr);
    CHECK(a->type == nullptr);
    CHECK(a->NumInputs() == 0);

    CHECK(b->RequiresGrad());
    CHECK(b->IsParameter());
    REQUIRE(b->grad != nullptr);
    CHECK(b->grad->rows == 2);
    CHECK(b->grad->cols == 3);
}

TEST_CASE("A failed Var::Create does not consume an index", "[autograd]")
{
    Arena tiny(KiB(1));
    ModelState* model = ModelState::Create(tiny);
    REQUIRE(model != nullptr);

    Var* v = Var::Create(tiny, *model, 100, 100);
    CHECK(v == nullptr);
    CHECK(model->num_vars == 0);
}

TEST_CASE("Graph-building ops reject invalid inputs without creating nodes", "[autograd]")
{
    GraphFixture f;
    Var* a = f.Leaf(2, 1, true);
    Var* b = f.Leaf(3, 1, true);
    Var* w = f.Leaf(2, 3, true);
    const u32 before = f.model->num_vars;

    CHECK(ops::Relu(f.arena, *f.model, nullptr) == nullptr);
    CHECK(ops::Add(f.arena, *f.model, a, nullptr) == nullptr);
    CHECK(ops::Add(f.arena, *f.model, nullptr, a) == nullptr);
    CHECK(ops::Add(f.arena, *f.model, a, b) == nullptr);        // 2x1 + 3x1
    CHECK(ops::Sub(f.arena, *f.model, a, b) == nullptr);
    CHECK(ops::MatMul(f.arena, *f.model, a, w) == nullptr);     // (2x1)(2x3)
    CHECK(ops::ReinforceLoss(f.arena, *f.model, a, b) == nullptr);
    CHECK(f.model->num_vars == before);

    SECTION("a valid op works and records its type and inputs")
    {
        Var* y = ops::MatMul(f.arena, *f.model, w, b);   // (2x3)(3x1) -> 2x1
        REQUIRE(y != nullptr);
        CHECK(y->val->rows == 2);
        CHECK(y->val->cols == 1);
        CHECK(y->type == &var_type::MatMul);
        CHECK(y->inputs[0] == w);
        CHECK(y->inputs[1] == b);
        CHECK(y->NumInputs() == 2);
        CHECK(y->RequiresGrad());
    }
}

TEST_CASE("RequiresGrad propagates only from inputs that need it", "[autograd]")
{
    GraphFixture f;
    Var* x  = f.Leaf(3, 1, false);
    Var* w1 = f.Leaf(2, 3, false);
    Var* w2 = f.Leaf(2, 3, true);

    Var* no_grad   = Must(ops::MatMul(f.arena, *f.model, w1, x));
    Var* with_grad = Must(ops::MatMul(f.arena, *f.model, w2, x));
    CHECK_FALSE(no_grad->RequiresGrad());
    CHECK(no_grad->grad == nullptr);
    CHECK(with_grad->RequiresGrad());
    CHECK(with_grad->grad != nullptr);
}

TEST_CASE("Graph::Build returns a duplicate-free topological order", "[autograd][graph]")
{
    GraphFixture f;
    Var* x  = f.Leaf(3, 1, false);
    Var* w1 = f.Param(4, 3);
    Var* b1 = f.Param(4, 1);
    Var* w2 = f.Param(2, 4);

    Var* wx   = Must(ops::MatMul(f.arena, *f.model, w1, x));
    Var* z    = Must(ops::Add(f.arena, *f.model, wx, b1));
    Var* h    = Must(ops::Relu(f.arena, *f.model, z));
    Var* skip = Must(ops::Add(f.arena, *f.model, h, h));     // h is consumed twice
    Var* lg   = Must(ops::MatMul(f.arena, *f.model, w2, skip));
    Var* y    = Must(ops::Softmax(f.arena, *f.model, lg));

    const Graph g = f.Build(y);

    CHECK(g.size == f.model->num_vars);           // everything created is reachable
    CHECK(g.vars[g.size - 1] == y);               // output is last

    std::unordered_map<const Var*, u32> position;
    for (u32 i = 0; i < g.size; ++i) {
        const bool inserted = position.emplace(g.vars[i], i).second;
        CHECK(inserted);                          // no duplicates
    }
    for (u32 i = 0; i < g.size; ++i) {
        const Var* v = g.vars[i];
        for (u32 k = 0; k < v->NumInputs(); ++k) {
            REQUIRE(position.contains(v->inputs[k]));
            CHECK(position[v->inputs[k]] < i);    // inputs come before consumers
        }
    }
}

TEST_CASE("Graph::Build handles degenerate inputs", "[autograd][graph]")
{
    GraphFixture f;
    CHECK(Graph::Build(f.arena, *f.model, nullptr).Empty());

    Var* lone = f.Leaf(2, 2, false);
    const Graph g = f.Build(lone);
    CHECK(g.size == 1);
    CHECK(g.vars[0] == lone);
}

TEST_CASE("Forward computes the expected values", "[autograd][forward]")
{
    GraphFixture f;
    Var* W = f.Leaf(2, 2, false);
    Var* x = f.Leaf(2, 1, false);
    Var* b = f.Leaf(2, 1, false);
    SetVal(W, {1, -1, 2, 0});
    SetVal(x, {3, 1});
    SetVal(b, {0, -10});

    Var* wx = Must(ops::MatMul(f.arena, *f.model, W, x));
    Var* z  = Must(ops::Add(f.arena, *f.model, wx, b));
    Var* y  = Must(ops::Relu(f.arena, *f.model, z));

    const Graph g = f.Build(y);
    g.Forward();

    ExpectMatNear(*wx, {2, 6});
    ExpectMatNear(*z, {2, -4});
    ExpectMatNear(*y, {2, 0});
}

TEST_CASE("Backward produces the exact hand-computed gradients", "[autograd][backward]")
{
    // L = sum(relu(W x + b)) with W=[[1,-1],[2,0]], x=[3,1], b=[0,-10]
    // z = [2,-4], relu mask = [1,0]  =>  dz = [1,0]
    // dW = dz x^T = [[3,1],[0,0]],  db = dz = [1,0],  dx = W^T dz = [1,-1]
    GraphFixture f;
    Var* W = f.Leaf(2, 2, true);
    Var* x = f.Leaf(2, 1, true);
    Var* b = f.Leaf(2, 1, true);
    SetVal(W, {1, -1, 2, 0});
    SetVal(x, {3, 1});
    SetVal(b, {0, -10});

    Var* wx = Must(ops::MatMul(f.arena, *f.model, W, x));
    Var* z  = Must(ops::Add(f.arena, *f.model, wx, b));
    Var* y  = Must(ops::Relu(f.arena, *f.model, z));

    const Graph g = f.Build(y);
    g.Forward();
    REQUIRE(g.Backward());

    ExpectMatNear(*W->grad, {3, 1, 0, 0});
    ExpectMatNear(*b->grad, {1, 0});
    ExpectMatNear(*x->grad, {1, -1});
    ExpectMatNear(*y->grad, {1, 1});   // seeded with ones
}

TEST_CASE("Backward: parameter gradients accumulate, other gradients are cleared", "[autograd][backward]")
{
    GraphFixture f;
    Var* W = f.Param(2, 3);                    // parameter: gradient persists across calls
    Var* x = f.Leaf(3, 1, false);
    SetVal(x, {1, 2, 3});
    Var* y = Must(ops::MatMul(f.arena, *f.model, W, x));

    const Graph g = f.Build(y);
    g.Forward();

    W->grad->Clear();
    REQUIRE(g.Backward());
    ExpectMatNear(*W->grad, {1, 2, 3, 1, 2, 3});

    REQUIRE(g.Backward());                     // second call: parameter grad doubles
    ExpectMatNear(*W->grad, {2, 4, 6, 2, 4, 6});

    W->grad->Clear();                          // explicit clear resets it
    REQUIRE(g.Backward());
    ExpectMatNear(*W->grad, {1, 2, 3, 1, 2, 3});
}

TEST_CASE("Backward clears non-parameter gradients on every call", "[autograd][backward]")
{
    GraphFixture f;
    Var* x = f.Leaf(3, 1, true);
    x->val->Fill(1.0f);
    Var* y = Must(ops::Relu(f.arena, *f.model, x));

    const Graph g = f.Build(y);
    g.Forward();
    REQUIRE(g.Backward());
    REQUIRE(g.Backward());
    ExpectMatNear(*x->grad, {1, 1, 1});        // not 2: intermediate and leaf grads reset
}

TEST_CASE("Backward refuses graphs that cannot be differentiated", "[autograd][backward]")
{
    GraphFixture f;

    SECTION("empty graph")
    {
        const Graph empty{};
        CHECK_FALSE(empty.Backward());
    }

    SECTION("output does not require a gradient")
    {
        Var* w = f.Leaf(2, 3, false);
        Var* x = f.Leaf(3, 1, false);
        Var* y = Must(ops::MatMul(f.arena, *f.model, w, x));
        const Graph g = f.Build(y);
        g.Forward();
        CHECK_FALSE(g.Backward());
    }
}

TEST_CASE("Graph::Build scales linearly on a large computation graph", "[autograd][graph]")
{
    // Build a 500-layer deep and branched DAG to verify linear DFS post-order ordering.
    GraphFixture f(42, MiB(32));
    Var* cur = f.Leaf(2, 2, true);
    Var* bias = f.Leaf(2, 2, true);
    cur->val->Fill(0.01f);
    bias->val->Fill(0.001f);

    constexpr u32 kDepth = 500;
    for (u32 i = 0; i < kDepth; ++i) {
        Var* next = Must(ops::Add(f.arena, *f.model, cur, bias));
        cur = Must(ops::Relu(f.arena, *f.model, next));
    }

    const Graph g = f.Build(cur);
    REQUIRE(g.size == 2 + kDepth * 2);
    CHECK(g.vars[g.size - 1] == cur);

    // Verify topological invariance: all dependencies precede their consumers
    std::unordered_map<const Var*, u32> pos;
    for (u32 i = 0; i < g.size; ++i) {
        const bool inserted = pos.emplace(g.vars[i], i).second;
        CHECK(inserted); // No duplicates
    }
    for (u32 i = 0; i < g.size; ++i) {
        const Var* v = g.vars[i];
        for (u32 k = 0; k < v->NumInputs(); ++k) {
            REQUIRE(pos.contains(v->inputs[k]));
            CHECK(pos[v->inputs[k]] < i);
        }
    }

    // Verify forward and backward execute without stack overflow
    g.Forward();
    REQUIRE(g.Backward());
}

