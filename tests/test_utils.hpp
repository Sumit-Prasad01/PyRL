#pragma once

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <span>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "arena.hpp"
#include "autograd.hpp"
#include "base.hpp"
#include "mat.hpp"
#include "prng.hpp"

namespace testutil {

// ------------------------------------------------------------ matrices

inline Mat* MakeMat(Arena& arena, u32 rows, u32 cols)
{
    Mat* m = Mat::Create(arena, rows, cols);
    REQUIRE(m != nullptr);
    return m;
}

inline Mat* MakeMat(Arena& arena, u32 rows, u32 cols, std::initializer_list<f32> values)
{
    Mat* m = MakeMat(arena, rows, cols);
    REQUIRE(values.size() == m->Size());
    std::ranges::copy(values, m->data);
    return m;
}

inline Mat* RandomMat(Arena& arena, Prng& rng, u32 rows, u32 cols,
                      f32 lo = -1.0f, f32 hi = 1.0f)
{
    Mat* m = MakeMat(arena, rows, cols);
    m->FillRand(lo, hi, rng);
    return m;
}

// |x| in [min_abs, max_abs] with a random sign. Keeps ReLU inputs away from
// the kink at 0, where finite differences are meaningless.
inline void FillAwayFromZero(Mat& m, Prng& rng, f32 min_abs = 0.1f, f32 max_abs = 1.0f)
{
    for (f32& v : m.Span()) {
        const f32 mag = min_abs + (max_abs - min_abs) * rng.NextFloat();
        v = (rng.NextBelow(2) == 1) ? mag : -mag;
    }
}

inline f64 SumOf(const Mat& m)
{
    f64 s = 0.0;
    for (f32 v : m.Span()) { s += v; }
    return s;
}

inline std::vector<f32> ToVector(const Mat& m)
{
    const auto s = m.Span();
    return {s.begin(), s.end()};
}

inline void ExpectMatNear(const Mat& actual, std::initializer_list<f32> expected,
                          f32 tol = 1e-5f)
{
    REQUIRE(actual.Size() == expected.size());
    u64 i = 0;
    for (f32 e : expected) {
        const f32 a = actual.data[i];
        CAPTURE(i, a, e);
        CHECK(std::fabs(a - e) <= tol);
        ++i;
    }
}

inline void ExpectMatNear(const Var& actual, std::initializer_list<f32> expected,
                          f32 tol = 1e-5f)
{
    REQUIRE(actual.val != nullptr);
    ExpectMatNear(*actual.val, expected, tol);
}

// Reference matmul in f64: op(a) * op(b).
inline std::vector<f64> NaiveMatMul(const Mat& a, const Mat& b, bool ta, bool tb)
{
    const u32 m = ta ? a.cols : a.rows;
    const u32 k = ta ? a.rows : a.cols;
    const u32 n = tb ? b.rows : b.cols;
    std::vector<f64> out(u64{m} * n, 0.0);
    for (u32 i = 0; i < m; i++) {
        for (u32 j = 0; j < n; j++) {
            f64 sum = 0.0;
            for (u32 p = 0; p < k; p++) {
                const f64 av = ta ? a.At(p, i) : a.At(i, p);
                const f64 bv = tb ? b.At(j, p) : b.At(p, j);
                sum += av * bv;
            }
            out[u64{i} * n + j] = sum;
        }
    }
    return out;
}

// ------------------------------------------------------------ graphs

inline Var* Must(Var* v)
{
    REQUIRE(v != nullptr);
    return v;
}

inline void SetVal(Var* v, std::initializer_list<f32> values)
{
    REQUIRE(values.size() == v->val->Size());
    std::ranges::copy(values, v->val->data);
}

// An arena + model + RNG bundle with helpers for building small graphs.
struct GraphFixture {
    Arena       arena;
    ModelState* model;
    Prng        rng;

    explicit GraphFixture(u64 seed = 1, u64 capacity = MiB(4))
        : arena(capacity), model(ModelState::Create(arena)), rng(seed, 7)
    {
        REQUIRE(model != nullptr);
    }

    // Leaf variable filled with uniform values in [lo, hi).
    Var* Leaf(u32 rows, u32 cols, bool requires_grad, f32 lo = -1.0f, f32 hi = 1.0f)
    {
        Var* v = Must(Var::Create(arena, *model, rows, cols,
                                  requires_grad ? VarFlags::RequiresGrad : VarFlags::None));
        v->val->FillRand(lo, hi, rng);
        return v;
    }

    Var* Param(u32 rows, u32 cols)
    {
        Var* v = Must(Var::Create(arena, *model, rows, cols,
                                  VarFlags::Parameter | VarFlags::RequiresGrad));
        v->val->FillRand(-1.0f, 1.0f, rng);
        return v;
    }

    Graph Build(Var* out)
    {
        Graph g = Graph::Build(arena, *model, out);
        REQUIRE_FALSE(g.Empty());
        return g;
    }
};

// Smallest |input| over all ReLU nodes (call after Forward()). If this is
// smaller than a few times the finite-difference step, the check is unreliable.
inline f32 ReluMargin(const Graph& g)
{
    f32 margin = std::numeric_limits<f32>::infinity();
    for (const Var* v : g.Vars()) {
        if (v->type != &var_type::Relu) { continue; }
        for (f32 z : v->inputs[0]->val->Span()) {
            margin = std::min(margin, std::fabs(z));
        }
    }
    return margin;
}

// ------------------------------------------------------------ gradient check

struct GradCheckOptions {
    f32 eps     = 1e-3f;  // finite-difference step
    f64 abs_tol = 2e-3;   // f32 forward pass => ~1e-4 numerical noise; keep slack
    f64 rel_tol = 2e-2;
};

// Compares autograd with central differences of L = sum(output->val) for every
// element of every Var in `wrt`. Each Var in `wrt` must require a gradient.
inline void ExpectGradientsMatch(const Graph& graph, std::span<Var* const> wrt,
                                 GradCheckOptions opt = {})
{
    REQUIRE_FALSE(graph.Empty());
    Var* output = graph.vars[graph.size - 1];

    // 1. Analytic gradients (parameter grads are not auto-cleared, so clear them).
    graph.Forward();
    for (Var* v : wrt) {
        REQUIRE(v->RequiresGrad());
        v->grad->Clear();
    }
    REQUIRE(graph.Backward());

    std::vector<std::vector<f32>> analytic;
    analytic.reserve(wrt.size());
    for (Var* v : wrt) { analytic.push_back(ToVector(*v->grad)); }

    // 2. Numerical gradients.
    u64 checked = 0, failures = 0;
    for (std::size_t vi = 0; vi < wrt.size(); ++vi) {
        Var* v = wrt[vi];
        const auto vals = v->val->Span();
        for (std::size_t i = 0; i < vals.size(); ++i) {
            const f32 orig = vals[i];
            const f32 hi   = orig + opt.eps;
            const f32 lo   = orig - opt.eps;

            vals[i] = hi;  graph.Forward();  const f64 l_hi = SumOf(*output->val);
            vals[i] = lo;  graph.Forward();  const f64 l_lo = SumOf(*output->val);
            vals[i] = orig;

            // Use the step actually representable in f32.
            const f64 numeric = (l_hi - l_lo) / (static_cast<f64>(hi) - static_cast<f64>(lo));
            const f64 a       = analytic[vi][i];
            const f64 err     = std::fabs(a - numeric);
            const f64 scale   = std::max(std::fabs(a), std::fabs(numeric));

            ++checked;
            if (err > opt.abs_tol + opt.rel_tol * scale) {
                ++failures;
                if (failures <= 5) {   // do not flood the log
                    FAIL_CHECK("gradient mismatch: var #" << v->index << ", element " << i
                               << ": analytic=" << a << " numeric=" << numeric);
                }
            }
        }
    }
    graph.Forward();  // leave values consistent with the unperturbed parameters
    INFO(checked << " gradient entries checked");
    CHECK(failures == 0);
}

inline void ExpectGradientsMatch(const Graph& graph, std::initializer_list<Var*> wrt,
                                 GradCheckOptions opt = {})
{
    const std::vector<Var*> v(wrt);
    ExpectGradientsMatch(graph, std::span<Var* const>(v), opt);
}

} // namespace testutil
