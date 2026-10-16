#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "arena.hpp"
#include "autograd.hpp"
#include "nn.hpp"
#include "prng.hpp"

using namespace nn;

TEST_CASE("Initializers: Orthogonal produces orthonormal vectors", "[nn][init]")
{
    Arena arena(KiB(64));
    Prng rng(1234, 5678);

    SECTION("Square matrix orthonormal check Q * Q^T == I")
    {
        Mat* Q = Mat::Create(arena, 8, 8);
        init::Orthogonal(*Q, rng, 1.0f);

        // Compute Q * Q^T
        Mat* I_est = Mat::Create(arena, 8, 8);
        REQUIRE(mat::MatMul(*I_est, *Q, *Q, true, false, true));

        for (u32 r = 0; r < 8; ++r) {
            for (u32 c = 0; c < 8; ++c) {
                const f32 expected = (r == c) ? 1.0f : 0.0f;
                CHECK(std::fabs(I_est->At(r, c) - expected) < 1e-4f);
            }
        }
    }

    SECTION("Orthogonal with gain")
    {
        const f32 gain = 2.5f;
        Mat* Q = Mat::Create(arena, 6, 6);
        init::Orthogonal(*Q, rng, gain);

        Mat* I_est = Mat::Create(arena, 6, 6);
        REQUIRE(mat::MatMul(*I_est, *Q, *Q, true, false, true));

        for (u32 r = 0; r < 6; ++r) {
            for (u32 c = 0; c < 6; ++c) {
                const f32 expected = (r == c) ? (gain * gain) : 0.0f;
                CHECK(std::fabs(I_est->At(r, c) - expected) < 1e-3f);
            }
        }
    }
}

TEST_CASE("Initializers: Xavier and Kaiming variance", "[nn][init][statistical]")
{
    Arena arena(MiB(2));
    Prng rng(999, 111);

    // Large matrix for statistical stability
    constexpr u32 R = 200;
    constexpr u32 C = 300;
    Mat* m = Mat::Create(arena, R, C);

    SECTION("XavierNormal variance = 2 / (fan_in + fan_out)")
    {
        init::XavierNormal(*m, rng, 1.0f);
        const f32 target_var = 2.0f / (R + C);

        f64 sum = 0.0, sum_sq = 0.0;
        for (f32 v : m->Span()) {
            sum += v;
            sum_sq += static_cast<f64>(v) * v;
        }
        const f64 mean = sum / m->Size();
        const f64 var  = (sum_sq / m->Size()) - (mean * mean);

        CHECK(std::fabs(mean) < 0.01);
        CHECK(std::fabs(var - target_var) / target_var < 0.08); // within 8% relative error
    }

    SECTION("KaimingNormal variance = 2 / fan_in")
    {
        init::KaimingNormal(*m, rng, 0.0f);
        const f32 target_var = 2.0f / C;

        f64 sum = 0.0, sum_sq = 0.0;
        for (f32 v : m->Span()) {
            sum += v;
            sum_sq += static_cast<f64>(v) * v;
        }
        const f64 mean = sum / m->Size();
        const f64 var  = (sum_sq / m->Size()) - (mean * mean);

        CHECK(std::fabs(mean) < 0.01);
        CHECK(std::fabs(var - target_var) / target_var < 0.08);
    }
}

TEST_CASE("Linear layer forward and parameter registration", "[nn][linear]")
{
    Arena arena(KiB(64));
    auto* model = ModelState::Create(arena);
    Prng rng(12, 34);

    Linear fc(arena, *model, 4, 3, true, true, &rng, 1.0f);
    CHECK(fc.Parameters().size() == 2); // weight and bias
    CHECK(fc.Weight()->val->rows == 3);
    CHECK(fc.Weight()->val->cols == 4);
    CHECK(fc.Bias()->val->rows == 3);
    CHECK(fc.Bias()->val->cols == 1);

    Var* x = Var::Create(arena, *model, 4, 1, VarFlags::RequiresGrad);
    x->val->Fill(1.0f);

    Var* out = fc.Forward(x);
    REQUIRE(out != nullptr);
    CHECK(out->val->rows == 3);
    CHECK(out->val->cols == 1);

    const Graph g = Graph::Build(arena, *model, out);
    g.Forward();
    for (f32 v : out->val->Span()) {
        CHECK(std::isfinite(v));
    }
}

TEST_CASE("Sequential module composition and Actor network rebuild", "[nn][sequential]")
{
    Arena arena(KiB(64));
    auto* model = ModelState::Create(arena);
    Prng rng(77, 88);

    // Rebuild the policy network in 4 lines!
    // 76 inputs -> 16 hidden -> 5 outputs -> Softmax
    Sequential policy(
        std::make_shared<Linear>(arena, *model, 76, 16, true, true, &rng, std::sqrt(2.0f)),
        std::make_shared<ReLU>(arena, *model),
        std::make_shared<Linear>(arena, *model, 16, 5, true, true, &rng, 0.01f),
        std::make_shared<Softmax>(arena, *model)
    );

    CHECK(policy.Size() == 4);
    CHECK(policy.Parameters().size() == 4); // W0, b0, W1, b1

    Var* state = Var::Create(arena, *model, 76, 1, VarFlags::None);
    state->val->Fill(0.5f);

    Var* probs = policy.Forward(state);
    REQUIRE(probs != nullptr);
    CHECK(probs->val->rows == 5);
    CHECK(probs->val->cols == 1);

    const Graph g = Graph::Build(arena, *model, probs);
    g.Forward();

    // Softmax probabilities sum to 1.0 and are all non-negative
    f32 sum_p = 0.0f;
    for (f32 p : probs->val->Span()) {
        CHECK(p >= 0.0f);
        CHECK(p <= 1.0f);
        sum_p += p;
    }
    CHECK(std::fabs(sum_p - 1.0f) < 1e-5f);
}

TEST_CASE("Sequential train/eval mode toggling", "[nn][sequential]")
{
    Arena arena(KiB(4));
    auto* model = ModelState::Create(arena);
    Sequential seq(
        std::make_shared<Linear>(arena, *model, 4, 4)
    );

    CHECK(seq.IsTraining());
    seq.Eval();
    CHECK_FALSE(seq.IsTraining());
    seq.Train(true);
    CHECK(seq.IsTraining());
}
