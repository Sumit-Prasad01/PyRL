#include <cmath>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "mat.hpp"
#include "prng.hpp"
#include "test_utils.hpp"

using namespace testutil;

// ------------------------------------------------------------ construction & in-place ops

TEST_CASE("Mat::Create returns a zeroed, row-major matrix", "[mat]")
{
    Arena arena(KiB(4));
    Mat* m = Mat::Create(arena, 3, 4);
    REQUIRE(m != nullptr);
    CHECK(m->rows == 3);
    CHECK(m->cols == 4);
    CHECK(m->Size() == 12);
    for (f32 v : m->Span()) { CHECK(v == 0.0f); }

    m->At(1, 2) = 7.0f;
    CHECK(m->data[1 * 4 + 2] == 7.0f);   // row-major layout
}

TEST_CASE("Mat::Create fails cleanly when the arena is too small", "[mat]")
{
    Arena tiny(16);
    CHECK(Mat::Create(tiny, 100, 100) == nullptr);

    Arena arena(KiB(1));
    CHECK(Mat::Create(arena, 65536, 65536) == nullptr);   // 2^32 elements: no u32 overflow
}

TEST_CASE("Clear, Fill and Scale", "[mat]")
{
    Arena arena(KiB(4));
    Mat* m = MakeMat(arena, 2, 3);

    m->Fill(2.5f);
    for (f32 v : m->Span()) { CHECK(v == 2.5f); }

    m->Scale(-2.0f);
    for (f32 v : m->Span()) { CHECK(v == -5.0f); }

    m->Clear();
    for (f32 v : m->Span()) { CHECK(v == 0.0f); }
}

TEST_CASE("FillRand respects bounds and the supplied generator", "[mat]")
{
    Arena arena(KiB(64));
    Mat* a = MakeMat(arena, 50, 50);
    Mat* b = MakeMat(arena, 50, 50);

    Prng r1(3, 4), r2(3, 4);
    a->FillRand(-2.0f, 3.0f, r1);
    b->FillRand(-2.0f, 3.0f, r2);

    for (f32 v : a->Span()) {
        CHECK(v >= -2.0f);
        CHECK(v < 3.0f);
    }
    CHECK(ToVector(*a) == ToVector(*b));   // same seed => same matrix
}

// ------------------------------------------------------------ add / sub

TEST_CASE("Add and Sub", "[mat]")
{
    Arena arena(KiB(4));
    Mat* a   = MakeMat(arena, 2, 2, {1, 2, 3, 4});
    Mat* b   = MakeMat(arena, 2, 2, {10, 20, 30, 40});
    Mat* out = MakeMat(arena, 2, 2);

    REQUIRE(mat::Add(*out, *a, *b));
    ExpectMatNear(*out, {11, 22, 33, 44});

    REQUIRE(mat::Sub(*out, *b, *a));
    ExpectMatNear(*out, {9, 18, 27, 36});

    SECTION("in-place use is allowed for element-wise ops")
    {
        REQUIRE(mat::Add(*a, *a, *b));
        ExpectMatNear(*a, {11, 22, 33, 44});
    }
}

TEST_CASE("Shape mismatches return false and write nothing", "[mat]")
{
    Arena arena(KiB(4));
    Mat* out = MakeMat(arena, 2, 2, {9, 9, 9, 9});
    Mat* a   = MakeMat(arena, 2, 2);
    Mat* b   = MakeMat(arena, 2, 3);

    CHECK_FALSE(mat::Add(*out, *a, *b));
    CHECK_FALSE(mat::Sub(*out, *a, *b));
    ExpectMatNear(*out, {9, 9, 9, 9});

    Mat* wrong_out = MakeMat(arena, 3, 2);
    CHECK_FALSE(mat::Add(*wrong_out, *a, *a));
}

// ------------------------------------------------------------ matmul

TEST_CASE("MatMul known values", "[mat][matmul]")
{
    Arena arena(KiB(4));
    Mat* a   = MakeMat(arena, 2, 2, {1, 2, 3, 4});
    Mat* b   = MakeMat(arena, 2, 2, {5, 6, 7, 8});
    Mat* out = MakeMat(arena, 2, 2);

    REQUIRE(mat::MatMul(*out, *a, *b));
    ExpectMatNear(*out, {19, 22, 43, 50});

    // A^T * B = [[1,3],[2,4]] * [[5,6],[7,8]]
    REQUIRE(mat::MatMul(*out, *a, *b, true, true, false));
    ExpectMatNear(*out, {26, 30, 38, 44});

    // A * B^T = [[1,2],[3,4]] * [[5,7],[6,8]]
    REQUIRE(mat::MatMul(*out, *a, *b, true, false, true));
    ExpectMatNear(*out, {17, 23, 39, 53});
}

TEST_CASE("MatMul matches a naive reference for all transpose modes", "[mat][matmul]")
{
    struct Shape { u32 m, k, n; };
    constexpr Shape kShapes[] = {{3, 5, 4}, {1, 7, 1}, {8, 8, 8}, {2, 1, 6}, {5, 3, 1}};

    Arena arena(MiB(1));
    Prng rng(2024, 1);

    for (const Shape& s : kShapes) {
        for (int mask = 0; mask < 4; ++mask) {
            const bool ta = (mask & 1) != 0;
            const bool tb = (mask & 2) != 0;
            INFO("m=" << s.m << " k=" << s.k << " n=" << s.n
                      << " transpose_a=" << ta << " transpose_b=" << tb);

            Mat* a   = RandomMat(arena, rng, ta ? s.k : s.m, ta ? s.m : s.k);
            Mat* b   = RandomMat(arena, rng, tb ? s.n : s.k, tb ? s.k : s.n);
            Mat* out = MakeMat(arena, s.m, s.n);

            REQUIRE(mat::MatMul(*out, *a, *b, true, ta, tb));
            const auto ref = NaiveMatMul(*a, *b, ta, tb);
            for (u64 i = 0; i < out->Size(); ++i) {
                CHECK(std::fabs(out->data[i] - ref[i]) <= 1e-4 + 1e-4 * std::fabs(ref[i]));
            }
        }
    }
}

TEST_CASE("MatMul with zero_out=false accumulates (needed for backprop)", "[mat][matmul]")
{
    Arena arena(KiB(4));
    Prng rng(1, 1);
    Mat* a   = RandomMat(arena, rng, 3, 4);
    Mat* b   = RandomMat(arena, rng, 4, 2);
    Mat* out = MakeMat(arena, 3, 2);
    out->Fill(1.0f);

    REQUIRE(mat::MatMul(*out, *a, *b, /*zero_out=*/false));
    const auto ref = NaiveMatMul(*a, *b, false, false);
    for (u64 i = 0; i < out->Size(); ++i) {
        CHECK(std::fabs(out->data[i] - (1.0 + ref[i])) <= 1e-4);
    }
}

TEST_CASE("MatMul rejects incompatible shapes", "[mat][matmul]")
{
    Arena arena(KiB(4));
    Mat* a   = MakeMat(arena, 2, 3);
    Mat* b   = MakeMat(arena, 2, 3);
    Mat* out = MakeMat(arena, 2, 2);

    CHECK_FALSE(mat::MatMul(*out, *a, *b));                       // (2x3)(2x3)
    CHECK(mat::MatMul(*out, *a, *b, true, false, true));          // (2x3)(3x2) works
    Mat* bad_out = MakeMat(arena, 3, 3);
    CHECK_FALSE(mat::MatMul(*bad_out, *a, *b, true, false, true));
}

// ------------------------------------------------------------ activations

TEST_CASE("Relu", "[mat]")
{
    Arena arena(KiB(4));
    Mat* in  = MakeMat(arena, 1, 4, {-1.0f, 0.0f, 2.0f, -0.5f});
    Mat* out = MakeMat(arena, 1, 4);
    REQUIRE(mat::Relu(*out, *in));
    ExpectMatNear(*out, {0, 0, 2, 0});

    Mat* wrong = MakeMat(arena, 4, 1);
    CHECK_FALSE(mat::Relu(*wrong, *in));
}

TEST_CASE("Softmax", "[mat]")
{
    Arena arena(KiB(4));

    SECTION("known values")
    {
        Mat* in  = MakeMat(arena, 3, 1, {0.0f, 1.0f, 2.0f});
        Mat* out = MakeMat(arena, 3, 1);
        REQUIRE(mat::Softmax(*out, *in));
        ExpectMatNear(*out, {0.09003057f, 0.24472847f, 0.66524096f}, 1e-6f);
    }

    SECTION("uniform input gives a uniform distribution")
    {
        Mat* in  = MakeMat(arena, 4, 1);
        Mat* out = MakeMat(arena, 4, 1);
        REQUIRE(mat::Softmax(*out, *in));
        ExpectMatNear(*out, {0.25f, 0.25f, 0.25f, 0.25f});
    }

    SECTION("is numerically stable for large logits")
    {
        Mat* in  = MakeMat(arena, 3, 1, {1000.0f, 1001.0f, 1002.0f});
        Mat* out = MakeMat(arena, 3, 1);
        REQUIRE(mat::Softmax(*out, *in));
        for (f32 v : out->Span()) { CHECK(std::isfinite(v)); }
        ExpectMatNear(*out, {0.09003057f, 0.24472847f, 0.66524096f}, 1e-6f);   // shift invariant
        CHECK(std::fabs(SumOf(*out) - 1.0) < 1e-6);
    }

    SECTION("softmax is over ALL elements, not per row")
    {
        Mat* in  = MakeMat(arena, 2, 2, {1, 2, 3, 4});
        Mat* out = MakeMat(arena, 2, 2);
        REQUIRE(mat::Softmax(*out, *in));
        CHECK(std::fabs(SumOf(*out) - 1.0) < 1e-6);
    }

    SECTION("empty matrix is a no-op, not an out-of-bounds read")
    {
        Mat* e = Mat::Create(arena, 0, 0);
        REQUIRE(e != nullptr);
        CHECK(mat::Softmax(*e, *e));
    }

    SECTION("shape mismatch")
    {
        Mat* in  = MakeMat(arena, 3, 1);
        Mat* out = MakeMat(arena, 1, 3);
        CHECK_FALSE(mat::Softmax(*out, *in));
    }
}

// ------------------------------------------------------------ loss + gradient kernels (known values)

TEST_CASE("ReinforceLoss", "[mat][loss]")
{
    Arena arena(KiB(4));
    Mat* out = MakeMat(arena, 1, 1);

    SECTION("-log(p) * advantage")
    {
        Mat* p   = MakeMat(arena, 1, 1, {0.5f});
        Mat* adv = MakeMat(arena, 1, 1, {2.0f});
        REQUIRE(mat::ReinforceLoss(*out, *p, *adv));
        ExpectMatNear(*out, {1.3862944f}, 1e-5f);
    }

    SECTION("probabilities are clamped at 1e-8 so the loss stays finite")
    {
        Mat* p   = MakeMat(arena, 1, 1, {0.0f});
        Mat* adv = MakeMat(arena, 1, 1, {1.0f});
        REQUIRE(mat::ReinforceLoss(*out, *p, *adv));
        ExpectMatNear(*out, {18.420681f}, 1e-4f);
    }

    SECTION("shape mismatch")
    {
        Mat* p   = MakeMat(arena, 1, 1);
        Mat* adv = MakeMat(arena, 2, 1);
        CHECK_FALSE(mat::ReinforceLoss(*out, *p, *adv));
    }
}

TEST_CASE("Gradient kernels accumulate into their output", "[mat][grad]")
{
    Arena arena(KiB(4));

    SECTION("ReinforceAddGrad: += grad * (-rt / p)")
    {
        Mat* gp   = MakeMat(arena, 1, 1, {0.0f});
        Mat* p    = MakeMat(arena, 1, 1, {0.5f});
        Mat* rt   = MakeMat(arena, 1, 1, {2.0f});
        Mat* grad = MakeMat(arena, 1, 1, {1.0f});
        REQUIRE(mat::ReinforceAddGrad(*gp, *p, *rt, *grad));
        ExpectMatNear(*gp, {-4.0f});
        REQUIRE(mat::ReinforceAddGrad(*gp, *p, *rt, *grad));   // accumulates
        ExpectMatNear(*gp, {-8.0f});
    }

    SECTION("ReluAddGrad passes gradient only where the input was positive")
    {
        Mat* out  = MakeMat(arena, 1, 2, {1.0f, 1.0f});
        Mat* in   = MakeMat(arena, 1, 2, {-1.0f, 2.0f});
        Mat* grad = MakeMat(arena, 1, 2, {5.0f, 7.0f});
        REQUIRE(mat::ReluAddGrad(*out, *in, *grad));
        ExpectMatNear(*out, {1.0f, 8.0f});
    }

    SECTION("ReluAddGrad at exactly zero passes zero gradient")
    {
        Mat* out  = MakeMat(arena, 1, 1, {0.0f});
        Mat* in   = MakeMat(arena, 1, 1, {0.0f});
        Mat* grad = MakeMat(arena, 1, 1, {5.0f});
        REQUIRE(mat::ReluAddGrad(*out, *in, *grad));
        ExpectMatNear(*out, {0.0f});
    }

    SECTION("SoftmaxAddGrad: a constant upstream gradient has zero effect")
    {
        Mat* s    = MakeMat(arena, 3, 1, {0.2f, 0.3f, 0.5f});
        Mat* grad = MakeMat(arena, 3, 1, {3.0f, 3.0f, 3.0f});
        Mat* out  = MakeMat(arena, 3, 1);
        REQUIRE(mat::SoftmaxAddGrad(*out, *s, *grad));
        ExpectMatNear(*out, {0.0f, 0.0f, 0.0f}, 1e-6f);
    }

    SECTION("SoftmaxAddGrad known values")
    {
        Mat* s    = MakeMat(arena, 2, 1, {0.5f, 0.5f});
        Mat* grad = MakeMat(arena, 2, 1, {1.0f, 0.0f});
        Mat* out  = MakeMat(arena, 2, 1);
        REQUIRE(mat::SoftmaxAddGrad(*out, *s, *grad));
        ExpectMatNear(*out, {0.25f, -0.25f}, 1e-6f);
    }
}
