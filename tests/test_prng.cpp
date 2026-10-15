#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "prng.hpp"

// Compile-time check: the generator is usable in constant expressions.
constexpr u32 kFirstOutput = [] { Prng p(42u, 54u); return p.Next(); }();
static_assert(kFirstOutput == 0xa15c02b7u);

TEST_CASE("Prng reproduces the reference PCG32 sequence", "[prng]")
{
    // Output of pcg32 seeded with (initstate=42, initseq=54), taken from the
    // pcg-c-basic demo. If this fails, the seeding or the rotation is wrong.
    Prng rng(42u, 54u);
    constexpr std::array<u32, 6> expected{
        0xa15c02b7u, 0x7b47f409u, 0xba1d3330u, 0x83d2f293u, 0xbfa4784bu, 0xcbed606eu};
    for (u32 e : expected) { CHECK(rng.Next() == e); }
}

TEST_CASE("Prng is deterministic and seedable", "[prng]")
{
    Prng a(123, 456), b(123, 456), c(124, 456), d(123, 457);

    bool same_ab = true, same_ac = true, same_ad = true;
    for (int i = 0; i < 1000; ++i) {
        const u32 va = a.Next();
        same_ab &= (va == b.Next());
        same_ac &= (va == c.Next());
        same_ad &= (va == d.Next());
    }
    CHECK(same_ab);
    CHECK_FALSE(same_ac);   // different state
    CHECK_FALSE(same_ad);   // different stream

    // Re-seeding restarts the sequence.
    Prng e(9, 9);
    const u32 first = e.Next();
    e.Next();
    e.Seed(9, 9);
    CHECK(e.Next() == first);
}

TEST_CASE("Prng copies are independent and equality compares state", "[prng]")
{
    Prng a(1, 2);
    Prng b = a;
    CHECK(a == b);
    const u32 va = a.Next();
    CHECK_FALSE(a == b);
    CHECK(b.Next() == va);
    CHECK(a == b);
}

TEST_CASE("NextFloat lies in [0, 1) and is roughly uniform", "[prng][statistical]")
{
    Prng rng(7, 7);
    constexpr int kN = 1'000'000;
    f64 sum = 0.0;
    f32 lo = 1.0f, hi = 0.0f;
    for (int i = 0; i < kN; ++i) {
        const f32 x = rng.NextFloat();
        lo = std::min(lo, x);
        hi = std::max(hi, x);
        sum += x;
    }
    CHECK(lo >= 0.0f);
    CHECK(hi < 1.0f);
    CHECK(std::fabs(sum / kN - 0.5) < 5e-3);   // standard error is ~3e-4
}

TEST_CASE("NextBelow stays in range and is unbiased", "[prng][statistical]")
{
    Prng rng(11, 3);

    SECTION("range 1 always yields 0")
    {
        for (int i = 0; i < 100; ++i) { CHECK(rng.NextBelow(1) == 0); }
    }

    SECTION("huge ranges stay in bounds")
    {
        constexpr u32 kRange = 0xFFFFFFFFu;
        for (int i = 0; i < 1000; ++i) { CHECK(rng.NextBelow(kRange) < kRange); }
    }

    SECTION("chi-squared uniformity for a non power of two range")
    {
        constexpr u32 kRange = 10;
        constexpr int kN = 100'000;
        std::array<int, kRange> counts{};
        for (int i = 0; i < kN; ++i) {
            const u32 v = rng.NextBelow(kRange);
            REQUIRE(v < kRange);
            counts[v]++;
        }
        const f64 expected = static_cast<f64>(kN) / kRange;
        f64 chi2 = 0.0;
        for (int c : counts) { chi2 += (c - expected) * (c - expected) / expected; }
        // 9 degrees of freedom: 27.9 is p = 0.001. 40 is far in the tail.
        CHECK(chi2 < 40.0);
    }
}

TEST_CASE("Prng works with the standard library", "[prng]")
{
    Prng rng(5, 5);

    SECTION("std::shuffle yields a permutation")
    {
        std::vector<int> v(100);
        std::iota(v.begin(), v.end(), 0);
        const auto original = v;
        std::shuffle(v.begin(), v.end(), rng);
        CHECK(v != original);
        std::sort(v.begin(), v.end());
        CHECK(v == original);
    }

    SECTION("uniform_int_distribution respects its bounds")
    {
        std::uniform_int_distribution<int> dist(0, 5);
        for (int i = 0; i < 1000; ++i) {
            const int x = dist(rng);
            CHECK((x >= 0 && x <= 5));
        }
    }

    SECTION("normal_distribution has mean ~0 and stddev ~1")
    {
        std::normal_distribution<f32> dist{};
        constexpr int kN = 20'000;
        f64 sum = 0.0, sq = 0.0;
        for (int i = 0; i < kN; ++i) {
            const f64 x = dist(rng);
            sum += x;
            sq += x * x;
        }
        const f64 mean = sum / kN;
        const f64 var  = sq / kN - mean * mean;
        CHECK(std::fabs(mean) < 0.05);
        CHECK(std::fabs(std::sqrt(var) - 1.0) < 0.05);
    }
}

TEST_CASE("The global generator is seedable and thread-local", "[prng]")
{
    prng::Seed(1, 2);
    const u32 a = prng::Rand();
    prng::Seed(1, 2);
    CHECK(prng::Rand() == a);

    prng::Seed(5, 6);
    const u32 x = prng::RandN(100);
    Prng reference(5, 6);
    CHECK(x == reference.NextBelow(100));

    prng::Seed(5, 6);
    CHECK(prng::RandF() == Prng(5, 6).NextFloat());

    // A new thread starts from the default state regardless of this thread's state.
    prng::Seed(77, 88);
    u32 from_thread = 0;
    std::thread t([&] { from_thread = prng::Rand(); });
    t.join();
    CHECK(from_thread == Prng{}.Next());
}
