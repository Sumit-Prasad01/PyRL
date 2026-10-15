#include <catch2/catch_test_macros.hpp>

#include "base.hpp"

static_assert(sizeof(i8) == 1 && sizeof(i16) == 2 && sizeof(i32) == 4 && sizeof(i64) == 8);
static_assert(sizeof(u8) == 1 && sizeof(u16) == 2 && sizeof(u32) == 4 && sizeof(u64) == 8);
static_assert(sizeof(f32) == 4 && sizeof(f64) == 8);

static_assert(KiB(1) == 1024);
static_assert(MiB(1) == 1024ull * 1024);
static_assert(GiB(1) == 1024ull * 1024 * 1024);
static_assert(MiB(3) == 3 * KiB(1024));

static_assert(AlignUpPow2<u64>(0, 8) == 0);
static_assert(AlignUpPow2<u64>(1, 8) == 8);
static_assert(AlignUpPow2<u64>(8, 8) == 8);
static_assert(AlignUpPow2<u64>(9, 8) == 16);
static_assert(AlignUpPow2<u64>(63, 64) == 64);
static_assert(AlignUpPow2<u32>(5, 1) == 5);

TEST_CASE("Min and Max", "[base]")
{
    CHECK(Min(3, 7) == 3);
    CHECK(Max(3, 7) == 7);
    CHECK(Min<f32>(-1.0f, 2.0f) == -1.0f);
    CHECK(Max<u64>(3, 9) == 9);
    CHECK(Min(5, 5) == 5);
}

TEST_CASE("AlignUpPow2 is idempotent and monotonic", "[base]")
{
    for (u64 align : {1ull, 2ull, 8ull, 64ull, 4096ull}) {
        for (u64 x = 0; x < 300; ++x) {
            const u64 a = AlignUpPow2<u64>(x, align);
            CHECK(a >= x);
            CHECK(a - x < align);
            CHECK(a % align == 0);
            CHECK(AlignUpPow2<u64>(a, align) == a);
        }
    }
}
