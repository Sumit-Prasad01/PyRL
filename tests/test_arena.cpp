#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <utility>

#include <catch2/catch_test_macros.hpp>

#include "arena.hpp"

namespace {

struct alignas(32) Wide { f32 v[8]; };

struct Pt {
    int x, y;
    Pt(int a, int b) : x(a), y(b) {}   // not default constructible: needs Emplace
};

bool IsAligned(const void* p, std::uintptr_t align)
{
    return reinterpret_cast<std::uintptr_t>(p) % align == 0;
}

} // namespace

TEST_CASE("Arena reports capacity and usage", "[arena]")
{
    Arena arena(1024);
    CHECK(arena.Capacity() == 1024);
    CHECK(arena.Used() == 0);
    CHECK(arena.Remaining() == 1024);

    const auto* p = arena.PushArray<f32>(10);
    REQUIRE(p != nullptr);
    CHECK(arena.Used() == 40);
    CHECK(arena.Remaining() == 1024 - 40);
}

TEST_CASE("Arena memory is zeroed by default and aligned", "[arena]")
{
    Arena arena(4096);

    const auto* z = arena.PushArray<u32>(64);
    REQUIRE(z != nullptr);
    CHECK(std::all_of(z, z + 64, [](u32 v) { return v == 0; }));
    CHECK(IsAligned(z, alignof(u32)));

    SECTION("natural alignment of typed pushes after an odd-sized push")
    {
        const void* odd = arena.Push(3);
        REQUIRE(odd != nullptr);
        const auto* d = arena.PushArray<f64>(1);
        REQUIRE(d != nullptr);
        CHECK(IsAligned(d, alignof(f64)));
    }

    SECTION("over-aligned types are honored")
    {
        const void* odd = arena.Push(1);
        REQUIRE(odd != nullptr);
        const Wide* w = arena.PushStruct<Wide>();
        REQUIRE(w != nullptr);
        CHECK(IsAligned(w, 32));
    }

    SECTION("explicit 64-byte alignment is exact (base block is 64-aligned)")
    {
        const void* odd = arena.Push(5);
        REQUIRE(odd != nullptr);
        const void* p = arena.Push(8, true, 64);
        REQUIRE(p != nullptr);
        CHECK(IsAligned(p, 64));
    }
}

TEST_CASE("Arena returns nullptr when full instead of overrunning", "[arena]")
{
    Arena arena(64);
    const void* a = arena.Push(64);
    REQUIRE(a != nullptr);
    CHECK(arena.Remaining() == 0);

    const void* b = arena.Push(1);
    CHECK(b == nullptr);

    const void* huge = arena.Push(std::numeric_limits<u64>::max());
    CHECK(huge == nullptr);
}

TEST_CASE("PushArray guards against size overflow", "[arena]")
{
    Arena arena(1024);
    const u64 too_many = std::numeric_limits<u64>::max() / sizeof(u64) + 1;
    const auto* p = arena.PushArray<u64>(too_many);   // sizeof(T) * n would wrap to 0
    CHECK(p == nullptr);
    CHECK(arena.Used() == 0);
}

TEST_CASE("Reset reclaims the arena and re-zeroes on request", "[arena]")
{
    Arena arena(256);
    auto* p = static_cast<u8*>(arena.Push(128, /*zero=*/false));
    REQUIRE(p != nullptr);
    std::memset(p, 0xFF, 128);

    arena.Reset();
    CHECK(arena.Used() == 0);

    auto* q = static_cast<u8*>(arena.Push(128, /*zero=*/true));
    REQUIRE(q != nullptr);
    CHECK(q == p);   // same memory is reused
    CHECK(std::all_of(q, q + 128, [](u8 b) { return b == 0; }));
}

TEST_CASE("Emplace constructs non-trivial types in place", "[arena]")
{
    Arena arena(256);
    Pt* p = arena.Emplace<Pt>(3, 4);
    REQUIRE(p != nullptr);
    CHECK(p->x == 3);
    CHECK(p->y == 4);

    Arena tiny(2);
    CHECK(tiny.Emplace<Pt>(1, 2) == nullptr);
}

TEST_CASE("Arena is movable and the moved-from arena is empty", "[arena]")
{
    Arena a(512);
    auto* p = a.PushArray<u32>(4);
    REQUIRE(p != nullptr);

    Arena b(std::move(a));
    CHECK(a.Capacity() == 0);   // NOLINT(bugprone-use-after-move): documented behavior
    CHECK(a.Used() == 0);
    CHECK(b.Capacity() == 512);
    CHECK(b.Used() == 16);
    CHECK(b.PushArray<u32>(4) != nullptr);

    Arena c(8);
    c = std::move(b);
    CHECK(c.Capacity() == 512);
    CHECK(b.Capacity() == 0);   // NOLINT(bugprone-use-after-move)

    // A moved-from arena can still refuse allocations safely.
    const void* none = a.Push(1);
    CHECK(none == nullptr);
}

TEST_CASE("Constructing an impossibly large arena throws", "[arena]")
{
    REQUIRE_THROWS_AS(Arena(std::numeric_limits<u64>::max()), std::bad_alloc);
}

TEST_CASE("Arena tracks peak memory usage", "[arena]")
{
    Arena arena(1024);
    CHECK(arena.PeakUsed() == 0);

    const auto* p1 = arena.PushArray<f32>(10); // 40 bytes
    REQUIRE(p1 != nullptr);
    CHECK(arena.Used() == 40);
    CHECK(arena.PeakUsed() == 40);

    const auto* p2 = arena.PushArray<f32>(20); // 80 bytes => total 120
    REQUIRE(p2 != nullptr);
    CHECK(arena.Used() == 120);
    CHECK(arena.PeakUsed() == 120);

    arena.Reset();
    CHECK(arena.Used() == 0);
    CHECK(arena.PeakUsed() == 120); // Peak is retained after Reset

    const auto* p3 = arena.PushArray<f32>(5); // 20 bytes
    REQUIRE(p3 != nullptr);
    CHECK(arena.Used() == 20);
    CHECK(arena.PeakUsed() == 120); // Peak remains 120
}

TEST_CASE("Arena Save and Restore checkpoints", "[arena]")
{
    Arena arena(1024);
    REQUIRE(arena.PushArray<u32>(10) != nullptr); // 40 bytes
    const u64 checkpoint = arena.Save();
    CHECK(checkpoint == 40);

    REQUIRE(arena.PushArray<u32>(25) != nullptr); // 100 bytes => 140
    CHECK(arena.Used() == 140);

    arena.Restore(checkpoint);
    CHECK(arena.Used() == 40);
    CHECK(arena.Remaining() == 1024 - 40);
}

TEST_CASE("ScopedCheckpoint automatically restores position on scope exit", "[arena]")
{
    Arena arena(1024);
    REQUIRE(arena.PushArray<f64>(5) != nullptr); // 40 bytes
    CHECK(arena.Used() == 40);

    {
        ScopedCheckpoint sc(arena);
        REQUIRE(arena.PushArray<f64>(10) != nullptr); // 80 bytes => 120
        CHECK(arena.Used() == 120);
    }

    CHECK(arena.Used() == 40);
}

#ifndef NDEBUG
TEST_CASE("Arena debug poisoning catches use of reset/restored memory", "[arena]")
{
    Arena arena(256);
    auto* p = static_cast<u8*>(arena.Push(64, /*zero=*/false));
    REQUIRE(p != nullptr);
    std::memset(p, 0x12, 64);

    arena.Reset();
    // After Reset, memory is poisoned with 0xCD
    CHECK(p[0] == 0xCD);
    CHECK(p[63] == 0xCD);

    // Save/Restore checkpoint poisoning
    auto* q = static_cast<u8*>(arena.Push(32, /*zero=*/true));
    REQUIRE(q != nullptr);
    const u64 cp = arena.Save();
    auto* r = static_cast<u8*>(arena.Push(32, /*zero=*/true));
    REQUIRE(r != nullptr);
    std::memset(r, 0x55, 32);

    arena.Restore(cp);
    CHECK(r[0] == 0xCD);
    CHECK(r[31] == 0xCD);
}
#endif

