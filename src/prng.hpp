#pragma once

#include <bit>
#include <cassert>
#include <concepts>
#include <limits>
#include <random>

#include "base.hpp"

class Prng {
public:
    using result_type = u32;

    constexpr Prng() noexcept = default;
    constexpr Prng(u64 initstate, u64 initseq) noexcept { Seed(initstate, initseq); }

    constexpr void Seed(u64 initstate, u64 initseq) noexcept
    {
        state_ = 0;
        inc_ = (initseq << 1u) | 1u;
        Next();
        state_ += initstate;
        Next();
    }

    // Uniform u32 over the full range.
    constexpr u32 Next() noexcept
    {
        const u64 oldstate = state_;
        state_ = oldstate * kMultiplier + inc_;
        const u32 xorshifted = static_cast<u32>(((oldstate >> 18u) ^ oldstate) >> 27u);
        const int rot = static_cast<int>(oldstate >> 59u);
        return std::rotr(xorshifted, rot);
    }

    // Unbiased uniform integer in [0, range). Returns 0 if range == 0.
    // Lemire's multiply-shift method with rejection (no modulo bias).
    constexpr u32 NextBelow(u32 range) noexcept
    {
        assert(range > 0);
        if (range == 0) return 0;

        u64 m = static_cast<u64>(Next()) * range;
        u32 low = static_cast<u32>(m);
        if (low < range) {
            const u32 threshold = (0u - range) % range;
            while (low < threshold) {
                m = static_cast<u64>(Next()) * range;
                low = static_cast<u32>(m);
            }
        }
        return static_cast<u32>(m >> 32);
    }

    // Uniform f32 in [0, 1). Uses the top 24 bits (the full f32 mantissa).
    constexpr f32 NextFloat() noexcept
    {
        return static_cast<f32>(Next() >> 8) * (1.0f / 16777216.0f); // 2^-24
    }

    // Uniform f64 in [0, 1). Uses 53 bits (the full f64 mantissa) from two draws.
    constexpr f64 NextDouble() noexcept
    {
        const u64 high = static_cast<u64>(Next()) << 21;
        const u64 low  = static_cast<u64>(Next()) >> 11;
        return static_cast<f64>(high | low) * (1.0 / 9007199254740992.0); // 2^-53
    }

    // UniformRandomBitGenerator interface
    constexpr result_type operator()() noexcept { return Next(); }
    static constexpr result_type min() noexcept { return std::numeric_limits<u32>::min(); }
    static constexpr result_type max() noexcept { return std::numeric_limits<u32>::max(); }

    friend constexpr bool operator==(const Prng&, const Prng&) noexcept = default;

private:
    static constexpr u64 kMultiplier = 6364136223846793005ULL;

    u64 state_ = 0x853c49e6748fea9bULL;
    u64 inc_   = 0xda3e39cb94b95bdbULL;
};

static_assert(std::uniform_random_bit_generator<Prng>);

// ---- Global convenience generator ----
// One instance per thread (thread_local), so there are no data races.
// Every thread starts from the same default state; call prng::Seed on each
// thread if you need distinct streams.
namespace prng {

Prng& Global() noexcept;

void Seed(u64 initstate, u64 initseq) noexcept;
u32 Rand() noexcept;
u32 RandN(u32 range) noexcept;
f32 RandF() noexcept;

}
