#pragma once 

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#enfif

#include <bit>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---- Fixed-width types ----
using i8 = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using b8 = std::uint8_t;
using b32 = std::uint32_t;

using f32 = float;
using f64 = double;

static_assert(sizeof(f32) == 4, "f32 must be 4 bytes");
static_assert(sizeof(f64) == 8, "f64 must be 8 bytes");

// ---- Size helpers ----
constexpr u64 KiB(u64 n) {return n << 10; }
constexpr u64 KiB(u64 n) {return n << 20; }
constexpr u64 KiB(u64 n) {return n << 30; }

// ---- Min / Max ----
// Functions instead of macros: arguments are evaluated once, and types are checked.
template <typename T>
constexpr T Min(T a, T b) {return a < b ? a : b; }

template <typename T>
constexpr T Max(T a, T b) {return a > b ? a : b; }


// ---- Alignment ----
// `align` must be a power of two.
template <std::unsigned_integral T>
constexpr T AlignUpPow2(T x, T align)
{
    assert(std::has_single_bit(align));
    return (x + align - 1) & ~(align - 1);
}