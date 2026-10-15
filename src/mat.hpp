#pragma once

#include <cassert>
#include <span>

#include "arena.hpp"
#include "base.hpp"

class Prng;

// Row-major f32 matrix backed by an Arena.
//
// - Mat is a *view*: it does not own `data`. The Arena that created it owns
//   the memory, so a Mat is valid only until that Arena is Reset()/destroyed.
// - Copying a Mat copies the view (shallow), not the elements.
// - Trivially destructible, so it can live inside an Arena.

struct Mat {
    u32 rows;
    u32 cols;
    f32* data;

    constexpr Mat(u32 r, u32 c, f32* d) noexcept : rows(r), cols(c), data(d) {}

    [[nodiscard]] static Mat* Create(Arena& arena, u32 rows, u32 cols) noexcept;

    [[nodiscard]] constexpr u64 Size() const noexcept { return u64{rows} * cols; }

    [[nodiscard]] std::span<f32>         Span() noexcept          { return {data, Size()}; }
    [[nodiscard]] std::span<const f32>   Span() const noexcept    { return {data, Size()}; }

    [[nodiscard]] f32& At(u32 r, u32 c) noexcept
    {
        assert(r < rows && c < cols);
        return data[u64{r} * cols + c];
    }
    [[nodiscard]] const f32& At(u32 r, u32 c) const noexcept
    {
        assert(r < rows && c < cols);
        return data[u64{r} * cols + c];
    }
    [[nodiscard]] constexpr bool SameShape(const Mat& o) const noexcept
    {
        return rows == o.rows && cols == o.cols;
    }

    void Clear() noexcept;
    void Fill(f32 x) noexcept;
    void Scale(f32 s) noexcept;

    void FillRand(f32 lower, f32 upper);
    void FillRand(f32 lower, f32 upper, Prng& rng);
};

namespace mat {

[[nodiscard]] bool Add(Mat& out, const Mat& a, const Mat& b) noexcept;
[[nodiscard]] bool Sub(Mat& out, const Mat& a, const Mat& b) noexcept;

[[nodiscard]] bool MatMul(Mat& out, const Mat& a, const Mat& b,
                          bool zero_out = true,
                          bool transpose_a = false,
                          bool transpose_b = false) noexcept;

[[nodiscard]] bool Relu(Mat& out, const Mat& in) noexcept;

[[nodiscard]] bool Softmax(Mat& out, const Mat& in) noexcept;

[[nodiscard]] bool ReinforceLoss(Mat& out, const Mat& probs,
                                 const Mat& advantages) noexcept;

[[nodiscard]] bool ReinforceAddGrad(Mat& prob_grads, const Mat& probs,
                                    const Mat& rt, const Mat& grad) noexcept;

[[nodiscard]] bool ReluAddGrad(Mat& out, const Mat& in, const Mat& grad) noexcept;

[[nodiscard]] bool SoftmaxAddGrad(Mat& out, const Mat& softmax_out,
                                  const Mat& grad) noexcept;
}
