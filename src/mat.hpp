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

template <typename T = f32>
struct BasicMat {
    u32 rows;
    u32 cols;
    T*  data;

    BasicMat() noexcept = default;
    constexpr BasicMat(u32 r, u32 c, T* d) noexcept : rows(r), cols(c), data(d) {}

    [[nodiscard]] static BasicMat* Create(Arena& arena, u32 rows, u32 cols) noexcept
    {
        BasicMat* m = arena.PushStruct<BasicMat>();
        if (m == nullptr) { return nullptr; }
        m->rows = rows;
        m->cols = cols;
        m->data = arena.PushArray<T>(u64{rows} * cols);
        if (m->data == nullptr && rows > 0 && cols > 0) { return nullptr; }
        return m;
    }

    [[nodiscard]] constexpr u64 Size() const noexcept { return u64{rows} * cols; }

    [[nodiscard]] std::span<T>         Span() noexcept          { return {data, Size()}; }
    [[nodiscard]] std::span<const T>   Span() const noexcept    { return {data, Size()}; }

    [[nodiscard]] T& At(u32 r, u32 c) noexcept
    {
        assert(r < rows && c < cols);
        return data[u64{r} * cols + c];
    }
    [[nodiscard]] const T& At(u32 r, u32 c) const noexcept
    {
        assert(r < rows && c < cols);
        return data[u64{r} * cols + c];
    }
    [[nodiscard]] constexpr bool SameShape(const BasicMat& o) const noexcept
    {
        return rows == o.rows && cols == o.cols;
    }

    void Clear() noexcept
    {
        if (data != nullptr) {
            std::fill_n(data, Size(), T{0});
        }
    }
    void Fill(T x) noexcept
    {
        if (data != nullptr) {
            std::fill_n(data, Size(), x);
        }
    }
    void Scale(T s) noexcept
    {
        if (data != nullptr) {
            for (u64 i = 0, n = Size(); i < n; ++i) {
                data[i] *= s;
            }
        }
    }

    void FillRand(T lower, T upper);
    void FillRand(T lower, T upper, Prng& rng);
};

using Mat   = BasicMat<f32>;
using Mat64 = BasicMat<f64>;

namespace mat {

template <typename T = f32>
[[nodiscard]] bool Add(BasicMat<T>& out, const BasicMat<T>& a, const BasicMat<T>& b) noexcept;

template <typename T = f32>
[[nodiscard]] bool Sub(BasicMat<T>& out, const BasicMat<T>& a, const BasicMat<T>& b) noexcept;

template <typename T = f32>
[[nodiscard]] bool MatMul(BasicMat<T>& out, const BasicMat<T>& a, const BasicMat<T>& b,
                          bool zero_out = true,
                          bool transpose_a = false,
                          bool transpose_b = false) noexcept;

template <typename T = f32>
[[nodiscard]] bool Relu(BasicMat<T>& out, const BasicMat<T>& in) noexcept;

template <typename T = f32>
[[nodiscard]] bool Softmax(BasicMat<T>& out, const BasicMat<T>& in) noexcept;

template <typename T = f32>
[[nodiscard]] bool LogSoftmax(BasicMat<T>& out, const BasicMat<T>& in) noexcept;

template <typename T = f32>
[[nodiscard]] bool ReinforceLoss(BasicMat<T>& out, const BasicMat<T>& probs,
                                 const BasicMat<T>& advantages) noexcept;

template <typename T = f32>
[[nodiscard]] bool LogProbLoss(BasicMat<T>& out, const BasicMat<T>& log_probs,
                               const BasicMat<T>& target) noexcept;

template <typename T = f32>
[[nodiscard]] bool CrossEntropy(BasicMat<T>& out, const BasicMat<T>& logits,
                                const BasicMat<T>& target) noexcept;

template <typename T = f32>
[[nodiscard]] bool ReinforceAddGrad(BasicMat<T>& prob_grads, const BasicMat<T>& probs,
                                    const BasicMat<T>& rt, const BasicMat<T>& grad) noexcept;

template <typename T = f32>
[[nodiscard]] bool LogProbLossAddGrad(BasicMat<T>& log_prob_grads, const BasicMat<T>& target,
                                      const BasicMat<T>& grad) noexcept;

template <typename T = f32>
[[nodiscard]] bool CrossEntropyAddGrad(BasicMat<T>& out, const BasicMat<T>& logits,
                                       const BasicMat<T>& target, const BasicMat<T>& grad) noexcept;

template <typename T = f32>
[[nodiscard]] bool ReluAddGrad(BasicMat<T>& out, const BasicMat<T>& in, const BasicMat<T>& grad) noexcept;

template <typename T = f32>
[[nodiscard]] bool SoftmaxAddGrad(BasicMat<T>& out, const BasicMat<T>& softmax_out,
                                  const BasicMat<T>& grad) noexcept;

template <typename T = f32>
[[nodiscard]] bool LogSoftmaxAddGrad(BasicMat<T>& out, const BasicMat<T>& log_softmax_out,
                                     const BasicMat<T>& grad) noexcept;
}
