#include "mat.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "prng.hpp"

namespace {
constexpr f32 kProbEpsilon = 1e-8f;

template <bool TransA, bool TransB>
void MatMulKernel(Mat& out, const Mat& a, const Mat& b) noexcept
{
    const u64 m = out.rows;
    const u64 n = out.cols;
    const u64 k_dim = TransA ? a.rows : a.cols;

    for (u64 i = 0; i < m; i++) {
        for (u64 k = 0; k < k_dim; k++) {
            const f32 a_ik = TransA ? a.data[i + k * a.cols]
                                    : a.data[k + i * a.cols];

            for (u64 j = 0; j < n; j++) {
                const f32 b_kj = TransB ? b.data[k + j * b.cols]
                                        : b.data[j + k * b.cols];
                    
                out.data[j + i * n] += a_ik * b_kj;
            }
        }
    }
}

} // namespace

Mat* Mat::Create(Arena& arena, u32 rows, u32 cols) noexcept
{
    f32* data = arena.PushArray<f32>(u64{rows} * cols);
    if (data == nullptr) {
        return nullptr;
    }
    return arena.Emplace<Mat>(rows, cols, data);
}

void Mat::Clear() noexcept
{
    std::ranges::fill(Span(), 0.0f);
}

void Mat::Fill(f32 x) noexcept
{
    std::ranges::fill(Span(), x);
}

void Mat::Scale(f32 s) noexcept
{
    for (f32& v : Span()) {
        v *= s;
    }
}

void Mat::FillRand(f32 lower, f32 upper, Prng& rng)
{
    const f32 range = upper - lower;
    for (f32& v : Span()) {
        v = rng.NextFloat() * range + lower;
    }
}

void Mat::FillRand(f32 lower, f32 upper) 
{
    FillRand(lower, upper, prng::Global());
}

namespace mat {

bool Add(Mat& out, const Mat& a, const Mat& b) noexcept
{
    if (!a.SameShape(b) || !out.SameShape(a)) {
        return false;
    }
    const u64 size = out.Size();
    for (u64 i = 0; i < size; i++) {
        out.data[i] = a.data[i] + b.data[i];
    }
    return true;
}

bool Sub(Mat& out, const Mat& a, const Mat& b) noexcept
{
    if (!a.SameShape(b) || !out.SameShape(a)) {
        return false;
    }
    const u64 size = out.Size();
    for (u64 i = 0; i < size; i++) {
        out.data[i] = a.data[i] - b.data[i];
    }
    return true;
}

bool MatMul(Mat& out, const Mat& a, const Mat& b,
            bool zero_out, bool transpose_a, bool transpose_b) noexcept
{
    const u32 a_rows = transpose_a ? a.cols : a.rows;
    const u32 a_cols = transpose_a ? a.rows : a.cols;
    const u32 b_rows = transpose_b ? b.cols : b.rows;
    const u32 b_cols = transpose_b ? b.rows : b.cols;

    if (a_cols != b_rows) { return false; }
    if (out.rows != a_rows || out.cols != b_cols) { return false; }

    assert(out.data != a.data && out.data != b.data);

    if (zero_out) {
        out.Clear();
    }

    if (!transpose_a && !transpose_b)       { MatMulKernel<false, false>(out, a, b); }
    else if (!transpose_a && transpose_b)   { MatMulKernel<false, true >(out, a, b); }
    else if (transpose_a && !transpose_b)   { MatMulKernel<true,  false>(out, a, b); }
    else                                    { MatMulKernel<true,  true >(out, a, b); }

    return true;
}   

bool Relu(Mat& out, const Mat& in) noexcept
{
    if (!out.SameShape(in)) {
        return false;
    }
    const u64 size = out.Size();
    for (u64 i = 0; i < size; i++) {
        out.data[i] = Max(0.0f, in.data[i]);
    }
    return true;
}

bool Softmax(Mat& out, const Mat& in) noexcept
{
    if (!out.SameShape(in)) {
        return false;
    }
    const u64 size = out.Size();
    if (size == 0) {
        return true;
    }

    f32 max_value = in.data[0];
    for (u64 i = 1; i < size; i++) {
        max_value = Max(max_value, in.data[i]);
    }

    f32 sum = 0.0f;
    for (u64 i = 0; i < size; i++) {
        out.data[i] = std::exp(in.data[i] - max_value);
        sum += out.data[i];
    }

    out.Scale(1.0f / sum);
    return true;
}

bool ReinforceLoss(Mat& out, const Mat& probs, const Mat& advantages) noexcept
{
    if (!probs.SameShape(advantages) || !out.SameShape(advantages)) {
        return false;
    }
    const u64 size = out.Size();
    for (u64 i = 0; i < size; i++) {
        const f32 p = Max(probs.data[i], kProbEpsilon);
        out.data[i] = -std::log(p) * advantages.data[i];
    }

    return true;
}

bool ReinforceAddGrad(Mat& prob_grads, const Mat& probs,
                      const Mat& rt, const Mat& grad) noexcept
{
    if (!probs.SameShape(rt)) { return false; }
    if (!grad.SameShape(probs)) { return false; }
    if (!prob_grads.SameShape(probs)) { return false; }

    const u64 size = probs.Size();
    for (u64 i = 0; i < size; i++) {
        const f32 p = Max(probs.data[i], kProbEpsilon);
        prob_grads.data[i] += grad.data[i] * (-rt.data[i] / p);
    }

    return true;
}

bool ReluAddGrad(Mat& out, const Mat& in, const Mat& grad) noexcept
{
    if (!out.SameShape(in) || !out.SameShape(grad)) {
        return false;
    }
    const u64 size = out.Size();
    for (u64 i = 0; i < size; i++) {
        out.data[i] += in.data[i] > 0.0f ? grad.data[i] : 0.0f;
    }
    
    return true;
}

bool SoftmaxAddGrad(Mat& out, const Mat& softmax_out, const Mat& grad) noexcept
{
    if (!out.SameShape(softmax_out) || !grad.SameShape(softmax_out)) {
        return false;
    }
    const u64 size = softmax_out.Size();

    f32 dot = 0.0f;
    for (u64 i = 0; i < size; i++) {
        dot += grad.data[i] * softmax_out.data[i];
    }
    for (u64 i = 0; i < size; i++) {
        out.data[i] += softmax_out.data[i] * (grad.data[i] - dot);
    }

    return true;
}

} // namespace mat
