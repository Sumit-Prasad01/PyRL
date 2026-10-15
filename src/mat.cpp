#include "mat.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "prng.hpp"

namespace {

template <typename T, bool TransA, bool TransB>
void MatMulKernel(BasicMat<T>& out, const BasicMat<T>& a, const BasicMat<T>& b) noexcept
{
    const u64 m = out.rows;
    const u64 n = out.cols;
    const u64 k_dim = TransA ? a.rows : a.cols;

    for (u64 i = 0; i < m; i++) {
        for (u64 k = 0; k < k_dim; k++) {
            const T a_ik = TransA ? a.data[i + k * a.cols]
                                  : a.data[k + i * a.cols];

            for (u64 j = 0; j < n; j++) {
                const T b_kj = TransB ? b.data[k + j * b.cols]
                                      : b.data[j + k * b.cols];
                    
                out.data[j + i * n] += a_ik * b_kj;
            }
        }
    }
}

template <typename T>
inline bool ParseTarget(const BasicMat<T>& target, u64 num_classes, u32& out_action, T& out_adv) noexcept
{
    if (num_classes == 0) { return false; }
    if (target.Size() == 2) {
        const i32 a = static_cast<i32>(target.data[0] >= T{0} ? target.data[0] + T{0.5} : -1);
        if (a < 0 || static_cast<u64>(a) >= num_classes) { return false; }
        out_action = static_cast<u32>(a);
        out_adv = target.data[1];
        return true;
    }
    if (target.Size() == num_classes) {
        u32 best_idx = 0;
        T max_abs = T{0};
        for (u64 i = 0; i < num_classes; ++i) {
            const T abs_val = std::fabs(target.data[i]);
            if (abs_val > max_abs) {
                max_abs = abs_val;
                best_idx = static_cast<u32>(i);
            }
        }
        out_action = best_idx;
        out_adv = target.data[best_idx];
        return true;
    }
    return false;
}

} // namespace

template <>
void BasicMat<f32>::FillRand(f32 lower, f32 upper, Prng& rng)
{
    const f32 range = upper - lower;
    for (f32& v : Span()) {
        v = rng.NextFloat() * range + lower;
    }
}

template <>
void BasicMat<f64>::FillRand(f64 lower, f64 upper, Prng& rng)
{
    const f64 range = upper - lower;
    for (f64& v : Span()) {
        v = rng.NextDouble() * range + lower;
    }
}

template <typename T>
void BasicMat<T>::FillRand(T lower, T upper) 
{
    FillRand(lower, upper, prng::Global());
}

namespace mat {

template <typename T>
bool Add(BasicMat<T>& out, const BasicMat<T>& a, const BasicMat<T>& b) noexcept
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

template <typename T>
bool Sub(BasicMat<T>& out, const BasicMat<T>& a, const BasicMat<T>& b) noexcept
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

template <typename T>
bool MatMul(BasicMat<T>& out, const BasicMat<T>& a, const BasicMat<T>& b,
            bool zero_out, bool transpose_a, bool transpose_b) noexcept
{
    const u32 a_rows = transpose_a ? a.cols : a.rows;
    const u32 a_cols = transpose_a ? a.rows : a.cols;
    const u32 b_rows = transpose_b ? b.cols : b.rows;
    const u32 b_cols = transpose_b ? b.rows : b.cols;

    if (a_cols != b_rows || out.rows != a_rows || out.cols != b_cols) {
        return false;
    }
    if (zero_out) {
        out.Clear();
    }

    if (!transpose_a && !transpose_b) {
        MatMulKernel<T, false, false>(out, a, b);
    } else if (transpose_a && !transpose_b) {
        MatMulKernel<T, true, false>(out, a, b);
    } else if (!transpose_a && transpose_b) {
        MatMulKernel<T, false, true>(out, a, b);
    } else {
        MatMulKernel<T, true, true>(out, a, b);
    }
    return true;
}

template <typename T>
bool Relu(BasicMat<T>& out, const BasicMat<T>& in) noexcept
{
    if (!out.SameShape(in)) {
        return false;
    }
    const u64 size = out.Size();
    for (u64 i = 0; i < size; i++) {
        out.data[i] = in.data[i] > T{0} ? in.data[i] : T{0};
    }
    return true;
}

template <typename T>
bool Softmax(BasicMat<T>& out, const BasicMat<T>& in) noexcept
{
    if (!out.SameShape(in)) {
        return false;
    }
    const u64 size = out.Size();
    if (size == 0) {
        return true;
    }

    T max_value = in.data[0];
    for (u64 i = 1; i < size; i++) {
        if (in.data[i] > max_value) { max_value = in.data[i]; }
    }

    T sum = T{0};
    for (u64 i = 0; i < size; i++) {
        out.data[i] = std::exp(in.data[i] - max_value);
        sum += out.data[i];
    }

    out.Scale(T{1} / sum);
    return true;
}

template <typename T>
bool LogSoftmax(BasicMat<T>& out, const BasicMat<T>& in) noexcept
{
    if (!out.SameShape(in)) { return false; }
    const u64 size = out.Size();
    if (size == 0) { return true; }

    T max_val = in.data[0];
    for (u64 i = 1; i < size; ++i) {
        if (in.data[i] > max_val) { max_val = in.data[i]; }
    }

    T sum_exp = T{0};
    for (u64 i = 0; i < size; ++i) {
        sum_exp += std::exp(in.data[i] - max_val);
    }

    const T log_sum_exp = max_val + std::log(sum_exp);
    for (u64 i = 0; i < size; ++i) {
        out.data[i] = in.data[i] - log_sum_exp;
    }
    return true;
}

template <typename T>
bool ReinforceLoss(BasicMat<T>& out, const BasicMat<T>& probs, const BasicMat<T>& advantages) noexcept
{
    if (!probs.SameShape(advantages) || !out.SameShape(advantages)) {
        return false;
    }
    const u64 size = out.Size();
    for (u64 i = 0; i < size; i++) {
        const T p = probs.data[i] > T{1e-8} ? probs.data[i] : T{1e-8};
        out.data[i] = -std::log(p) * advantages.data[i];
    }

    return true;
}

template <typename T>
bool ReinforceAddGrad(BasicMat<T>& prob_grads, const BasicMat<T>& probs,
                      const BasicMat<T>& rt, const BasicMat<T>& grad) noexcept
{
    if (!probs.SameShape(rt)) { return false; }
    if (!grad.SameShape(probs)) { return false; }
    if (!prob_grads.SameShape(probs)) { return false; }

    const u64 size = probs.Size();
    for (u64 i = 0; i < size; i++) {
        const T p = probs.data[i] > T{1e-8} ? probs.data[i] : T{1e-8};
        prob_grads.data[i] += grad.data[i] * (-rt.data[i] / p);
    }

    return true;
}

template <typename T>
bool ReluAddGrad(BasicMat<T>& out, const BasicMat<T>& in, const BasicMat<T>& grad) noexcept
{
    if (!out.SameShape(in) || !out.SameShape(grad)) {
        return false;
    }
    const u64 size = out.Size();
    for (u64 i = 0; i < size; i++) {
        out.data[i] += in.data[i] > T{0} ? grad.data[i] : T{0};
    }
    
    return true;
}

template <typename T>
bool SoftmaxAddGrad(BasicMat<T>& out, const BasicMat<T>& softmax_out, const BasicMat<T>& grad) noexcept
{
    if (!out.SameShape(softmax_out) || !grad.SameShape(softmax_out)) {
        return false;
    }
    const u64 size = softmax_out.Size();

    T dot = T{0};
    for (u64 i = 0; i < size; i++) {
        dot += grad.data[i] * softmax_out.data[i];
    }
    for (u64 i = 0; i < size; i++) {
        out.data[i] += softmax_out.data[i] * (grad.data[i] - dot);
    }

    return true;
}

template <typename T>
bool LogSoftmaxAddGrad(BasicMat<T>& out, const BasicMat<T>& log_softmax_out, const BasicMat<T>& grad) noexcept
{
    if (!out.SameShape(log_softmax_out) || !grad.SameShape(log_softmax_out)) {
        return false;
    }
    const u64 size = log_softmax_out.Size();
    if (size == 0) { return true; }

    T sum_grad = T{0};
    for (u64 i = 0; i < size; ++i) {
        sum_grad += grad.data[i];
    }

    for (u64 i = 0; i < size; ++i) {
        out.data[i] += grad.data[i] - std::exp(log_softmax_out.data[i]) * sum_grad;
    }
    return true;
}

template <typename T>
bool LogProbLoss(BasicMat<T>& out, const BasicMat<T>& log_probs, const BasicMat<T>& target) noexcept
{
    if (out.rows != 1 || out.cols != 1) { return false; }
    u32 action = 0;
    T adv = T{0};
    if (!ParseTarget(target, log_probs.Size(), action, adv)) { return false; }

    out.data[0] = -log_probs.data[action] * adv;
    return true;
}

template <typename T>
bool LogProbLossAddGrad(BasicMat<T>& log_prob_grads, const BasicMat<T>& target, const BasicMat<T>& grad) noexcept
{
    if (grad.rows != 1 || grad.cols != 1) { return false; }
    u32 action = 0;
    T adv = T{0};
    if (!ParseTarget(target, log_prob_grads.Size(), action, adv)) { return false; }

    log_prob_grads.data[action] += -adv * grad.data[0];
    return true;
}

template <typename T>
bool CrossEntropy(BasicMat<T>& out, const BasicMat<T>& logits, const BasicMat<T>& target) noexcept
{
    if (out.rows != 1 || out.cols != 1) { return false; }
    const u64 size = logits.Size();
    u32 action = 0;
    T weight = T{1};
    if (!ParseTarget(target, size, action, weight)) { return false; }

    T max_val = logits.data[0];
    for (u64 i = 1; i < size; ++i) {
        if (logits.data[i] > max_val) { max_val = logits.data[i]; }
    }

    T sum_exp = T{0};
    for (u64 i = 0; i < size; ++i) {
        sum_exp += std::exp(logits.data[i] - max_val);
    }

    const T log_sum_exp = max_val + std::log(sum_exp);
    out.data[0] = weight * (log_sum_exp - logits.data[action]);
    return true;
}

template <typename T>
bool CrossEntropyAddGrad(BasicMat<T>& out, const BasicMat<T>& logits, const BasicMat<T>& target, const BasicMat<T>& grad) noexcept
{
    if (grad.rows != 1 || grad.cols != 1) { return false; }
    if (!out.SameShape(logits)) { return false; }
    const u64 size = logits.Size();
    u32 action = 0;
    T weight = T{1};
    if (!ParseTarget(target, size, action, weight)) { return false; }

    T max_val = logits.data[0];
    for (u64 i = 1; i < size; ++i) {
        if (logits.data[i] > max_val) { max_val = logits.data[i]; }
    }

    T sum_exp = T{0};
    for (u64 i = 0; i < size; ++i) {
        sum_exp += std::exp(logits.data[i] - max_val);
    }
    const T inv_sum = T{1} / sum_exp;
    const T g = grad.data[0] * weight;

    for (u64 i = 0; i < size; ++i) {
        const T p = std::exp(logits.data[i] - max_val) * inv_sum;
        const T delta = (i == action) ? T{1} : T{0};
        out.data[i] += g * (p - delta);
    }
    return true;
}

} // namespace mat

// ---- Explicit template instantiations for f32 and f64 ----

template struct BasicMat<f32>;
template struct BasicMat<f64>;

template void BasicMat<f32>::FillRand(f32, f32);
template void BasicMat<f64>::FillRand(f64, f64);

template bool mat::Add<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::Add<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::Sub<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::Sub<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::MatMul<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&, bool, bool, bool) noexcept;
template bool mat::MatMul<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&, bool, bool, bool) noexcept;

template bool mat::Relu<f32>(BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::Relu<f64>(BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::Softmax<f32>(BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::Softmax<f64>(BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::LogSoftmax<f32>(BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::LogSoftmax<f64>(BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::ReinforceLoss<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::ReinforceLoss<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::ReinforceAddGrad<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::ReinforceAddGrad<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::ReluAddGrad<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::ReluAddGrad<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::SoftmaxAddGrad<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::SoftmaxAddGrad<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::LogSoftmaxAddGrad<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::LogSoftmaxAddGrad<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::LogProbLoss<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::LogProbLoss<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::LogProbLossAddGrad<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::LogProbLossAddGrad<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::CrossEntropy<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::CrossEntropy<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::CrossEntropyAddGrad<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::CrossEntropyAddGrad<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;
