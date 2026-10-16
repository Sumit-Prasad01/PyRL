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
    const u32 r_out = std::max(a.rows, b.rows);
    const u32 c_out = std::max(a.cols, b.cols);
    if ((a.rows != b.rows && a.rows != 1 && b.rows != 1) ||
        (a.cols != b.cols && a.cols != 1 && b.cols != 1) ||
        out.rows != r_out || out.cols != c_out) {
        return false;
    }
    if (a.SameShape(b) && out.SameShape(a)) {
        const u64 size = out.Size();
        for (u64 i = 0; i < size; i++) {
            out.data[i] = a.data[i] + b.data[i];
        }
        return true;
    }
    const u32 ar_step = (a.rows == 1) ? 0 : 1;
    const u32 ac_step = (a.cols == 1) ? 0 : 1;
    const u32 br_step = (b.rows == 1) ? 0 : 1;
    const u32 bc_step = (b.cols == 1) ? 0 : 1;
    for (u32 r = 0; r < out.rows; ++r) {
        const u32 ar = r * ar_step;
        const u32 br = r * br_step;
        for (u32 c = 0; c < out.cols; ++c) {
            out.At(r, c) = a.At(ar, c * ac_step) + b.At(br, c * bc_step);
        }
    }
    return true;
}

template <typename T>
bool Sub(BasicMat<T>& out, const BasicMat<T>& a, const BasicMat<T>& b) noexcept
{
    const u32 r_out = std::max(a.rows, b.rows);
    const u32 c_out = std::max(a.cols, b.cols);
    if ((a.rows != b.rows && a.rows != 1 && b.rows != 1) ||
        (a.cols != b.cols && a.cols != 1 && b.cols != 1) ||
        out.rows != r_out || out.cols != c_out) {
        return false;
    }
    if (a.SameShape(b) && out.SameShape(a)) {
        const u64 size = out.Size();
        for (u64 i = 0; i < size; i++) {
            out.data[i] = a.data[i] - b.data[i];
        }
        return true;
    }
    const u32 ar_step = (a.rows == 1) ? 0 : 1;
    const u32 ac_step = (a.cols == 1) ? 0 : 1;
    const u32 br_step = (b.rows == 1) ? 0 : 1;
    const u32 bc_step = (b.cols == 1) ? 0 : 1;
    for (u32 r = 0; r < out.rows; ++r) {
        const u32 ar = r * ar_step;
        const u32 br = r * br_step;
        for (u32 c = 0; c < out.cols; ++c) {
            out.At(r, c) = a.At(ar, c * ac_step) - b.At(br, c * bc_step);
        }
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
    if (out.Size() == 0) { return true; }

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

template <typename T>
bool AddAddGrad(BasicMat<T>& grad_a, BasicMat<T>& grad_b,
                const BasicMat<T>& a, const BasicMat<T>& b,
                const BasicMat<T>& grad_out) noexcept
{
    (void)a; (void)b;
    if (grad_a.SameShape(grad_out)) {
        for (u64 i = 0, n = grad_out.Size(); i < n; ++i) { grad_a.data[i] += grad_out.data[i]; }
    } else if (grad_a.rows == 1 && grad_a.cols == grad_out.cols) {
        for (u32 c = 0; c < grad_out.cols; ++c) {
            T s = T{0};
            for (u32 r = 0; r < grad_out.rows; ++r) { s += grad_out.At(r, c); }
            grad_a.At(0, c) += s;
        }
    } else if (grad_a.rows == grad_out.rows && grad_a.cols == 1) {
        for (u32 r = 0; r < grad_out.rows; ++r) {
            T s = T{0};
            for (u32 c = 0; c < grad_out.cols; ++c) { s += grad_out.At(r, c); }
            grad_a.At(r, 0) += s;
        }
    } else if (grad_a.rows == 1 && grad_a.cols == 1) {
        T s = T{0};
        for (u64 i = 0, n = grad_out.Size(); i < n; ++i) { s += grad_out.data[i]; }
        grad_a.At(0, 0) += s;
    }

    if (grad_b.SameShape(grad_out)) {
        for (u64 i = 0, n = grad_out.Size(); i < n; ++i) { grad_b.data[i] += grad_out.data[i]; }
    } else if (grad_b.rows == 1 && grad_b.cols == grad_out.cols) {
        for (u32 c = 0; c < grad_out.cols; ++c) {
            T s = T{0};
            for (u32 r = 0; r < grad_out.rows; ++r) { s += grad_out.At(r, c); }
            grad_b.At(0, c) += s;
        }
    } else if (grad_b.rows == grad_out.rows && grad_b.cols == 1) {
        for (u32 r = 0; r < grad_out.rows; ++r) {
            T s = T{0};
            for (u32 c = 0; c < grad_out.cols; ++c) { s += grad_out.At(r, c); }
            grad_b.At(r, 0) += s;
        }
    } else if (grad_b.rows == 1 && grad_b.cols == 1) {
        T s = T{0};
        for (u64 i = 0, n = grad_out.Size(); i < n; ++i) { s += grad_out.data[i]; }
        grad_b.At(0, 0) += s;
    }
    return true;
}

template <typename T>
bool SubAddGrad(BasicMat<T>& grad_a, BasicMat<T>& grad_b,
                const BasicMat<T>& a, const BasicMat<T>& b,
                const BasicMat<T>& grad_out) noexcept
{
    (void)a; (void)b;
    if (grad_a.SameShape(grad_out)) {
        for (u64 i = 0, n = grad_out.Size(); i < n; ++i) { grad_a.data[i] += grad_out.data[i]; }
    } else if (grad_a.rows == 1 && grad_a.cols == grad_out.cols) {
        for (u32 c = 0; c < grad_out.cols; ++c) {
            T s = T{0};
            for (u32 r = 0; r < grad_out.rows; ++r) { s += grad_out.At(r, c); }
            grad_a.At(0, c) += s;
        }
    } else if (grad_a.rows == grad_out.rows && grad_a.cols == 1) {
        for (u32 r = 0; r < grad_out.rows; ++r) {
            T s = T{0};
            for (u32 c = 0; c < grad_out.cols; ++c) { s += grad_out.At(r, c); }
            grad_a.At(r, 0) += s;
        }
    } else if (grad_a.rows == 1 && grad_a.cols == 1) {
        T s = T{0};
        for (u64 i = 0, n = grad_out.Size(); i < n; ++i) { s += grad_out.data[i]; }
        grad_a.At(0, 0) += s;
    }

    if (grad_b.SameShape(grad_out)) {
        for (u64 i = 0, n = grad_out.Size(); i < n; ++i) { grad_b.data[i] -= grad_out.data[i]; }
    } else if (grad_b.rows == 1 && grad_b.cols == grad_out.cols) {
        for (u32 c = 0; c < grad_out.cols; ++c) {
            T s = T{0};
            for (u32 r = 0; r < grad_out.rows; ++r) { s += grad_out.At(r, c); }
            grad_b.At(0, c) -= s;
        }
    } else if (grad_b.rows == grad_out.rows && grad_b.cols == 1) {
        for (u32 r = 0; r < grad_out.rows; ++r) {
            T s = T{0};
            for (u32 c = 0; c < grad_out.cols; ++c) { s += grad_out.At(r, c); }
            grad_b.At(r, 0) -= s;
        }
    } else if (grad_b.rows == 1 && grad_b.cols == 1) {
        T s = T{0};
        for (u64 i = 0, n = grad_out.Size(); i < n; ++i) { s += grad_out.data[i]; }
        grad_b.At(0, 0) -= s;
    }
    return true;
}

template <typename T>
bool Mul(BasicMat<T>& out, const BasicMat<T>& a, const BasicMat<T>& b) noexcept
{
    const u32 r_out = std::max(a.rows, b.rows);
    const u32 c_out = std::max(a.cols, b.cols);
    if ((a.rows != b.rows && a.rows != 1 && b.rows != 1) ||
        (a.cols != b.cols && a.cols != 1 && b.cols != 1) ||
        out.rows != r_out || out.cols != c_out) {
        return false;
    }
    if (a.SameShape(b) && out.SameShape(a)) {
        for (u64 i = 0, n = out.Size(); i < n; ++i) { out.data[i] = a.data[i] * b.data[i]; }
        return true;
    }
    const u32 ar_step = (a.rows == 1) ? 0 : 1;
    const u32 ac_step = (a.cols == 1) ? 0 : 1;
    const u32 br_step = (b.rows == 1) ? 0 : 1;
    const u32 bc_step = (b.cols == 1) ? 0 : 1;
    for (u32 r = 0; r < out.rows; ++r) {
        const u32 ar = r * ar_step;
        const u32 br = r * br_step;
        for (u32 c = 0; c < out.cols; ++c) {
            out.At(r, c) = a.At(ar, c * ac_step) * b.At(br, c * bc_step);
        }
    }
    return true;
}

template <typename T>
bool MulAddGrad(BasicMat<T>& grad_a, BasicMat<T>& grad_b,
                const BasicMat<T>& a, const BasicMat<T>& b,
                const BasicMat<T>& grad_out) noexcept
{
    const u32 ar_step = (a.rows == 1) ? 0 : 1;
    const u32 ac_step = (a.cols == 1) ? 0 : 1;
    const u32 br_step = (b.rows == 1) ? 0 : 1;
    const u32 bc_step = (b.cols == 1) ? 0 : 1;

    for (u32 r = 0; r < grad_out.rows; ++r) {
        const u32 ar = r * ar_step;
        const u32 br = r * br_step;
        for (u32 c = 0; c < grad_out.cols; ++c) {
            const u32 ac = c * ac_step;
            const u32 bc = c * bc_step;
            const T g = grad_out.At(r, c);
            grad_a.At(ar, ac) += g * b.At(br, bc);
            grad_b.At(br, bc) += g * a.At(ar, ac);
        }
    }
    return true;
}

template <typename T>
bool Div(BasicMat<T>& out, const BasicMat<T>& a, const BasicMat<T>& b) noexcept
{
    const u32 r_out = std::max(a.rows, b.rows);
    const u32 c_out = std::max(a.cols, b.cols);
    if ((a.rows != b.rows && a.rows != 1 && b.rows != 1) ||
        (a.cols != b.cols && a.cols != 1 && b.cols != 1) ||
        out.rows != r_out || out.cols != c_out) {
        return false;
    }
    const u32 ar_step = (a.rows == 1) ? 0 : 1;
    const u32 ac_step = (a.cols == 1) ? 0 : 1;
    const u32 br_step = (b.rows == 1) ? 0 : 1;
    const u32 bc_step = (b.cols == 1) ? 0 : 1;
    for (u32 r = 0; r < out.rows; ++r) {
        const u32 ar = r * ar_step;
        const u32 br = r * br_step;
        for (u32 c = 0; c < out.cols; ++c) {
            out.At(r, c) = a.At(ar, c * ac_step) / b.At(br, c * bc_step);
        }
    }
    return true;
}

template <typename T>
bool DivAddGrad(BasicMat<T>& grad_a, BasicMat<T>& grad_b,
                const BasicMat<T>& a, const BasicMat<T>& b,
                const BasicMat<T>& grad_out) noexcept
{
    const u32 ar_step = (a.rows == 1) ? 0 : 1;
    const u32 ac_step = (a.cols == 1) ? 0 : 1;
    const u32 br_step = (b.rows == 1) ? 0 : 1;
    const u32 bc_step = (b.cols == 1) ? 0 : 1;

    for (u32 r = 0; r < grad_out.rows; ++r) {
        const u32 ar = r * ar_step;
        const u32 br = r * br_step;
        for (u32 c = 0; c < grad_out.cols; ++c) {
            const u32 ac = c * ac_step;
            const u32 bc = c * bc_step;
            const T g  = grad_out.At(r, c);
            const T bv = b.At(br, bc);
            const T av = a.At(ar, ac);
            grad_a.At(ar, ac) += g / bv;
            grad_b.At(br, bc) += -g * av / (bv * bv);
        }
    }
    return true;
}

template <typename T>
bool Tanh(BasicMat<T>& out, const BasicMat<T>& in) noexcept
{
    if (!out.SameShape(in)) return false;
    for (u64 i = 0, n = out.Size(); i < n; ++i) {
        out.data[i] = std::tanh(in.data[i]);
    }
    return true;
}

template <typename T>
bool TanhAddGrad(BasicMat<T>& in_grad, const BasicMat<T>& out_val, const BasicMat<T>& grad) noexcept
{
    if (!in_grad.SameShape(out_val) || !in_grad.SameShape(grad)) return false;
    for (u64 i = 0, n = in_grad.Size(); i < n; ++i) {
        const T y = out_val.data[i];
        in_grad.data[i] += grad.data[i] * (T{1} - y * y);
    }
    return true;
}

template <typename T>
bool Sigmoid(BasicMat<T>& out, const BasicMat<T>& in) noexcept
{
    if (!out.SameShape(in)) return false;
    for (u64 i = 0, n = out.Size(); i < n; ++i) {
        out.data[i] = T{1} / (T{1} + std::exp(-in.data[i]));
    }
    return true;
}

template <typename T>
bool SigmoidAddGrad(BasicMat<T>& in_grad, const BasicMat<T>& out_val, const BasicMat<T>& grad) noexcept
{
    if (!in_grad.SameShape(out_val) || !in_grad.SameShape(grad)) return false;
    for (u64 i = 0, n = in_grad.Size(); i < n; ++i) {
        const T y = out_val.data[i];
        in_grad.data[i] += grad.data[i] * y * (T{1} - y);
    }
    return true;
}

template <typename T>
bool LeakyRelu(BasicMat<T>& out, const BasicMat<T>& in, T alpha) noexcept
{
    if (!out.SameShape(in)) return false;
    for (u64 i = 0, n = out.Size(); i < n; ++i) {
        out.data[i] = in.data[i] > T{0} ? in.data[i] : alpha * in.data[i];
    }
    return true;
}

template <typename T>
bool LeakyReluAddGrad(BasicMat<T>& in_grad, const BasicMat<T>& in, const BasicMat<T>& grad, T alpha) noexcept
{
    if (!in_grad.SameShape(in) || !in_grad.SameShape(grad)) return false;
    for (u64 i = 0, n = in_grad.Size(); i < n; ++i) {
        in_grad.data[i] += grad.data[i] * (in.data[i] > T{0} ? T{1} : alpha);
    }
    return true;
}

template <typename T>
bool Gelu(BasicMat<T>& out, const BasicMat<T>& in) noexcept
{
    if (!out.SameShape(in)) return false;
    const T inv_sqrt2 = T{0.7071067811865475};
    for (u64 i = 0, n = out.Size(); i < n; ++i) {
        const T x = in.data[i];
        out.data[i] = T{0.5} * x * (T{1} + std::erf(x * inv_sqrt2));
    }
    return true;
}

template <typename T>
bool GeluAddGrad(BasicMat<T>& in_grad, const BasicMat<T>& in, const BasicMat<T>& grad) noexcept
{
    if (!in_grad.SameShape(in) || !in_grad.SameShape(grad)) return false;
    const T inv_sqrt2   = T{0.7071067811865475};
    const T inv_sqrt2pi = T{0.3989422804014327};
    for (u64 i = 0, n = in_grad.Size(); i < n; ++i) {
        const T x = in.data[i];
        const T dy_dx = T{0.5} * (T{1} + std::erf(x * inv_sqrt2)) + x * inv_sqrt2pi * std::exp(-T{0.5} * x * x);
        in_grad.data[i] += grad.data[i] * dy_dx;
    }
    return true;
}

template <typename T>
bool Exp(BasicMat<T>& out, const BasicMat<T>& in) noexcept
{
    if (!out.SameShape(in)) return false;
    for (u64 i = 0, n = out.Size(); i < n; ++i) {
        out.data[i] = std::exp(in.data[i]);
    }
    return true;
}

template <typename T>
bool ExpAddGrad(BasicMat<T>& in_grad, const BasicMat<T>& out_val, const BasicMat<T>& grad) noexcept
{
    if (!in_grad.SameShape(out_val) || !in_grad.SameShape(grad)) return false;
    for (u64 i = 0, n = in_grad.Size(); i < n; ++i) {
        in_grad.data[i] += grad.data[i] * out_val.data[i];
    }
    return true;
}

template <typename T>
bool Log(BasicMat<T>& out, const BasicMat<T>& in, T eps) noexcept
{
    if (!out.SameShape(in)) return false;
    for (u64 i = 0, n = out.Size(); i < n; ++i) {
        out.data[i] = std::log(std::max(in.data[i], eps));
    }
    return true;
}

template <typename T>
bool LogAddGrad(BasicMat<T>& in_grad, const BasicMat<T>& in, const BasicMat<T>& grad, T eps) noexcept
{
    if (!in_grad.SameShape(in) || !in_grad.SameShape(grad)) return false;
    for (u64 i = 0, n = in_grad.Size(); i < n; ++i) {
        in_grad.data[i] += grad.data[i] / std::max(in.data[i], eps);
    }
    return true;
}

template <typename T>
bool Clip(BasicMat<T>& out, const BasicMat<T>& in, T min_val, T max_val) noexcept
{
    if (!out.SameShape(in)) return false;
    for (u64 i = 0, n = out.Size(); i < n; ++i) {
        out.data[i] = std::clamp(in.data[i], min_val, max_val);
    }
    return true;
}

template <typename T>
bool ClipAddGrad(BasicMat<T>& in_grad, const BasicMat<T>& in, const BasicMat<T>& grad, T min_val, T max_val) noexcept
{
    if (!in_grad.SameShape(in) || !in_grad.SameShape(grad)) return false;
    for (u64 i = 0, n = in_grad.Size(); i < n; ++i) {
        in_grad.data[i] += (in.data[i] >= min_val && in.data[i] <= max_val) ? grad.data[i] : T{0};
    }
    return true;
}

template <typename T>
bool Sum(BasicMat<T>& out, const BasicMat<T>& in, i32 axis) noexcept
{
    if (axis == 0) {
        if (out.rows != 1 || out.cols != in.cols) return false;
        out.Clear();
        for (u32 r = 0; r < in.rows; ++r) {
            for (u32 c = 0; c < in.cols; ++c) {
                out.At(0, c) += in.At(r, c);
            }
        }
    } else if (axis == 1) {
        if (out.rows != in.rows || out.cols != 1) return false;
        out.Clear();
        for (u32 r = 0; r < in.rows; ++r) {
            for (u32 c = 0; c < in.cols; ++c) {
                out.At(r, 0) += in.At(r, c);
            }
        }
    } else {
        if (out.rows != 1 || out.cols != 1) return false;
        T s = T{0};
        for (u64 i = 0, n = in.Size(); i < n; ++i) { s += in.data[i]; }
        out.data[0] = s;
    }
    return true;
}

template <typename T>
bool SumAddGrad(BasicMat<T>& in_grad, const BasicMat<T>& grad_out, i32 axis) noexcept
{
    if (axis == 0) {
        if (grad_out.rows != 1 || grad_out.cols != in_grad.cols) return false;
        for (u32 r = 0; r < in_grad.rows; ++r) {
            for (u32 c = 0; c < in_grad.cols; ++c) {
                in_grad.At(r, c) += grad_out.At(0, c);
            }
        }
    } else if (axis == 1) {
        if (grad_out.rows != in_grad.rows || grad_out.cols != 1) return false;
        for (u32 r = 0; r < in_grad.rows; ++r) {
            for (u32 c = 0; c < in_grad.cols; ++c) {
                in_grad.At(r, c) += grad_out.At(r, 0);
            }
        }
    } else {
        if (grad_out.rows != 1 || grad_out.cols != 1) return false;
        const T g = grad_out.data[0];
        for (u64 i = 0, n = in_grad.Size(); i < n; ++i) {
            in_grad.data[i] += g;
        }
    }
    return true;
}

template <typename T>
bool Mean(BasicMat<T>& out, const BasicMat<T>& in, i32 axis) noexcept
{
    if (!Sum(out, in, axis)) return false;
    T factor = T{1};
    if (axis == 0) {
        factor = static_cast<T>(in.rows);
    } else if (axis == 1) {
        factor = static_cast<T>(in.cols);
    } else {
        factor = static_cast<T>(in.Size());
    }
    if (factor > T{0}) {
        out.Scale(T{1} / factor);
    }
    return true;
}

template <typename T>
bool MeanAddGrad(BasicMat<T>& in_grad, const BasicMat<T>& grad_out, i32 axis) noexcept
{
    T factor = T{1};
    if (axis == 0) {
        factor = static_cast<T>(in_grad.rows);
    } else if (axis == 1) {
        factor = static_cast<T>(in_grad.cols);
    } else {
        factor = static_cast<T>(in_grad.Size());
    }
    if (factor <= T{0}) return false;

    if (axis == 0) {
        if (grad_out.rows != 1 || grad_out.cols != in_grad.cols) return false;
        for (u32 r = 0; r < in_grad.rows; ++r) {
            for (u32 c = 0; c < in_grad.cols; ++c) {
                in_grad.At(r, c) += grad_out.At(0, c) / factor;
            }
        }
    } else if (axis == 1) {
        if (grad_out.rows != in_grad.rows || grad_out.cols != 1) return false;
        for (u32 r = 0; r < in_grad.rows; ++r) {
            for (u32 c = 0; c < in_grad.cols; ++c) {
                in_grad.At(r, c) += grad_out.At(r, 0) / factor;
            }
        }
    } else {
        if (grad_out.rows != 1 || grad_out.cols != 1) return false;
        const T g = grad_out.data[0] / factor;
        for (u64 i = 0, n = in_grad.Size(); i < n; ++i) {
            in_grad.data[i] += g;
        }
    }
    return true;
}

template <typename T>
bool LayerNorm(BasicMat<T>& out, const BasicMat<T>& in,
               const BasicMat<T>& gamma, const BasicMat<T>& beta,
               T eps) noexcept
{
    if (!out.SameShape(in) || gamma.Size() != in.cols || beta.Size() != in.cols) {
        return false;
    }
    const u32 B = in.rows;
    const u32 D = in.cols;
    if (D == 0) return true;

    for (u32 b = 0; b < B; ++b) {
        T mean = T{0};
        for (u32 j = 0; j < D; ++j) {
            mean += in.At(b, j);
        }
        mean /= static_cast<T>(D);

        T var = T{0};
        for (u32 j = 0; j < D; ++j) {
            const T diff = in.At(b, j) - mean;
            var += diff * diff;
        }
        var /= static_cast<T>(D);

        const T inv_std = T{1} / std::sqrt(var + eps);
        for (u32 j = 0; j < D; ++j) {
            const T x_hat = (in.At(b, j) - mean) * inv_std;
            out.At(b, j) = gamma.data[j] * x_hat + beta.data[j];
        }
    }
    return true;
}

template <typename T>
bool LayerNormAddGrad(BasicMat<T>& in_grad, BasicMat<T>& gamma_grad,
                      BasicMat<T>& beta_grad, const BasicMat<T>& in,
                      const BasicMat<T>& gamma, const BasicMat<T>& grad_out,
                      T eps) noexcept
{
    if (!in_grad.SameShape(in) || gamma.Size() != in.cols || grad_out.SameShape(in) == false) {
        return false;
    }
    const u32 B = in.rows;
    const u32 D = in.cols;
    if (D == 0) return true;

    const T inv_D = T{1} / static_cast<T>(D);

    for (u32 b = 0; b < B; ++b) {
        T mean = T{0};
        for (u32 j = 0; j < D; ++j) {
            mean += in.At(b, j);
        }
        mean *= inv_D;

        T var = T{0};
        for (u32 j = 0; j < D; ++j) {
            const T diff = in.At(b, j) - mean;
            var += diff * diff;
        }
        var *= inv_D;

        const T inv_std = T{1} / std::sqrt(var + eps);

        T sum_g = T{0};
        T sum_gx = T{0};
        for (u32 j = 0; j < D; ++j) {
            const T dy = grad_out.At(b, j);
            const T x_hat = (in.At(b, j) - mean) * inv_std;
            if (gamma_grad.data) gamma_grad.data[j] += dy * x_hat;
            if (beta_grad.data)  beta_grad.data[j]  += dy;

            const T g = dy * gamma.data[j];
            sum_g += g;
            sum_gx += g * x_hat;
        }

        for (u32 j = 0; j < D; ++j) {
            const T x_hat = (in.At(b, j) - mean) * inv_std;
            const T g = grad_out.At(b, j) * gamma.data[j];
            const T dx = inv_std * (g - sum_g * inv_D - x_hat * sum_gx * inv_D);
            in_grad.At(b, j) += dx;
        }
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

template bool mat::AddAddGrad<f32>(BasicMat<f32>&, BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::AddAddGrad<f64>(BasicMat<f64>&, BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::SubAddGrad<f32>(BasicMat<f32>&, BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::SubAddGrad<f64>(BasicMat<f64>&, BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::Mul<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::Mul<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::MulAddGrad<f32>(BasicMat<f32>&, BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::MulAddGrad<f64>(BasicMat<f64>&, BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::Div<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::Div<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::DivAddGrad<f32>(BasicMat<f32>&, BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::DivAddGrad<f64>(BasicMat<f64>&, BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::Tanh<f32>(BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::Tanh<f64>(BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::TanhAddGrad<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::TanhAddGrad<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::Sigmoid<f32>(BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::Sigmoid<f64>(BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::SigmoidAddGrad<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::SigmoidAddGrad<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::LeakyRelu<f32>(BasicMat<f32>&, const BasicMat<f32>&, f32) noexcept;
template bool mat::LeakyRelu<f64>(BasicMat<f64>&, const BasicMat<f64>&, f64) noexcept;

template bool mat::LeakyReluAddGrad<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&, f32) noexcept;
template bool mat::LeakyReluAddGrad<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&, f64) noexcept;

template bool mat::Gelu<f32>(BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::Gelu<f64>(BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::GeluAddGrad<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::GeluAddGrad<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::Exp<f32>(BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::Exp<f64>(BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::ExpAddGrad<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&) noexcept;
template bool mat::ExpAddGrad<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&) noexcept;

template bool mat::Log<f32>(BasicMat<f32>&, const BasicMat<f32>&, f32) noexcept;
template bool mat::Log<f64>(BasicMat<f64>&, const BasicMat<f64>&, f64) noexcept;

template bool mat::LogAddGrad<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&, f32) noexcept;
template bool mat::LogAddGrad<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&, f64) noexcept;

template bool mat::Clip<f32>(BasicMat<f32>&, const BasicMat<f32>&, f32, f32) noexcept;
template bool mat::Clip<f64>(BasicMat<f64>&, const BasicMat<f64>&, f64, f64) noexcept;

template bool mat::ClipAddGrad<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&, f32, f32) noexcept;
template bool mat::ClipAddGrad<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&, f64, f64) noexcept;

template bool mat::Sum<f32>(BasicMat<f32>&, const BasicMat<f32>&, i32) noexcept;
template bool mat::Sum<f64>(BasicMat<f64>&, const BasicMat<f64>&, i32) noexcept;

template bool mat::SumAddGrad<f32>(BasicMat<f32>&, const BasicMat<f32>&, i32) noexcept;
template bool mat::SumAddGrad<f64>(BasicMat<f64>&, const BasicMat<f64>&, i32) noexcept;

template bool mat::Mean<f32>(BasicMat<f32>&, const BasicMat<f32>&, i32) noexcept;
template bool mat::Mean<f64>(BasicMat<f64>&, const BasicMat<f64>&, i32) noexcept;

template bool mat::MeanAddGrad<f32>(BasicMat<f32>&, const BasicMat<f32>&, i32) noexcept;
template bool mat::MeanAddGrad<f64>(BasicMat<f64>&, const BasicMat<f64>&, i32) noexcept;

template bool mat::LayerNorm<f32>(BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&, f32) noexcept;
template bool mat::LayerNorm<f64>(BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&, f64) noexcept;

template bool mat::LayerNormAddGrad<f32>(BasicMat<f32>&, BasicMat<f32>&, BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&, const BasicMat<f32>&, f32) noexcept;
template bool mat::LayerNormAddGrad<f64>(BasicMat<f64>&, BasicMat<f64>&, BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&, const BasicMat<f64>&, f64) noexcept;
