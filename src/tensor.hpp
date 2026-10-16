#pragma once

#include <algorithm>
#include <cassert>
#include <initializer_list>
#include <numeric>
#include <optional>
#include <span>
#include <type_traits>
#include <vector>

#include "arena.hpp"
#include "base.hpp"
#include "mat.hpp"

// Maximum rank supported by Tensor inline shape/strides array.
constexpr u32 kMaxTensorDims = 8;

struct TensorShape {
    u32 ndim = 0;
    u32 dims[kMaxTensorDims] = {};

    constexpr TensorShape() noexcept = default;

    constexpr TensorShape(std::initializer_list<u32> list) noexcept
    {
        assert(list.size() <= kMaxTensorDims);
        ndim = static_cast<u32>(list.size());
        u32 i = 0;
        for (u32 d : list) {
            dims[i++] = d;
        }
    }

    constexpr TensorShape(std::span<const u32> span) noexcept
    {
        assert(span.size() <= kMaxTensorDims);
        ndim = static_cast<u32>(span.size());
        for (u32 i = 0; i < ndim; ++i) {
            dims[i] = span[i];
        }
    }

    [[nodiscard]] constexpr u64 Numel() const noexcept
    {
        if (ndim == 0) { return 0; }
        u64 n = 1;
        for (u32 i = 0; i < ndim; ++i) {
            n *= dims[i];
        }
        return n;
    }

    [[nodiscard]] constexpr bool operator==(const TensorShape& o) const noexcept
    {
        if (ndim != o.ndim) { return false; }
        for (u32 i = 0; i < ndim; ++i) {
            if (dims[i] != o.dims[i]) { return false; }
        }
        return true;
    }

    [[nodiscard]] constexpr bool operator!=(const TensorShape& o) const noexcept
    {
        return !(*this == o);
    }

    [[nodiscard]] constexpr u32 operator[](u32 i) const noexcept
    {
        assert(i < ndim);
        return dims[i];
    }
};

// N-Dimensional Strided Tensor view.
//
// - Trivially copyable and trivially destructible: can live inside Arena or be passed by value.
// - Supports zero-copy views: Reshape, Transpose, Permute, Slice, BroadcastTo.
// - NumPy-style broadcasting semantics.
// - Compatible with BasicMat<T>.
template <typename T = f32>
struct Tensor {
    T*  data;
    u64 offset;
    u32 ndim;
    u32 shape[kMaxTensorDims];
    i64 strides[kMaxTensorDims];

    Tensor() noexcept = default;

    [[nodiscard]] static Tensor* Create(Arena& arena, std::span<const u32> shape_span) noexcept
    {
        assert(shape_span.size() <= kMaxTensorDims);
        Tensor* t = arena.PushStruct<Tensor>();
        if (t == nullptr) { return nullptr; }

        t->ndim   = static_cast<u32>(shape_span.size());
        t->offset = 0;

        u64 numel = 1;
        for (u32 i = 0; i < t->ndim; ++i) {
            t->shape[i] = shape_span[i];
            numel *= shape_span[i];
        }

        // Compute row-major (C-contiguous) strides
        i64 cur_stride = 1;
        for (i32 i = static_cast<i32>(t->ndim) - 1; i >= 0; --i) {
            t->strides[i] = cur_stride;
            cur_stride *= t->shape[i];
        }

        t->data = arena.PushArray<T>(numel);
        if (t->data == nullptr && numel > 0) {
            return nullptr;
        }
        return t;
    }

    [[nodiscard]] static Tensor* Create(Arena& arena, std::initializer_list<u32> shape_list) noexcept
    {
        return Create(arena, std::span<const u32>(shape_list.begin(), shape_list.size()));
    }

    [[nodiscard]] static Tensor FromMat(const BasicMat<T>& mat) noexcept
    {
        Tensor t{};
        t.data       = mat.data;
        t.offset     = 0;
        t.ndim       = 2;
        t.shape[0]   = mat.rows;
        t.shape[1]   = mat.cols;
        t.strides[0] = static_cast<i64>(mat.cols);
        t.strides[1] = 1;
        return t;
    }

    [[nodiscard]] BasicMat<T> ToMat() const noexcept
    {
        assert(ndim == 2);
        assert(IsContiguous());
        return BasicMat<T>(shape[0], shape[1], const_cast<T*>(Data()));
    }

    [[nodiscard]] T* Data() noexcept { return data + offset; }
    [[nodiscard]] const T* Data() const noexcept { return data + offset; }

    [[nodiscard]] constexpr u64 Numel() const noexcept
    {
        if (ndim == 0) { return 0; }
        u64 n = 1;
        for (u32 i = 0; i < ndim; ++i) {
            n *= shape[i];
        }
        return n;
    }

    [[nodiscard]] constexpr u64 Size() const noexcept { return Numel(); }

    [[nodiscard]] std::span<T> Span() noexcept
    {
        assert(IsContiguous());
        return {Data(), Numel()};
    }

    [[nodiscard]] std::span<const T> Span() const noexcept
    {
        assert(IsContiguous());
        return {Data(), Numel()};
    }

    [[nodiscard]] bool IsContiguous() const noexcept
    {
        if (ndim == 0) { return true; }
        u64 expected_stride = 1;
        for (i32 i = static_cast<i32>(ndim) - 1; i >= 0; --i) {
            if (shape[i] == 1) {
                continue;
            }
            if (strides[i] != static_cast<i64>(expected_stride)) {
                return false;
            }
            expected_stride *= shape[i];
        }
        return true;
    }

    [[nodiscard]] bool SameShape(const Tensor& o) const noexcept
    {
        if (ndim != o.ndim) { return false; }
        for (u32 i = 0; i < ndim; ++i) {
            if (shape[i] != o.shape[i]) { return false; }
        }
        return true;
    }

    [[nodiscard]] T& At(u32 i) noexcept
    {
        assert(ndim == 1 && i < shape[0]);
        return data[offset + static_cast<u64>(static_cast<i64>(i) * strides[0])];
    }

    [[nodiscard]] const T& At(u32 i) const noexcept
    {
        assert(ndim == 1 && i < shape[0]);
        return data[offset + static_cast<u64>(static_cast<i64>(i) * strides[0])];
    }

    [[nodiscard]] T& At(u32 r, u32 c) noexcept
    {
        assert(ndim == 2 && r < shape[0] && c < shape[1]);
        const i64 idx = static_cast<i64>(offset) + static_cast<i64>(r) * strides[0] + static_cast<i64>(c) * strides[1];
        assert(idx >= 0);
        return data[static_cast<u64>(idx)];
    }

    [[nodiscard]] const T& At(u32 r, u32 c) const noexcept
    {
        assert(ndim == 2 && r < shape[0] && c < shape[1]);
        const i64 idx = static_cast<i64>(offset) + static_cast<i64>(r) * strides[0] + static_cast<i64>(c) * strides[1];
        assert(idx >= 0);
        return data[static_cast<u64>(idx)];
    }

    [[nodiscard]] T& At(u32 d0, u32 d1, u32 d2) noexcept
    {
        assert(ndim == 3 && d0 < shape[0] && d1 < shape[1] && d2 < shape[2]);
        const i64 idx = static_cast<i64>(offset) + static_cast<i64>(d0) * strides[0] +
                        static_cast<i64>(d1) * strides[1] + static_cast<i64>(d2) * strides[2];
        assert(idx >= 0);
        return data[static_cast<u64>(idx)];
    }

    [[nodiscard]] const T& At(u32 d0, u32 d1, u32 d2) const noexcept
    {
        assert(ndim == 3 && d0 < shape[0] && d1 < shape[1] && d2 < shape[2]);
        const i64 idx = static_cast<i64>(offset) + static_cast<i64>(d0) * strides[0] +
                        static_cast<i64>(d1) * strides[1] + static_cast<i64>(d2) * strides[2];
        assert(idx >= 0);
        return data[static_cast<u64>(idx)];
    }

    [[nodiscard]] T& At(u32 d0, u32 d1, u32 d2, u32 d3) noexcept
    {
        assert(ndim == 4 && d0 < shape[0] && d1 < shape[1] && d2 < shape[2] && d3 < shape[3]);
        const i64 idx = static_cast<i64>(offset) + static_cast<i64>(d0) * strides[0] +
                        static_cast<i64>(d1) * strides[1] + static_cast<i64>(d2) * strides[2] +
                        static_cast<i64>(d3) * strides[3];
        assert(idx >= 0);
        return data[static_cast<u64>(idx)];
    }

    [[nodiscard]] const T& At(u32 d0, u32 d1, u32 d2, u32 d3) const noexcept
    {
        assert(ndim == 4 && d0 < shape[0] && d1 < shape[1] && d2 < shape[2] && d3 < shape[3]);
        const i64 idx = static_cast<i64>(offset) + static_cast<i64>(d0) * strides[0] +
                        static_cast<i64>(d1) * strides[1] + static_cast<i64>(d2) * strides[2] +
                        static_cast<i64>(d3) * strides[3];
        assert(idx >= 0);
        return data[static_cast<u64>(idx)];
    }

    [[nodiscard]] T& At(std::span<const u32> indices) noexcept
    {
        assert(indices.size() == ndim);
        i64 idx = static_cast<i64>(offset);
        for (u32 i = 0; i < ndim; ++i) {
            assert(indices[i] < shape[i]);
            idx += static_cast<i64>(indices[i]) * strides[i];
        }
        assert(idx >= 0);
        return data[static_cast<u64>(idx)];
    }

    [[nodiscard]] const T& At(std::span<const u32> indices) const noexcept
    {
        assert(indices.size() == ndim);
        i64 idx = static_cast<i64>(offset);
        for (u32 i = 0; i < ndim; ++i) {
            assert(indices[i] < shape[i]);
            idx += static_cast<i64>(indices[i]) * strides[i];
        }
        assert(idx >= 0);
        return data[static_cast<u64>(idx)];
    }

    void Clear() noexcept
    {
        Fill(T{0});
    }

    void Fill(T val) noexcept
    {
        if (data == nullptr) { return; }
        if (IsContiguous()) {
            std::fill_n(Data(), Numel(), val);
            return;
        }
        const u64 total = Numel();
        if (total == 0) { return; }
        u32 idx[kMaxTensorDims] = {};
        for (u64 n = 0; n < total; ++n) {
            i64 cur_idx = static_cast<i64>(offset);
            for (u32 d = 0; d < ndim; ++d) {
                cur_idx += static_cast<i64>(idx[d]) * strides[d];
            }
            data[cur_idx] = val;
            for (i32 d = static_cast<i32>(ndim) - 1; d >= 0; --d) {
                idx[d]++;
                if (idx[d] < shape[d]) { break; }
                idx[d] = 0;
            }
        }
    }

    [[nodiscard]] std::optional<Tensor> Reshape(std::span<const u32> new_shape) const noexcept
    {
        if (!IsContiguous()) {
            return std::nullopt;
        }
        if (new_shape.size() > kMaxTensorDims) {
            return std::nullopt;
        }
        u64 new_numel = 1;
        for (u32 d : new_shape) {
            new_numel *= d;
        }
        if (new_numel != Numel()) {
            return std::nullopt;
        }

        Tensor result{};
        result.data   = const_cast<T*>(Data());
        result.offset = 0;
        result.ndim   = static_cast<u32>(new_shape.size());

        i64 cur_stride = 1;
        for (i32 i = static_cast<i32>(result.ndim) - 1; i >= 0; --i) {
            result.shape[i]   = new_shape[i];
            result.strides[i] = cur_stride;
            cur_stride *= new_shape[i];
        }
        return result;
    }

    [[nodiscard]] std::optional<Tensor> Reshape(std::initializer_list<u32> new_shape) const noexcept
    {
        return Reshape(std::span<const u32>(new_shape.begin(), new_shape.size()));
    }

    [[nodiscard]] std::optional<Tensor> Transpose(u32 dim0, u32 dim1) const noexcept
    {
        if (dim0 >= ndim || dim1 >= ndim) {
            return std::nullopt;
        }
        Tensor result = *this;
        std::swap(result.shape[dim0], result.shape[dim1]);
        std::swap(result.strides[dim0], result.strides[dim1]);
        return result;
    }

    [[nodiscard]] std::optional<Tensor> Permute(std::span<const u32> order) const noexcept
    {
        if (order.size() != ndim) {
            return std::nullopt;
        }
        bool seen[kMaxTensorDims] = {};
        for (u32 d : order) {
            if (d >= ndim || seen[d]) {
                return std::nullopt;
            }
            seen[d] = true;
        }

        Tensor result = *this;
        for (u32 i = 0; i < ndim; ++i) {
            result.shape[i]   = shape[order[i]];
            result.strides[i] = strides[order[i]];
        }
        return result;
    }

    [[nodiscard]] std::optional<Tensor> Permute(std::initializer_list<u32> order) const noexcept
    {
        return Permute(std::span<const u32>(order.begin(), order.size()));
    }

    [[nodiscard]] std::optional<Tensor> Slice(u32 dim, u32 start, u32 end, u32 step = 1) const noexcept
    {
        if (dim >= ndim || start > end || end > shape[dim] || step == 0) {
            return std::nullopt;
        }
        Tensor result = *this;
        result.offset = static_cast<u64>(static_cast<i64>(offset) + static_cast<i64>(start) * strides[dim]);
        result.shape[dim] = (end - start + step - 1) / step;
        result.strides[dim] = strides[dim] * static_cast<i64>(step);
        return result;
    }

    [[nodiscard]] std::optional<Tensor> BroadcastTo(std::span<const u32> target_shape) const noexcept
    {
        const u32 target_ndim = static_cast<u32>(target_shape.size());
        if (target_ndim < ndim || target_ndim > kMaxTensorDims) {
            return std::nullopt;
        }

        Tensor result{};
        result.data   = const_cast<T*>(data);
        result.offset = offset;
        result.ndim   = target_ndim;

        const u32 pad = target_ndim - ndim;
        for (u32 j = 0; j < pad; ++j) {
            result.shape[j]   = target_shape[j];
            result.strides[j] = 0;
        }

        for (u32 i = 0; i < ndim; ++i) {
            const u32 j          = pad + i;
            const u32 src_dim    = shape[i];
            const u32 target_dim = target_shape[j];
            if (src_dim == target_dim) {
                result.shape[j]   = target_dim;
                result.strides[j] = strides[i];
            } else if (src_dim == 1) {
                result.shape[j]   = target_dim;
                result.strides[j] = 0;
            } else {
                return std::nullopt;
            }
        }
        return result;
    }

    [[nodiscard]] std::optional<Tensor> BroadcastTo(std::initializer_list<u32> target_shape) const noexcept
    {
        return BroadcastTo(std::span<const u32>(target_shape.begin(), target_shape.size()));
    }

    [[nodiscard]] Tensor* Contiguous(Arena& arena) const noexcept
    {
        if (IsContiguous() && offset == 0) {
            Tensor* copy = arena.PushStruct<Tensor>();
            if (copy == nullptr) { return nullptr; }
            *copy = *this;
            return copy;
        }

        Tensor* out = Tensor::Create(arena, std::span<const u32>(shape, ndim));
        if (out == nullptr) { return nullptr; }

        const u64 total = Numel();
        if (total == 0) { return out; }

        u32 idx[kMaxTensorDims] = {};
        for (u64 n = 0; n < total; ++n) {
            i64 src_idx = static_cast<i64>(offset);
            for (u32 d = 0; d < ndim; ++d) {
                src_idx += static_cast<i64>(idx[d]) * strides[d];
            }
            out->data[n] = data[src_idx];

            for (i32 d = static_cast<i32>(ndim) - 1; d >= 0; --d) {
                idx[d]++;
                if (idx[d] < shape[d]) { break; }
                idx[d] = 0;
            }
        }
        return out;
    }
};

static_assert(std::is_trivially_default_constructible_v<Tensor<f32>>);
static_assert(std::is_trivially_destructible_v<Tensor<f32>>);
static_assert(std::is_trivially_copyable_v<Tensor<f32>>);
static_assert(std::is_trivially_default_constructible_v<Tensor<f64>>);
static_assert(std::is_trivially_destructible_v<Tensor<f64>>);
static_assert(std::is_trivially_copyable_v<Tensor<f64>>);

// Helper for computing NumPy broadcasted shape between two shapes.
inline bool BroadcastShapes(std::span<const u32> a, std::span<const u32> b,
                            u32* out_shape, u32& out_ndim) noexcept
{
    const u32 na = static_cast<u32>(a.size());
    const u32 nb = static_cast<u32>(b.size());
    out_ndim = std::max(na, nb);
    if (out_ndim > kMaxTensorDims) { return false; }

    for (u32 i = 0; i < out_ndim; ++i) {
        const u32 da = (i < na) ? a[na - 1 - i] : 1;
        const u32 db = (i < nb) ? b[nb - 1 - i] : 1;
        if (da == db) {
            out_shape[out_ndim - 1 - i] = da;
        } else if (da == 1) {
            out_shape[out_ndim - 1 - i] = db;
        } else if (db == 1) {
            out_shape[out_ndim - 1 - i] = da;
        } else {
            return false;
        }
    }
    return true;
}

// RAII heap-owned tensor container.
template <typename T = f32>
class OwnedTensor {
public:
    OwnedTensor() noexcept = default;

    explicit OwnedTensor(std::span<const u32> shape_span)
    {
        Init(shape_span);
    }

    OwnedTensor(std::initializer_list<u32> shape_list)
    {
        Init(std::span<const u32>(shape_list.begin(), shape_list.size()));
    }

    [[nodiscard]] Tensor<T>& View() noexcept { return view_; }
    [[nodiscard]] const Tensor<T>& View() const noexcept { return view_; }

    operator Tensor<T>&() noexcept { return view_; }
    operator const Tensor<T>&() const noexcept { return view_; }

    [[nodiscard]] std::span<T> Span() noexcept { return storage_; }
    [[nodiscard]] std::span<const T> Span() const noexcept { return storage_; }

private:
    void Init(std::span<const u32> shape_span)
    {
        assert(shape_span.size() <= kMaxTensorDims);
        view_.ndim   = static_cast<u32>(shape_span.size());
        view_.offset = 0;

        u64 numel = 1;
        for (u32 i = 0; i < view_.ndim; ++i) {
            view_.shape[i] = shape_span[i];
            numel *= shape_span[i];
        }

        i64 cur_stride = 1;
        for (i32 i = static_cast<i32>(view_.ndim) - 1; i >= 0; --i) {
            view_.strides[i] = cur_stride;
            cur_stride *= view_.shape[i];
        }

        storage_.resize(numel, T{0});
        view_.data = storage_.data();
    }

    std::vector<T> storage_;
    Tensor<T>      view_;
};
