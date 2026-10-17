#pragma once

#include <span>
#include <vector>
#include <stdexcept>
#include <algorithm>

#include "base.hpp"
#include "prng.hpp"

namespace rl {

enum class SpaceType : u8 {
    Discrete = 0,
    Box      = 1,
};

class Space {
public:
    virtual ~Space() = default;
    [[nodiscard]] virtual SpaceType Type() const noexcept = 0;
    [[nodiscard]] virtual u32 Dim() const noexcept = 0;
};

class DiscreteSpace final : public Space {
public:
    u32 n = 0;

    explicit DiscreteSpace(u32 num_actions) noexcept : n(num_actions) {}

    [[nodiscard]] SpaceType Type() const noexcept override { return SpaceType::Discrete; }
    [[nodiscard]] u32 Dim() const noexcept override { return n; }

    [[nodiscard]] bool Contains(i64 action) const noexcept
    {
        return action >= 0 && static_cast<u32>(action) < n;
    }

    [[nodiscard]] u32 Sample(Prng& rng) const noexcept
    {
        return (n > 0) ? rng.NextBelow(n) : 0;
    }
};

class BoxSpace final : public Space {
public:
    std::vector<u32> shape;
    std::vector<f32> low;
    std::vector<f32> high;
    u32 total_dim = 1;

    BoxSpace(u32 dim, f32 min_val, f32 max_val)
        : shape{dim},
          low(dim, min_val),
          high(dim, max_val),
          total_dim(dim)
    {
    }

    BoxSpace(std::vector<u32> sp, f32 min_val, f32 max_val)
        : shape(std::move(sp))
    {
        total_dim = 1;
        for (u32 s : shape) { total_dim *= s; }
        low.assign(total_dim, min_val);
        high.assign(total_dim, max_val);
    }

    BoxSpace(std::vector<u32> sp, std::vector<f32> l, std::vector<f32> h)
        : shape(std::move(sp)), low(std::move(l)), high(std::move(h))
    {
        total_dim = 1;
        for (u32 s : shape) { total_dim *= s; }
        if (low.size() != total_dim || high.size() != total_dim) {
            throw std::invalid_argument("BoxSpace: low/high size mismatch with total shape dimension.");
        }
    }

    [[nodiscard]] SpaceType Type() const noexcept override { return SpaceType::Box; }
    [[nodiscard]] u32 Dim() const noexcept override { return total_dim; }

    [[nodiscard]] bool Contains(std::span<const f32> obs) const noexcept
    {
        if (obs.size() != total_dim) { return false; }
        for (u32 i = 0; i < total_dim; ++i) {
            if (obs[i] < low[i] || obs[i] > high[i]) {
                return false;
            }
        }
        return true;
    }

    void Sample(Prng& rng, std::vector<f32>& out) const
    {
        out.resize(total_dim);
        for (u32 i = 0; i < total_dim; ++i) {
            const f32 r = rng.NextFloat(); // [0, 1)
            out[i] = low[i] + r * (high[i] - low[i]);
        }
    }

    [[nodiscard]] std::vector<f32> Sample(Prng& rng) const
    {
        std::vector<f32> res(total_dim);
        Sample(rng, res);
        return res;
    }
};

} // namespace rl
