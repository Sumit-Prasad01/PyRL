#pragma once

#include <cmath>
#include <cassert>
#include <numbers>
#include <span>
#include <vector>

#include "base.hpp"
#include "mat.hpp"
#include "prng.hpp"

namespace dist {

inline constexpr f32 kPi = 3.14159265358979323846f;
inline constexpr f32 kTwoPi = 6.28318530717958647692f;
inline constexpr f32 kLogTwoPi = 1.83787706640934548356f;

// Standard normal sample using Box-Muller transform
inline f32 SampleStandardNormal(Prng& rng) noexcept
{
    f32 u1 = rng.NextFloat();
    while (u1 <= 1e-7f) {
        u1 = rng.NextFloat();
    }
    const f32 u2 = rng.NextFloat();
    const f32 radius = std::sqrt(-2.0f * std::log(u1));
    const f32 theta  = kTwoPi * u2;
    return radius * std::cos(theta);
}

// Categorical distribution over discrete outcomes {0, 1, ..., K-1}.
// Supports single distribution or batch of distributions [B, K].
class Categorical {
public:
    // Construct from unnormalized logits [B, K] or [K, 1]
    explicit Categorical(const Mat& logits)
    {
        InitFromLogits(logits.Span(), (logits.cols == 1) ? 1 : logits.rows, (logits.cols == 1) ? logits.rows : logits.cols);
    }

    explicit Categorical(std::span<const f32> logits, u32 batch_size = 1)
    {
        assert(batch_size > 0 && logits.size() % batch_size == 0);
        const u32 num_classes = static_cast<u32>(logits.size() / batch_size);
        InitFromLogits(logits, batch_size, num_classes);
    }

    [[nodiscard]] u32 BatchSize() const noexcept { return batch_size_; }
    [[nodiscard]] u32 NumClasses() const noexcept { return num_classes_; }

    [[nodiscard]] std::span<const f32> Probs() const noexcept { return probs_; }
    [[nodiscard]] std::span<const f32> LogProbs() const noexcept { return log_probs_; }

    // Sample an action for each batch element
    void Sample(Prng& rng, std::span<u32> out_actions) const
    {
        assert(out_actions.size() == batch_size_);
        for (u32 b = 0; b < batch_size_; ++b) {
            const u64 offset = static_cast<u64>(b) * num_classes_;
            const f32 u = rng.NextFloat();
            f32 cum = 0.0f;
            u32 chosen = num_classes_ - 1;
            for (u32 c = 0; c < num_classes_; ++c) {
                cum += probs_[offset + c];
                if (u <= cum) {
                    chosen = c;
                    break;
                }
            }
            out_actions[b] = chosen;
        }
    }

    [[nodiscard]] u32 Sample(Prng& rng) const
    {
        assert(batch_size_ == 1);
        u32 a = 0;
        Sample(rng, std::span<u32>(&a, 1));
        return a;
    }

    // Evaluate log-probability of given actions for each batch element
    void LogProb(std::span<const u32> actions, std::span<f32> out_log_probs) const
    {
        assert(actions.size() == batch_size_ && out_log_probs.size() == batch_size_);
        for (u32 b = 0; b < batch_size_; ++b) {
            const u32 a = actions[b];
            assert(a < num_classes_);
            out_log_probs[b] = log_probs_[static_cast<u64>(b) * num_classes_ + a];
        }
    }

    [[nodiscard]] f32 LogProb(u32 action) const
    {
        assert(batch_size_ == 1);
        f32 lp = 0.0f;
        LogProb(std::span<const u32>(&action, 1), std::span<f32>(&lp, 1));
        return lp;
    }

    // Entropy H = -sum(p * log_p) for each batch element
    void Entropy(std::span<f32> out_entropy) const
    {
        assert(out_entropy.size() == batch_size_);
        for (u32 b = 0; b < batch_size_; ++b) {
            const u64 offset = static_cast<u64>(b) * num_classes_;
            f32 h = 0.0f;
            for (u32 c = 0; c < num_classes_; ++c) {
                const f32 p = probs_[offset + c];
                if (p > 1e-12f) {
                    h -= p * log_probs_[offset + c];
                }
            }
            out_entropy[b] = h;
        }
    }

    [[nodiscard]] f32 Entropy() const
    {
        assert(batch_size_ == 1);
        f32 h = 0.0f;
        Entropy(std::span<f32>(&h, 1));
        return h;
    }

    // KL divergence KL(this || other) for each batch element
    void KLDivergence(const Categorical& other, std::span<f32> out_kl) const
    {
        assert(batch_size_ == other.batch_size_ && num_classes_ == other.num_classes_);
        assert(out_kl.size() == batch_size_);
        for (u32 b = 0; b < batch_size_; ++b) {
            const u64 offset = static_cast<u64>(b) * num_classes_;
            f32 kl = 0.0f;
            for (u32 c = 0; c < num_classes_; ++c) {
                const f32 p = probs_[offset + c];
                if (p > 1e-12f) {
                    const f32 log_p = log_probs_[offset + c];
                    const f32 log_q = other.log_probs_[offset + c];
                    kl += p * (log_p - log_q);
                }
            }
            out_kl[b] = std::max(0.0f, kl);
        }
    }

    [[nodiscard]] f32 KLDivergence(const Categorical& other) const
    {
        assert(batch_size_ == 1);
        f32 kl = 0.0f;
        KLDivergence(other, std::span<f32>(&kl, 1));
        return kl;
    }

private:
    void InitFromLogits(std::span<const f32> logits, u32 batch_size, u32 num_classes)
    {
        batch_size_  = batch_size;
        num_classes_ = num_classes;
        const u64 total = static_cast<u64>(batch_size) * num_classes;
        probs_.resize(total);
        log_probs_.resize(total);

        for (u32 b = 0; b < batch_size; ++b) {
            const u64 offset = static_cast<u64>(b) * num_classes;
            f32 max_val = logits[offset];
            for (u32 c = 1; c < num_classes; ++c) {
                if (logits[offset + c] > max_val) {
                    max_val = logits[offset + c];
                }
            }

            f32 sum_exp = 0.0f;
            for (u32 c = 0; c < num_classes; ++c) {
                probs_[offset + c] = std::exp(logits[offset + c] - max_val);
                sum_exp += probs_[offset + c];
            }

            const f32 log_sum_exp = max_val + std::log(sum_exp);
            const f32 inv_sum = 1.0f / sum_exp;
            for (u32 c = 0; c < num_classes; ++c) {
                probs_[offset + c]     *= inv_sum;
                log_probs_[offset + c]  = logits[offset + c] - log_sum_exp;
            }
        }
    }

    u32 batch_size_  = 0;
    u32 num_classes_ = 0;
    std::vector<f32> probs_;
    std::vector<f32> log_probs_;
};

// Diagonal Gaussian (Normal) distribution for continuous actions.
// Parameterized by mean and log_std (or std).
class DiagGaussian {
public:
    DiagGaussian(std::span<const f32> mean, std::span<const f32> log_std, u32 batch_size = 1)
        : batch_size_(batch_size)
    {
        assert(batch_size > 0 && mean.size() == log_std.size() && mean.size() % batch_size == 0);
        dim_ = static_cast<u32>(mean.size() / batch_size);
        mean_.assign(mean.begin(), mean.end());
        log_std_.assign(log_std.begin(), log_std.end());
        std_.resize(mean.size());
        for (u64 i = 0; i < mean.size(); ++i) {
            std_[i] = std::exp(log_std_[i]);
        }
    }

    [[nodiscard]] u32 BatchSize() const noexcept { return batch_size_; }
    [[nodiscard]] u32 Dim() const noexcept { return dim_; }

    [[nodiscard]] std::span<const f32> Mean() const noexcept { return mean_; }
    [[nodiscard]] std::span<const f32> Std() const noexcept { return std_; }
    [[nodiscard]] std::span<const f32> LogStd() const noexcept { return log_std_; }

    // Sample using reparameterization: x = mean + std * z
    void Sample(Prng& rng, std::span<f32> out_samples) const
    {
        assert(out_samples.size() == mean_.size());
        for (u64 i = 0; i < mean_.size(); ++i) {
            const f32 z = SampleStandardNormal(rng);
            out_samples[i] = mean_[i] + std_[i] * z;
        }
    }

    // Evaluate log-probability: sum over dimensions for each batch element
    void LogProb(std::span<const f32> value, std::span<f32> out_log_probs) const
    {
        assert(value.size() == mean_.size() && out_log_probs.size() == batch_size_);
        for (u32 b = 0; b < batch_size_; ++b) {
            const u64 offset = static_cast<u64>(b) * dim_;
            f32 lp = 0.0f;
            for (u32 d = 0; d < dim_; ++d) {
                const u64 idx = offset + d;
                const f32 diff = (value[idx] - mean_[idx]) / std_[idx];
                lp -= 0.5f * (diff * diff + 2.0f * log_std_[idx] + kLogTwoPi);
            }
            out_log_probs[b] = lp;
        }
    }

    [[nodiscard]] f32 LogProb(std::span<const f32> value) const
    {
        assert(batch_size_ == 1);
        f32 lp = 0.0f;
        LogProb(value, std::span<f32>(&lp, 1));
        return lp;
    }

    // Differential entropy: 0.5 * sum(1 + log(2*pi) + 2*log_std) for each batch element
    void Entropy(std::span<f32> out_entropy) const
    {
        assert(out_entropy.size() == batch_size_);
        for (u32 b = 0; b < batch_size_; ++b) {
            const u64 offset = static_cast<u64>(b) * dim_;
            f32 h = 0.0f;
            for (u32 d = 0; d < dim_; ++d) {
                const u64 idx = offset + d;
                h += 0.5f * (1.0f + kLogTwoPi + 2.0f * log_std_[idx]);
            }
            out_entropy[b] = h;
        }
    }

    [[nodiscard]] f32 Entropy() const
    {
        assert(batch_size_ == 1);
        f32 h = 0.0f;
        Entropy(std::span<f32>(&h, 1));
        return h;
    }

    // KL divergence KL(this || other) for each batch element
    // KL(p || q) = sum( log(std_q / std_p) + (std_p^2 + (mu_p - mu_q)^2)/(2 * std_q^2) - 0.5 )
    void KLDivergence(const DiagGaussian& other, std::span<f32> out_kl) const
    {
        assert(batch_size_ == other.batch_size_ && dim_ == other.dim_);
        assert(out_kl.size() == batch_size_);
        for (u32 b = 0; b < batch_size_; ++b) {
            const u64 offset = static_cast<u64>(b) * dim_;
            f32 kl = 0.0f;
            for (u32 d = 0; d < dim_; ++d) {
                const u64 idx = offset + d;
                const f32 sp = std_[idx];
                const f32 sq = other.std_[idx];
                const f32 diff = mean_[idx] - other.mean_[idx];
                kl += (other.log_std_[idx] - log_std_[idx]) +
                      (sp * sp + diff * diff) / (2.0f * sq * sq) - 0.5f;
            }
            out_kl[b] = std::max(0.0f, kl);
        }
    }

    [[nodiscard]] f32 KLDivergence(const DiagGaussian& other) const
    {
        assert(batch_size_ == 1);
        f32 kl = 0.0f;
        KLDivergence(other, std::span<f32>(&kl, 1));
        return kl;
    }

private:
    u32 batch_size_ = 0;
    u32 dim_        = 0;
    std::vector<f32> mean_;
    std::vector<f32> log_std_;
    std::vector<f32> std_;
};

} // namespace dist
