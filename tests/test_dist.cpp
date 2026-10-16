#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <numeric>

#include "dist.hpp"
#include "prng.hpp"

using namespace dist;

TEST_CASE("Categorical single distribution", "[dist][categorical]")
{
    // Uniform logits -> equal probabilities
    Categorical cat_uniform(std::vector<f32>{0.0f, 0.0f, 0.0f, 0.0f});
    CHECK(cat_uniform.NumClasses() == 4);
    for (f32 p : cat_uniform.Probs()) {
        CHECK(std::fabs(p - 0.25f) < 1e-6f);
    }
    CHECK(std::fabs(cat_uniform.Entropy() - std::log(4.0f)) < 1e-5f);

    // Exact known logits [0, ln(3)] -> p = [0.25, 0.75]
    const f32 ln3 = std::log(3.0f);
    Categorical cat(std::vector<f32>{0.0f, ln3});
    CHECK(std::fabs(cat.Probs()[0] - 0.25f) < 1e-6f);
    CHECK(std::fabs(cat.Probs()[1] - 0.75f) < 1e-6f);

    CHECK(std::fabs(cat.LogProb(0) - std::log(0.25f)) < 1e-6f);
    CHECK(std::fabs(cat.LogProb(1) - std::log(0.75f)) < 1e-6f);

    const f32 expected_h = -(0.25f * std::log(0.25f) + 0.75f * std::log(0.75f));
    CHECK(std::fabs(cat.Entropy() - expected_h) < 1e-5f);

    // KL divergence to self is 0
    CHECK(std::fabs(cat.KLDivergence(cat)) < 1e-6f);

    // KL divergence between [0.25, 0.75] and [0.5, 0.5]
    Categorical cat_half(std::vector<f32>{0.0f, 0.0f});
    const f32 expected_kl = 0.25f * std::log(0.25f / 0.5f) + 0.75f * std::log(0.75f / 0.5f);
    CHECK(std::fabs(cat.KLDivergence(cat_half) - expected_kl) < 1e-5f);
}

TEST_CASE("Categorical sampling distribution matches probabilities", "[dist][categorical][statistical]")
{
    Categorical cat(std::vector<f32>{0.0f, std::log(2.0f), std::log(7.0f)}); // p = [0.1, 0.2, 0.7]
    Prng rng(42, 1);

    constexpr int kN = 30000;
    int counts[3] = {};
    for (int i = 0; i < kN; ++i) {
        counts[cat.Sample(rng)]++;
    }

    CHECK(std::fabs(static_cast<f64>(counts[0]) / kN - 0.1) < 0.015);
    CHECK(std::fabs(static_cast<f64>(counts[1]) / kN - 0.2) < 0.015);
    CHECK(std::fabs(static_cast<f64>(counts[2]) / kN - 0.7) < 0.015);
}

TEST_CASE("Categorical batched", "[dist][categorical]")
{
    // Batch of 2, 3 classes each
    std::vector<f32> logits = {
        0.0f, 0.0f, 0.0f,       // uniform
        0.0f, std::log(3.0f), 0.0f // [0.2, 0.6, 0.2]
    };
    Categorical cat(logits, 2);
    CHECK(cat.BatchSize() == 2);
    CHECK(cat.NumClasses() == 3);

    std::vector<f32> entropy(2);
    cat.Entropy(entropy);
    CHECK(std::fabs(entropy[0] - std::log(3.0f)) < 1e-5f);

    std::vector<u32> actions = {0, 1};
    std::vector<f32> lps(2);
    cat.LogProb(actions, lps);
    CHECK(std::fabs(lps[0] - std::log(1.0f / 3.0f)) < 1e-5f);
    CHECK(std::fabs(lps[1] - std::log(0.6f)) < 1e-4f);
}

TEST_CASE("DiagGaussian single distribution", "[dist][gaussian]")
{
    std::vector<f32> mu = {0.0f, 2.0f};
    std::vector<f32> log_std = {0.0f, std::log(0.5f)}; // std = [1.0, 0.5]
    DiagGaussian gauss(mu, log_std);

    CHECK(gauss.Dim() == 2);
    CHECK(gauss.BatchSize() == 1);

    // Exact log_prob at mean:
    // -0.5 * (0 + 2*0 + log(2pi)) + -0.5 * (0 + 2*ln(0.5) + log(2pi))
    const f32 expected_lp_mean = -0.5f * (kLogTwoPi) - 0.5f * (2.0f * std::log(0.5f) + kLogTwoPi);
    CHECK(std::fabs(gauss.LogProb(mu) - expected_lp_mean) < 1e-5f);

    // Entropy: 0.5 * (1 + log(2pi) + 2*log_std) summed
    const f32 expected_h = 0.5f * (1.0f + kLogTwoPi) + 0.5f * (1.0f + kLogTwoPi + 2.0f * std::log(0.5f));
    CHECK(std::fabs(gauss.Entropy() - expected_h) < 1e-5f);

    // KL divergence to self is 0
    CHECK(std::fabs(gauss.KLDivergence(gauss)) < 1e-6f);

    // Analytical KL test between N(0, 1) and N(1, 1):
    // KL = (0 + 1^2) / 2 = 0.5
    DiagGaussian p(std::vector<f32>{0.0f}, std::vector<f32>{0.0f});
    DiagGaussian q(std::vector<f32>{1.0f}, std::vector<f32>{0.0f});
    CHECK(std::fabs(p.KLDivergence(q) - 0.5f) < 1e-5f);
}

TEST_CASE("DiagGaussian sampling statistical properties", "[dist][gaussian][statistical]")
{
    const f32 target_mu = 3.5f;
    const f32 target_std = 1.8f;
    DiagGaussian gauss(std::vector<f32>{target_mu}, std::vector<f32>{std::log(target_std)});

    Prng rng(1234, 5678);
    constexpr int kN = 25000;
    f64 sum = 0.0;
    f64 sum_sq = 0.0;
    std::vector<f32> sample(1);

    for (int i = 0; i < kN; ++i) {
        gauss.Sample(rng, sample);
        sum += sample[0];
        sum_sq += static_cast<f64>(sample[0]) * sample[0];
    }

    const f64 sample_mean = sum / kN;
    const f64 sample_var  = (sum_sq / kN) - (sample_mean * sample_mean);
    const f64 sample_std  = std::sqrt(sample_var);

    CHECK(std::fabs(sample_mean - target_mu) < 0.05);
    CHECK(std::fabs(sample_std - target_std) < 0.05);
}
