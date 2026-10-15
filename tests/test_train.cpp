#include <cmath>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "env.hpp"
#include "model.hpp"
#include "train.hpp"
#include "test_utils.hpp"

using namespace testutil;

namespace {

TrainConfig SmallConfig()
{
    TrainConfig cfg;
    cfg.epochs       = 3;
    cfg.rollout_size = 8;
    cfg.episode_len  = 30;
    return cfg;
}

// Builds everything from scratch and returns the per-epoch statistics.
std::vector<EpochStats> RunTraining(u64 init_seed, u64 sample_seed, const TrainConfig& cfg)
{
    Arena arena(MiB(4));
    SnakeEnv env(36, 11);
    ActorConfig ac;
    ac.input_size  = env.InputSize();
    ac.hidden_size = 16;
    Prng init_rng(init_seed, 1), sample_rng(sample_seed, 2);
    ModelState* model = CreateActorModel(arena, ac, init_rng);

    std::vector<EpochStats> log;
    Train(*model, env, cfg, sample_rng, [&](const EpochStats& s) { log.push_back(s); });
    return log;
}

std::vector<f32> ParamSnapshot(const ModelState& model)
{
    std::vector<f32> out;
    for (const Var* v : model.cost_graph.Vars()) {
        if (!v->IsParameter()) { continue; }
        const auto s = v->val->Span();
        out.insert(out.end(), s.begin(), s.end());
    }
    return out;
}

struct Rig {
    Arena       arena{MiB(4)};
    SnakeEnv    env{36, 11};
    Prng        init_rng{5, 1};
    Prng        sample_rng{9, 2};
    ModelState* model = nullptr;

    explicit Rig(u32 input_size = 0)
    {
        ActorConfig cfg;
        cfg.input_size  = input_size == 0 ? env.InputSize() : input_size;
        cfg.hidden_size = 16;
        model = CreateActorModel(arena, cfg, init_rng);
    }
};

} // namespace

// ------------------------------------------------------------ SampleAction

TEST_CASE("SampleAction picks the only action with non-zero probability", "[train]")
{
    Arena arena(KiB(4));
    Mat* p = MakeMat(arena, 5, 1, {0, 0, 1, 0, 0});
    Prng rng(1, 2);
    for (int i = 0; i < 1000; ++i) {
        CHECK(SampleAction(*p, rng) == Action::Up);
    }
}

TEST_CASE("SampleAction never returns a zero-probability action", "[train]")
{
    Arena arena(KiB(4));
    Mat* p = MakeMat(arena, 3, 1, {0.5f, 0.0f, 0.5f});
    Prng rng(3, 4);
    for (int i = 0; i < 10'000; ++i) {
        CHECK(SampleAction(*p, rng) != Action::Right);   // index 1
    }
}

TEST_CASE("SampleAction with zero probability on first action when r is near zero", "[train]")
{
    Arena arena(KiB(4));
    Mat* p = MakeMat(arena, 3, 1, {0.0f, 0.5f, 0.5f});
    Prng rng(123, 456);
    for (int i = 0; i < 5000; ++i) {
        CHECK(SampleAction(*p, rng) != Action::Left);
    }
}

TEST_CASE("SampleAction follows the given distribution", "[train][statistical]")
{
    Arena arena(KiB(4));
    Mat* p = MakeMat(arena, 2, 1, {0.7f, 0.3f});
    Prng rng(5, 6);

    constexpr int kN = 20'000;
    int zeros = 0;
    for (int i = 0; i < kN; ++i) {
        if (SampleAction(*p, rng) == Action::Left) { ++zeros; }
    }
    CHECK(std::fabs(static_cast<f64>(zeros) / kN - 0.7) < 0.02);   // ~6 standard errors
}

// ------------------------------------------------------------ model construction

TEST_CASE("CreateActorModel builds consistent graphs", "[model]")
{
    Rig rig;
    ModelState& m = *rig.model;

    CHECK(m.input->val->Size() == rig.env.InputSize());
    CHECK(m.output->val->Size() == kNumActions);
    CHECK(m.advantage->val->Size() == kNumActions);
    CHECK(m.forward_graph.vars[m.forward_graph.size - 1] == m.output);
    CHECK(m.cost_graph.vars[m.cost_graph.size - 1] == m.cost);

    // Forward pass produces a probability distribution.
    rig.env.EncodeState(*m.input->val, Point{1, 1}, Point{2, 3}, Action::Right);
    m.forward_graph.Forward();
    CHECK(std::fabs(SumOf(*m.output->val) - 1.0) < 1e-5);
    for (f32 p : m.output->val->Span()) {
        CHECK(p >= 0.0f);
        CHECK(p <= 1.0f);
    }

    // Weights are initialized; biases are zero.
    std::size_t nonzero_weights = 0;
    for (const Var* v : m.cost_graph.Vars()) {
        if (v->IsParameter() && v->val->cols > 1) {
            for (f32 w : v->val->Span()) { nonzero_weights += (w != 0.0f); }
        }
    }
    CHECK(nonzero_weights > 100);
}

TEST_CASE("CreateActorModel reports bad setups with exceptions", "[model]")
{
    Prng rng(1, 1);

    SECTION("arena too small")
    {
        Arena tiny(KiB(1));
        CHECK_THROWS_AS(CreateActorModel(tiny, ActorConfig{}, rng), std::runtime_error);
    }
    SECTION("zero-sized layer")
    {
        Arena arena(MiB(1));
        ActorConfig cfg;
        cfg.hidden_size = 0;
        CHECK_THROWS_AS(CreateActorModel(arena, cfg, rng), std::invalid_argument);
    }
}

// ------------------------------------------------------------ Train

TEST_CASE("Train validates that model, env and config agree", "[train]")
{
    SECTION("model input size differs from the environment's")
    {
        Rig rig(/*input_size=*/40);
        CHECK_THROWS_AS(Train(*rig.model, rig.env, SmallConfig(), rig.sample_rng),
                        std::invalid_argument);
    }
    SECTION("empty rollout")
    {
        Rig rig;
        TrainConfig cfg = SmallConfig();
        cfg.rollout_size = 0;
        CHECK_THROWS_AS(Train(*rig.model, rig.env, cfg, rig.sample_rng), std::invalid_argument);
    }
}

TEST_CASE("Train runs, reports every epoch, and updates the weights", "[train]")
{
    Rig rig;
    const std::vector<f32> before = ParamSnapshot(*rig.model);

    std::vector<EpochStats> log;
    Train(*rig.model, rig.env, SmallConfig(), rig.sample_rng,
          [&](const EpochStats& s) { log.push_back(s); });

    REQUIRE(log.size() == 3);
    for (u32 i = 0; i < log.size(); ++i) {
        CHECK(log[i].epoch == i);
        CHECK(log[i].samples > 0);
        CHECK(std::isfinite(log[i].average_return));
        CHECK(std::isfinite(log[i].return_std));
        CHECK(log[i].return_std >= 0.0f);
    }

    const std::vector<f32> after = ParamSnapshot(*rig.model);
    CHECK(before != after);
    for (f32 w : after) { CHECK(std::isfinite(w)); }
}

TEST_CASE("Training is bit-for-bit reproducible for fixed seeds", "[train][determinism]")
{
    const TrainConfig cfg = SmallConfig();
    const auto a = RunTraining(1, 2, cfg);
    const auto b = RunTraining(1, 2, cfg);
    const auto c = RunTraining(1, 3, cfg);   // different sampling seed

    REQUIRE(a.size() == b.size());
    bool differs_from_c = false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        CHECK(a[i].average_return == b[i].average_return);
        CHECK(a[i].foods_eaten    == b[i].foods_eaten);
        CHECK(a[i].samples        == b[i].samples);
        CHECK(a[i].return_std     == b[i].return_std);
        differs_from_c |= (a[i].average_return != c[i].average_return) ||
                          (a[i].samples != c[i].samples);
    }
    CHECK(differs_from_c);   // the seed actually matters
}

// Statistical: REINFORCE is noisy. This asserts only that late returns beat
// early returns. If it flakes on your machine, widen the windows or raise the
// epoch count rather than loosening it to a tautology.
TEST_CASE("REINFORCE improves the average return on Snake", "[slow][learning]")
{
    TrainConfig cfg;
    cfg.epochs       = 500;
    cfg.rollout_size = 32;
    cfg.episode_len  = 100;

    const auto log = RunTraining(1, 2, cfg);
    REQUIRE(log.size() == 500);

    auto mean = [&](std::size_t from, std::size_t to) {
        f64 s = 0.0;
        for (std::size_t i = from; i < to; ++i) { s += log[i].average_return; }
        return s / static_cast<f64>(to - from);
    };
    const f64 early = mean(0, 50);
    const f64 late  = mean(log.size() - 50, log.size());
    INFO("early=" << early << " late=" << late);
    CHECK(late > early);
}
