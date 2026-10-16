#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <vector>

#include "arena.hpp"
#include "autograd.hpp"
#include "nn.hpp"
#include "optimizer.hpp"
#include "prng.hpp"
#include "serialize.hpp"

namespace fs = std::filesystem;

struct TempFileGuard {
    fs::path path;
    explicit TempFileGuard(fs::path p) : path(std::move(p)) {}
    ~TempFileGuard() {
        std::error_code ec;
        fs::remove(path, ec);
    }
    TempFileGuard(const TempFileGuard&) = delete;
    TempFileGuard& operator=(const TempFileGuard&) = delete;
};

TEST_CASE("Serialization: Mat round-trip", "[serialize][mat]")
{
    Arena arena(KiB(16));
    Mat* m = Mat::Create(arena, 4, 7);
    Prng rng(42, 99);
    m->FillRand(-5.0f, 5.0f, rng);

    const fs::path test_path = fs::temp_directory_path() / "test_mat.bin";
    TempFileGuard guard(test_path);

    std::string err;
    REQUIRE(serialize::SaveMat(test_path, *m, &err));

    Mat* m_loaded = Mat::Create(arena, 4, 7);
    REQUIRE(serialize::LoadMat(test_path, *m_loaded, &err));

    // Bit-exact float comparison
    REQUIRE(std::memcmp(m->data, m_loaded->data, m->Size() * sizeof(f32)) == 0);

    SECTION("Shape mismatch rejection")
    {
        Mat* bad_shape = Mat::Create(arena, 5, 7);
        CHECK_FALSE(serialize::LoadMat(test_path, *bad_shape, &err));
    }
}

TEST_CASE("Serialization: Header inspection without full model", "[serialize][inspect]")
{
    Arena arena(KiB(32));
    auto* model = ModelState::Create(arena);
    Prng rng(111, 222);

    Var* w = Var::Create(arena, *model, 5, 5, VarFlags::Parameter | VarFlags::RequiresGrad);
    w->val->Fill(1.23f);

    std::vector<Var*> params = {w};
    optim::Adam opt(params, { .lr = 0.005f });

    const fs::path test_path = fs::temp_directory_path() / "test_inspect.bin";
    TempFileGuard guard(test_path);

    REQUIRE(serialize::SaveCheckpoint(test_path, 42, 7, rng, params, &opt));

    serialize::CheckpointInfo info;
    REQUIRE(serialize::InspectCheckpoint(test_path, info));

    CHECK(info.version == serialize::kCurrentVersion);
    CHECK(info.step == 42);
    CHECK(info.epoch == 7);
    CHECK(info.num_params == 1);
    CHECK(info.opt_type == optim::OptimizerType::Adam);
}

TEST_CASE("Serialization: Bit-exact checkpoint resume test", "[serialize][gate][resume]")
{
    // The Phase 2 Gate:
    // A network trained continuously for 10 steps must yield bit-for-bit
    // identical weights, moments, and predictions as one trained for 5 steps,
    // saved, restored into fresh memory, and trained for 5 more steps.

    const fs::path chk_path = fs::temp_directory_path() / "test_resume_checkpoint.bin";
    TempFileGuard guard(chk_path);

    constexpr u32 kInputDim  = 4;
    constexpr u32 kHiddenDim = 8;
    constexpr u32 kOutputDim = 2;

    // Fixed dummy batch of 3 samples
    const std::vector<f32> x_data = {
        0.1f, -0.2f, 0.5f, 0.9f,
        -0.8f, 0.3f, -0.1f, 0.4f,
        0.7f, 0.6f, -0.5f, -0.3f
    };
    const std::vector<f32> target_data = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        1.0f, 0.0f
    };

    // Helper to step a model on the dummy batch
    auto train_step = [&](Arena& arena, ModelState& model, nn::Module& net,
                          optim::Optimizer& opt) -> f32 {
        opt.ZeroGrad();

        Var* x = Var::Create(arena, model, 3, kInputDim);
        std::memcpy(x->val->data, x_data.data(), x_data.size() * sizeof(f32));

        Var* target = Var::Create(arena, model, 3, kOutputDim);
        std::memcpy(target->val->data, target_data.data(), target_data.size() * sizeof(f32));

        Var* logits = net.Forward(x);
        Var* loss   = ops::CrossEntropy(arena, model, logits, target);

        Graph g = Graph::Build(arena, model, loss);
        g.Forward();
        const f32 loss_val = loss->val->At(0, 0);
        REQUIRE(g.Backward());

        opt.Step();
        return loss_val;
    };

    std::vector<f32> path_a_w1_weights;
    std::vector<f32> path_a_adam_m1;
    std::vector<f32> path_a_adam_v1;
    u64 path_a_rng_state = 0, path_a_rng_inc = 0;

    // ----------------- RUN A: 10 continuous steps -----------------
    {
        Arena perm_arena(MiB(1));
        auto* model = ModelState::Create(perm_arena);
        Prng init_rng(12345, 67890);

        auto l1 = std::make_shared<nn::Linear>(perm_arena, *model, kInputDim, kHiddenDim, true, false, &init_rng);
        auto act = std::make_shared<nn::ReLU>(perm_arena, *model);
        auto l2 = std::make_shared<nn::Linear>(perm_arena, *model, kHiddenDim, kOutputDim, true, false, &init_rng);
        nn::Sequential net(l1, act, l2);

        optim::Adam opt(net.Parameters(), { .lr = 0.01f });
        Prng step_rng(555, 777);

        for (u32 step = 0; step < 10; ++step) {
            Arena step_arena(KiB(64));
            train_step(step_arena, *model, net, opt);
            step_rng.Next(); // Advance RNG
        }

        // Save reference weights and Adam state
        const auto& w1 = *l1->Weight()->val;
        path_a_w1_weights.assign(w1.data, w1.data + w1.Size());
        path_a_adam_m1 = opt.M()[0];
        path_a_adam_v1 = opt.V()[0];
        step_rng.GetState(path_a_rng_state, path_a_rng_inc);
    }

    // ----------------- RUN B: 5 steps -> Save -> Resume -> 5 steps -----------------
    {
        // Phase B1: 5 steps
        {
            Arena perm_arena(MiB(1));
            auto* model = ModelState::Create(perm_arena);
            Prng init_rng(12345, 67890); // Same seed

            auto l1 = std::make_shared<nn::Linear>(perm_arena, *model, kInputDim, kHiddenDim, true, false, &init_rng);
            auto act = std::make_shared<nn::ReLU>(perm_arena, *model);
            auto l2 = std::make_shared<nn::Linear>(perm_arena, *model, kHiddenDim, kOutputDim, true, false, &init_rng);
            nn::Sequential net(l1, act, l2);

            optim::Adam opt(net.Parameters(), { .lr = 0.01f });
            Prng step_rng(555, 777); // Same seed

            for (u32 step = 0; step < 5; ++step) {
                Arena step_arena(KiB(64));
                train_step(step_arena, *model, net, opt);
                step_rng.Next();
            }

            // Save checkpoint at step 5
            std::string err;
            REQUIRE(serialize::SaveCheckpoint(chk_path, 5, 1, step_rng, net.Parameters(), &opt, &err));
        }

        // Phase B2: Brand new instances loaded from checkpoint
        {
            Arena fresh_arena(MiB(1));
            auto* fresh_model = ModelState::Create(fresh_arena);
            Prng dummy_rng(0, 0); // Completely different dummy seed

            auto l1 = std::make_shared<nn::Linear>(fresh_arena, *fresh_model, kInputDim, kHiddenDim, true, false, &dummy_rng);
            auto act = std::make_shared<nn::ReLU>(fresh_arena, *fresh_model);
            auto l2 = std::make_shared<nn::Linear>(fresh_arena, *fresh_model, kHiddenDim, kOutputDim, true, false, &dummy_rng);
            nn::Sequential net(l1, act, l2);

            optim::Adam opt(net.Parameters(), { .lr = 0.01f });
            Prng restored_rng(0, 0);

            u64 step = 0;
            u64 epoch = 0;
            std::string err;
            REQUIRE(serialize::LoadCheckpoint(chk_path, step, epoch, restored_rng, net.Parameters(), &opt, &err));

            CHECK(step == 5);
            CHECK(epoch == 1);
            CHECK(opt.StepCount() == 5);

            // Run steps 6 through 10
            for (u32 s = static_cast<u32>(step); s < 10; ++s) {
                Arena step_arena(KiB(64));
                train_step(step_arena, *fresh_model, net, opt);
                restored_rng.Next();
            }

            // Check bit-for-bit exactness against Run A!
            const auto& w1 = *l1->Weight()->val;
            std::vector<f32> path_b_w1_weights(w1.data, w1.data + w1.Size());
            REQUIRE(path_a_w1_weights == path_b_w1_weights);

            // Bit-exact Adam moments
            REQUIRE(path_a_adam_m1 == opt.M()[0]);
            REQUIRE(path_a_adam_v1 == opt.V()[0]);

            // Bit-exact RNG state
            u64 path_b_rng_state = 0, path_b_rng_inc = 0;
            restored_rng.GetState(path_b_rng_state, path_b_rng_inc);
            CHECK(path_a_rng_state == path_b_rng_state);
            CHECK(path_a_rng_inc == path_b_rng_inc);
        }
    }
}
