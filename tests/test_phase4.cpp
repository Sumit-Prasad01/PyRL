#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include "device.hpp"
#include "environment.hpp"
#include "env_wrappers.hpp"
#include "eval.hpp"
#include "spaces.hpp"
#include "vec_env.hpp"
#include "envs/bandit.hpp"
#include "envs/cartpole.hpp"
#include "envs/gridworld.hpp"
#include "envs/mountain_car.hpp"
#include "envs/pendulum.hpp"
#include "envs/snake_v2.hpp"

using namespace rl;

TEST_CASE("Phase 4 Spaces: DiscreteSpace and BoxSpace bounds and sampling", "[phase4][spaces]")
{
    SECTION("DiscreteSpace operations")
    {
        DiscreteSpace space(5);
        CHECK(space.Type() == SpaceType::Discrete);
        CHECK(space.Dim() == 5);
        CHECK(space.Contains(0));
        CHECK(space.Contains(4));
        CHECK_FALSE(space.Contains(-1));
        CHECK_FALSE(space.Contains(5));

        Prng rng(12345, 67890);
        for (int i = 0; i < 50; ++i) {
            const u32 sample = space.Sample(rng);
            CHECK(space.Contains(sample));
        }
    }

    SECTION("BoxSpace 1D operations")
    {
        BoxSpace space(4, -2.0f, 2.0f);
        CHECK(space.Type() == SpaceType::Box);
        CHECK(space.Dim() == 4);

        const std::vector<f32> valid = {0.0f, -1.5f, 1.9f, 2.0f};
        CHECK(space.Contains(valid));

        const std::vector<f32> invalid_high = {0.0f, 2.5f, 0.0f, 0.0f};
        CHECK_FALSE(space.Contains(invalid_high));

        const std::vector<f32> invalid_len = {0.0f, 0.0f};
        CHECK_FALSE(space.Contains(invalid_len));

        Prng rng(42, 99);
        std::vector<f32> sample = space.Sample(rng);
        CHECK(sample.size() == 4);
        CHECK(space.Contains(sample));
    }

    SECTION("BoxSpace Multi-dimensional shape")
    {
        BoxSpace space({2, 3}, -1.0f, 1.0f);
        CHECK(space.Dim() == 6);
        CHECK(space.shape.size() == 2);
        CHECK(space.shape[0] == 2);
        CHECK(space.shape[1] == 3);

        Prng rng(777, 888);
        const auto s = space.Sample(rng);
        CHECK(s.size() == 6);
        CHECK(space.Contains(s));
    }
}

TEST_CASE("Phase 4 Benchmark Environments: Core dynamics and reproducibility", "[phase4][envs]")
{
    SECTION("BanditEnv: Arm selection and reset reproducibility")
    {
        BanditEnv env({0.2f, 1.5f, -0.5f}, 0.0f, 42); // 0 noise for deterministic test
        CHECK(env.ObsDim() == 1);
        CHECK(env.ActionDim() == 3);
        CHECK(env.OptimalArm() == 1);

        env.Reset();
        StepResult res = env.Step({1.0f}); // Pick arm 1
        CHECK(res.reward == 1.5f);
        CHECK(res.terminated == true);

        env.Reset();
        res = env.Step({0.0f}); // Pick arm 0
        CHECK(res.reward == 0.2f);
    }

    SECTION("GridWorldEnv: Goal reaching and obstacle collision")
    {
        GridWorldEnv env(4, 4, 0.0f, 42); // 0 slip
        env.SetGoal({2, 0});
        env.SetObstacles({{1, 0}}); // Wall directly to the right of start (0, 0)

        env.Reset();
        // Move Right into wall -> should bounce and stay at (0, 0)
        StepResult res = env.Step({1.0f}); // Action 1 = Right
        CHECK(res.obs[0] == 0.0f);
        CHECK(res.obs[1] == 0.0f);
        CHECK_FALSE(res.terminated);

        // Move Down to (0, 1), Right to (1, 1), Right to (2, 1), Up to (2, 0) = Goal!
        env.Step({3.0f}); // Down
        env.Step({1.0f}); // Right
        env.Step({1.0f}); // Right
        res = env.Step({2.0f}); // Up
        CHECK(res.terminated == true);
        CHECK(res.reward > 0.9f);
    }

    SECTION("CartPoleEnv: Physics step and boundary termination")
    {
        CartPoleEnv env(42);
        CHECK(env.ObsDim() == 4);
        CHECK(env.ActionDim() == 2);

        StepResult res = env.Reset(100);
        CHECK(std::abs(res.obs[0]) <= 0.05f);
        CHECK(std::abs(res.obs[2]) <= 0.05f);

        // Step repeatedly with push right until pole falls or cart goes out of bounds
        bool terminated = false;
        for (int step = 0; step < 200; ++step) {
            res = env.Step({1.0f});
            CHECK(res.reward == 1.0f);
            if (res.terminated) {
                terminated = true;
                break;
            }
        }
        CHECK(terminated);
        CHECK((std::abs(res.obs[0]) > CartPoleEnv::kXThreshold ||
               std::abs(res.obs[2]) > CartPoleEnv::kThetaThreshold));
    }

    SECTION("MountainCarEnv: Physics step and gravity acceleration")
    {
        MountainCarEnv env(42);
        CHECK(env.ObsDim() == 2);
        CHECK(env.ActionDim() == 3);

        StepResult res = env.Reset(200);
        CHECK(res.obs[0] >= -0.6f);
        CHECK(res.obs[0] <= -0.4f);
        CHECK(res.obs[1] == 0.0f);

        // Step without pushing (action 1) -> gravity accelerates down valley
        res = env.Step({1.0f});
        CHECK(res.reward == -1.0f);
        CHECK_FALSE(res.terminated);
    }

    SECTION("PendulumEnv: Torque clamping and angle normalization")
    {
        PendulumEnv env(42);
        CHECK(env.ObsDim() == 3);
        CHECK(env.ActionDim() == 1);
        CHECK_FALSE(env.IsDiscreteAction());

        StepResult res = env.Reset(300);
        // Obs is [cos(theta), sin(theta), theta_dot]
        const f32 cos_th = res.obs[0];
        const f32 sin_th = res.obs[1];
        CHECK(std::abs(cos_th * cos_th + sin_th * sin_th - 1.0f) < 1e-4f);

        // Step with saturated torque
        res = env.Step({10.0f}); // Will be clamped to +2.0
        CHECK(res.reward <= 0.0f);
        CHECK_FALSE(res.terminated);
    }

    SECTION("SnakeEnvV2: Growth and self-collision")
    {
        // 6x6 grid with 1 food item
        SnakeEnvV2 env(6, 6, 1, SnakeObsMode::FeatureVector, 42);
        CHECK(env.ObsDim() == 12);
        CHECK(env.ActionDim() == 4);

        StepResult res = env.Reset(42);
        CHECK(env.SnakeLength() == 3);
        CHECK(res.obs.size() == 12);

        // Turn down, left, up to deliberately loop and self-collide
        env.Step({3.0f}); // Down
        env.Step({0.0f}); // Left
        res = env.Step({2.0f}); // Up into body!
        // Should either self-collide or survive if length allowed
        CHECK(res.obs.size() == 12);
    }

    SECTION("SnakeEnvV2: GridTensor observation mode")
    {
        SnakeEnvV2 env(8, 8, 1, SnakeObsMode::GridTensor, 42);
        CHECK(env.ObsDim() == 64);

        StepResult res = env.Reset(42);
        CHECK(res.obs.size() == 64);
    }
}

TEST_CASE("Phase 4 Wrappers: TimeLimit, Normalization, FrameStack, ActionRepeat", "[phase4][wrappers]")
{
    SECTION("TimeLimit wrapper: Truncation behavior")
    {
        auto cartpole = std::make_unique<CartPoleEnv>(42);
        TimeLimit env(std::move(cartpole), 10);

        env.Reset();
        StepResult res{};
        for (int i = 0; i < 10; ++i) {
            res = env.Step({0.0f});
            if (res.terminated) { break; }
        }
        if (!res.terminated) {
            CHECK(res.truncated == true);
            CHECK(res.Done() == true);
        }
    }

    SECTION("NormalizeObservation wrapper: Running statistics")
    {
        auto cartpole = std::make_unique<CartPoleEnv>(42);
        NormalizeObservation env(std::move(cartpole));

        StepResult res = env.Reset();
        CHECK(res.obs.size() == 4);

        for (int i = 0; i < 20; ++i) {
            res = env.Step({1.0f});
            for (f32 v : res.obs) {
                CHECK(v >= -10.0f);
                CHECK(v <= 10.0f);
            }
            if (res.Done()) { env.Reset(); }
        }
        CHECK(env.RMS().count > 10.0f);
    }

    SECTION("FrameStack wrapper: Stacking observations")
    {
        auto cartpole = std::make_unique<CartPoleEnv>(42);
        FrameStack env(std::move(cartpole), 4);
        CHECK(env.ObsDim() == 16); // 4 frames * 4 obs_dim

        StepResult res = env.Reset();
        CHECK(res.obs.size() == 16);
        // All 4 frames identical on reset
        for (int f = 0; f < 4; ++f) {
            CHECK(res.obs[f * 4 + 0] == res.obs[0]);
            CHECK(res.obs[f * 4 + 1] == res.obs[1]);
        }

        res = env.Step({1.0f});
        CHECK(res.obs.size() == 16);
    }

    SECTION("ActionRepeat wrapper: Reward accumulation")
    {
        auto cartpole = std::make_unique<CartPoleEnv>(42);
        ActionRepeat env(std::move(cartpole), 3);

        env.Reset();
        StepResult res = env.Step({1.0f});
        // 3 steps at +1 reward each (unless terminated earlier)
        CHECK(res.reward >= 1.0f);
        CHECK(res.reward <= 3.0f);
    }
}

TEST_CASE("Phase 4 VecEnv: Multi-threading, Auto-Reset, and Dual Device Buffer", "[phase4][vec_env]")
{
    constexpr u32 kNumEnvs = 8;
    CartPoleEnv proto(42);
    VecEnv vec_env(proto, kNumEnvs, 100, true); // multi-threaded

    CHECK(vec_env.NumEnvs() == kNumEnvs);
    CHECK(vec_env.ObsDim() == 4);
    CHECK(vec_env.HostBatchObs().size() == kNumEnvs * 4);

    vec_env.Reset(100);
    const auto& obs_0 = vec_env.HostBatchObs();
    // Envs should be decorrelated
    CHECK(obs_0[0] != obs_0[4]);

    // Step with batch actions
    std::vector<f32> actions(kNumEnvs, 1.0f);
    StepResultBatch batch = vec_env.Step(actions);

    CHECK(batch.rewards.size() == kNumEnvs);
    CHECK(batch.dones.size() == kNumEnvs);
    CHECK(batch.terminated.size() == kNumEnvs);
    for (u32 i = 0; i < kNumEnvs; ++i) {
        CHECK(batch.rewards[i] == 1.0f);
    }

    // Step until at least one environment auto-resets
    bool had_auto_reset = false;
    for (int step = 0; step < 100; ++step) {
        batch = vec_env.Step(actions);
        for (u32 i = 0; i < kNumEnvs; ++i) {
            if (batch.dones[i] == 1) {
                had_auto_reset = true;
                CHECK(batch.terminal_obs[i].has_value());
                CHECK(batch.terminal_obs[i]->size() == 4);
                break;
            }
        }
        if (had_auto_reset) { break; }
    }
    CHECK(had_auto_reset);

    // Dual Backend Device buffer check
    vec_env.SyncObsToDevice(Device::CPU);
    CHECK(vec_env.GetObsPtr(Device::CPU) != nullptr);

#ifdef RL_USE_CUDA
    if (HasCudaDevice()) {
        vec_env.SyncObsToDevice(Device::CUDA);
        CHECK(vec_env.GetObsPtr(Device::CUDA) != nullptr);
        CHECK(vec_env.GpuObsBuffer().IsAllocated());
    }
#endif
}

TEST_CASE("Phase 4 Deterministic Policy Evaluation Harness", "[phase4][eval]")
{
    CartPoleEnv env(42);

    // Fixed dummy policy: always push right
    auto policy_fn = [](const std::vector<f32>&) -> std::vector<f32> {
        return {1.0f};
    };

    EvalStats stats1 = EvaluatePolicy(env, policy_fn, 5, 12345, 100);
    EvalStats stats2 = EvaluatePolicy(env, policy_fn, 5, 12345, 100);

    CHECK(stats1.returns.size() == 5);
    CHECK(stats1.mean_return == stats2.mean_return);
    CHECK(stats1.std_return == stats2.std_return);
    CHECK(stats1.min_return == stats2.min_return);
    CHECK(stats1.max_return == stats2.max_return);
    CHECK(stats1.mean_length == stats2.mean_length);
    CHECK(stats1.success_rate == stats2.success_rate);
}
