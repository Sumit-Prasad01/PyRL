#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <vector>

#include "arena.hpp"
#include "autograd.hpp"
#include "device.hpp"
#include "eval.hpp"
#include "prng.hpp"
#include "vec_env.hpp"
#include "envs/cartpole.hpp"
#include "envs/pendulum.hpp"
#include "algorithms/a2c.hpp"
#include "algorithms/buffer.hpp"
#include "algorithms/dqn.hpp"
#include "algorithms/policy.hpp"
#include "algorithms/ppo.hpp"
#include "algorithms/reinforce.hpp"
#include "algorithms/sac.hpp"

using namespace rl;

TEST_CASE("Phase 5 Buffers: RolloutBuffer, GAE calculation, and Replay Buffers", "[phase5][buffer]")
{
    SECTION("RolloutBuffer GAE and advantage normalization")
    {
        constexpr u32 kSteps = 4;
        constexpr u32 kEnvs  = 2;
        constexpr u32 kObsDim = 3;
        constexpr u32 kActDim = 1;

        RolloutBuffer buf(kSteps, kEnvs, kObsDim, kActDim, true);
        CHECK_FALSE(buf.IsFull());

        std::vector<f32> obs(kEnvs * kObsDim, 1.0f);
        std::vector<f32> acts(kEnvs * kActDim, 0.0f);
        std::vector<f32> rewards = {1.0f, 2.0f};
        std::vector<u8>  dones   = {0, 0};
        std::vector<f32> values  = {0.5f, 0.8f};
        std::vector<f32> log_probs = {-0.69f, -0.69f};

        for (u32 s = 0; s < kSteps; ++s) {
            buf.Insert(obs.data(), acts.data(), rewards.data(),
                       dones.data(), values.data(), log_probs.data());
        }
        CHECK(buf.IsFull());

        std::vector<f32> last_vals = {0.6f, 0.9f};
        std::vector<u8>  last_dones = {0, 0};
        buf.ComputeReturnsAndGAE(last_vals.data(), last_dones.data(), 0.99f, 0.95f);

        for (u32 i = 0; i < buf.TotalSamples(); ++i) {
            CHECK(buf.Returns()[i] > 0.0f);
            CHECK_FALSE(std::isnan(buf.Advantages()[i]));
        }

        buf.NormalizeAdvantages();
        f32 sum_adv = 0.0f;
        for (u32 i = 0; i < buf.TotalSamples(); ++i) {
            sum_adv += buf.Advantages()[i];
        }
        CHECK(std::abs(sum_adv) < 1e-4f); // Zero mean

        Prng rng(42, 99);
        auto minibatches = buf.GenerateMinibatches(4, rng);
        CHECK(minibatches.size() == 2);
        CHECK(minibatches[0].obs.size() == 4 * kObsDim);
        CHECK(minibatches[1].obs.size() == 4 * kObsDim);
    }

    SECTION("ReplayBuffer: Ring buffer mechanics and sampling")
    {
        ReplayBuffer rb(5, 2, 1, true);
        CHECK(rb.Size() == 0);

        Prng rng(111, 222);
        for (int i = 0; i < 7; ++i) {
            rb.Push({static_cast<f32>(i), 0.0f}, {0.0f}, static_cast<f32>(i), {0.0f, 0.0f}, false);
        }
        CHECK(rb.Size() == 5); // Clamped at capacity

        TransitionBatch batch = rb.Sample(3, rng);
        CHECK(batch.obs.size() == 3 * 2);
        CHECK(batch.rewards.size() == 3);
        CHECK(batch.weights.size() == 3);
    }

    SECTION("PrioritizedReplayBuffer: SumTree sampling and priority update")
    {
        PrioritizedReplayBuffer per(8, 2, 1);
        Prng rng(333, 444);

        for (int i = 0; i < 6; ++i) {
            per.Push({static_cast<f32>(i), 1.0f}, {1.0f}, 1.0f, {2.0f, 2.0f}, false);
        }
        CHECK(per.Size() == 6);

        TransitionBatch batch = per.Sample(4, rng);
        CHECK(batch.obs.size() == 4 * 2);
        CHECK(batch.weights.size() == 4);

        std::vector<f32> errors = {0.1f, 1.5f, 0.01f, 2.0f};
        per.UpdatePriorities(batch.indices, errors);
    }
}

TEST_CASE("Phase 5 Policies: ActorCritic and DuelingQNetwork", "[phase5][policy]")
{
    Arena arena(1024 * 1024);
    ModelState model;
    Prng rng(555, 666);

    SECTION("ActorCritic: Forward, Action Sampling, and Evaluation")
    {
        ActorCritic ac(arena, model, 4, 2, 32, &rng);
        CHECK(ac.ObsDim() == 4);
        CHECK(ac.ActDim() == 2);
        CHECK(ac.Parameters().size() >= 4);

        std::vector<f32> obs = {0.1f, -0.2f, 0.05f, 0.0f};
        std::vector<f32> actions;
        std::vector<f32> log_probs;
        std::vector<f32> values;
        std::vector<f32> entropies;

        ac.GetActionAndValue(obs.data(), 1, rng, actions, log_probs, values, entropies);
        CHECK(actions.size() == 1);
        CHECK((actions[0] == 0.0f || actions[0] == 1.0f));
        CHECK(log_probs[0] <= 0.0f);
        CHECK(entropies[0] > 0.0f);

        std::vector<f32> eval_log_probs;
        std::vector<f32> eval_entropies;
        std::vector<f32> eval_values;
        ac.EvaluateActions(obs.data(), actions.data(), 1, eval_log_probs, eval_entropies, eval_values);
        CHECK(std::abs(eval_log_probs[0] - log_probs[0]) < 1e-5f);
        CHECK(std::abs(eval_values[0] - values[0]) < 1e-5f);
    }

    SECTION("DuelingQNetwork: Dueling aggregation and soft target updates")
    {
        DuelingQNetwork q_online(arena, model, 4, 2, 32, true, &rng);
        DuelingQNetwork q_target(arena, model, 4, 2, 32, true, &rng);

        std::vector<f32> obs = {0.5f, -0.5f, 0.1f, -0.1f};
        std::vector<f32> q_vals;
        q_online.Forward(obs.data(), 1, q_vals);
        CHECK(q_vals.size() == 2);

        std::vector<u32> acts;
        q_online.ActGreedy(obs.data(), 1, acts);
        CHECK(acts.size() == 1);
        CHECK(acts[0] == ((q_vals[1] > q_vals[0]) ? 1u : 0u));

        // Soft update test
        q_target.CopyWeightsFrom(q_online);
        q_target.SoftUpdateFrom(q_online, 0.1f);
    }
}

TEST_CASE("Phase 5 Algorithms: REINFORCE and A2C training steps", "[phase5][onpolicy]")
{
    Arena arena(2 * 1024 * 1024);
    ModelState model;
    Prng rng(777, 888);

    SECTION("REINFORCE: TrainEpisode on CartPole")
    {
        CartPoleEnv env(42);
        REINFORCEConfig cfg{};
        cfg.lr = 1e-3f;
        cfg.hidden_dim = 32;

        REINFORCE agent(arena, model, 4, 2, cfg, &rng);
        REINFORCEStats stats = agent.TrainEpisode(env, rng);

        CHECK(stats.episode_length > 0);
        CHECK(stats.episode_return > 0.0f);
        CHECK_FALSE(std::isnan(stats.policy_loss));
        CHECK_FALSE(std::isnan(stats.value_loss));
    }

    SECTION("A2C: TrainStep across parallel VecEnv")
    {
        CartPoleEnv proto(42);
        VecEnv vec_env(proto, 4, 100, false);

        A2CConfig cfg{};
        cfg.n_steps = 4;
        cfg.hidden_dim = 32;

        A2C agent(arena, model, 4, 2, cfg, &rng);
        A2CStats stats = agent.TrainStep(vec_env, rng);

        CHECK_FALSE(std::isnan(stats.policy_loss));
        CHECK_FALSE(std::isnan(stats.value_loss));
        CHECK(stats.entropy > 0.0f);
        CHECK(stats.grad_norm >= 0.0f);
    }
}

TEST_CASE("Phase 5 Algorithms: PPO complete training iteration", "[phase5][ppo]")
{
    Arena arena(2 * 1024 * 1024);
    ModelState model;
    Prng rng(999, 1111);

    CartPoleEnv proto(42);
    VecEnv vec_env(proto, 4, 200, false);

    PPOConfig cfg{};
    cfg.num_steps       = 16;
    cfg.num_epochs      = 2;
    cfg.num_minibatches = 2;
    cfg.hidden_dim      = 32;

    PPO agent(arena, model, 4, 2, cfg, &rng);
    PPOStats stats = agent.TrainIteration(vec_env, rng, 1.0f);

    CHECK_FALSE(std::isnan(stats.policy_loss));
    CHECK_FALSE(std::isnan(stats.value_loss));
    CHECK(stats.entropy > 0.0f);
    CHECK(stats.approx_kl >= 0.0f);
    CHECK(stats.clip_fraction >= 0.0f);
    CHECK(stats.clip_fraction <= 1.0f);
}

TEST_CASE("Phase 5 Algorithms: DQN and SAC training steps", "[phase5][offpolicy]")
{
    Arena arena(4 * 1024 * 1024);
    ModelState model;
    Prng rng(2222, 3333);

    SECTION("DQN: Step on CartPole")
    {
        CartPoleEnv env(42);
        DQNConfig cfg{};
        cfg.batch_size  = 8;
        cfg.buffer_size = 100;
        cfg.hidden_dim  = 32;

        DQN agent(arena, model, 4, 2, cfg, &rng);

        StepResult step = env.Reset();
        DQNStats stats{};
        for (int i = 0; i < 15; ++i) {
            const u32 act = agent.Act(step.obs, rng);
            StepResult next_step = env.Step({static_cast<f32>(act)});
            stats = agent.TrainStep(step.obs, act, next_step.reward, next_step.obs, next_step.Done(), rng);
            if (next_step.Done()) { step = env.Reset(); }
            else { step = next_step; }
        }

        CHECK(stats.train_step == 15);
        CHECK(stats.epsilon <= 1.0f);
        CHECK_FALSE(std::isnan(stats.loss));
    }

    SECTION("SAC: Continuous Step on Pendulum")
    {
        PendulumEnv env(42);
        SACConfig cfg{};
        cfg.batch_size  = 8;
        cfg.buffer_size = 100;
        cfg.hidden_dim  = 32;

        SAC agent(arena, model, 3, 1, cfg, &rng);

        StepResult step = env.Reset();
        SACStats stats{};
        for (int i = 0; i < 15; ++i) {
            const auto act = agent.Act(step.obs, rng, false);
            CHECK(act.size() == 1);
            CHECK(act[0] >= -1.0f);
            CHECK(act[0] <= 1.0f);

            StepResult next_step = env.Step(act);
            stats = agent.TrainStep(step.obs, act, next_step.reward, next_step.obs, next_step.Done(), rng);
            step = next_step;
        }

        CHECK(stats.train_step == 15);
        CHECK(stats.alpha > 0.0f);
        CHECK_FALSE(std::isnan(stats.actor_loss));
        CHECK_FALSE(std::isnan(stats.critic_loss));
    }
}
