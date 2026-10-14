#pragma once

#include <functional>
#include <vector>

#include "autograd.hpp"
#include "base.hpp"
#include "env.hpp"
#include "prng.hpp"


struct TrainConfig {
    u32 epochs        = 2000;
    u32 rollout_size  = 64;
    u32 episode_len   = 100;
    u32 gamma         = 0.99f;
    f32 learning_rate = 0.05f;
};

struct Transition {
    Point  state;
    Point  food_state;
    Action pov;
    Action action;
    f32    reward;
    f32    ret;
    Point  next_state;
    bool   done;
};


struct Trajectory {
    std::vector<Transition> transitions;
};


struct EpochsStats {
    u32  epochs;
    f32  average_return;
    u32  foods_eaten;
    u32  samples;
    f32  return_std;
};


using EpochCallBack = std::function<void(const EpochsStats&)>;

[[nodiscard]] Action SampleAction(const Mat& probs, Prng& rng);


void Train(ModelState& model, SnakeEnv& env, const TrainConfig& config, Prng& rng, const EpochCallBack& on_epoch = {});