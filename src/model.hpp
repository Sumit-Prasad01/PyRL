#pragma once

#include "arena.hpp"
#include "autograd.hpp"
#include "base.hpp"
#include "prng.hpp"

struct ActorConfig {
    u32 input_size  = 76;
    u32 hidden_size = 128;
    u32 num_actions = 5;
};

[[nodiscard]] ModelState* CreateActorModel(Arena& arena, const ActorConfig& config, Prng& rng);
