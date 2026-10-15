#include <cstdio>
#include <cstdlib>
#include <exception>

#include "arena.hpp"
#include "env.hpp"
#include "model.hpp"
#include "prng.hpp"
#include "train.hpp"

int main() 
{
    try {
        Arena arena(MiB(16));

        SnakeEnv env(36);

        ActorConfig config;
        config.input_size = env.InputSize();

        Prng init_rng(42, 1);
        Prng sample_rng(7, 2);

        ModelState* model = CreateActorModel(arena, config, init_rng);

        Train(*model, env, TrainConfig{}, sample_rng, [](const EpochStats& s) {
            std::printf(
                "Epoch %u | Average return: %.3f | Foods: %u | Samples: %u | Return std: %.3f\n",
                s.epoch, s.average_return, s.foods_eaten, s.samples, s.return_std);
        });
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
