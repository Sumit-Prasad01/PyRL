#include <stdexcept>

#include <catch2/catch_test_macros.hpp>

#include "env.hpp"
#include "mat.hpp"
#include "test_utils.hpp"

using namespace testutil;

TEST_CASE("SnakeEnv validates its grid size", "[env]")
{
    CHECK_NOTHROW(SnakeEnv(36));
    CHECK_NOTHROW(SnakeEnv(4));
    CHECK_THROWS_AS(SnakeEnv(35), std::invalid_argument);   // not a perfect square
    CHECK_THROWS_AS(SnakeEnv(1), std::invalid_argument);    // 1x1: food could never be placed
    CHECK_THROWS_AS(SnakeEnv(0), std::invalid_argument);
}

TEST_CASE("Reset restores a valid initial state", "[env]")
{
    SnakeEnv env(36, 99);
    for (int i = 0; i < 200; ++i) {
        env.Reset();
        CHECK(env.Snake() == Point{3, 3});                  // centre of 6x6
        CHECK(env.Food() != env.Snake());
        CHECK((env.Food().x >= 0 && env.Food().x < 6));
        CHECK((env.Food().y >= 0 && env.Food().y < 6));
        CHECK(env.Score() == 0.0f);
        CHECK(env.FoodsEaten() == 0);
        CHECK(env.Steps() == 0);
        CHECK(env.Pov() == Action::Right);
        CHECK_FALSE(env.IsGameOver());
    }
}

TEST_CASE("Actions move the snake (Up increases y, as in the original C code)", "[env]")
{
    SnakeEnv env(36, 5);
    const Point start = env.Snake();   // (3,3)

    SECTION("Left")  { (void)env.Step(Action::Left);  CHECK(env.Snake() == Point{start.x - 1, start.y}); }
    SECTION("Right") { (void)env.Step(Action::Right); CHECK(env.Snake() == Point{start.x + 1, start.y}); }
    SECTION("Up")    { (void)env.Step(Action::Up);    CHECK(env.Snake() == Point{start.x, start.y + 1}); }
    SECTION("Down")  { (void)env.Step(Action::Down);  CHECK(env.Snake() == Point{start.x, start.y - 1}); }

    SECTION("None keeps the current heading")
    {
        (void)env.Step(Action::Right);
        (void)env.Step(Action::None);
        CHECK(env.Snake() == Point{start.x + 2, start.y});
        CHECK(env.Pov() == Action::Right);
    }

    SECTION("Steps are counted")
    {
        (void)env.Step(Action::Up);
        (void)env.Step(Action::Down);
        CHECK(env.Steps() == 2);
    }
}

TEST_CASE("Leaving the grid ends the episode with the wall penalty", "[env]")
{
    SnakeEnv env(36, 3);
    SnakeEnv::StepResult last{};
    int steps = 0;
    while (!last.done && steps < 10) {
        last = env.Step(Action::Left);
        ++steps;
    }
    CHECK(last.done);
    CHECK(steps == 4);                               // x: 3 -> 2 -> 1 -> 0 -> -1
    CHECK(last.reward == SnakeEnv::kWallPenalty);    // food can never be at x = -1
    CHECK(env.IsGameOver());
}

TEST_CASE("Eating food pays the food reward and respawns the food", "[env]")
{
    for (u64 seed = 1; seed <= 20; ++seed) {
        INFO("seed " << seed);
        SnakeEnv env(36, seed);
        const Point food = env.Food();
        Point s = env.Snake();

        SnakeEnv::StepResult r{};
        // Walk along x first, then y. This stays inside the grid and reaches
        // the food cell exactly on the final step.
        while (s.x < food.x) { r = env.Step(Action::Right); ++s.x; }
        while (s.x > food.x) { r = env.Step(Action::Left);  --s.x; }
        while (s.y < food.y) { r = env.Step(Action::Up);    ++s.y; }
        while (s.y > food.y) { r = env.Step(Action::Down);  --s.y; }

        CHECK(r.reward == SnakeEnv::kFoodReward);
        CHECK_FALSE(r.done);
        CHECK(env.FoodsEaten() == 1);
        CHECK(env.Food() != env.Snake());
        CHECK(env.Score() == SnakeEnv::kFoodReward);
    }
}

TEST_CASE("EncodeState writes exactly three one-hot entries", "[env]")
{
    SnakeEnv env(36);
    REQUIRE(env.InputSize() == 76);   // 2 * 36 + 4

    Arena arena(KiB(4));
    Mat* m = MakeMat(arena, env.InputSize(), 1);
    m->Fill(9.0f);                    // must be fully overwritten

    env.EncodeState(*m, Point{1, 2}, Point{3, 4}, Action::Up);

    CHECK(SumOf(*m) == 3.0);
    CHECK(m->data[2 * 6 + 1] == 1.0f);          // snake cell (y*cols + x)
    CHECK(m->data[36 + 4 * 6 + 3] == 1.0f);     // food cell, offset by grid_size
    CHECK(m->data[72 + 2] == 1.0f);             // heading slot for Up
}

TEST_CASE("Environments with the same seed behave identically", "[env]")
{
    SnakeEnv a(36, 1234), b(36, 1234), c(36, 4321);
    bool same_ab = true, same_ac = true;
    for (int i = 0; i < 50; ++i) {
        a.Reset(); b.Reset(); c.Reset();
        same_ab &= (a.Food() == b.Food());
        same_ac &= (a.Food() == c.Food());
    }
    CHECK(same_ab);
    CHECK_FALSE(same_ac);
}
