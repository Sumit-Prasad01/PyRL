#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "arena.hpp"
#include "autograd.hpp"
#include "optimizer.hpp"

using namespace optim;

TEST_CASE("SGD hand-computed steps and convergence on quadratic", "[optimizer][sgd]")
{
    Arena arena(KiB(64));
    auto* model = ModelState::Create(arena);
    Var* theta = Var::Create(arena, *model, 1, 1, VarFlags::Parameter | VarFlags::RequiresGrad);
    theta->val->data[0] = 4.0f;

    SECTION("vanilla SGD hand-computed first steps")
    {
        SGD opt({theta}, SGDConfig{.lr = 0.1f, .momentum = 0.0f, .dampening = 0.0f, .weight_decay = 0.0f, .nesterov = false});

        // Step 1: grad = 4.0 -> theta = 4.0 - 0.1 * 4.0 = 3.6
        theta->grad->data[0] = 4.0f;
        opt.Step();
        CHECK(std::fabs(theta->val->data[0] - 3.6f) < 1e-6f);

        // Step 2: grad = 3.6 -> theta = 3.6 - 0.1 * 3.6 = 3.24
        theta->grad->data[0] = 3.6f;
        opt.Step();
        CHECK(std::fabs(theta->val->data[0] - 3.24f) < 1e-6f);
    }

    SECTION("SGD with momentum hand-computed step")
    {
        SGD opt({theta}, SGDConfig{.lr = 0.1f, .momentum = 0.9f, .dampening = 0.0f, .weight_decay = 0.0f, .nesterov = false});

        // Step 1: v = 4.0, theta = 3.6
        theta->grad->data[0] = 4.0f;
        opt.Step();
        CHECK(std::fabs(theta->val->data[0] - 3.6f) < 1e-6f);

        // Step 2: v = 0.9 * 4.0 + 3.6 = 7.2, theta = 3.6 - 0.1 * 7.2 = 2.88
        theta->grad->data[0] = 3.6f;
        opt.Step();
        CHECK(std::fabs(theta->val->data[0] - 2.88f) < 1e-6f);
    }

    SECTION("SGD converges to minimum of convex bowl f(x) = x^2")
    {
        SGD opt({theta}, SGDConfig{.lr = 0.05f, .momentum = 0.8f});
        for (int step = 0; step < 100; ++step) {
            theta->grad->data[0] = 2.0f * theta->val->data[0]; // d(x^2)/dx = 2x
            opt.Step();
        }
        CHECK(std::fabs(theta->val->data[0]) < 1e-4f);
    }
}

TEST_CASE("Adam hand-computed first step and convergence", "[optimizer][adam]")
{
    Arena arena(KiB(64));
    auto* model = ModelState::Create(arena);
    Var* theta = Var::Create(arena, *model, 1, 1, VarFlags::Parameter | VarFlags::RequiresGrad);
    theta->val->data[0] = 4.0f;

    SECTION("hand-computed first step")
    {
        Adam opt({theta}, AdamConfig{.lr = 0.1f, .beta1 = 0.9f, .beta2 = 0.999f, .eps = 1e-8f, .weight_decay = 0.0f});

        // g = 4.0 -> m_hat = 4.0, v_hat = 16.0 -> step = 0.1 * 4.0 / 4.0 = 0.1 -> theta = 3.9
        theta->grad->data[0] = 4.0f;
        opt.Step();
        CHECK(std::fabs(theta->val->data[0] - 3.9f) < 1e-5f);
        CHECK(opt.StepCount() == 1);
    }

    SECTION("Adam converges on 2D Rosenbrock-like convex bowl f(x, y) = x^2 + 2y^2")
    {
        Var* p = Var::Create(arena, *model, 2, 1, VarFlags::Parameter | VarFlags::RequiresGrad);
        p->val->data[0] = 5.0f;
        p->val->data[1] = -4.0f;

        Adam opt({p}, AdamConfig{.lr = 0.05f});
        for (int step = 0; step < 350; ++step) {
            p->grad->data[0] = 2.0f * p->val->data[0];
            p->grad->data[1] = 4.0f * p->val->data[1];
            opt.Step();
        }

        CHECK(std::fabs(p->val->data[0]) < 1e-2f);
        CHECK(std::fabs(p->val->data[1]) < 1e-2f);
    }
}

TEST_CASE("AdamW decoupled weight decay difference", "[optimizer][adamw]")
{
    Arena arena(KiB(64));
    auto* model = ModelState::Create(arena);

    Var* p_adam  = Var::Create(arena, *model, 1, 1, VarFlags::Parameter | VarFlags::RequiresGrad);
    Var* p_adamw = Var::Create(arena, *model, 1, 1, VarFlags::Parameter | VarFlags::RequiresGrad);
    p_adam->val->data[0]  = 2.0f;
    p_adamw->val->data[0] = 2.0f;

    Adam  opt_adam({p_adam}, AdamConfig{.lr = 0.1f, .weight_decay = 0.1f});
    AdamW opt_adamw({p_adamw}, AdamConfig{.lr = 0.1f, .weight_decay = 0.1f});

    p_adam->grad->data[0]  = 1.0f;
    p_adamw->grad->data[0] = 1.0f;

    opt_adam.Step();
    opt_adamw.Step();

    // In Adam, weight decay alters gradient before moment estimates: g = 1.0 + 0.1 * 2.0 = 1.2
    // In AdamW, weight decay directly decays parameter: theta -= lr * wd * theta, while moments use raw g=1.0
    // They should produce different values!
    CHECK(p_adam->val->data[0] != p_adamw->val->data[0]);

    // Hand check AdamW:
    // raw g = 1.0 -> m_hat = 1.0, v_hat = 1.0 -> update = 0.1
    // decoupled decay = 2.0 - 0.1 * 0.1 * 2.0 = 1.98
    // final theta = 1.98 - 0.1 = 1.88
    CHECK(std::fabs(p_adamw->val->data[0] - 1.88f) < 1e-4f);
}

TEST_CASE("RMSProp convergence", "[optimizer][rmsprop]")
{
    Arena arena(KiB(64));
    auto* model = ModelState::Create(arena);
    Var* theta = Var::Create(arena, *model, 1, 1, VarFlags::Parameter | VarFlags::RequiresGrad);
    theta->val->data[0] = 3.0f;

    RMSProp opt({theta}, RMSPropConfig{.lr = 0.05f});
    for (int step = 0; step < 150; ++step) {
        theta->grad->data[0] = 2.0f * theta->val->data[0];
        opt.Step();
    }
    CHECK(std::fabs(theta->val->data[0]) < 1e-2f);
}

TEST_CASE("Gradient clipping", "[optimizer][clipping]")
{
    Arena arena(KiB(64));
    auto* model = ModelState::Create(arena);

    Var* a = Var::Create(arena, *model, 2, 1, VarFlags::Parameter | VarFlags::RequiresGrad);
    Var* b = Var::Create(arena, *model, 2, 1, VarFlags::Parameter | VarFlags::RequiresGrad);

    // a->grad = [3, 4] -> norm = 5
    // b->grad = [0, 0] -> total norm = 5
    a->grad->data[0] = 3.0f;
    a->grad->data[1] = 4.0f;
    b->grad->data[0] = 0.0f;
    b->grad->data[1] = 0.0f;

    SECTION("ClipGradNorm scales down when norm > max_norm")
    {
        const f32 norm = ClipGradNorm(std::vector<Var*>{a, b}, 2.5f);
        CHECK(std::fabs(norm - 5.0f) < 1e-5f);
        CHECK(std::fabs(a->grad->data[0] - 1.5f) < 1e-4f);
        CHECK(std::fabs(a->grad->data[1] - 2.0f) < 1e-4f);
    }

    SECTION("ClipGradNorm is no-op when norm <= max_norm")
    {
        const f32 norm = ClipGradNorm(std::vector<Var*>{a, b}, 10.0f);
        CHECK(std::fabs(norm - 5.0f) < 1e-5f);
        CHECK(a->grad->data[0] == 3.0f);
        CHECK(a->grad->data[1] == 4.0f);
    }

    SECTION("ClipGradValue clamps elementwise")
    {
        a->grad->data[0] = 10.0f;
        a->grad->data[1] = -12.0f;
        ClipGradValue(std::vector<Var*>{a, b}, 2.0f);
        CHECK(a->grad->data[0] == 2.0f);
        CHECK(a->grad->data[1] == -2.0f);
    }
}

TEST_CASE("Learning rate schedules", "[optimizer][scheduler]")
{
    Arena arena(KiB(64));
    auto* model = ModelState::Create(arena);
    Var* theta = Var::Create(arena, *model, 1, 1, VarFlags::Parameter | VarFlags::RequiresGrad);

    SGD opt({theta}, SGDConfig{.lr = 0.1f});

    SECTION("LinearAnnealLR")
    {
        LinearAnnealLR sched(opt, 0.1f, 0.01f, 100);
        sched.Step(0);
        CHECK(std::fabs(opt.GetLr() - 0.1f) < 1e-6f);

        sched.Step(50);
        CHECK(std::fabs(opt.GetLr() - 0.055f) < 1e-6f);

        sched.Step(100);
        CHECK(std::fabs(opt.GetLr() - 0.01f) < 1e-6f);

        sched.Step(150); // beyond total_steps clamped
        CHECK(std::fabs(opt.GetLr() - 0.01f) < 1e-6f);
    }

    SECTION("CosineAnnealLR")
    {
        CosineAnnealLR sched(opt, 0.1f, 0.0f, 100);
        sched.Step(0);
        CHECK(std::fabs(opt.GetLr() - 0.1f) < 1e-6f);

        sched.Step(50); // at half-period, cos(pi/2) = 0 -> lr = 0.05
        CHECK(std::fabs(opt.GetLr() - 0.05f) < 1e-5f);

        sched.Step(100); // cos(pi) = -1 -> lr = 0.0
        CHECK(std::fabs(opt.GetLr() - 0.0f) < 1e-6f);
    }
}
