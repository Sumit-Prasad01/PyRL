#include "train.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <span>
#include <stdexcept>
#include <string>

#include "timer.hpp"

namespace {

void Check(bool ok, const char* msg)
{
    if (!ok) { throw std::logic_error(msg); }
}  

void Validate(const ModelState& m, const SnakeEnv& env, const TrainConfig& cfg)
{
    auto bad = [](const char* msg) { throw std::invalid_argument(std::string("Train: ") + msg); };
    
    if (!m.input || !m.output || !m.advantage || !m.cost) { bad("model is not fully built"); }
    if (m.forward_graph.Empty() || m.cost_graph.Empty())  { bad("model graphs are empty"); }
    if (!m.cost->RequiresGrad())                          { bad("cost does not require a gradient"); }
    if (m.input->val->Size() != env.InputSize())          { bad("model input size != env.InputSize()"); }
    if (m.output->val->Size() != kNumActions ||
        m.advantage->val->Size() != kNumActions)          { bad("model must have kNumActions outputs"); }
    if (cfg.rollout_size == 0 || cfg.episode_len == 0)    { bad("rollout_size and episode_len must be > 0"); }
}   

void CollectTrajectory(ModelState& model, SnakeEnv& env, u32 episode_len, Prng& rng, Trajectory& traj)
{
    RL_PROFILE_SCOPE("Rollout::CollectTrajectory");
    env.Reset();
    traj.transitions.clear();

    for (u32 t = 0; t < episode_len; t++) {
        Transition tr{};
        tr.state      = env.Snake();
        tr.food_state = env.Food();
        tr.pov        = env.Pov();

        env.EncodeState(*model.input->val, tr.state, tr.food_state, tr.pov);
        model.forward_graph.Forward();
        tr.action = SampleAction(*model.output->val, rng);

        const auto step = env.Step(tr.action);
        tr.reward       = step.reward;
        tr.done         = step.done;
        tr.next_state   = env.Snake();
        traj.transitions.push_back(tr);

        if (step.done) { break; }
    }
}

struct ReturnStats {
    f32 mean             = 0.0f;
    f32 stddev           = 0.0f;
    u32 count            = 0;
    f32 avg_first_return = 0.0f;
};

ReturnStats ComputeReturns(std::span<Trajectory> buffer, f32 gamma) 
{
    f64 sum = 0.0, sq_sum = 0.0, first_sum = 0.0;
    u64 count = 0;

    for (Trajectory& traj : buffer) {
        f32 g = 0.0f;
        for (std::size_t t = traj.transitions.size(); t-- > 0; ) {
            Transition& tr = traj.transitions[t];
            g = tr.reward + gamma * g;
            tr.ret  = g;
            sum    += g;
            sq_sum += static_cast<f64>(g) * g;
            ++count;
        }
        if (!traj.transitions.empty()) {
            first_sum += traj.transitions.front().ret;
        }
    }

    ReturnStats s;
    s.count = static_cast<u32>(count);
    if (count > 0) {
        const f64 mean = sum / static_cast<f64>(count);
        const f64 var = sq_sum / static_cast<f64>(count) - mean * mean;
        s.mean   = static_cast<f32>(mean);
        s.stddev = static_cast<f32>(std::sqrt(std::max(var, 0.0)));
    }
    if (!buffer.empty()) {
        s.avg_first_return = static_cast<f32>(first_sum / static_cast<f64>(buffer.size()));
    }
    return s;
}

void ZeroParameterGrads(const Graph& graph) 
{
    for (Var* v : graph.Vars()) {
        if (v->IsParameter() && v->RequiresGrad()) { v->grad->Clear(); }
    }
}

void SgdStep(const Graph& graph, f32 grad_scale)
{
    RL_PROFILE_SCOPE("Optimizer::SgdStep");
    for (Var* v : graph.Vars()) {
        if (!v->IsParameter() || !v->RequiresGrad()) { continue; }
        v->grad->Scale(grad_scale);
        Check(mat::Sub(*v->val, *v->val, *v->grad), "SGD update shape mismatch");
    }
}

} // namespace

Action SampleAction(const Mat& probs, Prng& rng) 
{
    const auto p = probs.Span();
    assert(!p.empty());

    const f32 r = rng.NextFloat();
    f32 cdf = 0.0f;
    for (std::size_t i = 0; i < p.size(); i++) {
        cdf += p[i];
        if (r < cdf) { return static_cast<Action>(i); }
    }
    return static_cast<Action>(p.size() - 1);
}

void Train(ModelState& model, SnakeEnv& env, const TrainConfig& cfg, Prng& rng,
           const EpochCallback& on_epoch)
{
    Validate(model, env, cfg);

    std::vector<Trajectory> buffer(cfg.rollout_size);
    for (Trajectory& t : buffer) { t.transitions.reserve(cfg.episode_len); }

    for (u32 epoch = 0; epoch < cfg.epochs; epoch++) {
        RL_PROFILE_SCOPE("Training::Epoch");
        u32 foods_eaten = 0;
        for (Trajectory& traj : buffer) {
            CollectTrajectory(model, env, cfg.episode_len, rng, traj);
            foods_eaten += env.FoodsEaten();
        }

        const ReturnStats stats = ComputeReturns(buffer, cfg.gamma);

        ZeroParameterGrads(model.cost_graph);

        {
            RL_PROFILE_SCOPE("Training::BackwardPasses");
            for (const Trajectory& traj : buffer) {
                for (const Transition& tr : traj.transitions) {
                    env.EncodeState(*model.input->val, tr.state, tr.food_state, tr.pov);

                    model.advantage->val->Clear();
                    const f32 advantage = (tr.ret - stats.mean) / (stats.stddev + 1e-8f);
                    model.advantage->val->Span()[static_cast<u32>(tr.action)] = advantage;

                    model.cost_graph.Forward();
                    Check(model.cost_graph.Backward(), "backward pass failed");
                }
            }
        }

        if (stats.count > 0) {
            SgdStep(model.cost_graph, cfg.learning_rate / static_cast<f32>(stats.count));
        }

        if (on_epoch) {
            on_epoch({epoch, stats.avg_first_return, foods_eaten, stats.count, stats.stddev});
        }
    }
}
