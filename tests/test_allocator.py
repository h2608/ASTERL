import numpy as np

from asterl.common.config import Config
from asterl.controller.allocator import FixedStageController, SGSAController
from asterl.controller.signals import SignalTracker


def make_tracker(fitnesses, pop_size=5):
    tracker = SignalTracker(pop_size, window_k=5, improve_eps=0.01, s_max=50_000)
    for i, f in enumerate(fitnesses):
        tracker.record_eval(i, f, env_steps=i)
    return tracker


def test_softmax_near_uniform_at_g0():
    cfg = Config(pop_size=5)
    ctrl = SGSAController(cfg)
    tracker = make_tracker([10.0, 20.0, 30.0, 40.0, 50.0])
    tracker.last_improve_step = 0
    plan = ctrl.plan(tracker, env_steps=0, diversity=None)  # g = 0
    assert plan.g == 0.0
    # at tau_max the max/min allocation ratio stays below e^(1/tau_max)
    assert plan.probs.max() / plan.probs.min() <= np.exp(1.0 / cfg.tau_max) + 1e-9
    assert np.isclose(plan.probs.sum(), 1.0)


def test_softmax_collapses_at_g1():
    cfg = Config(pop_size=5)
    ctrl = SGSAController(cfg)
    tracker = make_tracker([10.0, 20.0, 30.0, 40.0, 50.0])
    tracker.last_improve_step = 0
    plan = ctrl.plan(tracker, env_steps=10 * cfg.s_max, diversity=None)  # g = 1
    assert plan.g == 1.0
    assert plan.probs[4] > 0.99  # collapses onto the best individual
    assert np.isclose(plan.probs.sum(), 1.0)


def test_alpha_anneal_collapses_despite_bad_delta():
    """The v1 failure: the best individual stagnates, so its delta-rank is
    bottom and half the score mass shifts off it exactly at g=1 (observed
    p_max ~0.8 instead of ~1). Annealing alpha -> 1 with g restores collapse."""
    tracker = SignalTracker(5, window_k=5, improve_eps=0.01, s_max=50_000)
    for t in range(3):
        for i in range(4):
            tracker.record_eval(i, 10.0 * i + t, env_steps=t)  # improving
        tracker.record_eval(4, 50.0, env_steps=t)  # best but flat
    tracker.last_improve_step = 0
    steps = 10 * Config().s_max

    plan = SGSAController(Config(pop_size=5)).plan(tracker, steps, diversity=None)
    assert plan.g == 1.0
    assert plan.probs[4] > 0.99

    v1 = SGSAController(Config(pop_size=5, alpha_anneal=False)).plan(
        tracker, steps, diversity=None
    )
    assert v1.probs[4] < 0.9  # mass leaks to the improvers


def test_kappa_sharpens_gradient_weights():
    cfg = Config(pop_size=5, kappa=2.0)
    ctrl = SGSAController(cfg)
    tracker = make_tracker([10.0, 20.0, 30.0, 40.0, 50.0])
    plan = ctrl.plan(tracker, env_steps=cfg.s_max // 2, diversity=None)  # 0 < g < 1
    assert np.isclose(plan.grad_weights.sum(), 1.0)
    # kappa > 1 concentrates gradient mass more than rollout mass
    assert plan.grad_weights.max() > plan.probs.max()


def test_diversity_floor_attenuates_g():
    cfg = Config(pop_size=5)
    ctrl = SGSAController(cfg)
    tracker = make_tracker([1.0, 2.0, 3.0, 4.0, 5.0])
    steps = 10 * cfg.s_max
    plan_ok = ctrl.plan(tracker, steps, diversity=cfg.d_min * 2)
    assert plan_ok.g == 1.0
    plan_low = ctrl.plan(tracker, steps, diversity=cfg.d_min / 2)
    assert plan_low.g == 0.5  # one halving
    plan_lower = ctrl.plan(tracker, steps, diversity=cfg.d_min / 2)
    assert plan_lower.g == 0.25


def test_unevaluated_individuals_get_top_score():
    cfg = Config(pop_size=5)
    ctrl = SGSAController(cfg)
    tracker = make_tracker([10.0, 20.0])  # individuals 2,3,4 never evaluated
    plan = ctrl.plan(tracker, env_steps=0, diversity=None)
    # optimism under uncertainty: unevaluated >= evaluated allocation
    assert plan.probs[2] >= plan.probs.max() - 1e-12


def test_fixed_stage_recovers_terl_schedule():
    cfg = Config(pop_size=5, ratio=0.25, max_timesteps=1_000_000)
    ctrl = FixedStageController(cfg)
    tracker = make_tracker([1.0, 2.0, 3.0, 4.0, 5.0])
    stage1 = ctrl.plan(tracker, env_steps=100_000, diversity=None)
    assert stage1.g == 0.0
    assert np.allclose(stage1.probs, 0.2)
    assert stage1.pso_interval == 1e4
    stage2 = ctrl.plan(tracker, env_steps=250_000, diversity=None)
    assert stage2.g == 1.0
    assert stage2.probs[4] == 1.0
    assert stage2.pso_interval == 1e3
