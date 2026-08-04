import numpy as np
import pytest

from asterl.algos.aterl import apportion
from asterl.common.config import Config
from asterl.controller.allocator import (
    FixedStageController,
    SGSAController,
    make_controller,
)
from asterl.controller.signals import SignalTracker


def make_tracker(fitnesses, pop_size=5):
    tracker = SignalTracker(pop_size, window_k=5, improve_eps=0.01, s_max=50_000)
    for i, f in enumerate(fitnesses):
        tracker.record_eval(i, f, env_steps=i)
    return tracker


def test_open_regime_is_winner_take_most():
    """TERL's stage 1 is not uniform: the most-recently-improved individual
    gets 6/10 of episodes AND gradients (paper Algorithm 1, line 8). At g=0
    the softmax generalization must give a hot leader (top level AND top
    delta rank) a comparable share, with the floor keeping coverage."""
    cfg = Config(pop_size=5)
    ctrl = SGSAController(cfg)
    tracker = make_tracker([10.0, 20.0, 30.0, 40.0])
    tracker.record_eval(4, 40.0, env_steps=4)
    tracker.record_eval(4, 50.0, env_steps=5)  # improving: top delta too
    tracker.last_improve_step = 0
    plan = ctrl.plan(tracker, env_steps=0, diversity=None, best_idx=4)  # g = 0
    assert plan.g == 0.0
    # raw rollout share ~0.64 lands at ~0.42 after the floor (TERL: 6/10)
    assert 0.35 <= plan.probs[4] <= 0.55
    assert plan.probs.min() >= cfg.rollout_floor - 1e-12
    # gradient share in TERL-stage-1 territory (kappa sharpens past 6/10)
    assert 0.55 <= plan.grad_weights[4] <= 0.9
    assert np.isclose(plan.probs.sum(), 1.0)


def test_softmax_collapses_at_g1():
    cfg = Config(pop_size=5, rollout_floor=0.0)  # pure softmax, no floor
    ctrl = SGSAController(cfg)
    tracker = make_tracker([10.0, 20.0, 30.0, 40.0, 50.0])
    tracker.last_improve_step = 0
    plan = ctrl.plan(tracker, env_steps=10 * cfg.s_max, diversity=None, best_idx=4)  # g = 1
    assert plan.g == 1.0
    assert plan.probs[4] > 0.99  # collapses onto the best individual
    assert np.isclose(plan.probs.sum(), 1.0)


def test_rollout_floor_recovers_terl_stage2():
    """g=1 with the default floor must reproduce TERL stage 2 exactly:
    rollouts 6/1/1/1/1, gradients all to the best."""
    cfg = Config(pop_size=5)  # defaults: rollout_floor=0.1, episodes_per_round=10
    ctrl = SGSAController(cfg)
    tracker = make_tracker([10.0, 20.0, 30.0, 40.0, 50.0])
    tracker.last_improve_step = 0
    plan = ctrl.plan(tracker, env_steps=10 * cfg.s_max, diversity=None, best_idx=4)
    assert plan.probs[4] == pytest.approx(0.6, abs=0.01)
    assert np.all(plan.probs[:4] >= 0.1 - 1e-12)
    assert plan.grad_weights[4] > 0.99  # gradients stay unfloored
    assert list(apportion(plan.probs, cfg.episodes_per_round)) == [1, 1, 1, 1, 6]


def test_infeasible_floor_rejected():
    with pytest.raises(ValueError):
        make_controller(Config(pop_size=5, rollout_floor=0.5))


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

    # concentration="free": both arms predate designation pinning, so this
    # stays a pure alpha_anneal A/B (pinning would mask the leak either way)
    plan = SGSAController(
        Config(pop_size=5, rollout_floor=0.0, concentration="free")
    ).plan(tracker, steps, diversity=None, best_idx=4)
    assert plan.g == 1.0
    assert plan.probs[4] > 0.99

    v1 = SGSAController(
        Config(pop_size=5, alpha_anneal=False, rollout_floor=0.0, concentration="free")
    ).plan(tracker, steps, diversity=None, best_idx=4)
    assert v1.probs[4] < 0.9  # mass leaks to the improvers


def test_kappa_sharpens_gradient_weights():
    cfg = Config(pop_size=5, kappa=2.0)
    ctrl = SGSAController(cfg)
    tracker = make_tracker([10.0, 20.0, 30.0, 40.0, 50.0])
    plan = ctrl.plan(tracker, env_steps=cfg.s_max // 2, diversity=None, best_idx=4)  # 0 < g < 1
    assert np.isclose(plan.grad_weights.sum(), 1.0)
    # kappa > 1 concentrates gradient mass more than rollout mass
    assert plan.grad_weights.max() > plan.probs.max()


def test_diversity_floor_attenuates_g():
    cfg = Config(pop_size=5)
    ctrl = SGSAController(cfg)
    tracker = make_tracker([1.0, 2.0, 3.0, 4.0, 5.0])
    steps = 10 * cfg.s_max
    plan_ok = ctrl.plan(tracker, steps, diversity=cfg.d_min * 2, best_idx=4)
    assert plan_ok.g == 1.0
    plan_low = ctrl.plan(tracker, steps, diversity=cfg.d_min / 2, best_idx=4)
    assert plan_low.g == 0.5  # one halving
    plan_lower = ctrl.plan(tracker, steps, diversity=cfg.d_min / 2, best_idx=4)
    assert plan_lower.g == 0.25


def test_unevaluated_individuals_get_top_score():
    cfg = Config(pop_size=5)
    ctrl = SGSAController(cfg)
    tracker = make_tracker([10.0, 20.0])  # individuals 2,3,4 never evaluated
    plan = ctrl.plan(tracker, env_steps=0, diversity=None, best_idx=4)
    # optimism under uncertainty: unevaluated >= evaluated allocation
    assert plan.probs[2] >= plan.probs.max() - 1e-12


def test_fixed_stage_recovers_terl_schedule_endpoints():
    """Endpoints only — switch time, PSO intervals, stage-2 rollout floor and
    gradient concentration. The full TERL protocol lives in TERLTrainer."""
    cfg = Config(pop_size=5, ratio=0.25, max_timesteps=1_000_000)
    ctrl = FixedStageController(cfg)
    tracker = make_tracker([1.0, 2.0, 3.0, 4.0, 5.0])
    stage1 = ctrl.plan(tracker, env_steps=100_000, diversity=None, best_idx=4)
    assert stage1.g == 0.0
    assert np.allclose(stage1.probs, 0.2)
    assert stage1.pso_interval == 1e4
    stage2 = ctrl.plan(tracker, env_steps=250_000, diversity=None, best_idx=4)
    assert stage2.g == 1.0
    assert stage2.concentrated
    assert stage2.probs[4] == pytest.approx(0.6)  # TERL stage 2: 6/10 to the best
    assert np.allclose(stage2.probs[:4], 0.1)  # 1/10 floor for challengers
    assert stage2.grad_weights[4] == 1.0  # gradients all to the best
    assert stage2.pso_interval == 1e3
    # stage 2 rides the DESIGNATED slot, not the fitness argmax
    pinned = ctrl.plan(tracker, env_steps=250_000, diversity=None, best_idx=1)
    assert pinned.grad_weights[1] == 1.0
    assert pinned.probs[1] == pytest.approx(0.6)


def test_hysteresis_and_designation_pinning():
    """v5: the regime bit enters above gate_enter, exits below gate_exit, and
    while concentrated the allocation rides the DESIGNATED slot even when
    another slot rank-leads (TERL stage 2 rides best_idx, terl.py:259)."""
    cfg = Config(pop_size=5)
    ctrl = SGSAController(cfg)
    tracker = make_tracker([10.0, 20.0, 30.0, 40.0, 50.0])
    tracker.last_improve_step = 0
    s = 50_000  # make_tracker's s_max, which drives g — not cfg.s_max
    mid = ctrl.plan(tracker, env_steps=int(0.55 * s), diversity=None, best_idx=0)
    assert not mid.concentrated  # 0.55 < gate_enter: still open
    assert mid.probs[4] == mid.probs.max()  # open regime follows the ranks
    hi = ctrl.plan(tracker, env_steps=10 * s, diversity=None, best_idx=0)
    assert hi.concentrated
    assert hi.probs[0] == pytest.approx(0.6, abs=0.01)  # pinned to designation
    assert hi.grad_weights[0] > 0.99
    assert list(apportion(hi.probs, cfg.episodes_per_round)) == [6, 1, 1, 1, 1]
    back = ctrl.plan(tracker, env_steps=int(0.5 * s), diversity=None, best_idx=0)
    assert back.concentrated  # 0.5 is inside the hysteresis band
    assert back.grad_weights[0] == back.grad_weights.max()
    out = ctrl.plan(tracker, env_steps=int(0.3 * s), diversity=None, best_idx=0)
    assert not out.concentrated  # 0.3 < gate_exit: regime exits


def test_promotion_survives_tied_and_unevaluated_scores():
    """A score *exchange* with the argmax would split the budget ~0.5/0.5
    whenever the max is not unique; the margin promotion must keep the
    designated slot strictly on top."""
    cfg = Config(pop_size=5, rollout_floor=0.0)
    # two slots tied at the top window mean, designation elsewhere
    ctrl = SGSAController(cfg)
    tracker = make_tracker([10.0, 20.0, 30.0, 50.0, 50.0])
    tracker.last_improve_step = 0
    plan = ctrl.plan(tracker, env_steps=10 * 50_000, diversity=None, best_idx=0)
    assert plan.concentrated
    assert plan.probs[0] > 0.99
    # two never-evaluated slots (coverage override scores both 1.0)
    ctrl = SGSAController(cfg)
    tracker = make_tracker([10.0, 20.0, 30.0])  # slots 3, 4 unevaluated
    tracker.last_improve_step = 0
    plan = ctrl.plan(tracker, env_steps=10 * 50_000, diversity=None, best_idx=2)
    assert plan.probs[2] > 0.99


def test_regime_boundaries_are_strict():
    """g exactly at gate_enter does not enter; exactly at gate_exit does not
    exit (enter needs g > enter, exit needs g < exit)."""
    cfg = Config(pop_size=5)
    ctrl = SGSAController(cfg)
    tracker = make_tracker([1.0, 2.0, 3.0, 4.0, 5.0])
    tracker.last_improve_step = 0
    s = 50_000  # make_tracker's s_max
    at_enter = ctrl.plan(
        tracker, env_steps=int(cfg.gate_enter * s), diversity=None, best_idx=0
    )
    assert not at_enter.concentrated
    ctrl.plan(tracker, env_steps=10 * s, diversity=None, best_idx=0)  # enter
    at_exit = ctrl.plan(
        tracker, env_steps=int(cfg.gate_exit * s), diversity=None, best_idx=0
    )
    assert at_exit.concentrated


def test_hysteresis_bit_survives_checkpoint():
    cfg = Config(pop_size=5)
    ctrl = SGSAController(cfg)
    tracker = make_tracker([1.0, 2.0, 3.0, 4.0, 5.0])
    tracker.last_improve_step = 0
    ctrl.plan(tracker, env_steps=10 * cfg.s_max, diversity=None, best_idx=0)
    assert ctrl.concentrated
    clone = SGSAController(cfg)
    clone.load_state_dict(ctrl.state_dict())
    assert clone.concentrated


def test_invalid_concentration_and_gates_rejected():
    with pytest.raises(ValueError):
        make_controller(Config(pop_size=5, concentration="viral"))
    with pytest.raises(ValueError):
        make_controller(Config(pop_size=5, gate_enter=0.3, gate_exit=0.5))
