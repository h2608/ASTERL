import copy

import numpy as np
import pytest
import torch

from asterl.algos.td3 import Actor
from asterl.controller.signals import SignalTracker, behavioral_diversity, rank_normalize


def test_rank_normalize():
    ranks = rank_normalize([30.0, 10.0, 20.0])
    assert np.allclose(ranks, [1.0, 0.0, 0.5])
    assert rank_normalize([5.0]).tolist() == [0.0]
    # ties get the average rank: equal signals -> equal allocation
    assert np.allclose(rank_normalize([1.0, 1.0, 2.0]), [0.25, 0.25, 1.0])
    assert np.allclose(rank_normalize([0.0, 0.0, 0.0]), [0.5, 0.5, 0.5])


def test_stagnation_gate():
    tracker = SignalTracker(pop_size=2, window_k=5, improve_eps=0.01, s_max=1000)
    tracker.record_eval(0, 100.0, env_steps=0)
    assert tracker.gate(500) == 0.5
    assert tracker.gate(2000) == 1.0
    # a real improvement resets the clock
    tracker.record_eval(1, 200.0, env_steps=800)
    assert tracker.gate(800) == 0.0
    # an improvement inside the epsilon margin does NOT reset it
    tracker.record_eval(1, 200.5, env_steps=1000)
    assert tracker.gate(1800) == 1.0


def test_ratchet_retains_stagnation():
    tracker = SignalTracker(pop_size=2, window_k=5, improve_eps=0.01, s_max=1000,
                            improve_decay=0.5)
    tracker.record_eval(0, 100.0, env_steps=0)
    assert tracker.gate_stagnation(1000) == 1.0
    # v1 (improve_decay=0) would reset the gate to 0 here; the ratchet
    # retains half the accumulated stagnation per improvement
    tracker.record_eval(1, 200.0, env_steps=1000)
    assert tracker.gate_stagnation(1000) == 0.5
    tracker.record_eval(1, 400.0, env_steps=1000)  # window mean 300 > ref 200
    assert tracker.gate_stagnation(1000) == 0.25


def test_progress_gate_tracks_marginal_return():
    tracker = SignalTracker(pop_size=1, window_k=5, improve_eps=0.01, s_max=500,
                            prog_gate=True)

    def run(fit_fn, lo, hi):
        gates = []
        for step in range(lo, hi + 1, 100):
            tracker.record_eval(0, fit_fn(step), env_steps=step)
            gates.append(tracker.gate_progress(step))
        return gates

    fast = run(lambda s: float(s), 0, 1000)  # +1 fitness / step
    assert fast[-1] == 0.0  # improving at the run's own peak rate
    slow = run(lambda s: 1000 + 0.05 * (s - 1000), 1100, 3000)
    assert slow[-1] > 0.9  # marginal return collapsed -> concentrate
    renewed = run(lambda s: 1100 + (s - 3000), 3100, 4000)
    assert renewed[-1] < 0.2  # breakthrough re-opens exploration


def test_first_eval_counts_as_improvement():
    """A -inf reference plus a relative epsilon margin is nan and compares
    False, so before the fix the very first fitness never reset the gate."""
    tracker = SignalTracker(pop_size=2, window_k=5, improve_eps=0.01, s_max=1000)
    _, improved = tracker.record_eval(0, 5.0, env_steps=100)
    assert improved
    assert tracker.gate_stagnation(100) == 0.0


def test_micro_improvements_accumulate():
    """Sub-epsilon gains measured against a moving best could never reset the
    gate; against the fixed gate_ref they add up."""
    tracker = SignalTracker(pop_size=1, window_k=1, improve_eps=0.01, s_max=1000)
    tracker.record_eval(0, 100.0, env_steps=0)
    resets = 0
    for step, f in enumerate([100.5, 101.0, 101.5, 102.0], 1):
        _, improved = tracker.record_eval(0, f, env_steps=step)
        resets += improved
    assert resets == 1  # cumulative +1.5 over ref=100 crosses eps=1% at 101.5


def test_nan_fitness_raises():
    tracker = SignalTracker(pop_size=1, window_k=5, improve_eps=0.01, s_max=1000)
    with pytest.raises(ValueError):
        tracker.record_eval(0, float("nan"), env_steps=0)


def test_noise_spike_decays_from_progress_curve():
    """One lucky episode must not permanently raise the progress reference:
    the window mean absorbs it at 1/window_k and lets it decay back out."""
    tracker = SignalTracker(pop_size=1, window_k=5, improve_eps=0.01, s_max=500,
                            prog_gate=True)
    for step in range(0, 2001, 100):
        tracker.record_eval(0, 100.0, step)
    tracker.record_eval(0, 200.0, 2100)  # single spike
    assert tracker.curve[-1][1] == pytest.approx(120.0)
    for step in range(2200, 2701, 100):
        tracker.record_eval(0, 100.0, step)
    assert tracker.curve[-1][1] == pytest.approx(100.0)  # spike left the window


def test_progress_gate_off_by_default():
    tracker = SignalTracker(pop_size=1, window_k=5, improve_eps=0.01, s_max=500)
    for step in range(0, 3001, 100):
        tracker.record_eval(0, 100.0, env_steps=step)
    assert tracker.gate_progress(3000) == 0.0


def test_deltas_measure_improvement():
    tracker = SignalTracker(pop_size=2, window_k=4, improve_eps=0.01, s_max=1000)
    for step, f in enumerate([1.0, 2.0, 3.0, 4.0]):
        tracker.record_eval(0, f, step)  # improving
    for step, f in enumerate([4.0, 3.0, 2.0, 1.0]):
        tracker.record_eval(1, f, step)  # declining
    deltas = tracker.deltas()
    assert deltas[0] > 0 > deltas[1]


def test_diversity_zero_for_identical_actors():
    torch.manual_seed(0)
    a1 = Actor(3, 2, 1.0)
    a2 = Actor(3, 2, 1.0)
    a2.load_state_dict(a1.state_dict())
    states = torch.randn(64, 3)
    assert behavioral_diversity([a1, a2], states, 1.0) == 0.0
    a3 = Actor(3, 2, 1.0)
    d = behavioral_diversity([a1, a3], states, 1.0)
    assert 0.0 < d <= 1.0


def test_swap_slots_exchanges_records_only():
    """v4 swap-overwrite: per-slot records exchange, global gate state does
    not (max-over-slots statistics are permutation-invariant, so a swap must
    never perturb the stagnation clock or the progress curve)."""
    tracker = SignalTracker(pop_size=3, window_k=3, improve_eps=0.01, s_max=1000,
                            prog_gate=True)
    for step, f in enumerate([10.0, 20.0], 1):
        tracker.record_eval(0, f, env_steps=100 * step)
    for step, f in enumerate([50.0, 60.0], 3):
        tracker.record_eval(2, f, env_steps=100 * step)
    gate_state = copy.deepcopy(
        (tracker.gate_ref, tracker.last_improve_step, list(tracker.curve),
         tracker.peak_delta, tracker.global_best)
    )
    tracker.swap_slots(2, 0)
    assert list(tracker.hist[0]) == [50.0, 60.0]
    assert list(tracker.hist[2]) == [10.0, 20.0]
    assert tracker.personal_best[0] == 60.0
    assert tracker.personal_best[2] == 20.0
    means = tracker.fitness_means()
    assert means[0] == 55.0 and means[2] == 15.0
    assert (tracker.gate_ref, tracker.last_improve_step, list(tracker.curve),
            tracker.peak_delta, tracker.global_best) == gate_state


def test_tracker_state_roundtrip():
    tracker = SignalTracker(pop_size=2, window_k=3, improve_eps=0.01, s_max=1000)
    tracker.record_eval(0, 5.0, 10)
    tracker.record_eval(1, 7.0, 20)
    clone = SignalTracker(pop_size=2, window_k=3, improve_eps=0.01, s_max=1000)
    clone.load_state_dict(tracker.state_dict())
    assert clone.fitness_means() == tracker.fitness_means()
    assert clone.global_best == tracker.global_best
    assert clone.gate_ref == tracker.gate_ref
    assert clone.last_improve_step == tracker.last_improve_step
    assert list(clone.curve) == list(tracker.curve)
    assert clone.peak_delta == tracker.peak_delta
