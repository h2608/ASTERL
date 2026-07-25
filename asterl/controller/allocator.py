from dataclasses import dataclass

import numpy as np

from asterl.controller.signals import rank_normalize


@dataclass
class RoundPlan:
    g_raw: float
    g: float
    tau: float
    probs: np.ndarray  # rollout allocation
    grad_weights: np.ndarray  # gradient-step allocation (probs ** kappa, renormalized)
    pso_interval: float
    diversity: float | None


class SGSAController:
    """Stagnation-Gated Softmax Allocation.

    One gate g in [0,1] (from the stagnation signal) drives:
      - rollout allocation:   p_i ∝ exp(score_i / tau(g)),
        tau interpolating tau_max (g=0, near-uniform = TERL stage 1) down to
        tau_min (g=1, collapse onto best = TERL stage 2)
      - gradient allocation:  w_i ∝ p_i ** kappa  (kappa >= 1: steeper, mirrors
        TERL's stage-2 concentration)
      - PSO pull interval:    10 ** (4 - g) learned steps, interpolating TERL's
        own hand-set endpoints (1e4 stage 1, 1e3 stage 2)

    A diversity floor attenuates g when behavioral diversity collapses, letting
    the controller re-open exploration — something no fixed ratio can do.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.attenuation = 1.0

    def plan(self, tracker, env_steps, diversity):
        cfg = self.cfg
        g_raw = tracker.gate(env_steps)

        if diversity is not None:
            if diversity < cfg.d_min:
                self.attenuation *= 0.5
            else:
                self.attenuation = min(1.0, self.attenuation * 1.5)
        g = g_raw * self.attenuation

        fitness = tracker.fitness_means()
        scores = cfg.alpha * rank_normalize(fitness) + (1 - cfg.alpha) * rank_normalize(
            tracker.deltas()
        )
        # +inf fitness (never-evaluated individual) outranks everything: force coverage
        for i, f in enumerate(fitness):
            if np.isinf(f):
                scores[i] = 1.0

        tau = cfg.tau_max - g * (cfg.tau_max - cfg.tau_min)
        logits = (scores - scores.max()) / tau
        probs = np.exp(logits)
        probs /= probs.sum()

        weights = probs ** cfg.kappa
        weights /= weights.sum()

        pso_interval = 10.0 ** (4.0 - g)
        return RoundPlan(g_raw, g, tau, probs, weights, pso_interval, diversity)

    def state_dict(self):
        return {"attenuation": self.attenuation}

    def load_state_dict(self, d):
        self.attenuation = d["attenuation"]


class FixedStageController:
    """Recovers TERL's hard two-stage schedule through the same interface:
    g jumps 0 -> 1 at ratio * max_timesteps. Used for the fallback design
    (adaptive hard switch = same class with a stagnation trigger) and as the
    regression harness against the faithful TERL port."""

    def __init__(self, cfg):
        self.cfg = cfg

    def plan(self, tracker, env_steps, diversity):
        cfg = self.cfg
        g = 1.0 if env_steps >= cfg.max_timesteps * cfg.ratio else 0.0
        fitness = tracker.fitness_means()
        finite = [f if not np.isinf(f) else -np.inf for f in fitness]
        if g == 0.0:
            probs = np.full(cfg.pop_size, 1.0 / cfg.pop_size)
        else:
            probs = np.zeros(cfg.pop_size)
            probs[int(np.argmax(finite))] = 1.0
        weights = probs.copy()
        return RoundPlan(g, g, 0.0, probs, weights, 1e4 if g == 0.0 else 1e3, diversity)

    def state_dict(self):
        return {}

    def load_state_dict(self, d):
        pass


def make_controller(cfg):
    if cfg.allocator == "softmax":
        return SGSAController(cfg)
    if cfg.allocator == "fixed":
        return FixedStageController(cfg)
    raise ValueError(
        f"Unknown allocator {cfg.allocator!r} (sw_ucb/exp3 are Phase-4 ablation arms, "
        "not implemented yet)"
    )
