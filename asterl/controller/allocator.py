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
    g_stag: float = 0.0  # stagnation component of g_raw
    g_prog: float = 0.0  # diminishing-marginal-return component of g_raw
    # Hysteretic regime bit: True between g crossing gate_enter upward and
    # gate_exit downward. Drives designation pinning, record-gated succession
    # and the champion protocol — not the (continuous) tau/PSO schedules.
    concentrated: bool = False


class SGSAController:
    """Stagnation-Gated Softmax Allocation.

    One gate g in [0,1] (from the stagnation signal) drives:
      - rollout allocation:   p_i ∝ exp(score_i / tau(g)),
        tau interpolating tau_max (g=0: winner-take-most on the blended
        level+delta rank — TERL stage 1 gives its most-recently-improved
        individual 6/10 of episodes and gradients, NOT a uniform split)
        down to tau_min (g=1, collapse onto best = TERL stage 2)
      - gradient allocation:  w_i ∝ p_i ** kappa  (kappa >= 1: steeper, mirrors
        TERL's stage-2 concentration)
      - PSO pull interval:    10 ** (4 - g) learned steps, interpolating TERL's
        own hand-set endpoints (1e4 stage 1, 1e3 stage 2)

    A diversity floor attenuates g when behavioral diversity collapses,
    re-opening the graded allocation. Note the concentrated-regime *bit* is
    hysteretic (gate_enter/gate_exit), so one attenuation halving from g=1
    lands inside the band and does not exit the regime — sustained collapse
    does. (Empirically the floor fired in 1/30 formal runs; treat the
    re-opening claim accordingly.)
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.attenuation = 1.0
        self.concentrated = False

    def plan(self, tracker, env_steps, diversity, best_idx):
        cfg = self.cfg
        g_stag = tracker.gate_stagnation(env_steps)
        g_prog = tracker.gate_progress(env_steps)
        g_raw = max(g_stag, g_prog)

        if diversity is not None:
            if diversity < cfg.d_min:
                self.attenuation *= 0.5
            else:
                self.attenuation = min(1.0, self.attenuation * 1.5)
        g = g_raw * self.attenuation

        # Hysteretic regime boundary: a memoryless g > 0.5 check would flap
        # the regime (and with it succession semantics) on gate noise.
        if self.concentrated:
            if g < cfg.gate_exit:
                self.concentrated = False
        elif g > cfg.gate_enter:
            self.concentrated = True

        # Anneal alpha -> 1 with g: at full concentration the score must be
        # pure fitness rank, otherwise the delta term shifts mass off the best
        # individual exactly when it stagnates (v1 never actually collapsed
        # onto the best: p_max ~0.8 at g=1 instead of ~1).
        alpha = cfg.alpha + (1.0 - cfg.alpha) * g if cfg.alpha_anneal else cfg.alpha
        fitness = tracker.fitness_means()
        scores = alpha * rank_normalize(fitness) + (1 - alpha) * rank_normalize(
            tracker.deltas()
        )
        # +inf fitness (never-evaluated individual) outranks everything: force coverage
        for i, f in enumerate(fitness):
            if np.isinf(f):
                scores[i] = 1.0

        # Pinned concentrated regime: allocation rides the DESIGNATED slot, as
        # TERL stage 2 rides best_idx — promote it a full adjacent-rank gap
        # above the top score (an *exchange* with the argmax would split the
        # budget ~0.5/0.5 whenever the max is not unique, e.g. tied window
        # means or two unevaluated slots under rollout_floor=0). Routing
        # concentration through the instantaneous window-mean ranking instead
        # (v1-v4) re-targets the budget at noise frequency on plateaus, which
        # is what kept the concentrated regime from ever matching TERL stage 2
        # on Swimmer. The swap/free ablation arms keep the raw ranking.
        if self.concentrated and cfg.concentration == "pinned":
            scores[best_idx] = scores.max() + 0.25

        tau = cfg.tau_max - g * (cfg.tau_max - cfg.tau_min)
        logits = (scores - scores.max()) / tau
        probs = np.exp(logits)
        probs /= probs.sum()

        weights = probs ** cfg.kappa
        weights /= weights.sum()

        # Rollout floor (v3): every individual keeps >= rollout_floor of the
        # episodes even at full concentration. g=1 then reproduces TERL stage 2
        # exactly — best 6/10, others 1/10 with the defaults: the non-best
        # members are PSO-perturbed samplers around gbest whose episodes keep
        # challenger fitness measurable. v2 collapsed rollouts together with
        # gradients, which starved that signal (the Swimmer regression).
        # Gradient weights above are intentionally taken from the unfloored
        # softmax so gradients still concentrate fully.
        if cfg.rollout_floor > 0.0:
            probs = cfg.rollout_floor + (1.0 - cfg.pop_size * cfg.rollout_floor) * probs

        pso_interval = 10.0 ** (4.0 - g)
        return RoundPlan(
            g_raw, g, tau, probs, weights, pso_interval, diversity, g_stag, g_prog,
            concentrated=self.concentrated,
        )

    def state_dict(self):
        return {"attenuation": self.attenuation, "concentrated": self.concentrated}

    def load_state_dict(self, d):
        self.attenuation = d["attenuation"]
        self.concentrated = d.get("concentrated", False)


class FixedStageController:
    """TERL's hard two-stage schedule through the controller interface: g jumps
    0 -> 1 at ratio * max_timesteps, recovering the schedule's endpoints —
    switch time, PSO intervals (1e4/1e3), stage-2 gradient concentration on
    the designated best slot, and (via the rollout floor) the stage-2
    6/1/1/1/1 episode split. It is NOT the complete TERL algorithm: TERL's
    stage 1 is itself winner-take-most — 6/10 of episodes AND gradients go to
    the most-recently-improved individual (paper Algorithm 1, line 8), while
    this arm splits stage 1 uniformly — and per-rollout gradient attribution
    plus the fitness-eval early break live only in the faithful TERLTrainer
    port. Basis of the fallback design (adaptive hard switch = this schedule
    with a stagnation trigger)."""

    def __init__(self, cfg):
        self.cfg = cfg

    def plan(self, tracker, env_steps, diversity, best_idx):
        cfg = self.cfg
        g = 1.0 if env_steps >= cfg.max_timesteps * cfg.ratio else 0.0
        if g == 0.0:
            probs = np.full(cfg.pop_size, 1.0 / cfg.pop_size)
            weights = probs.copy()
        else:
            # Stage 2 rides the designated slot, as TERL rides best_idx —
            # not the instantaneous fitness argmax.
            weights = np.zeros(cfg.pop_size)
            weights[best_idx] = 1.0
            probs = weights
            if cfg.rollout_floor > 0.0:
                probs = cfg.rollout_floor + (
                    1.0 - cfg.pop_size * cfg.rollout_floor
                ) * weights
        return RoundPlan(
            g, g, 0.0, probs, weights, 1e4 if g == 0.0 else 1e3, diversity, g, 0.0,
            concentrated=g == 1.0,
        )

    def state_dict(self):
        return {}

    def load_state_dict(self, d):
        pass


def make_controller(cfg):
    if not 0.0 <= cfg.rollout_floor * cfg.pop_size <= 1.0:
        raise ValueError(
            f"rollout_floor={cfg.rollout_floor} infeasible for pop_size={cfg.pop_size}: "
            "the floors must sum to at most 1"
        )
    if cfg.concentration not in ("pinned", "swap", "free"):
        raise ValueError(
            f"Unknown concentration {cfg.concentration!r} (pinned | swap | free)"
        )
    if not 0.0 <= cfg.gate_exit <= cfg.gate_enter <= 1.0:
        raise ValueError(
            f"need 0 <= gate_exit ({cfg.gate_exit}) <= gate_enter "
            f"({cfg.gate_enter}) <= 1: g lives in [0, 1], so thresholds "
            "outside it silently freeze the regime"
        )
    if cfg.allocator == "softmax":
        return SGSAController(cfg)
    if cfg.allocator == "fixed":
        return FixedStageController(cfg)
    raise ValueError(
        f"Unknown allocator {cfg.allocator!r} (sw_ucb/exp3 are Phase-4 ablation arms, "
        "not implemented yet)"
    )
