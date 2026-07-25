from collections import deque

import numpy as np
import torch


def rank_normalize(values):
    """Map values to average-tie ranks in [0, 1]: equal signals must get equal
    allocation (early rounds have many ties, e.g. all-zero deltas).

    Rank normalization makes every controller signal scale-free across
    environments — the basis of the no-per-env-tuning claim.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n <= 1:
        return np.zeros(n)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n)
    ranks[order] = np.arange(n, dtype=np.float64)
    for v in np.unique(values):
        mask = values == v
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks / (n - 1)


class SignalTracker:
    """Online signals for the SGSA controller: stagnation gate g, per-individual
    fitness windows and improvement deltas."""

    def __init__(self, pop_size, window_k, improve_eps, s_max):
        self.pop_size = pop_size
        self.window_k = window_k
        self.improve_eps = improve_eps
        self.s_max = s_max
        self.hist = [deque(maxlen=window_k) for _ in range(pop_size)]
        self.personal_best = [-np.inf] * pop_size
        self.global_best = -np.inf
        self.last_improve_step = 0

    def record_eval(self, idx, fitness, env_steps):
        """Returns (personal_improved, global_improved)."""
        self.hist[idx].append(float(fitness))
        personal = fitness > self.personal_best[idx]
        if personal:
            self.personal_best[idx] = fitness
        margin = self.improve_eps * max(abs(self.global_best), 1.0)
        global_improved = fitness > self.global_best + margin
        if fitness > self.global_best:
            self.global_best = fitness
        if global_improved:
            self.last_improve_step = env_steps
        return personal, global_improved

    def gate(self, env_steps):
        return float(np.clip((env_steps - self.last_improve_step) / self.s_max, 0.0, 1.0))

    def fitness_means(self):
        # Unevaluated individuals get +inf: optimism under uncertainty forces
        # initial coverage of the whole population.
        return [np.mean(h) if len(h) else np.inf for h in self.hist]

    def deltas(self):
        out = []
        for h in self.hist:
            if len(h) < 2:
                out.append(0.0)
            else:
                half = len(h) // 2
                vals = list(h)
                out.append(float(np.mean(vals[half:]) - np.mean(vals[:half])))
        return out

    def state_dict(self):
        return {
            "hist": [list(h) for h in self.hist],
            "personal_best": list(self.personal_best),
            "global_best": self.global_best,
            "last_improve_step": self.last_improve_step,
        }

    def load_state_dict(self, d):
        self.hist = [deque(h, maxlen=self.window_k) for h in d["hist"]]
        self.personal_best = list(d["personal_best"])
        self.global_best = d["global_best"]
        self.last_improve_step = d["last_improve_step"]


def behavioral_diversity(actors, states, max_action):
    """Mean pairwise L2 action distance on shared-buffer states, normalized to
    be ~scale-free: distance / (2 * max_action * sqrt(action_dim)) in [0, 1].

    Action-space distance is behaviorally meaningful (cf. DvD); the parameter-
    space distance from TERL is kept only as a logged diagnostic.
    """
    with torch.no_grad():
        acts = [actor(states) for actor in actors]
    n = len(acts)
    if n < 2:
        return 0.0
    action_dim = acts[0].shape[1]
    norm = 2.0 * max_action * (action_dim ** 0.5)
    total, pairs = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(torch.norm(acts[i] - acts[j], dim=1).mean()) / norm
            pairs += 1
    return total / pairs
