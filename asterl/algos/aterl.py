import numpy as np
import torch

from asterl.algos.terl import PopulationTrainerBase, copy_params
from asterl.controller.allocator import make_controller
from asterl.controller.signals import SignalTracker, behavioral_diversity


def apportion(weights, total):
    """Largest-remainder apportionment of `total` integer steps by `weights`."""
    raw = np.asarray(weights) * total
    base = np.floor(raw).astype(int)
    for idx in np.argsort(-(raw - base))[: total - base.sum()]:
        base[idx] += 1
    return base


def param_distance(net1, net2):
    """TERL's get_difference (parameter-space L1), kept as a logged diagnostic."""
    total = 0.0
    for p1, p2 in zip(net1.parameters(), net2.parameters()):
        total += float(torch.sum(torch.abs(p1.data - p2.data)))
    return total


class ATERLTrainer(PopulationTrainerBase):
    """Adaptive-TERL: TERL's population + shared buffer + PSO, with the hard
    two-stage schedule and the extra_idx heuristic replaced by the SGSA
    controller (rollout/gradient allocation + gated PSO interval)."""

    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)
        self.tracker = SignalTracker(cfg.pop_size, cfg.window_k, cfg.improve_eps, cfg.s_max)
        self.controller = make_controller(cfg)

    def _diversity(self):
        if self.buffer.size < self.cfg.diversity_states:
            return None
        states = self.buffer.sample_states(self.cfg.diversity_states)
        return behavioral_diversity(
            [ind.actor for ind in self.pop], states, self.max_action
        )

    def train_round(self):
        cfg = self.cfg
        st = self.state

        diversity = self._diversity()
        plan = self.controller.plan(self.tracker, self.timesteps, diversity)

        # -- rollouts sampled from the allocation ------------------------
        idx_list = np.random.choice(cfg.pop_size, size=cfg.episodes_per_round, p=plan.probs)
        steps_before = self.timesteps
        for i in idx_list:
            fitness = self.collect_episode(i, noise=True)
            personal, global_improved = self.tracker.record_eval(i, fitness, self.timesteps)
            if personal:
                # TERL's protective reset: an improving individual is spared
                # the next PSO pull toward gbest.
                st.learned_steps[i] = 0
                st.last_evo_point[i] = 0
                self.logger.log({f"fitness/{i}": fitness}, self.timesteps)
            if global_improved:
                copy_params(self.pop[i].actor, self.test_individual)

        # -- best-individual bookkeeping ---------------------------------
        means = self.tracker.fitness_means()
        finite = [m if np.isfinite(m) else -np.inf for m in means]
        cand = int(np.argmax(finite))
        if cand != st.best_idx and np.isfinite(finite[cand]):
            if plan.g > 0.5:
                # Concentrated regime: the challenger's actor overwrites the
                # incumbent learner (which keeps its critics), as in TERL stage 2.
                copy_params(self.pop[cand].actor, self.pop[st.best_idx].actor)
                self.tracker.hist[st.best_idx] = self.tracker.hist[cand].copy()
                self.tracker.personal_best[st.best_idx] = self.tracker.personal_best[cand]
            else:
                st.best_idx = cand

        # -- gradient allocation -----------------------------------------
        round_steps = self.timesteps - steps_before
        if self.buffer.size >= cfg.start_timesteps and round_steps > 0:
            grad_steps = apportion(plan.grad_weights, round_steps)
            for i, n in enumerate(grad_steps):
                for _ in range(n):
                    self.pop[i].train(self.buffer, cfg.batch_size)

        # -- PSO with the gated interval ----------------------------------
        self.pso_step(plan.pso_interval)

        # -- logging -------------------------------------------------------
        metrics = {
            "controller/g_raw": plan.g_raw,
            "controller/g": plan.g,
            "controller/tau": plan.tau,
            "controller/pso_interval": plan.pso_interval,
            "controller/best_idx": st.best_idx,
        }
        if diversity is not None:
            metrics["controller/diversity"] = diversity
            metrics["controller/param_dist_mean"] = np.mean(
                [
                    param_distance(self.pop[i].actor, self.pop[j].actor)
                    for i in range(cfg.pop_size)
                    for j in range(i + 1, cfg.pop_size)
                ]
            )
        for i in range(cfg.pop_size):
            metrics[f"controller/p_{i}"] = plan.probs[i]
            if np.isfinite(finite[i]):
                metrics[f"controller/fitness_mean_{i}"] = finite[i]
        self.logger.log(metrics, self.timesteps)

        self.maybe_test()

    def state_dict(self):
        d = super().state_dict()
        d["tracker"] = self.tracker.state_dict()
        d["controller"] = self.controller.state_dict()
        return d

    def load_state_dict(self, d):
        super().load_state_dict(d)
        self.tracker.load_state_dict(d["tracker"])
        self.controller.load_state_dict(d["controller"])
