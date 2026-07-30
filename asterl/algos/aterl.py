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
        self.tracker = SignalTracker(
            cfg.pop_size, cfg.window_k, cfg.improve_eps, cfg.s_max,
            improve_decay=cfg.improve_decay, prog_gate=cfg.prog_gate,
        )
        self.controller = make_controller(cfg)

    def _diversity(self):
        if self.buffer.size < self.cfg.diversity_states:
            return None
        states = self.buffer.sample_states(self.cfg.diversity_states)
        return behavioral_diversity(
            [ind.actor for ind in self.pop], states, self.max_action
        )

    def _update_champion(self, i, g):
        """TERL's champion protocol: in the concentrated regime a new fitness
        record must be confirmed with stable_eval_times noise-free episodes
        (stored in the buffer and counted toward the budget, exactly as in
        TERL stage 2) before the actor overwrites the test individual."""
        cfg, st = self.cfg, self.state
        if g > 0.5 and cfg.stable_eval_times > 1 and cfg.fitness_eval_times == 1:
            stable = 0.0
            for _ in range(cfg.stable_eval_times):
                stable += self.collect_episode(i, noise=False) / cfg.stable_eval_times
            if stable > st.test_individual_fitness:
                st.test_individual_fitness = stable
                copy_params(self.pop[i].actor, self.test_individual)
        else:
            copy_params(self.pop[i].actor, self.test_individual)

    def train_round(self):
        cfg = self.cfg
        st = self.state

        diversity = self._diversity()
        plan = self.controller.plan(self.tracker, self.timesteps, diversity)
        best_idx_pre = st.best_idx

        # -- rollouts: deterministic largest-remainder split of the round's
        # episodes by the (floored) allocation. Multinomial sampling could
        # starve an individual of episodes for many rounds, leaving stale
        # fitness estimates in the ranking.
        episode_counts = apportion(plan.probs, cfg.episodes_per_round)
        steps_before = self.timesteps
        for i in range(cfg.pop_size):
            for _ in range(episode_counts[i]):
                fitness = self.collect_episode(i, noise=True)
                personal, _ = self.tracker.record_eval(i, fitness, self.timesteps)
                if personal:
                    # TERL's protective reset: an improving individual is spared
                    # the next PSO pull toward gbest.
                    st.learned_steps[i] = 0
                    st.last_evo_point[i] = 0
                    self.logger.log({f"fitness/{i}": fitness}, self.timesteps)
                if self.tracker.personal_best[i] > st.max_best_f:
                    st.max_best_f = self.tracker.personal_best[i]
                    self._update_champion(i, plan.g)

        # -- best-individual bookkeeping: the designation follows the rank
        # leader. No actor overwrite — copying the challenger's actor and
        # fitness history into the incumbent slot created two identical
        # top-ranked slots whose tied ranks split the softmax ~0.5/0.5
        # exactly at g=1, defeating the alpha-anneal collapse.
        means = self.tracker.fitness_means()
        finite = [m if np.isfinite(m) else -np.inf for m in means]
        cand = int(np.argmax(finite))
        if np.isfinite(finite[cand]):
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
            "controller/g_stag": plan.g_stag,
            "controller/g_prog": plan.g_prog,
            "controller/g": plan.g,
            "controller/tau": plan.tau,
            "controller/pso_interval": plan.pso_interval,
            # controller/* is the plan-time snapshot (the state the probs were
            # built from); post-round population state is logged separately so
            # one record never mixes the two.
            "controller/best_idx": best_idx_pre,
            "population/best_idx": st.best_idx,
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
                metrics[f"population/fitness_mean_{i}"] = finite[i]
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
