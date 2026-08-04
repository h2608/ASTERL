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

    def _update_champion(self, i, concentrated):
        """TERL's champion protocol: in the concentrated regime a new fitness
        record must be confirmed with stable_eval_times noise-free episodes
        (stored in the buffer and counted toward the budget, exactly as in
        TERL stage 2) before the actor overwrites the test individual."""
        cfg, st = self.cfg, self.state
        if concentrated and cfg.stable_eval_times > 1 and cfg.fitness_eval_times == 1:
            stable = 0.0
            for _ in range(cfg.stable_eval_times):
                stable += self.collect_episode(i, noise=False) / cfg.stable_eval_times
            if stable > st.test_individual_fitness:
                st.test_individual_fitness = stable
                copy_params(self.pop[i].actor, self.test_individual)
        else:
            copy_params(self.pop[i].actor, self.test_individual)

    def _record_succession(self, i, concentrated):
        """Individual i just set an all-time fitness record (TERL's
        best_f[i] > max_best_f, terl.py:240). In the pinned concentrated
        regime that — and only that — is when a challenger takes over: its
        actor is donated into the designated best slot (whose critic and
        optimizer train continuously) and the two slots swap fitness records
        so the designated slot ranks top, exactly TERL's stage-2 overwrite.
        Records are rare on plateaus, so the trained actor is NOT clobbered
        at noise frequency (v4's swap-on-window-mean-lead overwrote it every
        ~2 rounds on Swimmer). Returns 1.0 if the takeover fired."""
        cfg, st = self.cfg, self.state
        st.max_best_f = self.tracker.personal_best[i]
        if concentrated and cfg.concentration == "pinned" and i != st.best_idx:
            copy_params(self.pop[i].actor, self.pop[st.best_idx].actor)
            self.tracker.swap_slots(i, st.best_idx)
            return 1.0
        return 0.0

    def _designate_best(self, concentrated):
        """Post-round designation bookkeeping.

        Open regime: the designation follows the window-mean rank leader —
        slots keep their own records and learners (a deliberate deviation
        from TERL stage 1, which moves best_idx on records).

        Concentrated regime, by cfg.concentration:
          pinned — nothing to do here: the designation is frozen, allocation
            rides it via rank promotion in the controller, and succession is
            record-gated in _record_succession (TERL stage-2 semantics; the
            paper's own rationale, p.5: "Even if other individuals achieve a
            higher reward, their value networks no longer adapt to their
            actors due to a long period without updating").
            Stale-record lockout (an incumbent decaying below an old record
            no challenger beats) is TERL's own behavior; the champion
            protocol protects the reported score.
          swap   — v4 ablation arm: a strictly window-mean-leading challenger
            donates its actor into the incumbent slot and histories swap.
          free   — v3 ablation arm: the designation moves freely (churn
            redirects the gradient budget onto starved critics).

        Returns 1.0 if a v4-style swap-overwrite happened."""
        cfg, st = self.cfg, self.state
        means = self.tracker.fitness_means()
        finite = [m if np.isfinite(m) else -np.inf for m in means]
        cand = int(np.argmax(finite))
        if not np.isfinite(finite[cand]) or cand == st.best_idx:
            return 0.0
        # An incumbent with no finite record cannot anchor pinning or a swap
        # (unreachable with rollout_floor > 0, which evaluates every slot
        # each round; kept for rollout_floor=0 ablation arms).
        if concentrated and np.isfinite(finite[st.best_idx]):
            if cfg.concentration == "pinned":
                return 0.0
            if cfg.concentration == "swap":
                if finite[cand] > finite[st.best_idx]:
                    copy_params(self.pop[cand].actor, self.pop[st.best_idx].actor)
                    self.tracker.swap_slots(cand, st.best_idx)
                    return 1.0
                return 0.0
        st.best_idx = cand
        return 0.0

    def train_round(self):
        cfg = self.cfg
        st = self.state

        diversity = self._diversity()
        plan = self.controller.plan(self.tracker, self.timesteps, diversity, st.best_idx)
        best_idx_pre = st.best_idx
        swaps = 0.0

        # -- rollouts: deterministic largest-remainder split of the round's
        # episodes by the (floored) allocation. Multinomial sampling could
        # starve an individual of episodes for many rounds, leaving stale
        # fitness estimates in the ranking.
        episode_counts = apportion(plan.probs, cfg.episodes_per_round)
        last_trained = self.timesteps
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
                    # TERL's ordering (terl.py:240-256): succession first,
                    # champion confirmation second.
                    swaps += self._record_succession(i, plan.concentrated)
                    self._update_champion(i, plan.concentrated)
                # -- interleaved gradients (paper Algorithm 1, lines 22-25):
                # TERL trains after EVERY evaluation with that episode's step
                # count, so later episodes in a round are collected by already-
                # updated actors. The chunk covers the fitness episode plus any
                # champion-confirmation episodes it triggered; per-chunk
                # largest-remainder keeps total gradient steps == env steps
                # (UTD=1) with no carry state to checkpoint.
                if self.buffer.size >= cfg.start_timesteps:
                    chunk = self.timesteps - last_trained
                    for j, n in enumerate(apportion(plan.grad_weights, chunk)):
                        for _ in range(n):
                            self.pop[j].train(self.buffer, cfg.batch_size)
                last_trained = self.timesteps

        swaps += self._designate_best(plan.concentrated)

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
            "controller/concentrated": float(plan.concentrated),
            "population/best_idx": st.best_idx,
            "population/swap": swaps,
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
        # post-round (hence post-swap) per-slot means, matching population/*
        means = self.tracker.fitness_means()
        for i in range(cfg.pop_size):
            metrics[f"controller/p_{i}"] = plan.probs[i]
            if np.isfinite(means[i]):
                metrics[f"population/fitness_mean_{i}"] = means[i]
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
