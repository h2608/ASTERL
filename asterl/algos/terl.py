from dataclasses import asdict, dataclass, field

import numpy as np
import torch

from asterl.algos.td3 import TD3, Actor
from asterl.common.buffer import ReplayBuffer
from asterl.common.checkpoint import mark_done, save_checkpoint
from asterl.common.evaluator import eval_policy, make_env, rollout_episode
from asterl.common.seeding import (
    get_env_rng_state,
    get_rng_state,
    seed_everything,
    set_env_rng_state,
    set_rng_state,
)


def copy_params(source_net, target_net):
    for target_param, source_param in zip(target_net.parameters(), source_net.parameters()):
        target_param.data.copy_(source_param.data)


def pso_pull(actor, velocity, gbest_params, inertia):
    """One PSO update of `actor` toward gbest, generic over the state_dict
    (replaces TERL-main's hardcoded l1/l2/l3 tensor lists). pbest == X in TERL,
    so the pbest term vanishes and V = inertia*V + rand*(gbest - X)."""
    X = actor.state_dict()
    for name in X:
        velocity[name] = inertia * velocity[name] + torch.rand_like(X[name]) * (
            gbest_params[name] - X[name]
        )
        X[name].copy_(X[name] + velocity[name])


@dataclass
class PopulationState:
    """De-globalized port of the module-level state in TERL-main/TERL.py:25-35."""

    pop_size: int
    new_steps: list = field(default_factory=list)
    best_f: list = field(default_factory=list)
    learned_steps: list = field(default_factory=list)
    total_learned_steps: list = field(default_factory=list)
    last_evo_point: list = field(default_factory=list)
    best_idx: int = 0
    extra_idx: int = 0
    stage: int = 1
    max_best_f: float = -np.inf
    test_individual_fitness: float = -np.inf
    last_test_point: int = 0

    def __post_init__(self):
        n = self.pop_size
        if not self.new_steps:
            self.new_steps = [0] * n
            self.best_f = [-np.inf] * n
            self.learned_steps = [0] * n
            self.total_learned_steps = [0] * n
            self.last_evo_point = [0] * n


class PopulationTrainerBase:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        if cfg.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device

        seed_everything(cfg.seed)
        self.env = make_env(cfg.env_id, cfg.seed, cfg.env_kwargs, cfg.max_episode_steps)
        self.eval_env = make_env(
            cfg.env_id, cfg.seed + 10_000, cfg.env_kwargs, cfg.max_episode_steps
        )
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        self.pop = [
            TD3(
                self.state_dim,
                self.action_dim,
                self.max_action,
                discount=cfg.discount,
                device=self.device,
            )
            for _ in range(cfg.pop_size)
        ]
        self.inertia_weight = 0.0
        self.V = [
            {name: torch.zeros_like(p) for name, p in ind.actor.state_dict().items()}
            for ind in self.pop
        ]
        self.test_individual = Actor(self.state_dim, self.action_dim, self.max_action).to(
            self.device
        )
        self.buffer = ReplayBuffer(
            self.state_dim, self.action_dim, cfg.buffer_size, device=self.device
        )

        self.timesteps = 0
        self.num_games = 0
        self.state = PopulationState(cfg.pop_size)

    # -- rollouts ---------------------------------------------------------

    def collect_episode(self, idx, noise=True):
        """One training episode by individual idx, stored into the shared buffer,
        with TERL's step bookkeeping."""
        st = self.state
        reward, transitions = rollout_episode(
            self.env,
            self.pop[idx],
            max_action=self.max_action,
            action_dim=self.action_dim,
            expl_noise=self.cfg.expl_noise if noise else 0.0,
            random_actions=self.timesteps < self.cfg.start_timesteps,
        )
        for state, action, next_state, r, terminated in transitions:
            self.buffer.add(state, action, next_state, r, terminated)
            self.timesteps += 1
            st.new_steps[idx] += 1
            if self.timesteps >= self.cfg.start_timesteps:
                st.learned_steps[idx] += 1
                st.total_learned_steps[idx] += 1
        self.num_games += 1
        return reward

    # -- PSO (generic over state_dict, replaces the hardcoded l1/l2/l3 lists) --

    def pso_step(self, update_frequency):
        st = self.state
        gbest = self.pop[st.best_idx].actor.state_dict()
        for i in range(self.cfg.pop_size):
            if i == st.best_idx:
                continue
            if st.learned_steps[i] - st.last_evo_point[i] > update_frequency:
                st.last_evo_point[i] = st.learned_steps[i]
                pso_pull(self.pop[i].actor, self.V[i], gbest, self.inertia_weight)

    # -- evaluation protocol ---------------------------------------------

    def maybe_test(self):
        st = self.state
        if self.timesteps - st.last_test_point >= self.cfg.eval_freq:
            st.last_test_point = self.timesteps
            score = eval_policy(self.eval_env, self.test_individual, self.cfg.eval_episodes)
            self.logger.log({"test_score": score}, self.timesteps)
            return score
        return None

    # -- main loop --------------------------------------------------------

    def train_round(self):
        raise NotImplementedError

    def run(self):
        last_ckpt = self.timesteps
        while self.timesteps <= self.cfg.max_timesteps:
            self.train_round()
            if self.timesteps - last_ckpt >= self.cfg.checkpoint_freq:
                last_ckpt = self.timesteps
                save_checkpoint(self.cfg.run_dir, self.state_dict())
        save_checkpoint(self.cfg.run_dir, self.state_dict())
        mark_done(self.cfg.run_dir)

    # -- checkpointing ----------------------------------------------------

    def state_dict(self):
        return {
            # config record: resume refuses to continue under different
            # behavior-affecting settings (see config.resume_mismatch)
            "config": asdict(self.cfg),
            "timesteps": self.timesteps,
            "num_games": self.num_games,
            "pop_state": vars(self.state).copy(),
            "pop": [ind.state_dict() for ind in self.pop],
            "V": self.V,
            "test_individual": self.test_individual.state_dict(),
            "buffer": self.buffer.state_dict(),
            "rng": get_rng_state(),
            "env_rng": get_env_rng_state(self.env),
            "eval_env_rng": get_env_rng_state(self.eval_env),
        }

    def load_state_dict(self, d):
        self.timesteps = d["timesteps"]
        self.num_games = d["num_games"]
        self.state = PopulationState(**d["pop_state"])
        for ind, s in zip(self.pop, d["pop"]):
            ind.load_state_dict(s)
        self.V = [
            {name: v.to(self.device) for name, v in vd.items()} for vd in d["V"]
        ]
        self.test_individual.load_state_dict(d["test_individual"])
        self.buffer.load_state_dict(d["buffer"])
        set_rng_state(d["rng"])
        set_env_rng_state(self.env, d["env_rng"])
        set_env_rng_state(self.eval_env, d["eval_env_rng"])


class TERLTrainer(PopulationTrainerBase):
    """Faithful gymnasium port of TERL-main/TERL.py Agent.train() (lines 214-292),
    with the fixed stage switch at ratio * max_timesteps."""

    def train_round(self):
        cfg = self.cfg
        st = self.state

        if self.timesteps >= cfg.max_timesteps * cfg.ratio:
            st.stage = 2
        max_eval_times = cfg.fitness_eval_times

        if st.learned_steps[st.extra_idx] > st.learned_steps[st.best_idx] or st.stage == 2:
            st.extra_idx = st.best_idx
        idx_list = [st.extra_idx] * 5 + list(range(cfg.pop_size))

        for i in idx_list:
            st.new_steps[i] = 0
            eval_times = 0
            fitness = 0.0
            while eval_times < max_eval_times:
                fitness = (fitness * eval_times + self.collect_episode(i, noise=True)) / (
                    eval_times + 1
                )
                eval_times += 1
                if fitness < st.max_best_f:
                    break

            if fitness > st.best_f[i]:
                st.best_f[i] = fitness
                st.learned_steps[i] = 0
                st.last_evo_point[i] = 0
                if st.stage == 1:
                    st.extra_idx = i
                    self.logger.log({f"fitness/{i}": st.best_f[i]}, self.timesteps)

            if st.best_f[i] > st.max_best_f:
                st.max_best_f = st.best_f[i]
                if st.stage == 1:
                    st.best_idx = i
                    copy_params(self.pop[i].actor, self.test_individual)
                else:
                    if i != st.best_idx:
                        copy_params(self.pop[i].actor, self.pop[st.best_idx].actor)
                    if cfg.stable_eval_times > 1 and max_eval_times == 1:
                        stable = 0.0
                        for _ in range(cfg.stable_eval_times):
                            stable += self.collect_episode(i, noise=False) / cfg.stable_eval_times
                        if stable > st.test_individual_fitness:
                            st.test_individual_fitness = stable
                            copy_params(self.pop[i].actor, self.test_individual)
                    else:
                        copy_params(self.pop[i].actor, self.test_individual)

            if self.buffer.size >= cfg.start_timesteps:
                learner = self.pop[i] if st.stage == 1 else self.pop[st.best_idx]
                for _ in range(st.new_steps[i]):
                    learner.train(self.buffer, cfg.batch_size)

        self.pso_step(1e4 if st.stage == 1 else 1e3)
        self.maybe_test()
