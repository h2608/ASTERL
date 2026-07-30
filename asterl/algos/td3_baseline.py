from dataclasses import asdict

import torch

from asterl.algos.td3 import TD3
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


class TD3Trainer:
    """Single-agent TD3 baseline under the same rollout/eval/checkpoint protocol
    as the population trainers (1 gradient step per env step after warmup)."""

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

        self.policy = TD3(
            self.state_dim,
            self.action_dim,
            self.max_action,
            discount=cfg.discount,
            device=self.device,
        )
        self.buffer = ReplayBuffer(
            self.state_dim, self.action_dim, cfg.buffer_size, device=self.device
        )
        self.timesteps = 0
        self.last_test_point = 0

    def train_round(self):
        cfg = self.cfg
        _, transitions = rollout_episode(
            self.env,
            self.policy,
            max_action=self.max_action,
            action_dim=self.action_dim,
            expl_noise=cfg.expl_noise,
            random_actions=self.timesteps < cfg.start_timesteps,
        )
        for state, action, next_state, reward, terminated in transitions:
            self.buffer.add(state, action, next_state, reward, terminated)
            self.timesteps += 1
        if self.buffer.size >= cfg.start_timesteps:
            for _ in range(len(transitions)):
                self.policy.train(self.buffer, cfg.batch_size)

        if self.timesteps - self.last_test_point >= cfg.eval_freq:
            self.last_test_point = self.timesteps
            score = eval_policy(self.eval_env, self.policy.actor, cfg.eval_episodes)
            self.logger.log({"test_score": score}, self.timesteps)

    def run(self):
        last_ckpt = self.timesteps
        while self.timesteps <= self.cfg.max_timesteps:
            self.train_round()
            if self.timesteps - last_ckpt >= self.cfg.checkpoint_freq:
                last_ckpt = self.timesteps
                save_checkpoint(self.cfg.run_dir, self.state_dict())
        save_checkpoint(self.cfg.run_dir, self.state_dict())
        mark_done(self.cfg.run_dir)

    def state_dict(self):
        return {
            "config": asdict(self.cfg),
            "timesteps": self.timesteps,
            "last_test_point": self.last_test_point,
            "policy": self.policy.state_dict(),
            "buffer": self.buffer.state_dict(),
            "rng": get_rng_state(),
            "env_rng": get_env_rng_state(self.env),
            "eval_env_rng": get_env_rng_state(self.eval_env),
        }

    def load_state_dict(self, d):
        self.timesteps = d["timesteps"]
        self.last_test_point = d["last_test_point"]
        self.policy.load_state_dict(d["policy"])
        self.buffer.load_state_dict(d["buffer"])
        set_rng_state(d["rng"])
        set_env_rng_state(self.env, d["env_rng"])
        set_env_rng_state(self.eval_env, d["eval_env_rng"])
