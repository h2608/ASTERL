import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def get_env_rng_state(env):
    return {
        "env": env.unwrapped.np_random.bit_generator.state,
        "action_space": env.action_space.np_random.bit_generator.state,
    }


def set_env_rng_state(env, state):
    env.unwrapped.np_random.bit_generator.state = state["env"]
    env.action_space.np_random.bit_generator.state = state["action_space"]
