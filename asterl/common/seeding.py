import random

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        # GPU kernels (rand_like in the PSO pull, TD3 target noise) draw from
        # this stream: without it a resumed GPU run diverges from the
        # uninterrupted one. (Bit-exact GPU resume additionally requires
        # deterministic kernels; this removes the RNG divergence.)
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def get_env_rng_state(env):
    return {
        "env": env.unwrapped.np_random.bit_generator.state,
        "action_space": env.action_space.np_random.bit_generator.state,
    }


def set_env_rng_state(env, state):
    env.unwrapped.np_random.bit_generator.state = state["env"]
    env.action_space.np_random.bit_generator.state = state["action_space"]
