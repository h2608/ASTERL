import numpy as np
import torch


class ReplayBuffer:
    """Port of TERL-main/utils.py ReplayBuffer: float32 storage, checkpointable."""

    def __init__(self, state_dim, action_dim, max_size=int(1e6), device="cpu"):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

        self.device = device

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.as_tensor(self.state[ind]).to(self.device),
            torch.as_tensor(self.action[ind]).to(self.device),
            torch.as_tensor(self.next_state[ind]).to(self.device),
            torch.as_tensor(self.reward[ind]).to(self.device),
            torch.as_tensor(self.not_done[ind]).to(self.device),
        )

    def sample_states(self, n):
        ind = np.random.randint(0, self.size, size=n)
        return torch.as_tensor(self.state[ind]).to(self.device)

    def state_dict(self):
        n = self.size
        return {
            "ptr": self.ptr,
            "size": self.size,
            "state": self.state[:n].copy(),
            "action": self.action[:n].copy(),
            "next_state": self.next_state[:n].copy(),
            "reward": self.reward[:n].copy(),
            "not_done": self.not_done[:n].copy(),
        }

    def load_state_dict(self, d):
        n = d["size"]
        self.ptr = d["ptr"]
        self.size = n
        self.state[:n] = d["state"]
        self.action[:n] = d["action"]
        self.next_state[:n] = d["next_state"]
        self.reward[:n] = d["reward"]
        self.not_done[:n] = d["not_done"]
