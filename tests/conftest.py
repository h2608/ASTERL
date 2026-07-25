import pytest

from asterl.common.config import Config


class NullLogger:
    def __init__(self):
        self.records = []

    def log(self, metrics, step):
        self.records.append((step, dict(metrics)))

    def close(self):
        pass


@pytest.fixture
def null_logger():
    return NullLogger()


def small_cfg(tmp_path, algo, **overrides):
    """Pendulum config small enough for CPU unit tests: warmup 500 steps,
    a few thousand total, tiny buffer sampling untouched."""
    cfg = Config(
        algo=algo,
        env_id="Pendulum-v1",
        seed=3,
        run_dir=str(tmp_path),
        max_timesteps=4_000,
        start_timesteps=500,
        batch_size=64,
        pop_size=3,
        episodes_per_round=4,
        fitness_eval_times=1,
        s_max=2_000,
        device="cpu",
        checkpoint_freq=10_000_000,
        diversity_states=128,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg
