import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


@dataclass
class Config:
    # run identity
    algo: str = "terl"
    env_id: str = "Hopper-v5"
    seed: int = 0
    run_dir: str = ""

    # environment
    env_kwargs: dict = field(default_factory=dict)
    max_episode_steps: int | None = None  # None -> gymnasium registry default

    # RL (TERL-main defaults)
    max_timesteps: int = 1_000_000
    start_timesteps: int = 25_000
    batch_size: int = 256
    expl_noise: float = 0.1
    discount: float = 0.99
    buffer_size: int = 1_000_000
    device: str = "auto"

    # population / TERL
    pop_size: int = 5
    ratio: float = 0.25  # fixed stage-switch fraction (TERL baseline)
    fitness_eval_times: int = 1  # >1 only for high-variance envs (LunarLander, Pendulum)
    stable_eval_times: int = 5  # stage-2 test-individual confirmation (1 for HalfCheetah)

    # evaluation protocol
    eval_freq: int = 5_000
    eval_episodes: int = 5

    # SGSA controller (aterl)
    allocator: str = "softmax"  # softmax | fixed (sw_ucb/exp3 are Phase-4 ablation arms)
    s_max: int = 75_000
    window_k: int = 5
    tau_max: float = 2.0
    # tau_min must make g=1 collapse onto the best individual (TERL stage 2):
    # adjacent-rank logit gap = alpha * 0.25 / tau_min -> ~99.8% at 0.02 for pop 5
    tau_min: float = 0.02
    alpha: float = 0.5
    kappa: float = 1.5
    d_min: float = 0.02
    improve_eps: float = 0.01
    episodes_per_round: int = 10  # pop_size + 5, matches TERL's per-round episode cadence
    diversity_states: int = 512

    # logging / checkpointing
    wandb_mode: str = "disabled"  # online | offline | disabled
    wandb_project: str = "adaptive-terl"
    tb: bool = False
    checkpoint_freq: int = 50_000


def _parse_value(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def load_config(algo, env_id, seed, overrides=()):
    """default.yaml -> env/<env_id>.yaml -> algo-specific block -> CLI overrides."""
    cfg = Config(algo=algo, env_id=env_id, seed=seed)
    merged = {}
    for path in (CONFIG_DIR / "default.yaml", CONFIG_DIR / "env" / f"{env_id}.yaml"):
        if path.exists():
            with open(path) as fh:
                merged.update(yaml.safe_load(fh) or {})
    for key, value in merged.items():
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown config key {key!r} in yaml")
        setattr(cfg, key, value)
    for item in overrides:
        key, _, value = item.partition("=")
        if not hasattr(cfg, key):
            raise KeyError(f"Unknown config override {key!r}")
        setattr(cfg, key, _parse_value(value))
    return cfg


def save_config(cfg, path):
    with open(path, "w") as fh:
        yaml.safe_dump(asdict(cfg), fh, sort_keys=False)
