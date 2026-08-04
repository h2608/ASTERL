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
    # suffix distinguishing config variants of the same algo (e.g. "r0.1" for a
    # ratio-sweep arm); empty = canonical config. Affects run dir + W&B id.
    variant: str = ""

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
    # TERL's exploration stage is NOT uniform (paper Algorithm 1, line 8): the
    # most-recently-improved individual gets 6/10 of episodes AND gradients.
    # tau_max = 0.3 calibrates the g=0 softmax to that winner-take-most share
    # (raw rollout share ~0.6 for a hot leader; kappa pushes gradients higher);
    # 2.0 recovers the near-uniform open regime of v1-v5.
    tau_max: float = 0.3
    # tau_min must make g=1 collapse onto the best individual (TERL stage 2):
    # adjacent-rank logit gap = alpha * 0.25 / tau_min -> ~99.8% at 0.02 for pop 5
    tau_min: float = 0.02
    alpha: float = 0.5
    kappa: float = 1.5
    d_min: float = 0.02
    improve_eps: float = 0.01
    # v2 controller fixes (each independently disableable -> ablation arms;
    # improve_decay=0.0, prog_gate=False, alpha_anneal=False recovers v1):
    improve_decay: float = 0.5  # stagnation retained on global improvement (anti saw-tooth)
    prog_gate: bool = True  # diminishing-marginal-return gate, max-combined with stagnation
    alpha_anneal: bool = True  # alpha -> 1 as g -> 1 so g=1 collapses onto best fitness
    # v3: minimum rollout share per individual (0.0 recovers v2). At g=1 the
    # floored allocation is exactly TERL stage 2 (best 6/10, others 1/10):
    # non-best rollouts are PSO-perturbed samplers around gbest that keep
    # challenger fitness measurable while gradients concentrate unfloored.
    rollout_floor: float = 0.1
    # v5: what the concentrated regime does (the g signal itself is unchanged —
    # fixed r0.1 concentrates EARLIER than SGSA saturates and goes 5/5 on
    # Swimmer, so gate timing was never the failure; regime semantics were):
    #   pinned — TERL stage-2 semantics: rollout/gradient allocation rides the
    #     DESIGNATED best slot (rank-promoted into the softmax top), the
    #     designation never moves, and a challenger takes over only by setting
    #     an all-time fitness record (actor donated into the slot + histories
    #     swapped). Records are rare on plateaus, so the trained actor runs
    #     uninterrupted — the property every earlier variant lacked.
    #   swap   — v4: designation pinned, but overwrite fires on any strict
    #     window-mean lead (noise-frequency actor churn on plateaus: Swimmer
    #     overwrote the trained actor every ~2 rounds, 0/5 gait success).
    #   free   — v3: designation follows the instantaneous rank leader
    #     (gradient-budget churn onto starved critics: 7-67 moves/run).
    concentration: str = "pinned"
    # Hysteretic boundary of the concentrated regime on g: enter above
    # gate_enter, exit below gate_exit (enter=exit=0.5 recovers the old
    # memoryless g > 0.5 check, modulo behavior at exactly 0.5).
    gate_enter: float = 0.6
    gate_exit: float = 0.4
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


def algo_label(algo, variant):
    """Run-dir / W&B naming: config variants of one algo must not collide."""
    return f"{algo}-{variant}" if variant else algo


def variant_from_overrides(overrides):
    """The variant exactly as load_config will parse it — launch.py must map a
    job to the same run dir train.py will use (a quoted value like
    variant='"v3"' would otherwise resolve to two different directories)."""
    value = ""
    for item in overrides:
        key, _, raw = item.partition("=")
        if key == "variant":
            value = _parse_value(raw)
    return str(value)


# Fields that do not change the training trajectory: safe to differ on resume.
FINGERPRINT_EXCLUDE = {"run_dir", "wandb_mode", "wandb_project", "tb", "device",
                       "checkpoint_freq"}


def resume_mismatch(saved, cfg):
    """Behavior-affecting config keys on which `cfg` differs from the config
    recorded in a checkpoint (missing on either side counts as a difference)."""
    current = {k: v for k, v in asdict(cfg).items() if k not in FINGERPRINT_EXCLUDE}
    saved = {k: v for k, v in saved.items() if k not in FINGERPRINT_EXCLUDE}
    missing = object()
    return sorted(
        k for k in set(current) | set(saved)
        if current.get(k, missing) != saved.get(k, missing)
    )


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
