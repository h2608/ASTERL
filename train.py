"""Single-run entry point.

    python train.py --algo aterl --env Hopper-v5 --seed 0 --set max_timesteps=100000

Run directories are deterministic (runs_v2/<env>/<algo>/seed<seed>) so a rerun
of the same command resumes from the latest checkpoint automatically.
"""

import argparse
import shutil
from pathlib import Path

from asterl.algos.aterl import ATERLTrainer
from asterl.algos.td3_baseline import TD3Trainer
from asterl.algos.terl import TERLTrainer
from asterl.common.checkpoint import is_done, load_checkpoint, truncate_metrics
from asterl.common.config import (
    REPO_ROOT,
    algo_label,
    load_config,
    resume_mismatch,
    save_config,
)
from asterl.common.logger import RunLogger

TRAINERS = {"td3": TD3Trainer, "terl": TERLTrainer, "aterl": ATERLTrainer}


def prepare_run_dir(run_dir, fresh):
    """Pre-flight before anything is written. A completed run's artifacts
    (config.yaml, DONE, metrics) are its authoritative record: the DONE skip
    happens before any write, and --fresh wipes the directory so a partial
    rerun can never masquerade as the finished run it replaced.
    Returns False if the run is already DONE and should be skipped."""
    if is_done(run_dir) and not fresh:
        print(f"{run_dir} already DONE; use --fresh to rerun.")
        return False
    if fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return True


def load_resume_checkpoint(run_dir, cfg):
    """Checkpoint for resuming, after refusing config mismatches and dropping
    metrics rows logged after the checkpoint (orphans from a crash: the resumed
    run re-logs that step range, not necessarily identically on GPU)."""
    ckpt = load_checkpoint(run_dir)
    if ckpt is None:
        return None
    saved_cfg = ckpt.get("config")
    if saved_cfg is None:
        print("WARNING: checkpoint predates config fingerprinting; resuming unchecked.")
    else:
        mismatch = resume_mismatch(saved_cfg, cfg)
        if mismatch:
            raise SystemExit(
                f"Refusing to resume {run_dir}: config differs from the checkpoint "
                f"on {mismatch}. Rerun with --fresh or use a new variant."
            )
    dropped = truncate_metrics(run_dir, ckpt["timesteps"])
    if dropped:
        print(f"Dropped {dropped} metrics rows logged after the checkpoint.")
    return ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=sorted(TRAINERS))
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--wandb", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--tb", action="store_true")
    parser.add_argument("--runs-root", default=str(REPO_ROOT / "runs_v2"))
    parser.add_argument(
        "--fresh", action="store_true", help="wipe the run directory and start over"
    )
    args = parser.parse_args()

    cfg = load_config(args.algo, args.env, args.seed, args.set)
    if args.wandb is not None:
        cfg.wandb_mode = args.wandb
    if args.tb:
        cfg.tb = True

    run_dir = (
        Path(args.runs_root)
        / cfg.env_id
        / algo_label(cfg.algo, cfg.variant)
        / f"seed{cfg.seed}"
    )
    if not prepare_run_dir(run_dir, args.fresh):
        return
    cfg.run_dir = str(run_dir)
    save_config(cfg, run_dir / "config.yaml")

    ckpt = None if args.fresh else load_resume_checkpoint(run_dir, cfg)

    logger = RunLogger(cfg, run_dir)
    trainer = TRAINERS[cfg.algo](cfg, logger)
    if ckpt is not None:
        trainer.load_state_dict(ckpt)
        print(f"Resumed from checkpoint at {trainer.timesteps} steps.")

    try:
        trainer.run()
    finally:
        logger.close()
    print(f"Finished: {run_dir}")


if __name__ == "__main__":
    main()
