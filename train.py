"""Single-run entry point.

    python train.py --algo aterl --env Hopper-v5 --seed 0 --set max_timesteps=100000

Run directories are deterministic (runs_v2/<env>/<algo>/seed<seed>) so a rerun
of the same command resumes from the latest checkpoint automatically.
"""

import argparse
from pathlib import Path

from asterl.algos.aterl import ATERLTrainer
from asterl.algos.td3_baseline import TD3Trainer
from asterl.algos.terl import TERLTrainer
from asterl.common.checkpoint import is_done, load_checkpoint
from asterl.common.config import REPO_ROOT, load_config, save_config
from asterl.common.logger import RunLogger

TRAINERS = {"td3": TD3Trainer, "terl": TERLTrainer, "aterl": ATERLTrainer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, choices=sorted(TRAINERS))
    parser.add_argument("--env", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--wandb", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--tb", action="store_true")
    parser.add_argument("--runs-root", default=str(REPO_ROOT / "runs_v2"))
    parser.add_argument("--fresh", action="store_true", help="ignore existing checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.algo, args.env, args.seed, args.set)
    if args.wandb is not None:
        cfg.wandb_mode = args.wandb
    if args.tb:
        cfg.tb = True

    algo_label = f"{cfg.algo}-{cfg.variant}" if cfg.variant else cfg.algo
    run_dir = Path(args.runs_root) / cfg.env_id / algo_label / f"seed{cfg.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.run_dir = str(run_dir)
    save_config(cfg, run_dir / "config.yaml")

    if is_done(run_dir) and not args.fresh:
        print(f"{run_dir} already DONE; use --fresh to rerun.")
        return

    logger = RunLogger(cfg, run_dir)
    trainer = TRAINERS[cfg.algo](cfg, logger)

    if not args.fresh:
        ckpt = load_checkpoint(run_dir)
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
