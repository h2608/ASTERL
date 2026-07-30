"""Sweep launcher: expands an (algo x env x seed) grid into train.py subprocesses.

    python launch.py --algo terl,td3 --env Hopper-v5,HalfCheetah-v5 --seeds 0-4 \
        --workers 5 --set max_timesteps=1000000

Each worker is pinned to OMP_NUM_THREADS=1 (MuJoCo + small MLPs are CPU-bound;
throughput comes from run-level parallelism). Completed runs (DONE marker) are
skipped, interrupted runs resume from their checkpoint, so the launcher is safe
to rerun after a crash or spot preemption.
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from asterl.common.checkpoint import is_done
from asterl.common.config import REPO_ROOT, algo_label, variant_from_overrides


def parse_seeds(text):
    if "-" in text and "," not in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(s) for s in text.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", required=True, help="comma-separated")
    parser.add_argument("--env", required=True, help="comma-separated")
    parser.add_argument("--seeds", default="0", help="e.g. 0-4 or 0,1,7")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--wandb", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("--runs-root", default=str(REPO_ROOT / "runs_v2"))
    args = parser.parse_args()

    # mirror train.py's run-dir naming (same parser) so the DONE-skip sees
    # exactly the directory train.py will use
    variant = variant_from_overrides(args.set)

    jobs = []
    for algo in args.algo.split(","):
        label = algo_label(algo, variant)
        for env in args.env.split(","):
            for seed in parse_seeds(args.seeds):
                run_dir = Path(args.runs_root) / env / label / f"seed{seed}"
                if is_done(run_dir):
                    print(f"skip (done): {algo} {env} seed{seed}")
                    continue
                cmd = [
                    sys.executable,
                    str(REPO_ROOT / "train.py"),
                    "--algo", algo,
                    "--env", env,
                    "--seed", str(seed),
                    "--runs-root", args.runs_root,
                ]
                for item in args.set:
                    cmd += ["--set", item]
                if args.wandb:
                    cmd += ["--wandb", args.wandb]
                jobs.append((algo, env, seed, cmd, run_dir))

    print(f"{len(jobs)} jobs, {args.workers} workers")
    env_vars = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")

    def run(job):
        algo, env, seed, cmd, run_dir = job
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "stdout.log", "a") as log:
            result = subprocess.run(cmd, env=env_vars, stdout=log, stderr=subprocess.STDOUT)
        status = "ok" if result.returncode == 0 else f"FAILED({result.returncode})"
        print(f"[{status}] {algo} {env} seed{seed}")
        return result.returncode

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        codes = list(pool.map(run, jobs))
    failed = sum(1 for c in codes if c != 0)
    print(f"done: {len(codes) - failed} ok, {failed} failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
