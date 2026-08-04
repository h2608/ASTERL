"""Phase-1 ratio-sensitivity summary (the paper's Figure-1 numbers).

Aggregates final test_score per (env, ratio arm) over seeds:
    conda run -n asterl python scripts/analyze_phase1.py
"""

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "runs_v2"
ENVS = ["Hopper-v5", "HalfCheetah-v5", "Swimmer-v5"]
ARMS = [
    ("0.10", "terl-r0.1"),
    ("0.25", "terl"),
    ("0.50", "terl-r0.5"),
    ("0.75", "terl-r0.75"),
    ("SGSA", "aterl"),
    ("SGSA2", "aterl-v2"),
    ("SGSA3", "aterl-v3"),
    ("SGSA4", "aterl-v4"),
    ("SGSA5", "aterl-v5"),
    ("SGSA6", "aterl-v6"),
]
MARKS = [500_000, 1_000_000]


def scores_at(run_dir, mark):
    """Last test_score logged at or before `mark` env steps. Ties on step keep
    the LAST record: after a crash-resume the replayed range is authoritative,
    not the orphaned pre-crash rows."""
    best_step, score = -1, None
    for line in (run_dir / "metrics.jsonl").open():
        rec = json.loads(line)
        if "test_score" in rec and best_step <= rec["step"] <= mark:
            best_step, score = rec["step"], rec["test_score"]
    return score


for env in ENVS:
    print(f"\n{env}")
    for label, algo_dir in ARMS:
        for mark in MARKS:
            vals = []
            for seed_dir in sorted((ROOT / env / algo_dir).glob("seed*")):
                s = scores_at(seed_dir, mark)
                if s is not None:
                    vals.append(s)
            if not vals:
                continue
            mean = statistics.mean(vals)
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            if mark == MARKS[-1]:
                seeds_txt = " ".join(f"{v:8.1f}" for v in sorted(vals))
                print(f"  ratio {label}  1.0M: {mean:8.1f} ± {sd:6.1f}  (n={len(vals)})  seeds: {seeds_txt}")
            else:
                print(f"  ratio {label}  0.5M: {mean:8.1f} ± {sd:6.1f}  (n={len(vals)})")
