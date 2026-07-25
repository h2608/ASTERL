import json
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

MARKS = [250_000, 500_000, 1_000_000]

def at_marks(pairs):
    out = []
    for m in MARKS:
        past = [v for s, v in pairs if s <= m]
        out.append(past[-1] if past else None)
    return out

print("paper's recorded TERL (v2 envs), per-seed test_score at 250k/500k/1M:")
for env in ["Hopper-v2", "HalfCheetah-v2"]:
    for run in sorted(Path(f"TERL-main/learning_curves/{env}").iterdir()):
        acc = EventAccumulator(str(run)); acc.Reload()
        tag = [t for t in acc.Tags()["scalars"] if "test_score" in t][0]
        pairs = [(e.step, e.value) for e in acc.Scalars(tag)]
        vals = at_marks(pairs)
        seed = run.name.split("__")[2]
        print(f"  {env:16s} seed{seed:>3s}: " + "  ".join(f"{v:8.0f}" if v is not None else "     n/a" for v in vals))

print("\nour v5 runs, per-seed test_score at 250k/500k/1M:")
for env in ["Hopper-v5", "HalfCheetah-v5"]:
    for algo in ["terl", "td3"]:
        for seed_dir in sorted(Path(f"runs_v2/{env}/{algo}").iterdir()):
            pairs = [(r["step"], r["test_score"]) for r in map(json.loads, open(seed_dir / "metrics.jsonl")) if "test_score" in r]
            vals = at_marks(pairs)
            print(f"  {env:16s} {algo:5s} {seed_dir.name}: " + "  ".join(f"{v:8.0f}" for v in vals))
