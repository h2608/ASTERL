"""Quick sweep status: python status.py [runs_root]"""

import json
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs_v2")
rows = []
for cfg_path in sorted(root.glob("*/*/seed*/config.yaml")):
    run_dir = cfg_path.parent
    env, algo, seed = run_dir.parts[-3], run_dir.parts[-2], run_dir.parts[-1]
    done = (run_dir / "DONE").exists()
    step, score = 0, None
    metrics = run_dir / "metrics.jsonl"
    if metrics.exists():
        for line in metrics.open():
            rec = json.loads(line)
            step = max(step, rec["step"])
            if "test_score" in rec:
                score = rec["test_score"]
    state = "DONE" if done else f"{step:>9,}"
    rows.append((env, algo, seed, state, score))

if not rows:
    print(f"no runs under {root}/")
for env, algo, seed, state, score in rows:
    score_txt = f"{score:9.1f}" if score is not None else "        -"
    print(f"{env:18s} {algo:6s} {seed:6s} {state:>9s}  last test_score {score_txt}")
