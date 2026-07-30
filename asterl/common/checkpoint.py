import json
import os
from pathlib import Path

import torch

CHECKPOINT_NAME = "checkpoint.pt"
DONE_MARKER = "DONE"


def checkpoint_path(run_dir):
    return Path(run_dir) / CHECKPOINT_NAME


def save_checkpoint(run_dir, payload):
    """Atomic save: write to tmp then rename, so a killed run never leaves a
    truncated checkpoint behind (required for spot instances / WSL2)."""
    path = checkpoint_path(run_dir)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(run_dir):
    path = checkpoint_path(run_dir)
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def truncate_metrics(run_dir, max_step):
    """Drop metrics.jsonl records logged after `max_step`. A crash between the
    last checkpoint and death leaves orphaned rows; the resumed run re-logs
    that step range (not necessarily identically on GPU), so the pre-crash
    branch must not survive in the file. Returns the number of dropped rows;
    malformed rows (e.g. a partial final line from the crash) are dropped too."""
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return 0
    tmp = path.with_suffix(".tmp")
    dropped = 0
    with open(path) as src, open(tmp, "w") as dst:
        for line in src:
            try:
                step = json.loads(line)["step"]
            except (json.JSONDecodeError, KeyError):
                dropped += 1
                continue
            if step <= max_step:
                dst.write(line)
            else:
                dropped += 1
    os.replace(tmp, path)
    return dropped


def mark_done(run_dir):
    (Path(run_dir) / DONE_MARKER).touch()


def is_done(run_dir):
    return (Path(run_dir) / DONE_MARKER).exists()
