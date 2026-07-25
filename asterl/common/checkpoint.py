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


def mark_done(run_dir):
    (Path(run_dir) / DONE_MARKER).touch()


def is_done(run_dir):
    return (Path(run_dir) / DONE_MARKER).exists()
