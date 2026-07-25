import json
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from asterl.common.config import REPO_ROOT


class RunLogger:
    """Always writes metrics.jsonl in run_dir; W&B and TensorBoard are optional.

    Controller internals (g, tau, D, p_i, ...) are logged with the same step
    axis (global env steps) as test_score — these are the paper's analysis
    figures, logged from day one.
    """

    def __init__(self, cfg, run_dir):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._jsonl = open(self.run_dir / "metrics.jsonl", "a", buffering=1)

        self._wandb = None
        if cfg.wandb_mode != "disabled":
            load_dotenv(REPO_ROOT / ".env")
            import wandb

            self._wandb = wandb.init(
                project=cfg.wandb_project,
                name=f"{cfg.env_id}__{cfg.algo}__s{cfg.seed}",
                group=f"{cfg.algo}/{cfg.env_id}",
                config=asdict(cfg),
                mode=cfg.wandb_mode,
                dir=str(self.run_dir),
                resume="allow",
                id=f"{cfg.env_id}-{cfg.algo}-s{cfg.seed}".replace("/", "-"),
            )

        self._tb = None
        if cfg.tb:
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(str(self.run_dir / "tb"))

    def log(self, metrics, step):
        record = {"step": int(step)}
        record.update({k: float(v) for k, v in metrics.items()})
        self._jsonl.write(json.dumps(record) + "\n")
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        if self._tb is not None:
            for key, value in metrics.items():
                self._tb.add_scalar(key, value, step)

    def close(self):
        self._jsonl.close()
        if self._wandb is not None:
            self._wandb.finish()
        if self._tb is not None:
            self._tb.close()
