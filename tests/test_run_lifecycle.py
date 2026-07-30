"""Run-identity and resume safety: DONE-skip ordering, --fresh lifecycle,
config fingerprinting, crash-orphan truncation, CUDA RNG round-trip."""

import json

import pytest
import torch

from asterl.common.checkpoint import mark_done, truncate_metrics
from asterl.common.config import Config, resume_mismatch, variant_from_overrides
from asterl.common.seeding import get_rng_state, set_rng_state
from train import prepare_run_dir


def test_done_skip_before_any_write(tmp_path):
    run_dir = tmp_path / "seed0"
    run_dir.mkdir()
    mark_done(run_dir)
    (run_dir / "config.yaml").write_text("authoritative: true\n")
    assert prepare_run_dir(run_dir, fresh=False) is False
    # the completed run's record must be untouched by the skipped invocation
    assert (run_dir / "config.yaml").read_text() == "authoritative: true\n"


def test_fresh_wipes_stale_artifacts(tmp_path):
    run_dir = tmp_path / "seed0"
    run_dir.mkdir()
    mark_done(run_dir)
    for name in ("checkpoint.pt", "metrics.jsonl", "config.yaml"):
        (run_dir / name).write_text("stale")
    assert prepare_run_dir(run_dir, fresh=True) is True
    # an interrupted --fresh rerun must not inherit the old DONE marker or data
    assert list(run_dir.iterdir()) == []


def test_resume_mismatch_detects_behavior_changes():
    saved = Config(algo="aterl", rollout_floor=0.0, prog_gate=False)
    current = Config(algo="aterl")  # today's defaults
    diff = resume_mismatch(vars_of(saved), current)
    assert "rollout_floor" in diff and "prog_gate" in diff


def test_resume_mismatch_ignores_cosmetic_fields():
    saved = Config(wandb_mode="online", tb=True, device="cuda", run_dir="/old")
    current = Config(wandb_mode="disabled", tb=False, device="auto", run_dir="/new")
    assert resume_mismatch(vars_of(saved), current) == []


def vars_of(cfg):
    from dataclasses import asdict

    return asdict(cfg)


def test_truncate_metrics_drops_orphans(tmp_path):
    path = tmp_path / "metrics.jsonl"
    rows = [
        {"step": 100, "test_score": 1.0},
        {"step": 200, "test_score": 2.0},  # checkpoint here
        {"step": 300, "test_score": 3.0},  # orphaned pre-crash branch
    ]
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        fh.write('{"step": 400, "test_sc')  # partial line from the crash
    dropped = truncate_metrics(tmp_path, max_step=200)
    assert dropped == 2
    kept = [json.loads(line)["step"] for line in open(path)]
    assert kept == [100, 200]


def test_variant_parsing_matches_train(tmp_path):
    # launch.py and train.py must resolve the same directory name even for
    # quoted override values
    assert variant_from_overrides(["variant=v3"]) == "v3"
    assert variant_from_overrides(["variant='v3'"]) == "v3"
    assert variant_from_overrides(["ratio=0.1", "variant=r0.1"]) == "r0.1"
    assert variant_from_overrides(["ratio=0.1"]) == ""


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_cuda_rng_roundtrip():
    saved = get_rng_state()
    assert "torch_cuda" in saved
    first = torch.rand(8, device="cuda")
    set_rng_state(saved)
    replay = torch.rand(8, device="cuda")
    assert torch.equal(first, replay)
