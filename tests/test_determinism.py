import numpy as np
import pytest
import torch

from asterl.algos.aterl import ATERLTrainer
from asterl.algos.td3_baseline import TD3Trainer
from asterl.algos.terl import TERLTrainer
from tests.conftest import NullLogger, small_cfg

TRAINERS = {"td3": TD3Trainer, "terl": TERLTrainer, "aterl": ATERLTrainer}


def run_steps(trainer, until):
    while trainer.timesteps < until:
        trainer.train_round()
    return trainer


def actors_equal(t1, t2):
    if hasattr(t1, "pop"):
        nets1 = [ind.actor for ind in t1.pop] + [t1.test_individual]
        nets2 = [ind.actor for ind in t2.pop] + [t2.test_individual]
    else:
        nets1, nets2 = [t1.policy.actor], [t2.policy.actor]
    for n1, n2 in zip(nets1, nets2):
        for p1, p2 in zip(n1.parameters(), n2.parameters()):
            if not torch.equal(p1, p2):
                return False
    return True


@pytest.mark.parametrize("algo", ["td3", "terl", "aterl"])
def test_same_seed_same_result(tmp_path, algo):
    cfg = small_cfg(tmp_path / "a", algo)
    t1 = run_steps(TRAINERS[algo](cfg, NullLogger()), 2_000)
    cfg2 = small_cfg(tmp_path / "b", algo)
    t2 = run_steps(TRAINERS[algo](cfg2, NullLogger()), 2_000)
    assert t1.timesteps == t2.timesteps
    assert actors_equal(t1, t2)


@pytest.mark.parametrize("algo", ["td3", "terl", "aterl"])
def test_checkpoint_resume_equivalence(tmp_path, algo):
    cfg = small_cfg(tmp_path / "straight", algo)
    straight = run_steps(TRAINERS[algo](cfg, NullLogger()), 3_000)

    cfg_a = small_cfg(tmp_path / "part1", algo)
    part = run_steps(TRAINERS[algo](cfg_a, NullLogger()), 1_500)
    snapshot = part.state_dict()

    cfg_b = small_cfg(tmp_path / "part2", algo)
    resumed = TRAINERS[algo](cfg_b, NullLogger())
    resumed.load_state_dict(snapshot)
    assert resumed.timesteps == part.timesteps
    run_steps(resumed, 3_000)

    assert resumed.timesteps == straight.timesteps
    assert actors_equal(straight, resumed)
    if hasattr(straight, "state"):
        assert np.allclose(straight.state.best_f, resumed.state.best_f)
