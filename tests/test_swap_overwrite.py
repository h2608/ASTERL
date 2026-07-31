"""v4 swap-overwrite: in the concentrated regime (g > 0.5) a strictly leading
challenger donates its actor into the incumbent best slot and the two tracker
slots swap fitness records — best_idx never moves, so the gradient budget
stays on the continuously-trained critic (zero best-slot churn). In the open
regime the designation follows the rank leader as before."""

import torch

from asterl.algos.aterl import ATERLTrainer
from tests.conftest import NullLogger, small_cfg


def make_trainer(tmp_path, **overrides):
    return ATERLTrainer(small_cfg(tmp_path, "aterl", **overrides), NullLogger())


def seed_leader(trainer, leader=2, incumbent=0):
    """Give `leader` the best window mean while `incumbent` holds best_idx."""
    trainer.state.best_idx = incumbent
    for i in range(trainer.cfg.pop_size):
        fitness = 100.0 if i == leader else 10.0 * (i + 1)
        trainer.tracker.record_eval(i, fitness, env_steps=100 * (i + 1))


def actor_params(trainer, i):
    return [p.detach().clone() for p in trainer.pop[i].actor.parameters()]


def params_equal(params, actor):
    return all(torch.equal(p, q) for p, q in zip(params, actor.parameters()))


def test_swap_at_high_g(tmp_path):
    trainer = make_trainer(tmp_path)
    seed_leader(trainer)
    leader_params = actor_params(trainer, 2)
    assert not params_equal(leader_params, trainer.pop[0].actor)
    assert trainer._designate_best(1.0) == 1.0
    # incumbent slot keeps the designation but takes the challenger's actor
    # and record; the challenger slot inherits the incumbent's old record
    assert trainer.state.best_idx == 0
    assert params_equal(leader_params, trainer.pop[0].actor)
    assert params_equal(leader_params, trainer.pop[2].actor)  # donated, not moved
    means = trainer.tracker.fitness_means()
    assert means[0] == 100.0 and means[2] == 10.0


def test_designation_moves_at_low_g(tmp_path):
    trainer = make_trainer(tmp_path)
    seed_leader(trainer)
    incumbent_params = actor_params(trainer, 0)
    assert trainer._designate_best(0.0) == 0.0
    assert trainer.state.best_idx == 2
    assert params_equal(incumbent_params, trainer.pop[0].actor)  # no overwrite


def test_boundary_g_is_open_regime(tmp_path):
    """g exactly 0.5 must behave as the open regime (designation moves, no
    overwrite) — same boundary convention as _update_champion's g > 0.5."""
    trainer = make_trainer(tmp_path)
    seed_leader(trainer)
    incumbent_params = actor_params(trainer, 0)
    assert trainer._designate_best(0.5) == 0.0
    assert trainer.state.best_idx == 2
    assert params_equal(incumbent_params, trainer.pop[0].actor)


def test_swap_survives_checkpoint_roundtrip(tmp_path):
    """Everything a swap mutates (actor params, tracker records, best_idx)
    must land in state_dict so a kill-and-resume at the next round boundary
    is bit-exact."""
    trainer = make_trainer(tmp_path)
    seed_leader(trainer)
    assert trainer._designate_best(1.0) == 1.0
    clone = make_trainer(tmp_path)
    clone.load_state_dict(trainer.state_dict())
    assert clone.state.best_idx == trainer.state.best_idx == 0
    assert clone.tracker.fitness_means() == trainer.tracker.fitness_means()
    assert clone.tracker.personal_best == trainer.tracker.personal_best
    for i in range(trainer.cfg.pop_size):
        assert params_equal(actor_params(trainer, i), clone.pop[i].actor)


def test_flag_off_recovers_v3(tmp_path):
    trainer = make_trainer(tmp_path, swap_overwrite=False)
    seed_leader(trainer)
    assert trainer._designate_best(1.0) == 0.0
    assert trainer.state.best_idx == 2  # v3: designation moves even at g=1


def test_no_swap_without_strict_lead(tmp_path):
    """A tie is not a lead: cand (argmax picks the lowest tied index, 0) must
    not displace or overwrite an equally-ranked incumbent (slot 1)."""
    trainer = make_trainer(tmp_path)
    trainer.state.best_idx = 1
    for i in range(trainer.cfg.pop_size):
        trainer.tracker.record_eval(i, 50.0, env_steps=100 * (i + 1))
    incumbent_params = actor_params(trainer, 1)
    assert trainer._designate_best(1.0) == 0.0
    assert trainer.state.best_idx == 1
    assert params_equal(incumbent_params, trainer.pop[1].actor)
