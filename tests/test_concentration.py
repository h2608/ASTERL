"""Concentrated-regime semantics (cfg.concentration):

pinned (v5) — TERL stage 2: the designation is frozen, allocation rides it,
and a challenger takes over only on an all-time fitness record (actor donated
into the designated slot + histories swapped). swap (v4) — overwrite on any
strict window-mean lead. free (v3) — the designation follows the rank leader.
"""

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


# -- pinned (v5) -----------------------------------------------------------


def test_pinned_freezes_designation(tmp_path):
    """A window-mean lead alone moves nothing: no designation change, no
    overwrite (v4 overwrote the trained actor on every such lead)."""
    trainer = make_trainer(tmp_path)
    seed_leader(trainer)
    incumbent_params = actor_params(trainer, 0)
    assert trainer._designate_best(True) == 0.0
    assert trainer.state.best_idx == 0
    assert params_equal(incumbent_params, trainer.pop[0].actor)


def test_pinned_succession_on_record(tmp_path):
    """An all-time record is the only takeover path: the record-setter's
    actor is donated into the designated slot and the records swap, with
    best_idx unchanged — TERL's stage-2 overwrite."""
    trainer = make_trainer(tmp_path)
    seed_leader(trainer)
    trainer.state.max_best_f = 60.0  # slot 2's personal best 100 is a record
    leader_params = actor_params(trainer, 2)
    assert trainer._record_succession(2, True) == 1.0
    assert trainer.state.best_idx == 0
    assert trainer.state.max_best_f == 100.0
    assert params_equal(leader_params, trainer.pop[0].actor)
    assert params_equal(leader_params, trainer.pop[2].actor)  # donated, not moved
    means = trainer.tracker.fitness_means()
    assert means[0] == 100.0 and means[2] == 10.0


def test_record_by_incumbent_is_not_a_takeover(tmp_path):
    trainer = make_trainer(tmp_path)
    seed_leader(trainer, leader=0, incumbent=0)
    trainer.state.max_best_f = 60.0
    assert trainer._record_succession(0, True) == 0.0
    assert trainer.state.max_best_f == 100.0


def test_record_in_open_regime_only_updates_the_record(tmp_path):
    trainer = make_trainer(tmp_path)
    seed_leader(trainer)
    trainer.state.max_best_f = 60.0
    incumbent_params = actor_params(trainer, 0)
    assert trainer._record_succession(2, False) == 0.0
    assert trainer.state.max_best_f == 100.0
    assert params_equal(incumbent_params, trainer.pop[0].actor)


def test_record_in_swap_and_free_modes_only_updates_the_record(tmp_path):
    """_record_succession's takeover path is pinned-mode-only; the other
    modes must still advance max_best_f (it gates the champion protocol)."""
    for mode in ("swap", "free"):
        trainer = make_trainer(tmp_path, concentration=mode)
        seed_leader(trainer)
        trainer.state.max_best_f = 60.0
        incumbent_params = actor_params(trainer, 0)
        assert trainer._record_succession(2, True) == 0.0
        assert trainer.state.max_best_f == 100.0
        assert params_equal(incumbent_params, trainer.pop[0].actor)


def test_succession_survives_checkpoint_roundtrip(tmp_path):
    trainer = make_trainer(tmp_path)
    seed_leader(trainer)
    trainer.state.max_best_f = 60.0
    assert trainer._record_succession(2, True) == 1.0
    clone = make_trainer(tmp_path)
    clone.load_state_dict(trainer.state_dict())
    assert clone.state.best_idx == trainer.state.best_idx == 0
    assert clone.state.max_best_f == 100.0
    assert clone.tracker.fitness_means() == trainer.tracker.fitness_means()
    assert clone.tracker.personal_best == trainer.tracker.personal_best
    for i in range(trainer.cfg.pop_size):
        assert params_equal(actor_params(trainer, i), clone.pop[i].actor)


def test_interleaved_gradients_preserve_utd(tmp_path):
    """The gradient budget is applied in per-episode chunks (TERL's cadence,
    paper Algorithm 1 lines 22-25); total gradient steps per round must still
    equal the round's env steps exactly (UTD=1, the compute-fairness
    invariant)."""
    trainer = make_trainer(tmp_path)
    counter = {"n": 0}
    for ind in trainer.pop:
        orig = ind.train

        def counting(buffer, batch_size, _orig=orig):
            counter["n"] += 1
            return _orig(buffer, batch_size)

        ind.train = counting
    while trainer.buffer.size < trainer.cfg.start_timesteps:
        trainer.train_round()
    counter["n"] = 0
    t0 = trainer.timesteps
    trainer.train_round()
    assert counter["n"] == trainer.timesteps - t0 > 0


# -- swap (v4) and free (v3) ablation arms ---------------------------------


def test_swap_mode_overwrites_on_lead(tmp_path):
    trainer = make_trainer(tmp_path, concentration="swap")
    seed_leader(trainer)
    leader_params = actor_params(trainer, 2)
    assert trainer._designate_best(True) == 1.0
    assert trainer.state.best_idx == 0
    assert params_equal(leader_params, trainer.pop[0].actor)
    means = trainer.tracker.fitness_means()
    assert means[0] == 100.0 and means[2] == 10.0


def test_swap_mode_ignores_ties(tmp_path):
    """A tie is not a lead: cand (argmax picks the lowest tied index, 0) must
    not displace or overwrite an equally-ranked incumbent (slot 1)."""
    trainer = make_trainer(tmp_path, concentration="swap")
    trainer.state.best_idx = 1
    for i in range(trainer.cfg.pop_size):
        trainer.tracker.record_eval(i, 50.0, env_steps=100 * (i + 1))
    incumbent_params = actor_params(trainer, 1)
    assert trainer._designate_best(True) == 0.0
    assert trainer.state.best_idx == 1
    assert params_equal(incumbent_params, trainer.pop[1].actor)


def test_free_mode_moves_designation(tmp_path):
    trainer = make_trainer(tmp_path, concentration="free")
    seed_leader(trainer)
    incumbent_params = actor_params(trainer, 0)
    assert trainer._designate_best(True) == 0.0
    assert trainer.state.best_idx == 2
    assert params_equal(incumbent_params, trainer.pop[0].actor)  # no overwrite


def test_open_regime_moves_designation_in_all_modes(tmp_path):
    for mode in ("pinned", "swap", "free"):
        trainer = make_trainer(tmp_path, concentration=mode)
        seed_leader(trainer)
        assert trainer._designate_best(False) == 0.0
        assert trainer.state.best_idx == 2
