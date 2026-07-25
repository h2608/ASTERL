import torch

from asterl.algos.td3 import Actor
from asterl.algos.terl import pso_pull


def make_pair():
    torch.manual_seed(0)
    actor = Actor(4, 2, 1.0)
    gbest_actor = Actor(4, 2, 1.0)
    velocity = {name: torch.zeros_like(p) for name, p in actor.state_dict().items()}
    return actor, gbest_actor.state_dict(), velocity


def test_pull_moves_every_param_toward_gbest():
    actor, gbest, velocity = make_pair()
    before = {n: p.clone() for n, p in actor.state_dict().items()}
    torch.manual_seed(1)
    pso_pull(actor, velocity, gbest, inertia=0.0)
    after = actor.state_dict()
    for name in before:
        gap_before = (before[name] - gbest[name]).abs()
        gap_after = (after[name] - gbest[name]).abs()
        assert torch.all(gap_after <= gap_before + 1e-7)


def test_pull_with_unit_rand_reaches_gbest_exactly(monkeypatch):
    actor, gbest, velocity = make_pair()
    monkeypatch.setattr(torch, "rand_like", lambda t: torch.ones_like(t))
    pso_pull(actor, velocity, gbest, inertia=0.0)
    for name, p in actor.state_dict().items():
        assert torch.equal(p, gbest[name])


def test_generic_matches_terl_reference_order(monkeypatch):
    """With a constant rand factor the generic state_dict iteration must give
    exactly what TERL-main's hardcoded l1/l2/l3 weight+bias list update gives."""
    monkeypatch.setattr(torch, "rand_like", lambda t: torch.full_like(t, 0.37))

    actor, gbest, velocity = make_pair()
    reference = {n: p.clone() for n, p in actor.state_dict().items()}
    # TERL-main/TERL.py:168-194 reference computation (inertia 0, pbest term = 0)
    for name in ["l1.weight", "l2.weight", "l3.weight", "l1.bias", "l2.bias", "l3.bias"]:
        v = 0.37 * (gbest[name] - reference[name])
        reference[name] = reference[name] + v

    pso_pull(actor, velocity, gbest, inertia=0.0)
    for name, p in actor.state_dict().items():
        assert torch.allclose(p, reference[name])
