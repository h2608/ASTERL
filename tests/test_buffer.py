import numpy as np

from asterl.common.buffer import ReplayBuffer


def test_add_sample_roundtrip():
    buf = ReplayBuffer(state_dim=3, action_dim=2, max_size=10)
    s = np.arange(3.0)
    a = np.arange(2.0)
    buf.add(s, a, s + 1, 0.5, 0.0)
    assert buf.size == 1
    state, action, next_state, reward, not_done = buf.sample(4)
    assert state.shape == (4, 3)
    assert np.allclose(state.numpy(), s)
    assert np.allclose(not_done.numpy(), 1.0)


def test_wraparound():
    buf = ReplayBuffer(state_dim=1, action_dim=1, max_size=5)
    for i in range(8):
        buf.add([float(i)], [0.0], [0.0], 0.0, 0.0)
    assert buf.size == 5
    assert buf.ptr == 3
    # oldest surviving entries are 3..7
    assert set(buf.state[:, 0].tolist()) == {3.0, 4.0, 5.0, 6.0, 7.0}


def test_state_dict_roundtrip():
    buf = ReplayBuffer(state_dim=2, action_dim=1, max_size=100)
    for i in range(7):
        buf.add([i, i], [i], [i + 1, i + 1], float(i), float(i % 2))
    clone = ReplayBuffer(state_dim=2, action_dim=1, max_size=100)
    clone.load_state_dict(buf.state_dict())
    assert clone.size == buf.size and clone.ptr == buf.ptr
    assert np.array_equal(clone.state[:7], buf.state[:7])
    assert np.array_equal(clone.not_done[:7], buf.not_done[:7])
