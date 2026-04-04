"""State vector format and correctness tests."""

import math

from qdk_gpu_sim import GpuQuantumSim

TOLERANCE = 1e-5


def test_get_state_format(sim):
    """Single qubit in |0>: one entry with correct types."""
    q = sim.allocate()
    state = sim.get_state()
    assert len(state) == 1
    idx, amp = state[0]
    assert isinstance(idx, int)
    assert isinstance(amp, complex)
    assert idx == 0
    assert abs(amp - 1.0) < TOLERANCE
    sim.release(q)


def test_get_state_filters_near_zero(sim):
    """get_state should omit amplitudes that are effectively zero."""
    q = sim.allocate()
    sim.x(q)
    state = sim.get_state()
    assert len(state) == 1
    idx, amp = state[0]
    assert idx == 1
    assert abs(abs(amp) - 1.0) < TOLERANCE
    sim.release(q)


def test_get_state_bell_pair(sim):
    """Bell state: two non-zero entries at |00> and |11>."""
    q0 = sim.allocate()
    q1 = sim.allocate()
    sim.h(q0)
    sim.mcx([q0], q1)
    state = sim.get_state()
    assert len(state) == 2
    amps = {int(idx): amp for idx, amp in state}
    inv_sqrt2 = 1 / math.sqrt(2)
    for amp in amps.values():
        assert abs(abs(amp) - inv_sqrt2) < TOLERANCE
    sim.release(q0)
    sim.release(q1)


def test_get_state_returns_python_types(sim):
    """Verify types are native Python, not wrapped Rust objects."""
    q = sim.allocate()
    state = sim.get_state()
    idx, amp = state[0]
    assert type(idx).__name__ == "int"
    assert type(amp).__name__ == "complex"
    sim.release(q)


def test_get_state_normalization(sim):
    """Sum of |amplitude|^2 should be approximately 1.0."""
    q0 = sim.allocate()
    q1 = sim.allocate()
    sim.h(q0)
    sim.h(q1)
    sim.mcx([q0], q1)
    state = sim.get_state()
    total_prob = sum(abs(amp) ** 2 for _, amp in state)
    assert abs(total_prob - 1.0) < 1e-4
    sim.release(q0)
    sim.release(q1)
