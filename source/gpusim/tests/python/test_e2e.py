"""End-to-end integration tests exercising the full Python -> Rust -> GPU stack."""

import math

from qdk_gpu_sim import GpuQuantumSim

TOLERANCE = 1e-5


def test_bell_state_end_to_end():
    """Complete workflow: create sim, build Bell state, inspect, clean up."""
    sim = GpuQuantumSim(seed=42)

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


def test_many_shot_bell_correlation():
    """1000 shots: Bell pair outcomes must always be correlated."""
    results = [0, 0]
    for seed in range(1000):
        sim = GpuQuantumSim(seed=seed)
        q0 = sim.allocate()
        q1 = sim.allocate()
        sim.h(q0)
        sim.mcx([q0], q1)
        r0 = sim.measure(q0)
        r1 = sim.measure(q1)
        assert r0 == r1, f"Correlation violated at seed={seed}"
        results[int(r0)] += 1
        sim.release(q0)
        sim.release(q1)

    assert 400 < results[0] < 600, f"Skewed distribution: {results}"
    assert 400 < results[1] < 600, f"Skewed distribution: {results}"


def test_ghz_state():
    """Create a 4-qubit GHZ state and verify entanglement."""
    sim = GpuQuantumSim(seed=42)
    qubits = [sim.allocate() for _ in range(4)]
    sim.h(qubits[0])
    for i in range(1, 4):
        sim.mcx([qubits[0]], qubits[i])

    state = sim.get_state()
    assert len(state) == 2  # |0000> and |1111>
    amps = {int(idx): amp for idx, amp in state}
    inv_sqrt2 = 1 / math.sqrt(2)
    for amp in amps.values():
        assert abs(abs(amp) - inv_sqrt2) < TOLERANCE

    for q in reversed(qubits):
        sim.release(q)


def test_dump_returns_string():
    sim = GpuQuantumSim(seed=42)
    q = sim.allocate()
    sim.h(q)
    dump_str = sim.dump()
    assert isinstance(dump_str, str)
    assert len(dump_str) > 0
    sim.release(q)


def test_max_qubits():
    sim = GpuQuantumSim(seed=42)
    max_q = sim.max_qubits()
    assert isinstance(max_q, int)
    assert max_q > 0


def test_swap_qubit_ids():
    sim = GpuQuantumSim(seed=42)
    q0 = sim.allocate()
    q1 = sim.allocate()
    sim.x(q0)
    sim.swap_qubit_ids(q0, q1)
    assert sim.measure(q1) is True
    assert sim.measure(q0) is False
    sim.release(q0)
    sim.release(q1)


def test_set_rng_seed():
    """Changing the seed mid-simulation should be allowed."""
    sim = GpuQuantumSim(seed=42)
    q = sim.allocate()
    sim.h(q)
    sim.set_rng_seed(12345)
    result = sim.measure(q)
    assert isinstance(result, bool)
    sim.release(q)
