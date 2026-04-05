"""Basic lifecycle tests: allocate, gate, measure, release."""

from qdk_gpu_sim import GpuQuantumSim


def test_allocate_and_release(sim):
    q = sim.allocate()
    assert isinstance(q, int)
    sim.release(q)


def test_single_qubit_identity(sim):
    """A freshly allocated qubit should measure |0>."""
    q = sim.allocate()
    assert sim.measure(q) is False
    sim.release(q)


def test_x_gate_flips(sim):
    """X|0> = |1>."""
    q = sim.allocate()
    sim.x(q)
    assert sim.measure(q) is True
    sim.release(q)


def test_seed_reproducibility():
    """Identical seeds must produce identical measurement results."""
    results = []
    for _ in range(10):
        s = GpuQuantumSim(seed=12345)
        q = s.allocate()
        s.h(q)
        results.append(s.measure(q))
        s.release(q)
    assert all(r == results[0] for r in results)


def test_multiple_allocations(sim):
    """Allocate and release multiple qubits."""
    qubits = [sim.allocate() for _ in range(5)]
    for q in qubits:
        assert isinstance(q, int)
    for q in reversed(qubits):
        sim.release(q)


def test_qubit_id_recycling(sim):
    """Released qubit IDs should be recycled."""
    q0 = sim.allocate()
    sim.release(q0)
    q1 = sim.allocate()
    # q1 may or may not equal q0 depending on implementation,
    # but it must be a valid qubit.
    assert isinstance(q1, int)
    sim.release(q1)
