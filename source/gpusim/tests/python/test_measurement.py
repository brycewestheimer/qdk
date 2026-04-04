"""Measurement statistics and entanglement correlation tests."""

from qdk_gpu_sim import GpuQuantumSim


def test_bell_state_correlation():
    """Measuring one qubit of a Bell pair determines the other."""
    for trial in range(100):
        sim = GpuQuantumSim(seed=trial)
        q0 = sim.allocate()
        q1 = sim.allocate()
        sim.h(q0)
        sim.mcx([q0], q1)
        r0 = sim.measure(q0)
        r1 = sim.measure(q1)
        assert r0 == r1, f"Bell state correlation violated on trial {trial}"
        sim.release(q0)
        sim.release(q1)


def test_hadamard_statistics():
    """H|0> should give roughly 50/50 measurement outcomes over many trials."""
    counts = [0, 0]
    for seed in range(1000):
        sim = GpuQuantumSim(seed=seed)
        q = sim.allocate()
        sim.h(q)
        result = sim.measure(q)
        counts[int(result)] += 1
        sim.release(q)
    # 5-sigma tolerance: expect 400-600 for each outcome
    assert 350 < counts[0] < 650, f"Unexpected distribution: {counts}"
    assert 350 < counts[1] < 650, f"Unexpected distribution: {counts}"


def test_x_always_measures_one():
    """X|0> = |1>, so measurement should always return True."""
    for seed in range(50):
        sim = GpuQuantumSim(seed=seed)
        q = sim.allocate()
        sim.x(q)
        assert sim.measure(q) is True
        sim.release(q)


def test_joint_probability():
    """P(|00>) for a Bell state should be approximately 0.5."""
    sim = GpuQuantumSim(seed=42)
    q0 = sim.allocate()
    q1 = sim.allocate()
    sim.h(q0)
    sim.mcx([q0], q1)
    prob = sim.joint_probability([q0, q1])
    assert abs(prob - 0.5) < 0.01
    sim.release(q0)
    sim.release(q1)


def test_joint_measure():
    """Joint measurement returns a boolean."""
    sim = GpuQuantumSim(seed=42)
    q0 = sim.allocate()
    q1 = sim.allocate()
    sim.h(q0)
    sim.mcx([q0], q1)
    result = sim.joint_measure([q0, q1])
    assert isinstance(result, bool)
    sim.release(q0)
    sim.release(q1)


def test_qubit_is_zero():
    """Freshly allocated qubit should be in |0>."""
    sim = GpuQuantumSim(seed=42)
    q = sim.allocate()
    assert sim.qubit_is_zero(q) is True
    sim.x(q)
    assert sim.qubit_is_zero(q) is False
    sim.release(q)
