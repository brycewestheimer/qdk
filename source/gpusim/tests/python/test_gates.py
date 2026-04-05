"""Gate correctness tests: verify each gate produces the expected state."""

import cmath
import math

from qdk_gpu_sim import GpuQuantumSim

TOLERANCE = 1e-5


def approx_eq(a, b, tol=TOLERANCE):
    return abs(a - b) < tol


def get_amps(sim):
    """Return state as a dict: {basis_state_int: complex_amplitude}."""
    return {int(idx): amp for idx, amp in sim.get_state()}


def test_hadamard_state(sim):
    """H|0> = (|0> + |1>) / sqrt(2)."""
    q = sim.allocate()
    sim.h(q)
    amps = get_amps(sim)
    inv_sqrt2 = 1 / math.sqrt(2)
    assert approx_eq(abs(amps[0]), inv_sqrt2)
    assert approx_eq(abs(amps[1]), inv_sqrt2)
    sim.release(q)


def test_x_gate_state(sim):
    """X|0> = |1>."""
    q = sim.allocate()
    sim.x(q)
    amps = get_amps(sim)
    assert 0 not in amps or approx_eq(abs(amps[0]), 0)
    assert approx_eq(abs(amps[1]), 1.0)
    sim.release(q)


def test_y_gate_state(sim):
    """Y|0> = i|1>."""
    q = sim.allocate()
    sim.y(q)
    amps = get_amps(sim)
    assert approx_eq(amps[1].imag, 1.0)
    assert approx_eq(amps[1].real, 0.0)
    sim.release(q)


def test_z_gate_on_plus(sim):
    """Z|+> = |->: the |1> amplitude becomes negative."""
    q = sim.allocate()
    sim.h(q)
    sim.z(q)
    amps = get_amps(sim)
    inv_sqrt2 = 1 / math.sqrt(2)
    assert approx_eq(amps[0].real, inv_sqrt2)
    assert approx_eq(amps[1].real, -inv_sqrt2)
    sim.release(q)


def test_s_gate(sim):
    """S|+> = (|0> + i|1>) / sqrt(2)."""
    q = sim.allocate()
    sim.h(q)
    sim.s(q)
    amps = get_amps(sim)
    assert approx_eq(amps[1].imag, 1 / math.sqrt(2))
    assert approx_eq(amps[1].real, 0.0)
    sim.release(q)


def test_sadj_undoes_s(sim):
    """S followed by S^dag should return to |+>."""
    q = sim.allocate()
    sim.h(q)
    sim.s(q)
    sim.sadj(q)
    amps = get_amps(sim)
    inv_sqrt2 = 1 / math.sqrt(2)
    assert approx_eq(amps[0].real, inv_sqrt2)
    assert approx_eq(amps[1].real, inv_sqrt2)
    sim.release(q)


def test_t_gate(sim):
    """T|+> = (|0> + exp(i*pi/4)|1>) / sqrt(2)."""
    q = sim.allocate()
    sim.h(q)
    sim.t(q)
    amps = get_amps(sim)
    expected_phase = cmath.exp(1j * math.pi / 4)
    expected_amp = expected_phase / math.sqrt(2)
    assert approx_eq(amps[1].real, expected_amp.real)
    assert approx_eq(amps[1].imag, expected_amp.imag)
    sim.release(q)


def test_rx_pi(sim):
    """RX(pi)|0> = -i|1>."""
    q = sim.allocate()
    sim.rx(math.pi, q)
    amps = get_amps(sim)
    assert approx_eq(abs(amps.get(1, 0)), 1.0)
    sim.release(q)


def test_ry_pi(sim):
    """RY(pi)|0> = |1>."""
    q = sim.allocate()
    sim.ry(math.pi, q)
    amps = get_amps(sim)
    assert approx_eq(abs(amps.get(1, 0)), 1.0)
    sim.release(q)


def test_rz_rotation(sim):
    """RZ on |+> should change relative phase without affecting magnitudes."""
    q = sim.allocate()
    sim.h(q)
    sim.rz(math.pi / 2, q)
    amps = get_amps(sim)
    inv_sqrt2 = 1 / math.sqrt(2)
    assert approx_eq(abs(amps[0]), inv_sqrt2)
    assert approx_eq(abs(amps[1]), inv_sqrt2)
    sim.release(q)


def test_all_single_qubit_gates_smoke(sim):
    """Smoke test: every single-qubit gate runs without error."""
    q = sim.allocate()
    for gate in [sim.h, sim.x, sim.y, sim.z, sim.s, sim.sadj,
                 sim.t, sim.tadj, sim.sx, sim.sxadj]:
        gate(q)
    for rotation in [sim.rx, sim.ry, sim.rz]:
        rotation(0.5, q)
    sim.release(q)


def test_bell_state_via_mcx(sim):
    """H + CNOT produces a Bell state: (|00> + |11>) / sqrt(2)."""
    q0 = sim.allocate()
    q1 = sim.allocate()
    sim.h(q0)
    sim.mcx([q0], q1)
    amps = get_amps(sim)
    inv_sqrt2 = 1 / math.sqrt(2)
    assert approx_eq(abs(amps.get(0, 0)), inv_sqrt2)
    assert approx_eq(abs(amps.get(3, 0)), inv_sqrt2)
    sim.release(q0)
    sim.release(q1)


def test_toffoli_via_mcx(sim):
    """MCX with two controls: only flips target when both controls are |1>."""
    q0 = sim.allocate()
    q1 = sim.allocate()
    q2 = sim.allocate()
    sim.x(q0)
    sim.x(q1)
    sim.mcx([q0, q1], q2)
    amps = get_amps(sim)
    # |111> = index 7
    assert approx_eq(abs(amps.get(7, 0)), 1.0)
    sim.release(q0)
    sim.release(q1)
    sim.release(q2)


def test_multi_controlled_gates_smoke(sim):
    """Smoke test: every multi-controlled gate runs without error."""
    q0 = sim.allocate()
    q1 = sim.allocate()
    for gate in [sim.mcx, sim.mcy, sim.mcz, sim.mch, sim.mcs,
                 sim.mcsadj, sim.mct, sim.mctadj]:
        gate([q0], q1)
    sim.mcrz([q0], 0.5, q1)
    sim.mcphase([q0], math.cos(0.5), math.sin(0.5), q1)
    sim.release(q0)
    sim.release(q1)
