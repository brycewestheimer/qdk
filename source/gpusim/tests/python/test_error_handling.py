"""Error handling tests: verify Rust panics surface as Python exceptions."""

import pytest

from qdk_gpu_sim import GpuQuantumSim


def test_invalid_qubit_id():
    sim = GpuQuantumSim(seed=42)
    with pytest.raises(RuntimeError):
        sim.h(9999)


def test_release_invalid_qubit():
    sim = GpuQuantumSim(seed=42)
    with pytest.raises(RuntimeError):
        sim.release(9999)


def test_measure_invalid_qubit():
    sim = GpuQuantumSim(seed=42)
    with pytest.raises(RuntimeError):
        sim.measure(9999)


def test_double_release():
    sim = GpuQuantumSim(seed=42)
    q = sim.allocate()
    sim.release(q)
    with pytest.raises(RuntimeError):
        sim.release(q)


def test_gate_after_release():
    sim = GpuQuantumSim(seed=42)
    q = sim.allocate()
    sim.release(q)
    with pytest.raises(RuntimeError):
        sim.h(q)


def test_mcx_invalid_control():
    sim = GpuQuantumSim(seed=42)
    q = sim.allocate()
    with pytest.raises(RuntimeError):
        sim.mcx([9999], q)
    sim.release(q)


def test_joint_measure_invalid():
    sim = GpuQuantumSim(seed=42)
    with pytest.raises(RuntimeError):
        sim.joint_measure([9999])


def test_joint_probability_invalid():
    sim = GpuQuantumSim(seed=42)
    with pytest.raises(RuntimeError):
        sim.joint_probability([9999])
