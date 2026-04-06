// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#![allow(dead_code)] // Not all benchmarks use all utilities
#![allow(clippy::must_use_candidate)]
#![allow(clippy::cast_precision_loss)]

use rand::{Rng, SeedableRng, rngs::StdRng};

/// Fixed seed for all benchmark circuit generation. Deterministic results across runs.
pub const BENCH_SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;

/// A gate operation for benchmark circuits.
///
/// Simplified subset of Phase 4's `TestGate` -- no `Allocate`/`Release` needed since
/// benchmarks operate on fixed qubit counts allocated during setup.
#[derive(Clone, Debug)]
pub enum BenchGate {
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    T(usize),
    Rx(f64, usize),
    Ry(f64, usize),
    Rz(f64, usize),
    Cx(usize, usize),
    /// Physical SWAP gate (applies the 4×4 SWAP unitary).
    /// NOT a qubit ID relabeling — see `swap_qubit_ids` for that.
    Swap(usize, usize),
}

/// Generate a random Clifford+T circuit.
///
/// Each layer applies a random gate to every qubit (single-qubit) plus
/// random CNOT gates between approximately 30% of adjacent qubit pairs.
/// This produces dense, highly entangled states that defeat sparse simulation.
pub fn random_clifford_t_circuit(num_qubits: usize, depth: usize) -> Vec<BenchGate> {
    let mut rng = StdRng::seed_from_u64(BENCH_SEED);
    let mut gates = Vec::with_capacity(num_qubits * depth * 2);

    for _ in 0..depth {
        // Single-qubit layer: random gate on every qubit
        for q in 0..num_qubits {
            let gate = match rng.gen_range(0..5u32) {
                0 => BenchGate::H(q),
                1 => BenchGate::X(q),
                2 => BenchGate::T(q),
                3 => BenchGate::S(q),
                _ => BenchGate::Z(q),
            };
            gates.push(gate);
        }
        // Two-qubit layer: random CNOTs between ~30% of adjacent pairs
        for q in 0..num_qubits.saturating_sub(1) {
            if rng.gen_bool(0.3) {
                gates.push(BenchGate::Cx(q, q + 1));
            }
        }
    }
    gates
}

/// Generate a random universal circuit with non-Clifford gates.
///
/// Includes H, T, CNOT, `Rx`(irrational), `Ry`(irrational), `Rz`(irrational) to ensure
/// the state cannot be represented exactly at any finite precision. This is the
/// circuit type used for precision characterization -- pure Clifford circuits would
/// show zero precision loss, which is unrepresentative.
pub fn random_universal_circuit(num_qubits: usize, depth: usize, seed: u64) -> Vec<BenchGate> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut gates = Vec::new();

    for _ in 0..depth {
        for q in 0..num_qubits {
            let gate = match rng.gen_range(0..6u32) {
                0 => BenchGate::H(q),
                1 => BenchGate::T(q),
                2 => BenchGate::Rx(rng.r#gen::<f64>() * std::f64::consts::TAU, q),
                3 => BenchGate::Ry(rng.r#gen::<f64>() * std::f64::consts::TAU, q),
                4 => BenchGate::Rz(rng.r#gen::<f64>() * std::f64::consts::TAU, q),
                _ => BenchGate::S(q),
            };
            gates.push(gate);
        }
        // Entangling layer: random CNOTs between ~40% of adjacent pairs
        for q in 0..num_qubits.saturating_sub(1) {
            if rng.gen_bool(0.4) {
                gates.push(BenchGate::Cx(q, q + 1));
            }
        }
    }
    gates
}

/// Generate a QFT (Quantum Fourier Transform) circuit on `num_qubits`.
///
/// The QFT applies, for each qubit j from 0 to N-1:
///   1. H on qubit j
///   2. For each qubit k > j: controlled-`R_k` gate where `R_k` = diag(1, e^(2*pi*i/2^(k-j+1)))
///
/// Followed by a reversal of qubit order via SWAP gates.
///
/// Controlled phase rotations are decomposed into: CNOT + `Rz` sequences.
/// Specifically, controlled-`Rz`(theta) = CNOT(control, target) `Rz`(-theta/2, target)
/// CNOT(control, target) `Rz`(theta/2, target) with an additional `Rz`(theta/2) on
/// the control qubit.
pub fn qft_circuit(num_qubits: usize) -> Vec<BenchGate> {
    let mut gates = Vec::new();

    for j in 0..num_qubits {
        // Hadamard on qubit j
        gates.push(BenchGate::H(j));

        // Controlled phase rotations from higher qubits
        for k in (j + 1)..num_qubits {
            let exponent = k - j + 1;
            let theta = std::f64::consts::TAU / (1u64 << exponent) as f64;

            // Decompose controlled-Rz(theta) with control=k, target=j:
            //   Rz(theta/2, j)
            //   CNOT(k, j)
            //   Rz(-theta/2, j)
            //   CNOT(k, j)
            //   Rz(theta/2, k)      (phase on control)
            gates.push(BenchGate::Rz(theta / 2.0, j));
            gates.push(BenchGate::Cx(k, j));
            gates.push(BenchGate::Rz(-theta / 2.0, j));
            gates.push(BenchGate::Cx(k, j));
            gates.push(BenchGate::Rz(theta / 2.0, k));
        }
    }

    // Bit-reversal via SWAPs
    for i in 0..num_qubits / 2 {
        gates.push(BenchGate::Swap(i, num_qubits - 1 - i));
    }

    gates
}

/// Generate a simplified QPE circuit for chemistry benchmarking.
///
/// This constructs the structural skeleton of a QPE circuit:
/// - `system_qubits` qubits representing the molecular orbital basis
/// - `ancilla_qubits` qubits for phase readout precision
/// - Repeated controlled Trotter layers simulating Hamiltonian evolution
///
/// The Hamiltonian coefficients are synthetic (not physically accurate H2),
/// but the circuit structure is representative: repeated layers of `Rz`, `Rx`
/// rotations and CNOT entangling gates on the system register.
pub fn chemistry_qpe_circuit(
    system_qubits: usize,
    ancilla_qubits: usize,
    trotter_steps: usize,
) -> Vec<BenchGate> {
    let mut gates = Vec::new();

    // Hadamard on all ancilla qubits (phase kickback preparation)
    for a in system_qubits..(system_qubits + ancilla_qubits) {
        gates.push(BenchGate::H(a));
    }

    // Controlled Trotter layers
    for step in 0..trotter_steps {
        let angle = std::f64::consts::PI * (step as f64) / (trotter_steps as f64);

        // Trotter step: alternating CNOT-Rz-CNOT layers on system register
        for q in 0..system_qubits.saturating_sub(1) {
            gates.push(BenchGate::Cx(q, q + 1));
            gates.push(BenchGate::Rz(angle, q + 1));
            gates.push(BenchGate::Cx(q, q + 1));
        }
        // Single-qubit rotations on system qubits
        for q in 0..system_qubits {
            gates.push(BenchGate::Rx(angle * 0.5, q));
        }
    }

    // Inverse QFT on ancilla register (simplified: just Hadamards)
    // A proper inverse QFT would include controlled-phase gates, but for
    // benchmarking the Trotter layer performance, this is sufficient.
    for a in system_qubits..(system_qubits + ancilla_qubits) {
        gates.push(BenchGate::H(a));
    }

    gates
}

/// Execute a `BenchGate` sequence on `GpuQuantumSim`.
///
/// The simulator must already have `num_qubits` qubits allocated (qubit IDs 0..`num_qubits`).
pub fn apply_circuit_gpu(sim: &mut qdk_gpu_sim::GpuQuantumSim, circuit: &[BenchGate]) {
    for gate in circuit {
        match gate {
            BenchGate::H(q) => sim.h(*q),
            BenchGate::X(q) => sim.x(*q),
            BenchGate::Y(q) => sim.y(*q),
            BenchGate::Z(q) => sim.z(*q),
            BenchGate::S(q) => sim.s(*q),
            BenchGate::T(q) => sim.t(*q),
            BenchGate::Rx(theta, q) => sim.rx(*theta, *q),
            BenchGate::Ry(theta, q) => sim.ry(*theta, *q),
            BenchGate::Rz(theta, q) => sim.rz(*theta, *q),
            BenchGate::Cx(c, t) => sim.mcx(&[*c], *t),
            BenchGate::Swap(a, b) => sim.swap(*a, *b),
        }
    }
}

/// Execute a `BenchGate` sequence on `quantum_sparse_sim::QuantumSim`.
///
/// `qubit_ids` maps logical qubit indices (0..`num_qubits`) to the IDs returned
/// by `sim.allocate()`.
pub fn apply_circuit_sparse(
    sim: &mut quantum_sparse_sim::QuantumSim,
    qubit_ids: &[usize],
    circuit: &[BenchGate],
) {
    for gate in circuit {
        match gate {
            BenchGate::H(q) => sim.h(qubit_ids[*q]),
            BenchGate::X(q) => sim.x(qubit_ids[*q]),
            BenchGate::Y(q) => sim.y(qubit_ids[*q]),
            BenchGate::Z(q) => sim.z(qubit_ids[*q]),
            BenchGate::S(q) => sim.s(qubit_ids[*q]),
            BenchGate::T(q) => sim.t(qubit_ids[*q]),
            BenchGate::Rx(theta, q) => sim.rx(*theta, qubit_ids[*q]),
            BenchGate::Ry(theta, q) => sim.ry(*theta, qubit_ids[*q]),
            BenchGate::Rz(theta, q) => sim.rz(*theta, qubit_ids[*q]),
            BenchGate::Cx(c, t) => sim.mcx(&[qubit_ids[*c]], qubit_ids[*t]),
            BenchGate::Swap(a, b) => {
                // Decompose SWAP into 3 CNOTs to match GPU's physical SWAP gate.
                sim.mcx(&[qubit_ids[*a]], qubit_ids[*b]);
                sim.mcx(&[qubit_ids[*b]], qubit_ids[*a]);
                sim.mcx(&[qubit_ids[*a]], qubit_ids[*b]);
            }
        }
    }
}

/// Create a `GpuQuantumSim` with `n` qubits pre-allocated.
/// Returns the simulator. Qubit IDs will be 0..n.
pub fn make_gpu_sim(n: usize) -> qdk_gpu_sim::GpuQuantumSim {
    let mut sim =
        qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU simulator should initialize");
    for _ in 0..n {
        sim.allocate().expect("qubit allocation should succeed");
    }
    sim
}

/// Create a `QuantumSim` with `n` qubits pre-allocated.
/// Returns (simulator, `qubit_ids`).
pub fn make_sparse_sim(n: usize) -> (quantum_sparse_sim::QuantumSim, Vec<usize>) {
    let rng = StdRng::seed_from_u64(42);
    let mut sim = quantum_sparse_sim::QuantumSim::new(Some(rng));
    let ids: Vec<usize> = (0..n).map(|_| sim.allocate()).collect();
    (sim, ids)
}
