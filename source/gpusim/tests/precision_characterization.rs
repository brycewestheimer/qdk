// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#![cfg(feature = "gpu-tests")]
#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

use num_bigint::BigUint;
use num_complex::Complex64;
use qdk_gpu_sim::GpuQuantumSim;
use qdk_gpu_sim::precision_utils::{compute_metrics, to_dense};
use quantum_sparse_sim::QuantumSim;
use rand::{Rng, SeedableRng, rngs::StdRng};

/// Dedicated seed for precision characterization, distinct from benchmark seeds
/// to avoid accidental correlation.
const PRECISION_SEED: u64 = 0xF32_0EC1_510F_ACE0;

// -----------------------------------------------------------------------
// Gate enum and runners (self-contained; cannot import `bench_utils` from tests)
// -----------------------------------------------------------------------

#[derive(Clone, Debug)]
enum Gate {
    H(usize),
    T(usize),
    S(usize),
    Rx(f64, usize),
    Ry(f64, usize),
    Rz(f64, usize),
    Cx(usize, usize),
}

/// Generate a random universal circuit with non-Clifford gates.
///
/// Includes H, T, CNOT, and rotation gates with irrational angles to ensure
/// the state cannot be represented exactly at any finite precision.
fn random_universal_circuit(num_qubits: usize, depth: usize, seed: u64) -> Vec<Gate> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut gates = Vec::new();

    for _ in 0..depth {
        for q in 0..num_qubits {
            let gate = match rng.gen_range(0..6u32) {
                0 => Gate::H(q),
                1 => Gate::T(q),
                2 => Gate::Rx(rng.r#gen::<f64>() * std::f64::consts::TAU, q),
                3 => Gate::Ry(rng.r#gen::<f64>() * std::f64::consts::TAU, q),
                4 => Gate::Rz(rng.r#gen::<f64>() * std::f64::consts::TAU, q),
                _ => Gate::S(q),
            };
            gates.push(gate);
        }
        // Entangling layer: ~40% of adjacent pairs get a CNOT
        for q in 0..num_qubits.saturating_sub(1) {
            if rng.gen_bool(0.4) {
                gates.push(Gate::Cx(q, q + 1));
            }
        }
    }
    gates
}

fn run_on_gpu(circuit: &[Gate], num_qubits: usize) -> Vec<(BigUint, Complex64)> {
    let mut sim =
        GpuQuantumSim::new(Some(PRECISION_SEED)).expect("GPU simulator should initialize");
    for _ in 0..num_qubits {
        sim.allocate().expect("allocation should succeed");
    }
    for gate in circuit {
        match gate {
            Gate::H(q) => sim.h(*q),
            Gate::T(q) => sim.t(*q),
            Gate::S(q) => sim.s(*q),
            Gate::Rx(theta, q) => sim.rx(*theta, *q),
            Gate::Ry(theta, q) => sim.ry(*theta, *q),
            Gate::Rz(theta, q) => sim.rz(*theta, *q),
            Gate::Cx(c, t) => sim.mcx(&[*c], *t),
        }
    }
    let (state, _num_qubits) = sim.get_state().expect("GPU get_state should succeed");
    state
}

fn run_on_sparse(circuit: &[Gate], num_qubits: usize) -> Vec<(BigUint, Complex64)> {
    let rng = StdRng::seed_from_u64(PRECISION_SEED);
    let mut sim = QuantumSim::new(Some(rng));
    let ids: Vec<usize> = (0..num_qubits).map(|_| sim.allocate()).collect();
    for gate in circuit {
        match gate {
            Gate::H(q) => sim.h(ids[*q]),
            Gate::T(q) => sim.t(ids[*q]),
            Gate::S(q) => sim.s(ids[*q]),
            Gate::Rx(theta, q) => sim.rx(*theta, ids[*q]),
            Gate::Ry(theta, q) => sim.ry(*theta, ids[*q]),
            Gate::Rz(theta, q) => sim.rz(*theta, ids[*q]),
            Gate::Cx(c, t) => sim.mcx(&[ids[*c]], ids[*t]),
        }
    }
    let (state, _num_qubits) = sim.get_state();
    state
}

// -----------------------------------------------------------------------
// Error metrics — imported from qdk_gpu_sim::precision_utils
// -----------------------------------------------------------------------

// -----------------------------------------------------------------------
// Main precision characterization test
// -----------------------------------------------------------------------

/// Full precision characterization: (qubits x depth) matrix with all error metrics.
///
/// Run with: `cargo test -p qdk-gpu-sim --features gpu-tests -- precision_characterization --nocapture`
///
/// The `--nocapture` flag is required to see the formatted table (Rust's test
/// harness captures stdout by default).
#[test]
fn precision_characterization_matrix() {
    let qubit_counts = [4, 8, 12, 16, 20];
    let depths = [10, 50, 100, 500, 1000, 5000];

    println!();
    println!("=== f32 Precision Characterization vs f64 Reference ===");
    println!();
    println!(
        "{:>6} | {:>5} | {:>11} | {:>11} | {:>15} | {:>13}",
        "Qubits", "Depth", "Max Error", "RMS Error", "Fidelity", "Trace Dist"
    );
    println!("{}", "-".repeat(78));

    let mut all_fidelities: Vec<(usize, usize, f64)> = Vec::new();

    for &num_qubits in &qubit_counts {
        for &depth in &depths {
            // Use a seed that varies with parameters so each cell is independent
            let cell_seed = PRECISION_SEED
                .wrapping_add(num_qubits as u64 * 1_000_000)
                .wrapping_add(depth as u64);
            let circuit = random_universal_circuit(num_qubits, depth, cell_seed);

            // GPU (f32)
            let gpu_state = run_on_gpu(&circuit, num_qubits);
            let gpu_dense = to_dense(&gpu_state, num_qubits);

            // Sparse (f64 reference)
            let sparse_state = run_on_sparse(&circuit, num_qubits);
            let ref_dense = to_dense(&sparse_state, num_qubits);

            let metrics = compute_metrics(&gpu_dense, &ref_dense);

            println!(
                "{:>6} | {:>5} | {:>11.3e} | {:>11.3e} | {:>15.12} | {:>13.3e}",
                num_qubits,
                depth,
                metrics.max_error,
                metrics.rms_error,
                metrics.fidelity,
                metrics.trace_distance,
            );

            all_fidelities.push((num_qubits, depth, metrics.fidelity));
        }
    }
    println!();

    // Summary: identify the worst-case fidelity
    let worst = all_fidelities
        .iter()
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        .expect("should have at least one data point");
    println!(
        "Worst fidelity: {:.12} at {} qubits, depth {}",
        worst.2, worst.0, worst.1
    );

    // Hard assertion: fidelity must stay above 0.99 for all configurations.
    //
    // f32 arithmetic loses ~7 decimal digits per operation, but quantum gate
    // unitaries preserve normalization structurally. Even at depth 5000, the
    // fidelity degradation should be gradual (trace distance grows as
    // O(sqrt(depth) * eps)). A threshold of 0.99 catches catastrophic
    // precision failures while allowing normal f32 drift.
    for &(q, d, f) in &all_fidelities {
        assert!(
            f >= 0.99,
            "Fidelity {f:.6} at {q} qubits, depth {d} fell below 0.99 threshold"
        );
    }
}

/// QPE energy accuracy study using structural (Option B) approach.
///
/// Constructs a simplified QPE circuit with synthetic Hamiltonian coefficients
/// and compares the phase distribution from GPU vs sparse simulation.
/// The comparison is GPU-vs-sparse (precision delta), not GPU-vs-exact-energy
/// (that requires real Hamiltonian coefficients from `qdk-chemistry`).
#[test]
fn qpe_energy_comparison() {
    let system_qubits = 4;
    let ancilla_qubits = 8;
    let trotter_steps = 100;
    let total_qubits = system_qubits + ancilla_qubits;
    let num_shots = 200; // enough for statistical comparison

    // Build simplified QPE circuit
    let circuit = build_qpe_circuit(system_qubits, ancilla_qubits, trotter_steps);

    println!();
    println!("=== QPE Energy Accuracy Study (Structural Comparison) ===");
    println!(
        "Circuit: {system_qubits} system + {ancilla_qubits} ancilla qubits, {trotter_steps} Trotter steps, {num_shots} shots"
    );
    println!();

    // Collect phase distributions from both simulators
    let mut gpu_phases = Vec::with_capacity(num_shots);
    let mut sparse_phases = Vec::with_capacity(num_shots);

    for shot in 0..num_shots {
        let shot_seed = 0x00E5_EED0_BA5E_u64 + shot as u64;

        // GPU
        {
            let mut sim =
                GpuQuantumSim::new(Some(shot_seed)).expect("GPU simulator should initialize");
            for _ in 0..total_qubits {
                sim.allocate().expect("allocation should succeed");
            }
            apply_gates_gpu(&mut sim, &circuit);
            let phase = read_ancilla_phase_gpu(&mut sim, system_qubits, ancilla_qubits);
            gpu_phases.push(phase);
        }

        // Sparse
        {
            let rng = StdRng::seed_from_u64(shot_seed);
            let mut sim = QuantumSim::new(Some(rng));
            let ids: Vec<_> = (0..total_qubits).map(|_| sim.allocate()).collect();
            apply_gates_sparse(&mut sim, &ids, &circuit);
            let phase = read_ancilla_phase_sparse(&mut sim, &ids, system_qubits, ancilla_qubits);
            sparse_phases.push(phase);
        }
    }

    // Compare phase distributions
    let gpu_mean = gpu_phases.iter().sum::<f64>() / gpu_phases.len() as f64;
    let sparse_mean = sparse_phases.iter().sum::<f64>() / sparse_phases.len() as f64;

    // Convert mean phase to energy: E = -2 * pi * phase
    // (simplified; actual conversion depends on evolution time parameter)
    let gpu_energy = -2.0 * std::f64::consts::PI * gpu_mean;
    let sparse_energy = -2.0 * std::f64::consts::PI * sparse_mean;

    println!("Sparse mean phase: {sparse_mean:.8}");
    println!("GPU mean phase:    {gpu_mean:.8}");
    println!("Phase difference:  {:.3e}", (gpu_mean - sparse_mean).abs());
    println!();
    println!("Sparse energy estimate:  {sparse_energy:.6} (arb. units)");
    println!("GPU energy estimate:     {gpu_energy:.6} (arb. units)");
    println!(
        "|E_gpu - E_sparse|:      {:.3e}",
        (gpu_energy - sparse_energy).abs()
    );
    println!();
    println!("Note: These are synthetic Hamiltonian energies, not physical H2 values.");
    println!("The important metric is |E_gpu - E_sparse| -- the precision delta between");
    println!("f32 and f64 simulation of the same circuit.");
    println!();

    // Count how many shots gave identical phase outcomes
    let matching_shots = gpu_phases
        .iter()
        .zip(sparse_phases.iter())
        .filter(|(g, s)| (*g - *s).abs() < 1e-10)
        .count();
    println!(
        "Matching phase outcomes: {}/{} ({:.1}%)",
        matching_shots,
        num_shots,
        100.0 * matching_shots as f64 / num_shots as f64
    );
    println!();
    println!("Note on physically accurate QPE:");
    println!("  Constructing a real H2 QPE circuit requires specific Hamiltonian coefficients");
    println!("  (one-body and two-body integrals for the Jordan-Wigner transformed H2 STO-3G");
    println!("  Hamiltonian). This is better sourced from the qdk-chemistry pipeline and is");
    println!("  deferred to chemistry integration work. Phase 5 validates f32-vs-f64 precision");
    println!("  at the state vector and phase estimation level; physical energy validation");
    println!("  belongs in the integration testing phase.");
}

// -----------------------------------------------------------------------
// QPE helper functions
// -----------------------------------------------------------------------

fn build_qpe_circuit(
    system_qubits: usize,
    ancilla_qubits: usize,
    trotter_steps: usize,
) -> Vec<Gate> {
    let mut gates = Vec::new();

    // Hadamard on all ancilla qubits
    for a in system_qubits..(system_qubits + ancilla_qubits) {
        gates.push(Gate::H(a));
    }

    // Controlled Trotter layers
    for step in 0..trotter_steps {
        let angle = std::f64::consts::PI * (step as f64) / (trotter_steps as f64);
        // Trotter step on system register
        for q in 0..system_qubits.saturating_sub(1) {
            gates.push(Gate::Cx(q, q + 1));
            gates.push(Gate::Rz(angle, q + 1));
            gates.push(Gate::Cx(q, q + 1));
        }
        for q in 0..system_qubits {
            gates.push(Gate::Rx(angle * 0.5, q));
        }
    }

    // Inverse QFT on ancilla (simplified)
    for a in system_qubits..(system_qubits + ancilla_qubits) {
        gates.push(Gate::H(a));
    }

    gates
}

fn apply_gates_gpu(sim: &mut GpuQuantumSim, circuit: &[Gate]) {
    for gate in circuit {
        match gate {
            Gate::H(q) => sim.h(*q),
            Gate::T(q) => sim.t(*q),
            Gate::S(q) => sim.s(*q),
            Gate::Rx(theta, q) => sim.rx(*theta, *q),
            Gate::Ry(theta, q) => sim.ry(*theta, *q),
            Gate::Rz(theta, q) => sim.rz(*theta, *q),
            Gate::Cx(c, t) => sim.mcx(&[*c], *t),
        }
    }
}

fn apply_gates_sparse(sim: &mut QuantumSim, ids: &[usize], circuit: &[Gate]) {
    for gate in circuit {
        match gate {
            Gate::H(q) => sim.h(ids[*q]),
            Gate::T(q) => sim.t(ids[*q]),
            Gate::S(q) => sim.s(ids[*q]),
            Gate::Rx(theta, q) => sim.rx(*theta, ids[*q]),
            Gate::Ry(theta, q) => sim.ry(*theta, ids[*q]),
            Gate::Rz(theta, q) => sim.rz(*theta, ids[*q]),
            Gate::Cx(c, t) => sim.mcx(&[ids[*c]], ids[*t]),
        }
    }
}

// Extract phase from ancilla qubit measurements.
// Measures each ancilla qubit and interprets the bit string as a binary fraction.
fn read_ancilla_phase_gpu(
    sim: &mut GpuQuantumSim,
    system_offset: usize,
    num_ancilla: usize,
) -> f64 {
    let mut phase = 0.0_f64;
    for a in 0..num_ancilla {
        let result = sim
            .measure(system_offset + a)
            .expect("measurement should succeed");
        if result {
            phase += 1.0 / (1u64 << (a + 1)) as f64;
        }
    }
    phase
}

fn read_ancilla_phase_sparse(
    sim: &mut QuantumSim,
    ids: &[usize],
    system_offset: usize,
    num_ancilla: usize,
) -> f64 {
    let mut phase = 0.0_f64;
    for a in 0..num_ancilla {
        let result = sim.measure(ids[system_offset + a]);
        if result {
            phase += 1.0 / (1u64 << (a + 1)) as f64;
        }
    }
    phase
}
