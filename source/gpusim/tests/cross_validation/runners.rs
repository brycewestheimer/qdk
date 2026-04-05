use num_bigint::BigUint;
use num_complex::Complex64;
use rustc_hash::{FxHashMap, FxHashSet};

use super::circuit::{TestCircuit, TestGate};

/// Run a `TestCircuit` on `quantum-sparse-sim` and return the state vector.
///
/// Returns `(entries, num_qubits)` where entries are `(basis_index, amplitude)` pairs.
/// The sparse simulator may omit zero-amplitude entries.
pub fn run_on_sparse(circuit: &TestCircuit) -> (Vec<(BigUint, Complex64)>, usize) {
    let mut sim = quantum_sparse_sim::QuantumSim::new(None);
    let mut qubit_ids: Vec<usize> = (0..circuit.num_qubits).map(|_| sim.allocate()).collect();

    for gate in &circuit.gates {
        match gate {
            TestGate::H(q) => sim.h(qubit_ids[*q]),
            TestGate::X(q) => sim.x(qubit_ids[*q]),
            TestGate::Y(q) => sim.y(qubit_ids[*q]),
            TestGate::Z(q) => sim.z(qubit_ids[*q]),
            TestGate::S(q) => sim.s(qubit_ids[*q]),
            TestGate::Sadj(q) => sim.sadj(qubit_ids[*q]),
            TestGate::SX(q) => {
                // `QuantumSim` does not have native SX; decompose as H-S-H
                sim.h(qubit_ids[*q]);
                sim.s(qubit_ids[*q]);
                sim.h(qubit_ids[*q]);
            }
            TestGate::SXadj(q) => {
                // Decompose as H-Sadj-H
                sim.h(qubit_ids[*q]);
                sim.sadj(qubit_ids[*q]);
                sim.h(qubit_ids[*q]);
            }
            TestGate::T(q) => sim.t(qubit_ids[*q]),
            TestGate::Tadj(q) => sim.tadj(qubit_ids[*q]),
            TestGate::Rx(theta, q) => sim.rx(*theta, qubit_ids[*q]),
            TestGate::Ry(theta, q) => sim.ry(*theta, qubit_ids[*q]),
            TestGate::Rz(theta, q) => sim.rz(*theta, qubit_ids[*q]),
            TestGate::Cx(c, t) => sim.mcx(&[qubit_ids[*c]], qubit_ids[*t]),
            TestGate::Cy(c, t) => sim.mcy(&[qubit_ids[*c]], qubit_ids[*t]),
            TestGate::Cz(c, t) => sim.mcz(&[qubit_ids[*c]], qubit_ids[*t]),
            TestGate::Swap(a, b) => sim.swap_qubit_ids(qubit_ids[*a], qubit_ids[*b]),
            TestGate::Mcx(cs, t) => {
                let ctrl_ids: Vec<usize> = cs.iter().map(|c| qubit_ids[*c]).collect();
                sim.mcx(&ctrl_ids, qubit_ids[*t]);
            }
            TestGate::Mcy(cs, t) => {
                let ctrl_ids: Vec<usize> = cs.iter().map(|c| qubit_ids[*c]).collect();
                sim.mcy(&ctrl_ids, qubit_ids[*t]);
            }
            TestGate::Mcz(cs, t) => {
                let ctrl_ids: Vec<usize> = cs.iter().map(|c| qubit_ids[*c]).collect();
                sim.mcz(&ctrl_ids, qubit_ids[*t]);
            }
            TestGate::Allocate => {
                qubit_ids.push(sim.allocate());
            }
            TestGate::Release(q) => sim.release(qubit_ids[*q]),
        }
    }

    sim.get_state()
}

/// Run a `TestCircuit` on `GpuQuantumSim` and return the state vector.
///
/// Returns `(entries, num_qubits)` in the same format as `run_on_sparse`.
/// The GPU simulator returns ALL basis states (dense), but `get_state()` filters
/// near-zero amplitudes, so the format is compatible.
pub fn run_on_gpu(circuit: &TestCircuit) -> (Vec<(BigUint, Complex64)>, usize) {
    use qdk_gpu_sim::GpuQuantumSim;

    let mut sim = GpuQuantumSim::new(None).expect("GPU simulator should initialize");
    let mut qubit_ids: Vec<usize> = (0..circuit.num_qubits)
        .map(|_| sim.allocate().expect("allocation should succeed"))
        .collect();

    for gate in &circuit.gates {
        match gate {
            TestGate::H(q) => sim.h(qubit_ids[*q]),
            TestGate::X(q) => sim.x(qubit_ids[*q]),
            TestGate::Y(q) => sim.y(qubit_ids[*q]),
            TestGate::Z(q) => sim.z(qubit_ids[*q]),
            TestGate::S(q) => sim.s(qubit_ids[*q]),
            TestGate::Sadj(q) => sim.sadj(qubit_ids[*q]),
            TestGate::SX(q) => sim.sx(qubit_ids[*q]),
            TestGate::SXadj(q) => sim.sxadj(qubit_ids[*q]),
            TestGate::T(q) => sim.t(qubit_ids[*q]),
            TestGate::Tadj(q) => sim.tadj(qubit_ids[*q]),
            TestGate::Rx(theta, q) => sim.rx(*theta, qubit_ids[*q]),
            TestGate::Ry(theta, q) => sim.ry(*theta, qubit_ids[*q]),
            TestGate::Rz(theta, q) => sim.rz(*theta, qubit_ids[*q]),
            TestGate::Cx(c, t) => sim.mcx(&[qubit_ids[*c]], qubit_ids[*t]),
            TestGate::Cy(c, t) => sim.mcy(&[qubit_ids[*c]], qubit_ids[*t]),
            TestGate::Cz(c, t) => sim.mcz(&[qubit_ids[*c]], qubit_ids[*t]),
            TestGate::Swap(a, b) => sim.swap_qubit_ids(qubit_ids[*a], qubit_ids[*b]),
            TestGate::Mcx(cs, t) => {
                let ctrl_ids: Vec<usize> = cs.iter().map(|c| qubit_ids[*c]).collect();
                sim.mcx(&ctrl_ids, qubit_ids[*t]);
            }
            TestGate::Mcy(cs, t) => {
                let ctrl_ids: Vec<usize> = cs.iter().map(|c| qubit_ids[*c]).collect();
                sim.mcy(&ctrl_ids, qubit_ids[*t]);
            }
            TestGate::Mcz(cs, t) => {
                let ctrl_ids: Vec<usize> = cs.iter().map(|c| qubit_ids[*c]).collect();
                sim.mcz(&ctrl_ids, qubit_ids[*t]);
            }
            TestGate::Allocate => {
                qubit_ids.push(sim.allocate().expect("allocation should succeed"));
            }
            TestGate::Release(q) => sim.release(qubit_ids[*q]),
        }
    }

    sim.get_state().expect("GPU get_state should succeed")
}

/// Align global phase of a state vector by normalizing the first nonzero
/// amplitude to have zero imaginary part and positive real part.
///
/// Two unitary-equivalent states may differ by a global phase factor e^(i*phi).
/// This function removes that ambiguity by finding the first nonzero amplitude
/// and rotating the entire state so that amplitude is real and positive.
pub fn phase_normalize(state: &mut [(BigUint, Complex64)]) {
    // Find the first amplitude with magnitude above threshold
    if let Some((_, first_nonzero)) = state.iter().find(|(_, amp)| amp.norm() > 1e-12) {
        // The phase of first_nonzero is: first_nonzero / |first_nonzero|
        // To remove it, multiply everything by the conjugate of this phase
        let phase = first_nonzero / first_nonzero.norm();
        let conjugate = phase.conj();
        for (_, amp) in state.iter_mut() {
            *amp *= conjugate;
        }
    }
}

/// Compare two state vectors component-wise.
///
/// The sparse simulator may omit zero-amplitude entries; the GPU simulator
/// may include them (with very small magnitudes). This function handles the
/// asymmetry by treating missing entries as zero.
///
/// Returns `(max_component_error, passed)` where `passed` is true when every
/// component difference is below `tolerance`.
pub fn compare_states(
    sparse: &[(BigUint, Complex64)],
    gpu: &[(BigUint, Complex64)],
    tolerance: f64,
) -> (f64, bool) {
    let sparse_map: FxHashMap<&BigUint, &Complex64> =
        sparse.iter().map(|(idx, amp)| (idx, amp)).collect();
    let gpu_map: FxHashMap<&BigUint, &Complex64> =
        gpu.iter().map(|(idx, amp)| (idx, amp)).collect();

    // Union of all basis indices from both states
    let mut all_indices: FxHashSet<&BigUint> = FxHashSet::default();
    all_indices.extend(sparse_map.keys());
    all_indices.extend(gpu_map.keys());

    let zero = Complex64::new(0.0, 0.0);
    let mut max_error: f64 = 0.0;

    for idx in &all_indices {
        let s_amp = sparse_map.get(idx).copied().unwrap_or(&zero);
        let g_amp = gpu_map.get(idx).copied().unwrap_or(&zero);
        let diff = (s_amp - g_amp).norm();
        max_error = max_error.max(diff);
    }

    (max_error, max_error < tolerance)
}

/// Full comparison pipeline: phase-normalize both states, then compare.
///
/// Use this for unitary-only circuits (no measurement). For circuits with
/// measurements, the states are already in a definite basis and global phase
/// is not an issue -- use `compare_states` directly.
pub fn compare_states_phase_normalized(
    sparse: &mut [(BigUint, Complex64)],
    gpu: &mut [(BigUint, Complex64)],
    tolerance: f64,
) -> (f64, bool) {
    phase_normalize(sparse);
    phase_normalize(gpu);
    compare_states(sparse, gpu, tolerance)
}
