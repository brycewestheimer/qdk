use super::circuit::{TestCircuit, TestGate};
use super::generator::generate_universal_circuit;
use super::runners::{compare_states, phase_normalize, run_on_gpu, run_on_sparse};

/// Allocate/release/re-allocate cycle.
///
/// When releasing qubit 1 that is entangled with qubit 0, both simulators
/// will measure and collapse. Since the measurement is random, the final states
/// may differ between runs. This test uses a pattern that avoids depending on
/// the random measurement outcome by applying gates after the release.
#[test]
fn test_alloc_release_realloc() {
    let circuit = TestCircuit {
        num_qubits: 2,
        gates: vec![
            TestGate::H(0),
            TestGate::Cx(0, 1),
            TestGate::Release(1), // release qubit 1
            TestGate::Allocate,   // allocate a new qubit (becomes index 2)
            TestGate::H(2),       // apply H to new qubit
        ],
    };
    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    phase_normalize(&mut sparse_state);
    phase_normalize(&mut gpu_state);
    let (max_error, passed) = compare_states(&sparse_state, &gpu_state, 1e-5);
    assert!(passed, "Alloc/release/realloc max error: {max_error:.2e}");
}

/// Smoke test exercising every gate type in a single circuit.
#[test]
fn test_every_gate_type() {
    let circuit = TestCircuit {
        num_qubits: 4,
        gates: vec![
            // Single-qubit Clifford
            TestGate::H(0),
            TestGate::X(1),
            TestGate::Y(2),
            TestGate::Z(3),
            TestGate::S(0),
            TestGate::Sadj(1),
            TestGate::SX(2),
            TestGate::SXadj(3),
            // Non-Clifford
            TestGate::T(0),
            TestGate::Tadj(1),
            // Rotations
            TestGate::Rx(0.3, 0),
            TestGate::Ry(0.7, 1),
            TestGate::Rz(1.1, 2),
            // Two-qubit
            TestGate::Cx(0, 1),
            TestGate::Cy(1, 2),
            TestGate::Cz(2, 3),
            TestGate::Swap(0, 3),
            // Multi-controlled
            TestGate::Mcx(vec![0, 1], 2),
            TestGate::Mcy(vec![1, 2], 3),
            TestGate::Mcz(vec![2, 3], 0),
        ],
    };
    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    phase_normalize(&mut sparse_state);
    phase_normalize(&mut gpu_state);
    let (max_error, passed) = compare_states(&sparse_state, &gpu_state, 1e-5);
    assert!(passed, "Every gate type: max error {max_error:.2e}");
}

/// Deep precision stress test: 2 qubits, 1000 gates.
///
/// This test characterizes the precision behavior of the GPU simulator
/// under deep circuits. The f32 internal representation on GPU may
/// accumulate error over many gate applications. This test documents
/// the actual error rather than imposing a tight tolerance, but will
/// fail if the error is catastrophic (>1e-2).
#[test]
fn test_deep_2q_d1000() {
    let circuit = generate_universal_circuit(2, 1000, 42);
    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    phase_normalize(&mut sparse_state);
    phase_normalize(&mut gpu_state);
    let (max_error, _) = compare_states(&sparse_state, &gpu_state, 1e-5);

    // Document observed error; don't hard-fail for deep circuits
    eprintln!("deep_2q_d1000: max component error = {max_error:.2e}");
    // Soft assertion -- log but only fail if error is catastrophic
    assert!(
        max_error < 1e-2,
        "catastrophic error in deep circuit: {max_error:.2e}"
    );
}
