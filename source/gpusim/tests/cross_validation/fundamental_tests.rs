use super::circuit::{TestCircuit, TestGate};
use super::runners::{compare_states, phase_normalize, run_on_gpu, run_on_sparse};

/// Bell state: (|00> + |11>) / sqrt(2)
#[test]
fn test_bell_state() {
    let circuit = TestCircuit {
        num_qubits: 2,
        gates: vec![TestGate::H(0), TestGate::Cx(0, 1)],
    };

    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    phase_normalize(&mut sparse_state);
    phase_normalize(&mut gpu_state);
    let (max_error, passed) = compare_states(&sparse_state, &gpu_state, 1e-10);
    assert!(passed, "Bell state max error: {max_error:.2e}");
}

fn build_ghz_circuit(n: usize) -> TestCircuit {
    let mut gates = vec![TestGate::H(0)];
    for i in 1..n {
        gates.push(TestGate::Cx(0, i));
    }
    TestCircuit {
        num_qubits: n,
        gates,
    }
}

/// GHZ state for N=3: (|000> + |111>) / sqrt(2)
#[test]
fn test_ghz_n3() {
    let circuit = build_ghz_circuit(3);
    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    phase_normalize(&mut sparse_state);
    phase_normalize(&mut gpu_state);
    let (max_error, passed) = compare_states(&sparse_state, &gpu_state, 1e-10);
    assert!(passed, "GHZ-3 max error: {max_error:.2e}");
}

/// GHZ state for N=4
#[test]
fn test_ghz_n4() {
    let circuit = build_ghz_circuit(4);
    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    phase_normalize(&mut sparse_state);
    phase_normalize(&mut gpu_state);
    let (max_error, passed) = compare_states(&sparse_state, &gpu_state, 1e-10);
    assert!(passed, "GHZ-4 max error: {max_error:.2e}");
}

/// GHZ state for N=5
#[test]
fn test_ghz_n5() {
    let circuit = build_ghz_circuit(5);
    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    phase_normalize(&mut sparse_state);
    phase_normalize(&mut gpu_state);
    let (max_error, passed) = compare_states(&sparse_state, &gpu_state, 1e-10);
    assert!(passed, "GHZ-5 max error: {max_error:.2e}");
}

/// GHZ state for N=8
#[test]
fn test_ghz_n8() {
    let circuit = build_ghz_circuit(8);
    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    phase_normalize(&mut sparse_state);
    phase_normalize(&mut gpu_state);
    let (max_error, passed) = compare_states(&sparse_state, &gpu_state, 1e-10);
    assert!(passed, "GHZ-8 max error: {max_error:.2e}");
}

/// GHZ state for N=10
#[test]
fn test_ghz_n10() {
    let circuit = build_ghz_circuit(10);
    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    phase_normalize(&mut sparse_state);
    phase_normalize(&mut gpu_state);
    let (max_error, passed) = compare_states(&sparse_state, &gpu_state, 1e-10);
    assert!(passed, "GHZ-10 max error: {max_error:.2e}");
}

/// Single qubit: H-T-S-Rx-Ry-Rz chain
#[test]
fn test_single_qubit() {
    let circuit = TestCircuit {
        num_qubits: 1,
        gates: vec![
            TestGate::H(0),
            TestGate::T(0),
            TestGate::S(0),
            TestGate::Rx(std::f64::consts::FRAC_PI_3, 0),
            TestGate::Ry(std::f64::consts::FRAC_PI_4 * 0.4, 0),
            TestGate::Rz(std::f64::consts::FRAC_PI_6 * 0.8, 0),
        ],
    };
    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    phase_normalize(&mut sparse_state);
    phase_normalize(&mut gpu_state);
    let (max_error, passed) = compare_states(&sparse_state, &gpu_state, 1e-10);
    assert!(passed, "Single qubit max error: {max_error:.2e}");
}

/// Identity circuit: no gates, state should be |0000>
#[test]
fn test_identity_circuit() {
    let circuit = TestCircuit {
        num_qubits: 4,
        gates: vec![], // No gates -- state should be |0000>
    };
    let sparse_state = run_on_sparse(&circuit).0;
    let gpu_state = run_on_gpu(&circuit).0;
    let (max_error, passed) = compare_states(&sparse_state, &gpu_state, 1e-10);
    assert!(passed, "Identity circuit max error: {max_error:.2e}");
}
