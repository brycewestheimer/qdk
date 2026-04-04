use super::generator::{generate_clifford_t_circuit, generate_universal_circuit};
use super::runners::{compare_states_phase_normalized, run_on_gpu, run_on_sparse};

#[test]
fn test_random_clifford_t_4q_d50() {
    let circuit = generate_clifford_t_circuit(4, 50, 42);
    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    let (max_error, passed) =
        compare_states_phase_normalized(&mut sparse_state, &mut gpu_state, 1e-5);
    assert!(
        passed,
        "Random Clifford+T 4q/d50: max error {max_error:.2e}"
    );
}

#[test]
fn test_random_clifford_t_8q_d100() {
    let circuit = generate_clifford_t_circuit(8, 100, 42);
    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    let (max_error, passed) =
        compare_states_phase_normalized(&mut sparse_state, &mut gpu_state, 1e-5);
    assert!(
        passed,
        "Random Clifford+T 8q/d100: max error {max_error:.2e}"
    );
}

#[test]
fn test_random_universal_4q_d50() {
    let circuit = generate_universal_circuit(4, 50, 42);
    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    let (max_error, passed) =
        compare_states_phase_normalized(&mut sparse_state, &mut gpu_state, 1e-5);
    assert!(passed, "Random universal 4q/d50: max error {max_error:.2e}");
}

#[test]
fn test_random_universal_8q_d100() {
    let circuit = generate_universal_circuit(8, 100, 42);
    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    let (max_error, passed) =
        compare_states_phase_normalized(&mut sparse_state, &mut gpu_state, 1e-5);
    assert!(
        passed,
        "Random universal 8q/d100: max error {max_error:.2e}"
    );
}

#[test]
fn test_random_universal_12q_d200() {
    let circuit = generate_universal_circuit(12, 200, 42);
    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    let (max_error, passed) =
        compare_states_phase_normalized(&mut sparse_state, &mut gpu_state, 1e-5);
    assert!(
        passed,
        "Random universal 12q/d200: max error {max_error:.2e}"
    );
}
