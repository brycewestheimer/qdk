use super::circuit::{TestCircuit, TestGate};
use super::runners::{compare_states, phase_normalize, run_on_gpu, run_on_sparse};

/// Build a QFT circuit on `n` qubits.
///
/// The QFT applies:
///   for i in 0..n:
///     H(i)
///     for j in (i+1)..n:
///       CRz(pi/2^(j-i), control=j, target=i)
///   // Reverse qubit order with SWAPs
///   for i in 0..(n/2):
///     SWAP(i, n-1-i)
///
/// `CRz` is decomposed as: `CX(j,i)`, `Rz(-angle/2, i)`, `CX(j,i)`, `Rz(angle/2, i)`, `Rz(angle/2, j)`.
/// This is the standard decomposition of `CRz` into single-qubit `Rz` and `CX` gates.
fn build_qft_circuit(n: usize) -> TestCircuit {
    let mut gates = Vec::new();

    for i in 0..n {
        gates.push(TestGate::H(i));
        for j in (i + 1)..n {
            let angle = std::f64::consts::PI / f64::from(1u32 << (j - i));
            // Decompose CRz(angle, control=j, target=i):
            gates.push(TestGate::Cx(j, i));
            gates.push(TestGate::Rz(-angle / 2.0, i));
            gates.push(TestGate::Cx(j, i));
            gates.push(TestGate::Rz(angle / 2.0, i));
            gates.push(TestGate::Rz(angle / 2.0, j));
        }
    }

    // Reverse qubit order using physical SWAPs decomposed as 3 CX gates
    // (TestGate::Swap maps to swap_qubit_ids which is metadata-only)
    for i in 0..(n / 2) {
        let a = i;
        let b = n - 1 - i;
        gates.push(TestGate::Cx(a, b));
        gates.push(TestGate::Cx(b, a));
        gates.push(TestGate::Cx(a, b));
    }

    TestCircuit {
        num_qubits: n,
        gates,
    }
}

/// Compute the adjoint of a single gate.
fn adjoint_gate(gate: &TestGate) -> TestGate {
    match gate {
        // Self-adjoint gates
        TestGate::H(q) => TestGate::H(*q),
        TestGate::X(q) => TestGate::X(*q),
        TestGate::Y(q) => TestGate::Y(*q),
        TestGate::Z(q) => TestGate::Z(*q),
        TestGate::Cx(c, t) => TestGate::Cx(*c, *t),
        TestGate::Cy(c, t) => TestGate::Cy(*c, *t),
        TestGate::Cz(c, t) => TestGate::Cz(*c, *t),
        TestGate::Swap(a, b) => TestGate::Swap(*a, *b),
        TestGate::Mcx(cs, t) => TestGate::Mcx(cs.clone(), *t),
        TestGate::Mcy(cs, t) => TestGate::Mcy(cs.clone(), *t),
        TestGate::Mcz(cs, t) => TestGate::Mcz(cs.clone(), *t),
        // Adjoint pairs
        TestGate::S(q) => TestGate::Sadj(*q),
        TestGate::Sadj(q) => TestGate::S(*q),
        TestGate::T(q) => TestGate::Tadj(*q),
        TestGate::Tadj(q) => TestGate::T(*q),
        TestGate::SX(q) => TestGate::SXadj(*q),
        TestGate::SXadj(q) => TestGate::SX(*q),
        // Negate rotation angles
        TestGate::Rx(theta, q) => TestGate::Rx(-theta, *q),
        TestGate::Ry(theta, q) => TestGate::Ry(-theta, *q),
        TestGate::Rz(theta, q) => TestGate::Rz(-theta, *q),
        // State management -- pass through
        TestGate::Allocate => TestGate::Allocate,
        TestGate::Release(q) => TestGate::Release(*q),
    }
}

/// Invert a circuit: reverse gate order and take the adjoint of each gate.
fn invert_circuit(circuit: &TestCircuit) -> TestCircuit {
    let gates: Vec<TestGate> = circuit.gates.iter().rev().map(adjoint_gate).collect();
    TestCircuit {
        num_qubits: circuit.num_qubits,
        gates,
    }
}

fn build_qft_roundtrip_circuit(n: usize) -> TestCircuit {
    let qft = build_qft_circuit(n);
    let inv_qft = invert_circuit(&qft);
    let mut gates = qft.gates;
    gates.extend(inv_qft.gates);
    TestCircuit {
        num_qubits: n,
        gates,
    }
}

/// QFT on 4 qubits starting from |1000>
#[test]
fn test_qft_4() {
    let mut circuit = build_qft_circuit(4);
    circuit.gates.insert(0, TestGate::X(0));

    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    phase_normalize(&mut sparse_state);
    phase_normalize(&mut gpu_state);
    let (max_error, passed) = compare_states(&sparse_state, &gpu_state, 1e-5);
    assert!(passed, "QFT-4 max error: {max_error:.2e}");
}

/// QFT on 8 qubits starting from |10000000>
#[test]
fn test_qft_8() {
    let mut circuit = build_qft_circuit(8);
    circuit.gates.insert(0, TestGate::X(0));

    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    phase_normalize(&mut sparse_state);
    phase_normalize(&mut gpu_state);
    let (max_error, passed) = compare_states(&sparse_state, &gpu_state, 1e-5);
    assert!(passed, "QFT-8 max error: {max_error:.2e}");
}

/// QFT on 12 qubits starting from |100000000000>
#[test]
fn test_qft_12() {
    let mut circuit = build_qft_circuit(12);
    circuit.gates.insert(0, TestGate::X(0));

    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    phase_normalize(&mut sparse_state);
    phase_normalize(&mut gpu_state);
    let (max_error, passed) = compare_states(&sparse_state, &gpu_state, 1e-5);
    assert!(passed, "QFT-12 max error: {max_error:.2e}");
}

/// QFT inverse round-trip on 4 qubits starting from |1010>
#[test]
fn test_qft_inverse_roundtrip_4() {
    let mut circuit = build_qft_roundtrip_circuit(4);
    // Start from a non-trivial state: |1010>
    circuit.gates.insert(0, TestGate::X(0));
    circuit.gates.insert(1, TestGate::X(2));

    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    phase_normalize(&mut sparse_state);
    phase_normalize(&mut gpu_state);
    let (max_error, passed) = compare_states(&sparse_state, &gpu_state, 1e-5);
    assert!(passed, "QFT roundtrip-4 max error: {max_error:.2e}");
}

/// Build a Grover circuit for 3 qubits searching for |101>.
///
/// The circuit consists of:
/// 1. Superposition: H on all qubits
/// 2. Grover iterations (2 iterations optimal for 3 qubits, 1 solution):
///    a. Oracle: flip the phase of |101>
///    b. Diffusion: reflect about the mean
fn build_grover_3q_circuit() -> TestCircuit {
    let mut gates = Vec::new();

    // Step 1: Create uniform superposition
    gates.push(TestGate::H(0));
    gates.push(TestGate::H(1));
    gates.push(TestGate::H(2));

    // Step 2: Two Grover iterations
    for _ in 0..2 {
        // --- Oracle for |101> ---
        // |101> means qubit 0 = 1, qubit 1 = 0, qubit 2 = 1
        // To mark this state: flip qubit 1 (so all controls are |1>),
        // apply multi-controlled Z, flip qubit 1 back.
        gates.push(TestGate::X(1)); // flip qubit 1
        gates.push(TestGate::H(2)); // convert target for MCZ -> MCX+H trick
        gates.push(TestGate::Mcx(vec![0, 1], 2)); // Toffoli
        gates.push(TestGate::H(2)); // undo H
        gates.push(TestGate::X(1)); // unflip qubit 1

        // --- Diffusion operator ---
        // Reflect about |+++>: H on all, then reflect about |000>, then H on all
        gates.push(TestGate::H(0));
        gates.push(TestGate::H(1));
        gates.push(TestGate::H(2));

        // Reflect about |000>: flip all, multi-controlled Z, flip all
        gates.push(TestGate::X(0));
        gates.push(TestGate::X(1));
        gates.push(TestGate::X(2));

        // Multi-controlled Z = H on target, Toffoli, H on target
        gates.push(TestGate::H(2));
        gates.push(TestGate::Mcx(vec![0, 1], 2));
        gates.push(TestGate::H(2));

        gates.push(TestGate::X(0));
        gates.push(TestGate::X(1));
        gates.push(TestGate::X(2));

        gates.push(TestGate::H(0));
        gates.push(TestGate::H(1));
        gates.push(TestGate::H(2));
    }

    TestCircuit {
        num_qubits: 3,
        gates,
    }
}

/// Grover's 3-qubit search for |101>
#[test]
fn test_grover_3q() {
    let circuit = build_grover_3q_circuit();
    let mut sparse_state = run_on_sparse(&circuit).0;
    let mut gpu_state = run_on_gpu(&circuit).0;
    phase_normalize(&mut sparse_state);
    phase_normalize(&mut gpu_state);
    let (max_error, passed) = compare_states(&sparse_state, &gpu_state, 1e-5);
    assert!(passed, "Grover 3q max error: {max_error:.2e}");
}
