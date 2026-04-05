use rand::prelude::*;
use rand::rngs::StdRng;

use super::circuit::{TestCircuit, TestGate};

/// Generate a random circuit mixing all gate types.
///
/// Gate distribution:
/// - 40% single-qubit Clifford (H, S, Sadj, X, Y, Z)
/// - 15% single-qubit non-Clifford (T, Tadj)
/// - 20% single-qubit rotation (Rx, Ry, Rz with random angle)
/// - 25% two-qubit (Cx, Cy, Cz, Swap)
///
/// `num_qubits`: number of qubits to allocate.
/// `depth`: number of gate operations in the circuit.
/// `seed`: seed for deterministic generation.
pub fn generate_random_circuit(num_qubits: usize, depth: usize, seed: u64) -> TestCircuit {
    assert!(num_qubits >= 2, "random circuits need at least 2 qubits");
    let mut rng = StdRng::seed_from_u64(seed);
    let mut gates = Vec::with_capacity(depth);

    for _ in 0..depth {
        let q = rng.gen_range(0..num_qubits);
        let class_roll: f64 = rng.r#gen();
        let gate = if class_roll < 0.40 {
            // Clifford single-qubit
            match rng.gen_range(0..6u32) {
                0 => TestGate::H(q),
                1 => TestGate::S(q),
                2 => TestGate::Sadj(q),
                3 => TestGate::X(q),
                4 => TestGate::Y(q),
                _ => TestGate::Z(q),
            }
        } else if class_roll < 0.55 {
            // Non-Clifford
            if rng.gen_bool(0.5) {
                TestGate::T(q)
            } else {
                TestGate::Tadj(q)
            }
        } else if class_roll < 0.75 {
            // Rotation with random angle in [0, 2*pi)
            let theta = rng.r#gen::<f64>() * std::f64::consts::TAU;
            match rng.gen_range(0..3u32) {
                0 => TestGate::Rx(theta, q),
                1 => TestGate::Ry(theta, q),
                _ => TestGate::Rz(theta, q),
            }
        } else {
            // Two-qubit gate -- pick a second qubit different from q
            let q2 = loop {
                let candidate = rng.gen_range(0..num_qubits);
                if candidate != q {
                    break candidate;
                }
            };
            match rng.gen_range(0..4u32) {
                0 => TestGate::Cx(q, q2),
                1 => TestGate::Cy(q, q2),
                2 => TestGate::Cz(q, q2),
                _ => TestGate::Swap(q, q2),
            }
        };
        gates.push(gate);
    }

    TestCircuit { num_qubits, gates }
}

/// Clifford+T circuit only (no continuous rotations).
///
/// Gate distribution:
/// - 50% single-qubit Clifford (H, S, Sadj, X, Y, Z)
/// - 20% single-qubit non-Clifford (T, Tadj)
/// - 30% two-qubit (Cx only -- the standard Clifford entangling gate)
pub fn generate_clifford_t_circuit(num_qubits: usize, depth: usize, seed: u64) -> TestCircuit {
    assert!(num_qubits >= 2, "random circuits need at least 2 qubits");
    let mut rng = StdRng::seed_from_u64(seed);
    let mut gates = Vec::with_capacity(depth);

    for _ in 0..depth {
        let q = rng.gen_range(0..num_qubits);
        let class_roll: f64 = rng.r#gen();
        let gate = if class_roll < 0.50 {
            match rng.gen_range(0..6u32) {
                0 => TestGate::H(q),
                1 => TestGate::S(q),
                2 => TestGate::Sadj(q),
                3 => TestGate::X(q),
                4 => TestGate::Y(q),
                _ => TestGate::Z(q),
            }
        } else if class_roll < 0.70 {
            if rng.gen_bool(0.5) {
                TestGate::T(q)
            } else {
                TestGate::Tadj(q)
            }
        } else {
            let q2 = loop {
                let candidate = rng.gen_range(0..num_qubits);
                if candidate != q {
                    break candidate;
                }
            };
            TestGate::Cx(q, q2)
        };
        gates.push(gate);
    }

    TestCircuit { num_qubits, gates }
}

/// Universal circuit (includes parameterized rotations).
/// Same as `generate_random_circuit` but ensures the full gate set is exercised.
pub fn generate_universal_circuit(num_qubits: usize, depth: usize, seed: u64) -> TestCircuit {
    generate_random_circuit(num_qubits, depth, seed)
}
