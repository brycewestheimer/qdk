#![cfg(feature = "gpu-tests")]

use std::f32::consts::FRAC_1_SQRT_2;

const TOL: f32 = 1e-6;

// ============================================================================
// Test helpers
// ============================================================================

/// Expand the sparse state from `get_state()` into a dense amplitude vector.
fn dense_state(sim: &qdk_gpu_sim::GpuQuantumSim) -> Vec<(f32, f32)> {
    let (sparse, num_qubits) = sim.get_state().expect("get_state should succeed");
    let n = 1usize << num_qubits;
    let mut dense = vec![(0.0f32, 0.0f32); n];
    for (idx, amp) in sparse {
        let i: usize = idx.try_into().expect("index should fit");
        #[allow(clippy::cast_possible_truncation)]
        {
            dense[i] = (amp.re as f32, amp.im as f32);
        }
    }
    dense
}

fn assert_state_approx_eq(actual: &[(f32, f32)], expected: &[(f32, f32)], tol: f32) {
    assert_eq!(actual.len(), expected.len(), "state vector length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a.0 - e.0).abs() < tol && (a.1 - e.1).abs() < tol,
            "amplitude mismatch at index {i}: got ({}, {}), expected ({}, {})",
            a.0,
            a.1,
            e.0,
            e.1,
        );
    }
}

/// Zero complex amplitude.
const Z: (f32, f32) = (0.0, 0.0);
/// One (real) complex amplitude.
const ONE: (f32, f32) = (1.0, 0.0);

fn two_qubit_sim() -> (qdk_gpu_sim::GpuQuantumSim, usize, usize) {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should initialize");
    let q0 = sim.allocate().expect("allocation should succeed");
    let q1 = sim.allocate().expect("allocation should succeed");
    (sim, q0, q1)
}

fn three_qubit_sim() -> (qdk_gpu_sim::GpuQuantumSim, usize, usize, usize) {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should initialize");
    let q0 = sim.allocate().expect("allocation should succeed");
    let q1 = sim.allocate().expect("allocation should succeed");
    let q2 = sim.allocate().expect("allocation should succeed");
    (sim, q0, q1, q2)
}

// ============================================================================
// Single-qubit gate tests
// ============================================================================

// --- X gate ---

#[test]
fn x_gate_on_zero() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.x(q0);
    assert_state_approx_eq(&dense_state(&sim), &[Z, Z, ONE, Z], TOL);
}

#[test]
fn x_gate_on_one() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.x(q0);
    sim.x(q0);
    assert_state_approx_eq(&dense_state(&sim), &[ONE, Z, Z, Z], TOL);
}

#[test]
fn x_gate_on_qubit_1() {
    let (mut sim, _q0, q1) = two_qubit_sim();
    sim.x(q1);
    assert_state_approx_eq(&dense_state(&sim), &[Z, ONE, Z, Z], TOL);
}

#[test]
fn x_gate_is_self_inverse() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.h(q0);
    let before = dense_state(&sim);
    sim.x(q0);
    sim.x(q0);
    assert_state_approx_eq(&dense_state(&sim), &before, TOL);
}

// --- Y gate ---

#[test]
fn y_gate_on_zero() {
    // Y|0> = i|1>
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.y(q0);
    assert_state_approx_eq(&dense_state(&sim), &[Z, Z, (0.0, 1.0), Z], TOL);
}

#[test]
fn y_gate_on_one() {
    // Y|1> = -i|0>
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.x(q0);
    sim.y(q0);
    assert_state_approx_eq(&dense_state(&sim), &[(0.0, -1.0), Z, Z, Z], TOL);
}

#[test]
fn y_gate_is_self_inverse() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.h(q0);
    let before = dense_state(&sim);
    sim.y(q0);
    sim.y(q0);
    assert_state_approx_eq(&dense_state(&sim), &before, TOL);
}

// --- Z gate ---

#[test]
fn z_gate_on_zero() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.z(q0);
    assert_state_approx_eq(&dense_state(&sim), &[ONE, Z, Z, Z], TOL);
}

#[test]
fn z_gate_on_one() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.x(q0);
    sim.z(q0);
    assert_state_approx_eq(&dense_state(&sim), &[Z, Z, (-1.0, 0.0), Z], TOL);
}

#[test]
fn z_gate_on_plus() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.h(q0);
    sim.z(q0);
    let f = FRAC_1_SQRT_2;
    assert_state_approx_eq(&dense_state(&sim), &[(f, 0.0), Z, (-f, 0.0), Z], TOL);
}

// --- S gate ---

#[test]
fn s_gate_on_zero() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.s(q0);
    assert_state_approx_eq(&dense_state(&sim), &[ONE, Z, Z, Z], TOL);
}

#[test]
fn s_gate_on_one() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.x(q0);
    sim.s(q0);
    assert_state_approx_eq(&dense_state(&sim), &[Z, Z, (0.0, 1.0), Z], TOL);
}

#[test]
fn s_sadj_inverse() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.h(q0);
    let before = dense_state(&sim);
    sim.s(q0);
    sim.sadj(q0);
    assert_state_approx_eq(&dense_state(&sim), &before, TOL);
}

// --- Sadj gate ---

#[test]
fn sadj_gate_on_one() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.x(q0);
    sim.sadj(q0);
    assert_state_approx_eq(&dense_state(&sim), &[Z, Z, (0.0, -1.0), Z], TOL);
}

// --- T gate ---

#[test]
fn t_gate_on_zero() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.t(q0);
    assert_state_approx_eq(&dense_state(&sim), &[ONE, Z, Z, Z], TOL);
}

#[test]
fn t_gate_on_one() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.x(q0);
    sim.t(q0);
    let f = FRAC_1_SQRT_2;
    assert_state_approx_eq(&dense_state(&sim), &[Z, Z, (f, f), Z], TOL);
}

#[test]
fn t_tadj_inverse() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.h(q0);
    let before = dense_state(&sim);
    sim.t(q0);
    sim.tadj(q0);
    assert_state_approx_eq(&dense_state(&sim), &before, TOL);
}

// --- Tadj gate ---

#[test]
fn tadj_gate_on_one() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.x(q0);
    sim.tadj(q0);
    let f = FRAC_1_SQRT_2;
    assert_state_approx_eq(&dense_state(&sim), &[Z, Z, (f, -f), Z], TOL);
}

// --- SX gate ---

#[test]
fn sx_gate_on_zero() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.sx(q0);
    assert_state_approx_eq(&dense_state(&sim), &[(0.5, 0.5), Z, (0.5, -0.5), Z], TOL);
}

#[test]
fn sx_gate_on_one() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.x(q0);
    sim.sx(q0);
    assert_state_approx_eq(&dense_state(&sim), &[(0.5, -0.5), Z, (0.5, 0.5), Z], TOL);
}

#[test]
fn sx_sxadj_inverse() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.h(q0);
    let before = dense_state(&sim);
    sim.sx(q0);
    sim.sxadj(q0);
    assert_state_approx_eq(&dense_state(&sim), &before, TOL);
}

#[test]
fn sx_squared_is_x() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.sx(q0);
    sim.sx(q0);
    assert_state_approx_eq(&dense_state(&sim), &[Z, Z, ONE, Z], TOL);
}

// --- SXadj gate ---

#[test]
fn sxadj_gate_on_zero() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.sxadj(q0);
    assert_state_approx_eq(&dense_state(&sim), &[(0.5, -0.5), Z, (0.5, 0.5), Z], TOL);
}

// ============================================================================
// Rotation gate tests
// ============================================================================

// --- Rx ---

#[test]
fn rx_zero_is_identity() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.h(q0);
    let before = dense_state(&sim);
    sim.rx(0.0, q0);
    assert_state_approx_eq(&dense_state(&sim), &before, TOL);
}

#[test]
fn rx_pi_is_minus_i_x() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.rx(std::f64::consts::PI, q0);
    assert_state_approx_eq(&dense_state(&sim), &[Z, Z, (0.0, -1.0), Z], TOL);
}

#[test]
fn rx_pi_on_one() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.x(q0);
    sim.rx(std::f64::consts::PI, q0);
    assert_state_approx_eq(&dense_state(&sim), &[(0.0, -1.0), Z, Z, Z], TOL);
}

#[test]
fn rx_half_pi() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.rx(std::f64::consts::FRAC_PI_2, q0);
    let f = FRAC_1_SQRT_2;
    assert_state_approx_eq(&dense_state(&sim), &[(f, 0.0), Z, (0.0, -f), Z], TOL);
}

#[test]
fn rx_composition() {
    let a = 0.3;
    let b = 0.7;
    let (mut sim1, q0_1, _) = two_qubit_sim();
    sim1.h(q0_1);
    sim1.rx(a, q0_1);
    sim1.rx(b, q0_1);
    let state1 = dense_state(&sim1);

    let (mut sim2, q0_2, _) = two_qubit_sim();
    sim2.h(q0_2);
    sim2.rx(a + b, q0_2);
    let state2 = dense_state(&sim2);

    assert_state_approx_eq(&state1, &state2, TOL);
}

// --- Ry ---

#[test]
fn ry_zero_is_identity() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.h(q0);
    let before = dense_state(&sim);
    sim.ry(0.0, q0);
    assert_state_approx_eq(&dense_state(&sim), &before, TOL);
}

#[test]
fn ry_pi() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.ry(std::f64::consts::PI, q0);
    assert_state_approx_eq(&dense_state(&sim), &[Z, Z, ONE, Z], TOL);
}

#[test]
fn ry_pi_on_one() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.x(q0);
    sim.ry(std::f64::consts::PI, q0);
    assert_state_approx_eq(&dense_state(&sim), &[(-1.0, 0.0), Z, Z, Z], TOL);
}

#[test]
fn ry_half_pi() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.ry(std::f64::consts::FRAC_PI_2, q0);
    let f = FRAC_1_SQRT_2;
    assert_state_approx_eq(&dense_state(&sim), &[(f, 0.0), Z, (f, 0.0), Z], TOL);
}

// --- Rz ---

#[test]
fn rz_zero_is_identity() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.h(q0);
    let before = dense_state(&sim);
    sim.rz(0.0, q0);
    assert_state_approx_eq(&dense_state(&sim), &before, TOL);
}

#[test]
fn rz_pi() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.rz(std::f64::consts::PI, q0);
    assert_state_approx_eq(&dense_state(&sim), &[(0.0, -1.0), Z, Z, Z], TOL);
}

#[test]
fn rz_pi_on_one() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.x(q0);
    sim.rz(std::f64::consts::PI, q0);
    assert_state_approx_eq(&dense_state(&sim), &[Z, Z, (0.0, 1.0), Z], TOL);
}

#[test]
fn rz_half_pi_on_plus() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.h(q0);
    sim.rz(std::f64::consts::FRAC_PI_2, q0);
    assert_state_approx_eq(&dense_state(&sim), &[(0.5, -0.5), Z, (0.5, 0.5), Z], TOL);
}

#[test]
fn rz_composition() {
    let a = 1.2;
    let b = 0.8;
    let (mut sim1, q0_1, _) = two_qubit_sim();
    sim1.h(q0_1);
    sim1.rz(a, q0_1);
    sim1.rz(b, q0_1);
    let state1 = dense_state(&sim1);

    let (mut sim2, q0_2, _) = two_qubit_sim();
    sim2.h(q0_2);
    sim2.rz(a + b, q0_2);
    let state2 = dense_state(&sim2);

    assert_state_approx_eq(&state1, &state2, TOL);
}

// ============================================================================
// Multi-controlled gate tests
// ============================================================================

#[test]
fn cnot_truth_table() {
    for control_val in [false, true] {
        for target_val in [false, true] {
            let (mut sim, q0, q1) = two_qubit_sim();
            if control_val {
                sim.x(q0);
            }
            if target_val {
                sim.x(q1);
            }

            sim.mcx(&[q0], q1);

            let state = dense_state(&sim);
            let expected_target = if control_val { !target_val } else { target_val };
            let expected_idx = usize::from(control_val) * 2 + usize::from(expected_target);
            for (i, amp) in state.iter().enumerate() {
                if i == expected_idx {
                    assert!(
                        (amp.0 - 1.0).abs() < TOL && amp.1.abs() < TOL,
                        "input |{}{}>: expected 1.0 at index {expected_idx}, got {amp:?}",
                        u8::from(control_val),
                        u8::from(target_val),
                    );
                } else {
                    assert!(
                        amp.0.abs() < TOL && amp.1.abs() < TOL,
                        "input |{}{}>: expected 0 at index {i}, got {amp:?}",
                        u8::from(control_val),
                        u8::from(target_val),
                    );
                }
            }
        }
    }
}

#[test]
fn toffoli_truth_table() {
    for c0 in [false, true] {
        for c1 in [false, true] {
            for t in [false, true] {
                let (mut sim, q0, q1, q2) = three_qubit_sim();
                if c0 {
                    sim.x(q0);
                }
                if c1 {
                    sim.x(q1);
                }
                if t {
                    sim.x(q2);
                }

                sim.mcx(&[q0, q1], q2);

                let state = dense_state(&sim);
                let expected_t = if c0 && c1 { !t } else { t };
                let expected_idx =
                    usize::from(c0) * 4 + usize::from(c1) * 2 + usize::from(expected_t);
                for (i, amp) in state.iter().enumerate() {
                    if i == expected_idx {
                        assert!(
                            (amp.0 - 1.0).abs() < TOL && amp.1.abs() < TOL,
                            "Toffoli |{}{}{}> -> expected 1.0 at {expected_idx}, got {amp:?}",
                            u8::from(c0),
                            u8::from(c1),
                            u8::from(t),
                        );
                    } else {
                        assert!(
                            amp.0.abs() < TOL && amp.1.abs() < TOL,
                            "Toffoli |{}{}{}> -> expected 0 at {i}, got {amp:?}",
                            u8::from(c0),
                            u8::from(c1),
                            u8::from(t),
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn mcz_on_11() {
    let (mut sim, q0, q1) = two_qubit_sim();
    sim.x(q0);
    sim.x(q1);
    sim.mcz(&[q0], q1);
    assert_state_approx_eq(&dense_state(&sim), &[Z, Z, Z, (-1.0, 0.0)], TOL);
}

#[test]
fn mcz_on_01() {
    let (mut sim, q0, q1) = two_qubit_sim();
    sim.x(q1);
    sim.mcz(&[q0], q1);
    assert_state_approx_eq(&dense_state(&sim), &[Z, ONE, Z, Z], TOL);
}

#[test]
fn mcz_on_10() {
    let (mut sim, q0, q1) = two_qubit_sim();
    sim.x(q0);
    sim.mcz(&[q0], q1);
    assert_state_approx_eq(&dense_state(&sim), &[Z, Z, ONE, Z], TOL);
}

#[test]
fn three_control_x() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("should init");
    let q0 = sim.allocate().expect("allocation should succeed");
    let q1 = sim.allocate().expect("allocation should succeed");
    let q2 = sim.allocate().expect("allocation should succeed");
    let q3 = sim.allocate().expect("allocation should succeed");

    sim.x(q0);
    sim.x(q1);
    sim.x(q2);
    // State: |1110> = index 14

    sim.mcx(&[q0, q1, q2], q3);
    // All controls |1> -> flip q3: |1111> = index 15

    let state = dense_state(&sim);
    for (i, amp) in state.iter().enumerate() {
        if i == 15 {
            assert!(
                (amp.0 - 1.0).abs() < TOL && amp.1.abs() < TOL,
                "expected 1.0 at index 15, got {amp:?}"
            );
        } else {
            assert!(
                amp.0.abs() < TOL && amp.1.abs() < TOL,
                "expected 0 at index {i}, got {amp:?}"
            );
        }
    }
}

#[test]
fn three_control_x_partial() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("should init");
    let q0 = sim.allocate().expect("allocation should succeed");
    let q1 = sim.allocate().expect("allocation should succeed");
    let q2 = sim.allocate().expect("allocation should succeed");
    let q3 = sim.allocate().expect("allocation should succeed");

    sim.x(q0);
    sim.x(q1);
    // q2 stays |0>. State: |0110> = index 6

    sim.mcx(&[q0, q1, q2], q3);
    // Not all controls |1> -> no flip. State unchanged: index 6

    let state = dense_state(&sim);
    for (i, amp) in state.iter().enumerate() {
        if i == 6 {
            assert!(
                (amp.0 - 1.0).abs() < TOL && amp.1.abs() < TOL,
                "expected 1.0 at index 6, got {amp:?}"
            );
        } else {
            assert!(
                amp.0.abs() < TOL && amp.1.abs() < TOL,
                "expected 0 at index {i}, got {amp:?}"
            );
        }
    }
}

#[test]
fn mcx_zero_controls_is_x() {
    let (mut sim, q0, _q1) = two_qubit_sim();
    sim.mcx(&[], q0);
    assert_state_approx_eq(&dense_state(&sim), &[Z, Z, ONE, Z], TOL);
}

// ============================================================================
// Two-qubit gate tests (SWAP)
// ============================================================================

#[test]
fn swap_basic() {
    let (mut sim, q0, q1) = two_qubit_sim();
    sim.x(q0); // |10>
    sim.swap(q0, q1);
    assert_state_approx_eq(&dense_state(&sim), &[Z, ONE, Z, Z], TOL);
}

#[test]
fn swap_01_to_10() {
    let (mut sim, q0, q1) = two_qubit_sim();
    sim.x(q1); // |01>
    sim.swap(q0, q1);
    assert_state_approx_eq(&dense_state(&sim), &[Z, Z, ONE, Z], TOL);
}

#[test]
fn swap_superposition() {
    let (mut sim, q0, q1) = two_qubit_sim();
    sim.h(q0); // (|00> + |10>) / sqrt(2)
    sim.swap(q0, q1);
    let f = FRAC_1_SQRT_2;
    assert_state_approx_eq(&dense_state(&sim), &[(f, 0.0), (f, 0.0), Z, Z], TOL);
}

#[test]
fn swap_is_self_inverse() {
    let (mut sim, q0, q1) = two_qubit_sim();
    sim.h(q0);
    sim.t(q1);
    let before = dense_state(&sim);
    sim.swap(q0, q1);
    sim.swap(q0, q1);
    assert_state_approx_eq(&dense_state(&sim), &before, TOL);
}

#[test]
fn swap_unchanged_for_00_and_11() {
    let (mut sim, q0, q1) = two_qubit_sim();
    sim.swap(q0, q1);
    assert_state_approx_eq(&dense_state(&sim), &[ONE, Z, Z, Z], TOL);

    let (mut sim2, q0_2, q1_2) = two_qubit_sim();
    sim2.x(q0_2);
    sim2.x(q1_2);
    sim2.swap(q0_2, q1_2);
    assert_state_approx_eq(&dense_state(&sim2), &[Z, Z, Z, ONE], TOL);
}

// ============================================================================
// Composition tests
// ============================================================================

#[test]
fn bell_state() {
    let (mut sim, q0, q1) = two_qubit_sim();
    sim.h(q0);
    sim.mcx(&[q0], q1);
    let f = FRAC_1_SQRT_2;
    assert_state_approx_eq(&dense_state(&sim), &[(f, 0.0), Z, Z, (f, 0.0)], TOL);
}

#[test]
fn ghz_3_qubit() {
    let (mut sim, q0, q1, q2) = three_qubit_sim();
    sim.h(q0);
    sim.mcx(&[q0], q1);
    sim.mcx(&[q0], q2);
    let f = FRAC_1_SQRT_2;
    let mut expected = vec![Z; 8];
    expected[0] = (f, 0.0);
    expected[7] = (f, 0.0);
    assert_state_approx_eq(&dense_state(&sim), &expected, TOL);
}

#[test]
fn ghz_4_qubit() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("should init");
    let q0 = sim.allocate().expect("allocation should succeed");
    let q1 = sim.allocate().expect("allocation should succeed");
    let q2 = sim.allocate().expect("allocation should succeed");
    let q3 = sim.allocate().expect("allocation should succeed");
    sim.h(q0);
    sim.mcx(&[q0], q1);
    sim.mcx(&[q0], q2);
    sim.mcx(&[q0], q3);
    let f = FRAC_1_SQRT_2;
    let mut expected = vec![Z; 16];
    expected[0] = (f, 0.0);
    expected[15] = (f, 0.0);
    assert_state_approx_eq(&dense_state(&sim), &expected, TOL);
}

// ============================================================================
// swap_qubit_ids tests
// ============================================================================

#[test]
fn swap_qubit_ids_no_state_change() {
    let (mut sim, q0, q1) = two_qubit_sim();
    sim.x(q0); // |10>
    let before = dense_state(&sim);
    sim.swap_qubit_ids(q0, q1);
    assert_state_approx_eq(&dense_state(&sim), &before, TOL);
}

#[test]
fn swap_qubit_ids_affects_subsequent_gates() {
    let (mut sim, q0, q1) = two_qubit_sim();
    sim.x(q0); // |10> (bit 0 = 1, bit 1 = 0)
    sim.swap_qubit_ids(q0, q1); // Now q0 maps to bit 1, q1 maps to bit 0
    sim.x(q0); // Applies X to bit 1 -> bit0=1, bit1=1 = |11> = index 3
    assert_state_approx_eq(&dense_state(&sim), &[Z, Z, Z, ONE], TOL);
}

// ============================================================================
// SparseSim compatibility test
// ============================================================================

#[test]
fn hsh_equals_sx() {
    // SparseSim implements sx as H;S;H. Verify it matches our native SX.
    let (mut sim_native, q0_n, _) = two_qubit_sim();
    sim_native.h(q0_n);
    sim_native.sx(q0_n);
    let state_native = dense_state(&sim_native);

    let (mut sim_decomp, q0_d, _) = two_qubit_sim();
    sim_decomp.h(q0_d);
    sim_decomp.h(q0_d);
    sim_decomp.s(q0_d);
    sim_decomp.h(q0_d);
    let state_decomp = dense_state(&sim_decomp);

    assert_state_approx_eq(&state_native, &state_decomp, TOL);
}

// ============================================================================
// Multi-controlled gate tests -- additional gates (M-08)
// ============================================================================

/// MCY truth table: controlled-Y on 2-qubit system.
/// Y|0> = i|1>, Y|1> = -i|0>. Control |0> leaves target unchanged.
#[test]
fn mcy_truth_table() {
    for control_val in [false, true] {
        for target_val in [false, true] {
            let (mut sim, q0, q1) = two_qubit_sim();
            if control_val {
                sim.x(q0);
            }
            if target_val {
                sim.x(q1);
            }
            sim.mcy(&[q0], q1);
            let state = dense_state(&sim);

            if !control_val {
                // No gate: state unchanged
                let idx = usize::from(control_val) * 2 + usize::from(target_val);
                for (i, amp) in state.iter().enumerate() {
                    if i == idx {
                        assert!(
                            (amp.0 - 1.0).abs() < TOL && amp.1.abs() < TOL,
                            "MCY |{}{}>: expected 1.0 at {idx}, got {amp:?}",
                            u8::from(control_val),
                            u8::from(target_val)
                        );
                    } else {
                        assert!(
                            amp.0.abs() < TOL && amp.1.abs() < TOL,
                            "MCY |{}{}>: expected 0 at {i}, got {amp:?}",
                            u8::from(control_val),
                            u8::from(target_val)
                        );
                    }
                }
            } else if !target_val {
                // |10>, Y|0> = i|1> => |11> with amplitude i (index 3)
                assert_state_approx_eq(&state, &[Z, Z, Z, (0.0, 1.0)], TOL);
            } else {
                // |11>, Y|1> = -i|0> => |10> with amplitude -i (index 2)
                assert_state_approx_eq(&state, &[Z, Z, (0.0, -1.0), Z], TOL);
            }
        }
    }
}

/// MCH truth table: controlled-Hadamard on 2-qubit system.
/// H|0> = (|0>+|1>)/sqrt(2), H|1> = (|0>-|1>)/sqrt(2).
#[test]
fn mch_truth_table() {
    let f = FRAC_1_SQRT_2;
    for control_val in [false, true] {
        for target_val in [false, true] {
            let (mut sim, q0, q1) = two_qubit_sim();
            if control_val {
                sim.x(q0);
            }
            if target_val {
                sim.x(q1);
            }
            sim.mch(&[q0], q1);
            let state = dense_state(&sim);

            if !control_val {
                let idx = usize::from(control_val) * 2 + usize::from(target_val);
                for (i, amp) in state.iter().enumerate() {
                    if i == idx {
                        assert!(
                            (amp.0 - 1.0).abs() < TOL && amp.1.abs() < TOL,
                            "MCH |{}{}>: expected 1.0 at {idx}, got {amp:?}",
                            u8::from(control_val),
                            u8::from(target_val)
                        );
                    } else {
                        assert!(
                            amp.0.abs() < TOL && amp.1.abs() < TOL,
                            "MCH |{}{}>: expected 0 at {i}, got {amp:?}",
                            u8::from(control_val),
                            u8::from(target_val)
                        );
                    }
                }
            } else if !target_val {
                // |10>, H|0> = (+,+)/sqrt2 => indices 2,3
                assert_state_approx_eq(&state, &[Z, Z, (f, 0.0), (f, 0.0)], TOL);
            } else {
                // |11>, H|1> = (+,-)/sqrt2 => indices 2,3
                assert_state_approx_eq(&state, &[Z, Z, (f, 0.0), (-f, 0.0)], TOL);
            }
        }
    }
}

/// Helper: run a diagonal multi-controlled gate truth table.
///
/// `gate_fn` applies the gate. `expected_11` is the amplitude at index 3
/// when input is |11> (the only case where the gate activates on |1>).
fn diagonal_mc_truth_table(
    gate_fn: fn(&mut qdk_gpu_sim::GpuQuantumSim, &[usize], usize),
    expected_11: (f32, f32),
    label: &str,
) {
    for control_val in [false, true] {
        for target_val in [false, true] {
            let (mut sim, q0, q1) = two_qubit_sim();
            if control_val {
                sim.x(q0);
            }
            if target_val {
                sim.x(q1);
            }

            gate_fn(&mut sim, &[q0], q1);

            let state = dense_state(&sim);

            if !control_val || !target_val {
                let expected_idx = usize::from(control_val) * 2 + usize::from(target_val);
                for (i, amp) in state.iter().enumerate() {
                    if i == expected_idx {
                        assert!(
                            (amp.0 - 1.0).abs() < TOL && amp.1.abs() < TOL,
                            "{label} |{}{}>: expected 1.0 at {expected_idx}, got {amp:?}",
                            u8::from(control_val),
                            u8::from(target_val),
                        );
                    } else {
                        assert!(
                            amp.0.abs() < TOL && amp.1.abs() < TOL,
                            "{label} |{}{}>: expected 0 at {i}, got {amp:?}",
                            u8::from(control_val),
                            u8::from(target_val),
                        );
                    }
                }
            } else {
                let expected = [Z, Z, Z, expected_11];
                assert_state_approx_eq(&state, &expected, TOL);
            }
        }
    }
}

/// MCS: S = diag(1, i). Target |1> -> i|1>.
#[test]
fn mcs_truth_table() {
    diagonal_mc_truth_table(qdk_gpu_sim::GpuQuantumSim::mcs, (0.0, 1.0), "MCS");
}

/// `MCSadj`: Sadj = diag(1, -i). Target |1> -> -i|1>.
#[test]
fn mcsadj_truth_table() {
    diagonal_mc_truth_table(qdk_gpu_sim::GpuQuantumSim::mcsadj, (0.0, -1.0), "MCSadj");
}

/// MCT: T = diag(1, (1+i)/sqrt(2)). Target |1> -> (1+i)/sqrt(2) |1>.
#[test]
fn mct_truth_table() {
    let f = FRAC_1_SQRT_2;
    diagonal_mc_truth_table(qdk_gpu_sim::GpuQuantumSim::mct, (f, f), "MCT");
}

/// `MCTadj`: Tadj = diag(1, (1-i)/sqrt(2)). Target |1> -> (1-i)/sqrt(2) |1>.
#[test]
fn mctadj_truth_table() {
    let f = FRAC_1_SQRT_2;
    diagonal_mc_truth_table(qdk_gpu_sim::GpuQuantumSim::mctadj, (f, -f), "MCTadj");
}

/// MCRZ(pi/2) truth table. Rz(pi/2) = diag((1-i)/sqrt(2), (1+i)/sqrt(2)).
/// Unlike the `diagonal_mc_truth_table` helper, Rz has non-trivial `mat[0]`.
#[test]
fn mcrz_truth_table() {
    let f = FRAC_1_SQRT_2;
    let theta = std::f64::consts::FRAC_PI_2;
    for control_val in [false, true] {
        for target_val in [false, true] {
            let (mut sim, q0, q1) = two_qubit_sim();
            if control_val {
                sim.x(q0);
            }
            if target_val {
                sim.x(q1);
            }
            sim.mcrz(&[q0], theta, q1);
            let state = dense_state(&sim);

            if !control_val {
                let idx = usize::from(control_val) * 2 + usize::from(target_val);
                for (i, amp) in state.iter().enumerate() {
                    if i == idx {
                        assert!(
                            (amp.0 - 1.0).abs() < TOL && amp.1.abs() < TOL,
                            "MCRZ |{}{}>: expected 1 at {idx}",
                            u8::from(control_val),
                            u8::from(target_val)
                        );
                    } else {
                        assert!(
                            amp.0.abs() < TOL && amp.1.abs() < TOL,
                            "MCRZ |{}{}>: expected 0 at {i}",
                            u8::from(control_val),
                            u8::from(target_val)
                        );
                    }
                }
            } else if !target_val {
                // |10>, Rz|0> = e^(-i*pi/4)|0> => index 2 gets (f, -f)
                assert_state_approx_eq(&state, &[Z, Z, (f, -f), Z], TOL);
            } else {
                // |11>, Rz|1> = e^(i*pi/4)|1> => index 3 gets (f, f)
                assert_state_approx_eq(&state, &[Z, Z, Z, (f, f)], TOL);
            }
        }
    }
}

/// `MCPhase` with phase=e^(i*pi/3). Phase gate = diag(1, phase).
#[test]
fn mcphase_truth_table() {
    use num_complex::Complex64;
    let phase = Complex64::new(0.5, (3.0_f64).sqrt() / 2.0);

    // Only |11> gets the phase; use diagonal_mc helper pattern manually
    // because mcphase has a different signature (takes Complex64).
    #[allow(clippy::cast_possible_truncation)]
    let expected_11 = (phase.re as f32, phase.im as f32);

    for control_val in [false, true] {
        for target_val in [false, true] {
            let (mut sim, q0, q1) = two_qubit_sim();
            if control_val {
                sim.x(q0);
            }
            if target_val {
                sim.x(q1);
            }
            sim.mcphase(&[q0], phase, q1);
            let state = dense_state(&sim);

            if !control_val || !target_val {
                let idx = usize::from(control_val) * 2 + usize::from(target_val);
                for (i, amp) in state.iter().enumerate() {
                    if i == idx {
                        assert!(
                            (amp.0 - 1.0).abs() < TOL && amp.1.abs() < TOL,
                            "MCPhase |{}{}>: expected 1 at {idx}",
                            u8::from(control_val),
                            u8::from(target_val)
                        );
                    } else {
                        assert!(
                            amp.0.abs() < TOL && amp.1.abs() < TOL,
                            "MCPhase |{}{}>: expected 0 at {i}",
                            u8::from(control_val),
                            u8::from(target_val)
                        );
                    }
                }
            } else {
                assert_state_approx_eq(&state, &[Z, Z, Z, expected_11], TOL);
            }
        }
    }
}

/// Verify MCS composed with `MCSadj` is identity (with controls active).
#[test]
fn mcs_mcsadj_inverse() {
    let (mut sim, q0, q1) = two_qubit_sim();
    sim.x(q0); // Activate control
    sim.h(q1); // Put target in superposition
    let before = dense_state(&sim);
    sim.mcs(&[q0], q1);
    sim.mcsadj(&[q0], q1);
    assert_state_approx_eq(&dense_state(&sim), &before, TOL);
}

/// Verify MCT composed with `MCTadj` is identity (with controls active).
#[test]
fn mct_mctadj_inverse() {
    let (mut sim, q0, q1) = two_qubit_sim();
    sim.x(q0);
    sim.h(q1);
    let before = dense_state(&sim);
    sim.mct(&[q0], q1);
    sim.mctadj(&[q0], q1);
    assert_state_approx_eq(&dense_state(&sim), &before, TOL);
}

// ============================================================================
// Large-system gate tests (D5)
// ============================================================================

/// Helper to create a 16-qubit simulator and return (sim, `qubit_ids`).
fn sixteen_qubit_sim() -> (qdk_gpu_sim::GpuQuantumSim, Vec<usize>) {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should initialize");
    let ids: Vec<usize> = (0..16)
        .map(|_| sim.allocate().expect("allocation should succeed"))
        .collect();
    (sim, ids)
}

/// X gate at position 0 in a 16-qubit system.
#[test]
fn x_gate_16q_position_0() {
    let (mut sim, ids) = sixteen_qubit_sim();
    sim.x(ids[0]);
    // Only qubit 0 is |1> => index 1 (bit 0 set)
    let (state, _) = sim.get_state().expect("get_state should succeed");
    assert_eq!(state.len(), 1, "should have exactly one non-zero amplitude");
    assert_eq!(state[0].0, num_bigint::BigUint::from(1u32));
}

/// X gate at position n/2 (qubit 8) in a 16-qubit system.
#[test]
fn x_gate_16q_position_mid() {
    let (mut sim, ids) = sixteen_qubit_sim();
    sim.x(ids[8]);
    let (state, _) = sim.get_state().expect("get_state should succeed");
    assert_eq!(state.len(), 1);
    // Qubit ID 8 -> bit 8 => index 256
    assert_eq!(state[0].0, num_bigint::BigUint::from(256u32));
}

/// X gate at position n-1 (qubit 15) in a 16-qubit system.
#[test]
fn x_gate_16q_position_high() {
    let (mut sim, ids) = sixteen_qubit_sim();
    sim.x(ids[15]);
    let (state, _) = sim.get_state().expect("get_state should succeed");
    assert_eq!(state.len(), 1);
    // Qubit ID 15 -> bit 15 => index 32768
    assert_eq!(state[0].0, num_bigint::BigUint::from(32768u32));
}

/// Hadamard at various positions in 16-qubit system produces correct
/// superposition (2 non-zero amplitudes).
#[test]
fn h_gate_16q_all_positions() {
    for &pos in &[0, 8, 15] {
        let (mut sim, ids) = sixteen_qubit_sim();
        sim.h(ids[pos]);
        let (state, _) = sim.get_state().expect("get_state should succeed");
        assert_eq!(
            state.len(),
            2,
            "H on qubit {pos} should produce 2 non-zero amplitudes"
        );
    }
}

/// CNOT between distant qubits in a 16-qubit system.
#[test]
fn cnot_16q_distant() {
    let (mut sim, ids) = sixteen_qubit_sim();
    sim.x(ids[0]); // Set control qubit
    sim.mcx(&[ids[0]], ids[15]); // CNOT across full register
    let (state, _) = sim.get_state().expect("get_state should succeed");
    assert_eq!(state.len(), 1);
    // Both qubit 0 and qubit 15 are |1> => index = 1 + 32768 = 32769
    assert_eq!(state[0].0, num_bigint::BigUint::from(32769u32));
}

/// Bell state between qubit 0 and qubit 15 in a 16-qubit system.
#[test]
fn bell_state_16q_distant() {
    let (mut sim, ids) = sixteen_qubit_sim();
    sim.h(ids[0]);
    sim.mcx(&[ids[0]], ids[15]);
    let (state, _) = sim.get_state().expect("get_state should succeed");
    assert_eq!(
        state.len(),
        2,
        "Bell state should have 2 non-zero amplitudes"
    );
    // Should be |0...0> and |1...1> with index 0 and 32769
    let indices: Vec<u64> = state
        .iter()
        .map(|(idx, _)| idx.to_u64_digits().first().copied().unwrap_or(0))
        .collect();
    assert!(indices.contains(&0), "should contain |00...0>");
    assert!(indices.contains(&32769), "should contain |10...01>");
}
