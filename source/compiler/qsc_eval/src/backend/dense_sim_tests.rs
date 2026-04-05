// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

use crate::backend::{Backend, DenseSim, SparseSim};
use crate::val::{self, Value};

// =============================================================================
// State capture endianness
// =============================================================================

#[test]
fn capture_quantum_state_matches_sparse() {
    let mut dense = DenseSim::new();
    let mut sparse = SparseSim::new();

    let q0_d = dense.qubit_allocate();
    let q1_d = dense.qubit_allocate();
    let q0_s = sparse.qubit_allocate();
    let q1_s = sparse.qubit_allocate();

    dense.h(q0_d);
    dense.cx(q0_d, q1_d);
    sparse.h(q0_s);
    sparse.cx(q0_s, q1_s);

    let (state_d, count_d) = dense.capture_quantum_state();
    let (state_s, count_s) = sparse.capture_quantum_state();

    assert_eq!(count_d, count_s, "qubit counts should match");
    assert_eq!(
        state_d.len(),
        state_s.len(),
        "state entry counts should match"
    );
    for ((idx_d, amp_d), (idx_s, amp_s)) in state_d.iter().zip(state_s.iter()) {
        assert_eq!(idx_d, idx_s, "basis state indices should match");
        assert!(
            (amp_d - amp_s).norm() < 1e-5,
            "amplitude mismatch at index {idx_d}: dense={amp_d}, sparse={amp_s}"
        );
    }
}

#[test]
fn capture_quantum_state_3qubit_endianness() {
    let mut dense = DenseSim::new();
    let mut sparse = SparseSim::new();

    let q0_d = dense.qubit_allocate();
    let _q1_d = dense.qubit_allocate();
    let _q2_d = dense.qubit_allocate();
    let q0_s = sparse.qubit_allocate();
    let _q1_s = sparse.qubit_allocate();
    let _q2_s = sparse.qubit_allocate();

    dense.x(q0_d);
    sparse.x(q0_s);

    let (state_d, _) = dense.capture_quantum_state();
    let (state_s, _) = sparse.capture_quantum_state();

    assert_eq!(state_d.len(), 1);
    assert_eq!(state_s.len(), 1);
    assert_eq!(state_d[0].0, state_s[0].0);
}

// =============================================================================
// GlobalPhase intrinsic
// =============================================================================

#[test]
fn custom_intrinsic_global_phase_recognized() {
    let mut sim = DenseSim::new();
    let _q = sim.qubit_allocate();

    let arg = Value::Tuple(
        vec![
            Value::Array(vec![].into()),
            Value::Double(std::f64::consts::FRAC_PI_4),
        ]
        .into(),
        None,
    );

    let result = sim.custom_intrinsic("GlobalPhase", arg);
    assert!(
        matches!(result, Some(Ok(_))),
        "GlobalPhase should be recognized and succeed"
    );
}

#[test]
fn custom_intrinsic_global_phase_matches_sparse() {
    let mut dense = DenseSim::new();
    let mut sparse = SparseSim::new();

    let q_d = dense.qubit_allocate();
    let q_s = sparse.qubit_allocate();

    dense.h(q_d);
    sparse.h(q_s);

    let theta = std::f64::consts::FRAC_PI_3;
    let make_arg = || {
        Value::Tuple(
            vec![Value::Array(vec![].into()), Value::Double(theta)].into(),
            None,
        )
    };

    dense.custom_intrinsic("GlobalPhase", make_arg());
    sparse.custom_intrinsic("GlobalPhase", make_arg());

    let (state_d, _) = dense.capture_quantum_state();
    let (state_s, _) = sparse.capture_quantum_state();

    for ((_, amp_d), (_, amp_s)) in state_d.iter().zip(state_s.iter()) {
        assert!(
            (amp_d - amp_s).norm() < 1e-5,
            "GlobalPhase states should match: dense={amp_d}, sparse={amp_s}"
        );
    }
}

// =============================================================================
// MResetZ
// =============================================================================

#[test]
fn mresetz_resets_to_zero() {
    let mut sim = DenseSim::new();
    sim.set_seed(Some(42));
    let q = sim.qubit_allocate();
    sim.x(q);

    let result = sim.mresetz(q);
    assert!(
        matches!(result, val::Result::Val(true)),
        "should measure |1>"
    );
    assert!(sim.qubit_is_zero(q), "qubit should be |0> after mresetz");
}

#[test]
fn mresetz_zero_state() {
    let mut sim = DenseSim::new();
    let q = sim.qubit_allocate();

    let result = sim.mresetz(q);
    assert!(
        matches!(result, val::Result::Val(false)),
        "should measure |0>"
    );
    assert!(sim.qubit_is_zero(q), "qubit should remain |0>");
}

// =============================================================================
// Qubit release
// =============================================================================

#[test]
fn qubit_release_returns_false_for_nonzero() {
    let mut sim = DenseSim::new();
    let q = sim.qubit_allocate();
    sim.x(q);

    assert!(!sim.qubit_release(q));
}

#[test]
fn qubit_release_returns_true_for_zero() {
    let mut sim = DenseSim::new();
    let q = sim.qubit_allocate();

    assert!(sim.qubit_release(q));
}

// =============================================================================
// Reset
// =============================================================================

#[test]
fn reset_puts_qubit_in_zero() {
    let mut sim = DenseSim::new();
    sim.set_seed(Some(42));
    let q = sim.qubit_allocate();
    sim.h(q);

    sim.reset(q);
    assert!(sim.qubit_is_zero(q), "qubit should be |0> after reset");
}

// =============================================================================
// Two-qubit rotation decompositions
// =============================================================================

#[test]
fn rxx_matches_sparse() {
    let mut dense = DenseSim::new();
    let mut sparse = SparseSim::new();

    let q0_d = dense.qubit_allocate();
    let q1_d = dense.qubit_allocate();
    let q0_s = sparse.qubit_allocate();
    let q1_s = sparse.qubit_allocate();

    dense.h(q0_d);
    sparse.h(q0_s);

    let theta = 0.7;
    dense.rxx(theta, q0_d, q1_d);
    sparse.rxx(theta, q0_s, q1_s);

    let (state_d, _) = dense.capture_quantum_state();
    let (state_s, _) = sparse.capture_quantum_state();

    for ((_, amp_d), (_, amp_s)) in state_d.iter().zip(state_s.iter()) {
        assert!(
            (amp_d - amp_s).norm() < 1e-5,
            "Rxx mismatch: dense={amp_d}, sparse={amp_s}"
        );
    }
}

#[test]
fn ryy_matches_sparse() {
    let mut dense = DenseSim::new();
    let mut sparse = SparseSim::new();

    let q0_d = dense.qubit_allocate();
    let q1_d = dense.qubit_allocate();
    let q0_s = sparse.qubit_allocate();
    let q1_s = sparse.qubit_allocate();

    dense.h(q0_d);
    sparse.h(q0_s);

    let theta = 0.7;
    dense.ryy(theta, q0_d, q1_d);
    sparse.ryy(theta, q0_s, q1_s);

    let (state_d, _) = dense.capture_quantum_state();
    let (state_s, _) = sparse.capture_quantum_state();

    for ((_, amp_d), (_, amp_s)) in state_d.iter().zip(state_s.iter()) {
        assert!(
            (amp_d - amp_s).norm() < 1e-5,
            "Ryy mismatch: dense={amp_d}, sparse={amp_s}"
        );
    }
}

#[test]
fn rzz_matches_sparse() {
    let mut dense = DenseSim::new();
    let mut sparse = SparseSim::new();

    let q0_d = dense.qubit_allocate();
    let q1_d = dense.qubit_allocate();
    let q0_s = sparse.qubit_allocate();
    let q1_s = sparse.qubit_allocate();

    dense.h(q0_d);
    sparse.h(q0_s);

    let theta = 0.7;
    dense.rzz(theta, q0_d, q1_d);
    sparse.rzz(theta, q0_s, q1_s);

    let (state_d, _) = dense.capture_quantum_state();
    let (state_s, _) = sparse.capture_quantum_state();

    for ((_, amp_d), (_, amp_s)) in state_d.iter().zip(state_s.iter()) {
        assert!(
            (amp_d - amp_s).norm() < 1e-5,
            "Rzz mismatch: dense={amp_d}, sparse={amp_s}"
        );
    }
}
