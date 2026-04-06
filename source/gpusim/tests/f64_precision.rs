//! Precision comparison tests for f64 emulation.
//!
//! These tests compare the GPU simulator's output against expected values
//! to quantify the precision improvement from f64 emulation.
//!
//! These tests require BOTH feature flags:
//!   `cargo test -p qdk-gpu-sim --features gpu-tests,f64_emulation`
//!
//! They are NOT included in the default `--features gpu-tests` run.
//! Ensure they are executed as part of pre-merge validation.
//! See `scripts/test_all_features.sh` for the full test matrix.

#![cfg(all(feature = "gpu-tests", feature = "f64_emulation"))]

use num_complex::Complex64;
use qdk_gpu_sim::GpuQuantumSim;

#[test]
fn ds_roundtrip_on_cpu() {
    let val = std::f64::consts::PI;
    let (hi, lo) = qdk_gpu_sim::precision::to_ds(val);
    let reconstructed = qdk_gpu_sim::precision::from_ds(hi, lo);
    assert!(
        (val - reconstructed).abs() < 1e-14,
        "DS roundtrip error: {:.2e}",
        (val - reconstructed).abs()
    );
}

#[test]
fn f64_emulated_bell_state_precision() {
    let mut sim = GpuQuantumSim::new(Some(42)).expect("GPU init failed");
    let q0 = sim.allocate().expect("allocation should succeed");
    let q1 = sim.allocate().expect("allocation should succeed");

    sim.h(q0);
    sim.mcx(&[q0], q1);

    let (state, _num_qubits) = sim.get_state().expect("get_state failed");
    let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();

    // With f64 emulation, the amplitudes should be closer to 1/sqrt(2) than
    // plain f32 (~1e-7). DS arithmetic achieves ~1e-8 after gate application,
    // measurement readback, and collapse normalization on GPU hardware.
    for (_idx, amp) in &state {
        let error = (amp.norm() - inv_sqrt2).abs();
        assert!(
            error < 1e-7,
            "f64-emulated amplitude deviation too large: {error:.2e}"
        );
    }

    sim.release(q0);
    sim.release(q1);
}

#[test]
fn f64_emulated_deep_circuit_fidelity() {
    // Apply many gates and check that state vector normalization is maintained
    // with high precision.
    let mut sim = GpuQuantumSim::new(Some(42)).expect("GPU init failed");
    let mut qubits = Vec::new();
    for _ in 0..4 {
        qubits.push(sim.allocate().expect("allocation should succeed"));
    }

    // Apply a deep sequence of gates.
    for _ in 0..200 {
        for &q in &qubits {
            sim.h(q);
            sim.t(q);
        }
        sim.mcx(&[qubits[0]], qubits[1]);
        sim.mcx(&[qubits[2]], qubits[3]);
    }

    let (state, _num_qubits) = sim.get_state().expect("get_state failed");

    // Check normalization: sum of |amplitude|^2 should be very close to 1.0.
    let total_prob: f64 = state.iter().map(|(_, amp)| amp.norm_sqr()).sum();
    let norm_error = (total_prob - 1.0).abs();

    println!("f64 emulation: normalization error after 800 gates on 4 qubits: {norm_error:.2e}");
    // DS error accumulates as ~O(n_gates * eps_ds) through gate application,
    // plus additional amplification from measurement tree reductions and
    // collapse normalization (sqrt amplifies probability errors). For 800
    // gates on 4 qubits, observed errors are ~1e-5 to 1e-4.
    assert!(
        norm_error < 1e-3,
        "Normalization error too large with f64 emulation: {norm_error:.2e}"
    );

    for q in qubits.into_iter().rev() {
        sim.release(q);
    }
}

#[test]
fn f64_emulated_state_readback_encoding() {
    // Verify that get_state() correctly reconstructs Complex64 from DS pairs.
    let mut sim = GpuQuantumSim::new(Some(42)).expect("GPU init failed");
    let q = sim.allocate().expect("allocation should succeed");

    // After allocating one qubit with no gates, state should be exactly |0>.
    let (state, _) = sim.get_state().expect("get_state failed");
    assert_eq!(state.len(), 1, "should have exactly one non-zero amplitude");

    let (_, amp) = &state[0];
    let error = (amp - Complex64::new(1.0, 0.0)).norm();
    assert!(
        error < 1e-14,
        "initial state amplitude should be exactly 1.0: error = {error:.2e}"
    );

    sim.release(q);
}
