#![cfg(feature = "gpu-tests")]

use num_bigint::BigUint;
use num_complex::Complex64;

const TOLERANCE: f64 = 1e-6;

/// Assert two complex numbers are approximately equal.
fn assert_complex_approx(actual: Complex64, expected: Complex64, label: &str) {
    assert!(
        (actual.re - expected.re).abs() < TOLERANCE && (actual.im - expected.im).abs() < TOLERANCE,
        "{label}: expected ({}, {}) but got ({}, {})",
        expected.re,
        expected.im,
        actual.re,
        actual.im,
    );
}

/// Find an amplitude by basis state index.
fn find_amplitude(state: &[(BigUint, Complex64)], index: u64) -> Complex64 {
    let target = BigUint::from(index);
    state
        .iter()
        .find(|(idx, _)| *idx == target)
        .map_or(Complex64::new(0.0, 0.0), |(_, amp)| *amp)
}

#[test]
fn gpu_device_creation() {
    let sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42));
    assert!(sim.is_ok(), "GPU simulator should initialize");
    let sim = sim.expect("already checked");
    let info = sim.adapter_info();
    eprintln!(
        "GPU adapter: {} ({:?}, {:?})",
        info.name, info.device_type, info.backend
    );
    eprintln!("Max qubits (f32): {}", sim.max_qubits());
}

#[test]
fn state_initialization() {
    let mut sim =
        qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU simulator should initialize");
    let _q0 = sim.allocate();
    let _q1 = sim.allocate();

    let (state, num_qubits) = sim.get_state().expect("get_state should succeed");

    assert_eq!(num_qubits, 2);
    // Only |00> should have non-zero amplitude
    assert_eq!(state.len(), 1, "only |00> should be non-zero");
    let amp = find_amplitude(&state, 0);
    assert_complex_approx(amp, Complex64::new(1.0, 0.0), "|00> amplitude");
}

#[test]
fn hadamard_on_zero() {
    let mut sim =
        qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU simulator should initialize");
    let q = sim.allocate();
    sim.h(q);

    let (state, num_qubits) = sim.get_state().expect("get_state should succeed");

    assert_eq!(num_qubits, 1);
    assert_eq!(
        state.len(),
        2,
        "both |0> and |1> should have non-zero amplitude"
    );

    let expected = 1.0 / std::f64::consts::SQRT_2;
    for (idx, amp) in &state {
        let i: u64 = idx.try_into().expect("index should fit in u64");
        assert!(i < 2, "unexpected basis state index {i}");
        assert_complex_approx(
            *amp,
            Complex64::new(expected, 0.0),
            &format!("|{i}> amplitude"),
        );
    }
}

#[test]
fn hadamard_twice_is_identity() {
    let mut sim =
        qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU simulator should initialize");
    let q = sim.allocate();

    // Apply H twice: H*H = I, so state should return to |0>
    sim.h(q);
    sim.h(q);

    let (state, num_qubits) = sim.get_state().expect("get_state should succeed");

    assert_eq!(num_qubits, 1);
    assert_eq!(state.len(), 1, "only |0> should be non-zero after H*H");
    let amp = find_amplitude(&state, 0);
    assert_complex_approx(amp, Complex64::new(1.0, 0.0), "|0> amplitude after H*H");
}

#[test]
fn hadamard_on_two_qubit_system() {
    let mut sim =
        qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU simulator should initialize");
    let q0 = sim.allocate();
    let _q1 = sim.allocate();

    // Apply H to q0 only. State should be (|00> + |01>) / sqrt(2)
    // (qubit 0 is the LSB, so flipping q0 flips bit 0)
    // Actually: q0 maps to bit 0, so H on q0 gives:
    //   (|0> + |1>)/sqrt(2) tensor |0> = (|00> + |01>) / sqrt(2)
    // Wait -- bit ordering: index 0 = |00>, index 1 = bit 0 set = |01>,
    // index 2 = bit 1 set = |10>, index 3 = |11>.
    // H on bit 0 (q0): |00> -> (|00> + |01>) / sqrt(2)
    // So indices 0 and 1 should have amplitude 1/sqrt(2).
    sim.h(q0);

    let (state, num_qubits) = sim.get_state().expect("get_state should succeed");

    assert_eq!(num_qubits, 2);
    assert_eq!(state.len(), 2);

    let expected = 1.0 / std::f64::consts::SQRT_2;
    let amp0 = find_amplitude(&state, 0);
    let amp1 = find_amplitude(&state, 1);
    assert_complex_approx(amp0, Complex64::new(expected, 0.0), "|00> amplitude");
    assert_complex_approx(amp1, Complex64::new(expected, 0.0), "|01> amplitude");
}
