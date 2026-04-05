#![cfg(feature = "gpu-tests")]

use num_bigint::BigUint;
use num_complex::Complex64;

fn assert_probability_approx(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() < tol,
        "probability mismatch: got {actual}, expected {expected} (tol {tol})",
    );
}

fn assert_amplitude_approx(actual: Complex64, expected: Complex64, tol: f64) {
    assert!(
        (actual.re - expected.re).abs() < tol && (actual.im - expected.im).abs() < tol,
        "amplitude mismatch: got ({}, {}), expected ({}, {})",
        actual.re,
        actual.im,
        expected.re,
        expected.im,
    );
}

// ============================================================================
// Measurement probability tests
// ============================================================================

#[test]
fn test_measure_zero_state() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q = sim.allocate().expect("allocation should succeed");

    for _ in 0..100 {
        assert!(
            !sim.measure(q).expect("measurement should succeed"),
            "|0> should always measure as false"
        );
    }
}

#[test]
fn test_measure_one_state() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q = sim.allocate().expect("allocation should succeed");
    sim.x(q); // |1>

    assert!(
        sim.measure(q).expect("measurement should succeed"),
        "|1> should always measure as true"
    );
}

#[test]
fn test_measure_plus_state_statistics() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q = sim.allocate().expect("allocation should succeed");

    let n = 10_000u32;
    let mut ones = 0u32;

    for _ in 0..n {
        sim.h(q); // Prepare |+>
        if sim.measure(q).expect("measurement should succeed") {
            ones += 1;
            sim.x(q); // Reset to |0> for next iteration
        }
        // If measured |0>, qubit is already in |0> -- no reset needed.
    }

    let fraction = f64::from(ones) / f64::from(n);
    // Expected: 0.5. 3-sigma for binomial(10000, 0.5): 3 * sqrt(0.25/10000) = 0.015.
    // Use a generous 5% bound for GPU floating-point variation.
    assert!(
        (fraction - 0.5).abs() < 0.05,
        "measured fraction {fraction} too far from 0.5",
    );
}

// ============================================================================
// Entangled measurement tests
// ============================================================================

#[test]
fn test_bell_state_measurement_correlation() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");

    for _ in 0..100 {
        let q0 = sim.allocate().expect("allocation should succeed");
        let q1 = sim.allocate().expect("allocation should succeed");

        sim.h(q0);
        sim.mcx(&[q0], q1); // Bell state: (|00> + |11>) / sqrt(2)

        let r0 = sim.measure(q0).expect("measurement should succeed");
        let r1 = sim.measure(q1).expect("measurement should succeed");
        assert_eq!(r0, r1, "Bell pair measurements must be correlated");

        // Reset and release for next iteration
        if r0 {
            sim.x(q0);
        }
        if r1 {
            sim.x(q1);
        }
        sim.release(q0);
        sim.release(q1);
    }
}

#[test]
fn test_ghz_state_measurement() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");

    for _ in 0..100 {
        let q0 = sim.allocate().expect("allocation should succeed");
        let q1 = sim.allocate().expect("allocation should succeed");
        let q2 = sim.allocate().expect("allocation should succeed");

        sim.h(q0);
        sim.mcx(&[q0], q1);
        sim.mcx(&[q0], q2); // GHZ: (|000> + |111>) / sqrt(2)

        let r0 = sim.measure(q0).expect("measurement should succeed");
        let r1 = sim.measure(q1).expect("measurement should succeed");
        let r2 = sim.measure(q2).expect("measurement should succeed");
        assert_eq!(r0, r1, "GHZ: q0 and q1 must agree");
        assert_eq!(r1, r2, "GHZ: q1 and q2 must agree");

        if r0 {
            sim.x(q0);
        }
        if r1 {
            sim.x(q1);
        }
        if r2 {
            sim.x(q2);
        }
        sim.release(q0);
        sim.release(q1);
        sim.release(q2);
    }
}

// ============================================================================
// Joint measurement tests
// ============================================================================

#[test]
fn test_joint_probability_bell_even_parity() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q0 = sim.allocate().expect("allocation should succeed");
    let q1 = sim.allocate().expect("allocation should succeed");

    sim.h(q0);
    sim.mcx(&[q0], q1); // (|00> + |11>) / sqrt(2)

    let p = sim
        .joint_probability(&[q0, q1])
        .expect("probability computation should succeed");
    assert_probability_approx(p, 0.0, 1e-6);
}

#[test]
fn test_joint_probability_bell_odd_parity() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q0 = sim.allocate().expect("allocation should succeed");
    let q1 = sim.allocate().expect("allocation should succeed");

    sim.x(q1); // |01>
    sim.h(q0); // (|01> + |11>) / sqrt(2)
    sim.mcx(&[q0], q1); // (|01> + |10>) / sqrt(2)

    let p = sim
        .joint_probability(&[q0, q1])
        .expect("probability computation should succeed");
    assert_probability_approx(p, 1.0, 1e-6);
}

#[test]
fn test_joint_measure_collapses_correctly() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q0 = sim.allocate().expect("allocation should succeed");
    let q1 = sim.allocate().expect("allocation should succeed");
    sim.h(q0);
    sim.h(q1); // |++> = (|00> + |01> + |10> + |11>) / 2

    let result = sim
        .joint_measure(&[q0, q1])
        .expect("joint measurement should succeed");

    let (state, _) = sim.get_state().expect("get_state should succeed");
    let f = std::f64::consts::FRAC_1_SQRT_2;

    if result {
        // Odd parity survived: |01> and |10>, each with amplitude 1/sqrt(2)
        assert_eq!(state.len(), 2);
        for (idx, amp) in &state {
            let i = idx.to_u32_digits();
            let i_val = if i.is_empty() { 0 } else { i[0] };
            assert!(
                i_val == 1 || i_val == 2,
                "unexpected index {i_val} for odd parity"
            );
            assert_amplitude_approx(*amp, Complex64::new(f, 0.0), 1e-5);
        }
    } else {
        // Even parity survived: |00> and |11>, each with amplitude 1/sqrt(2)
        assert_eq!(state.len(), 2);
        for (idx, amp) in &state {
            let i = idx.to_u32_digits();
            let i_val = if i.is_empty() { 0 } else { i[0] };
            assert!(
                i_val == 0 || i_val == 3,
                "unexpected index {i_val} for even parity"
            );
            assert_amplitude_approx(*amp, Complex64::new(f, 0.0), 1e-5);
        }
    }
}

// ============================================================================
// State collapse verification
// ============================================================================

#[test]
fn test_collapse_plus_to_basis_state() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q = sim.allocate().expect("allocation should succeed");
    sim.h(q); // |+>

    let result = sim.measure(q).expect("measurement should succeed");

    let (state, _) = sim.get_state().expect("get_state should succeed");
    assert_eq!(
        state.len(),
        1,
        "collapsed state should have exactly one basis state"
    );

    let (idx, amp) = &state[0];
    if result {
        assert_eq!(*idx, BigUint::from(1u32));
    } else {
        assert_eq!(*idx, BigUint::from(0u32));
    }
    assert_amplitude_approx(*amp, Complex64::new(1.0, 0.0), 1e-5);
}

#[test]
fn test_bell_collapse_entangled_partner() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q0 = sim.allocate().expect("allocation should succeed");
    let q1 = sim.allocate().expect("allocation should succeed");

    sim.h(q0);
    sim.mcx(&[q0], q1); // (|00> + |11>) / sqrt(2)

    let r0 = sim.measure(q0).expect("measurement should succeed");

    let (state, _) = sim.get_state().expect("get_state should succeed");
    assert_eq!(
        state.len(),
        1,
        "Bell state collapses to a single basis state after measuring one qubit"
    );

    let (idx, amp) = &state[0];
    let expected_idx = if r0 { 3u32 } else { 0u32 }; // |11> or |00>
    assert_eq!(*idx, BigUint::from(expected_idx));
    assert_amplitude_approx(*amp, Complex64::new(1.0, 0.0), 1e-5);
}

// ============================================================================
// qubit_is_zero tests
// ============================================================================

#[test]
fn test_qubit_is_zero_on_zero_state() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q = sim.allocate().expect("allocation should succeed");
    assert!(
        sim.qubit_is_zero(q).expect("qubit_is_zero should succeed"),
        "|0> should be zero"
    );
}

#[test]
fn test_qubit_is_zero_on_one_state() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q = sim.allocate().expect("allocation should succeed");
    sim.x(q);
    assert!(
        !sim.qubit_is_zero(q).expect("qubit_is_zero should succeed"),
        "|1> should not be zero"
    );
}

#[test]
fn test_qubit_is_zero_after_round_trip() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q = sim.allocate().expect("allocation should succeed");
    sim.h(q);
    sim.h(q); // H*H = I, back to |0>
    assert!(
        sim.qubit_is_zero(q).expect("qubit_is_zero should succeed"),
        "|0> after H*H should be zero"
    );
}

// ============================================================================
// Release tests
// ============================================================================

#[test]
fn test_release_zero_qubit() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q = sim.allocate().expect("allocation should succeed");
    // Qubit is |0>. Release should succeed without measurement.
    sim.release(q);
    // Qubit ID should be recycled.
    let q2 = sim.allocate().expect("allocation should succeed");
    assert_eq!(q, q2, "released ID should be recycled");
}

#[test]
fn test_release_one_qubit() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q = sim.allocate().expect("allocation should succeed");
    sim.x(q); // |1>
    // Release should: measure (returns true), flip to |0>, deactivate.
    sim.release(q);
    let q2 = sim.allocate().expect("allocation should succeed");
    assert_eq!(q, q2, "released ID should be recycled");
    assert!(
        sim.qubit_is_zero(q2).expect("qubit_is_zero should succeed"),
        "recycled qubit should be in |0>"
    );
}

#[test]
fn test_release_and_reallocate() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q0 = sim.allocate().expect("allocation should succeed");
    let _q1 = sim.allocate().expect("allocation should succeed");
    sim.release(q0);
    let q2 = sim.allocate().expect("allocation should succeed"); // Should recycle q0's ID and bit position
    assert_eq!(q2, q0, "should recycle released ID");

    // Verify the recycled qubit is usable.
    sim.h(q2);
    assert!(
        !sim.qubit_is_zero(q2).expect("qubit_is_zero should succeed"),
        "|+> should not be identified as |0>"
    );
}

// ============================================================================
// get_state() tests
// ============================================================================

#[test]
fn test_get_state_simple() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q0 = sim.allocate().expect("allocation should succeed");
    let q1 = sim.allocate().expect("allocation should succeed");

    sim.h(q0);
    sim.mcx(&[q0], q1); // Bell state (|00> + |11>) / sqrt(2)

    let (state, num_qubits) = sim.get_state().expect("get_state should succeed");
    assert_eq!(num_qubits, 2);
    assert_eq!(state.len(), 2);

    let f = std::f64::consts::FRAC_1_SQRT_2;
    assert_eq!(state[0].0, BigUint::from(0u32));
    assert_amplitude_approx(state[0].1, Complex64::new(f, 0.0), 1e-6);
    assert_eq!(state[1].0, BigUint::from(3u32));
    assert_amplitude_approx(state[1].1, Complex64::new(f, 0.0), 1e-6);
}

#[test]
fn test_get_state_qubit_id_reordering() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");

    let q0 = sim.allocate().expect("allocation should succeed"); // ID 0, bit 0
    let q1 = sim.allocate().expect("allocation should succeed"); // ID 1, bit 1

    sim.release(q0); // ID 0 and bit 0 go to recycle pool

    let q2 = sim.allocate().expect("allocation should succeed"); // Gets recycled ID 0, recycled bit 0
    assert_eq!(q2, 0, "should recycle ID 0");

    // Now: qubit ID 0 (q2) -> bit 0, qubit ID 1 (q1) -> bit 1
    sim.x(q1); // Put qubit ID 1 in |1>

    // Internal: bit 1 set => raw index 2. Since ID 1 -> bit 1 (identity mapping),
    // the permuted index should also be 2 (bit 1 set = qubit ID 1 is |1>).
    let (state, num_qubits) = sim.get_state().expect("get_state should succeed");
    assert_eq!(num_qubits, 2);
    assert_eq!(state.len(), 1);
    assert_eq!(state[0].0, BigUint::from(2u32));
    assert_amplitude_approx(state[0].1, Complex64::new(1.0, 0.0), 1e-6);
}

#[test]
fn test_get_state_nontrivial_reordering() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");

    let q0 = sim.allocate().expect("allocation should succeed"); // ID 0 -> bit 0
    let q1 = sim.allocate().expect("allocation should succeed"); // ID 1 -> bit 1
    let q2 = sim.allocate().expect("allocation should succeed"); // ID 2 -> bit 2

    // Release q0 and q1
    sim.release(q0);
    sim.release(q1);

    // Reallocate (LIFO recycling)
    let _q3 = sim.allocate().expect("allocation should succeed"); // ID 1, bit 1
    let q4 = sim.allocate().expect("allocation should succeed"); // ID 0, bit 0

    // Scramble with swap_qubit_ids: swap the ID-to-bit mappings of ID 0 and ID 2
    sim.swap_qubit_ids(q4, q2); // After: ID 0 -> bit 2, ID 2 -> bit 0

    // Apply X to q4 (ID 0, now mapped to bit 2)
    sim.x(q4);

    // Internal state: bit 2 is set => raw index 4 (binary 100)
    // Permutation:
    //   (bit_pos=2, id=0): bit 2 of 4 = set => set bit 0 in output
    //   (bit_pos=1, id=1): bit 1 of 4 = clear => skip
    //   (bit_pos=0, id=2): bit 0 of 4 = clear => skip
    // Output index = 1 (only qubit ID 0 is |1>)

    let (state, num_qubits) = sim.get_state().expect("get_state should succeed");
    assert_eq!(num_qubits, 3);
    assert_eq!(state.len(), 1);
    assert_eq!(state[0].0, BigUint::from(1u32)); // Only qubit ID 0 is |1>
    assert_amplitude_approx(state[0].1, Complex64::new(1.0, 0.0), 1e-6);
}

#[test]
fn test_get_state_near_zero_filtering() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let _q = sim.allocate().expect("allocation should succeed");

    // |0> state: only one non-zero amplitude
    let (state, _) = sim.get_state().expect("get_state should succeed");
    assert_eq!(state.len(), 1, "should filter out zero amplitudes");
    assert_eq!(state[0].0, BigUint::from(0u32));
}

// ============================================================================
// dump() tests
// ============================================================================

#[test]
fn test_dump_zero_state() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let _q = sim.allocate().expect("allocation should succeed");

    let output = sim.dump().expect("dump should succeed");
    assert!(output.contains("1 qubits"), "dump should show qubit count");
    assert!(output.contains("|0>"), "dump should show |0> basis state");
}

#[test]
fn test_dump_bell_state() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q0 = sim.allocate().expect("allocation should succeed");
    let q1 = sim.allocate().expect("allocation should succeed");
    sim.h(q0);
    sim.mcx(&[q0], q1);

    let output = sim.dump().expect("dump should succeed");
    assert!(output.contains("2 qubits"), "dump should show qubit count");
    assert!(output.contains("|00>"), "dump should show |00>");
    assert!(output.contains("|11>"), "dump should show |11>");
}

// ============================================================================
// RNG determinism tests
// ============================================================================

#[test]
fn test_set_rng_seed_determinism() {
    let mut sim1 = qdk_gpu_sim::GpuQuantumSim::new(Some(123)).expect("GPU sim should init");
    let mut sim2 = qdk_gpu_sim::GpuQuantumSim::new(Some(123)).expect("GPU sim should init");

    let q1 = sim1.allocate().expect("allocation should succeed");
    let q2 = sim2.allocate().expect("allocation should succeed");

    let mut results1 = Vec::new();
    let mut results2 = Vec::new();

    for _ in 0..50 {
        sim1.h(q1);
        let r1 = sim1.measure(q1).expect("measurement should succeed");
        results1.push(r1);
        if r1 {
            sim1.x(q1);
        }

        sim2.h(q2);
        let r2 = sim2.measure(q2).expect("measurement should succeed");
        results2.push(r2);
        if r2 {
            sim2.x(q2);
        }
    }

    assert_eq!(results1, results2, "same seed should give same results");
}

#[test]
fn test_set_rng_seed_resets_sequence() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q = sim.allocate().expect("allocation should succeed");

    let mut results1 = Vec::new();
    for _ in 0..20 {
        sim.h(q);
        let r = sim.measure(q).expect("measurement should succeed");
        results1.push(r);
        if r {
            sim.x(q);
        }
    }

    // Reset seed and replay
    sim.set_rng_seed(42);

    let mut results2 = Vec::new();
    for _ in 0..20 {
        sim.h(q);
        let r = sim.measure(q).expect("measurement should succeed");
        results2.push(r);
        if r {
            sim.x(q);
        }
    }

    assert_eq!(
        results1, results2,
        "resetting seed should replay same sequence"
    );
}
