#![cfg(feature = "gpu-tests")]

/// H-02: Allocating after gates preserves existing state.
///
/// Regression test: previously, allocating a new qubit after applying gates
/// would reinitialize the state vector to |0...0>, destroying the quantum state.
#[test]
fn allocate_after_gates_preserves_state() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");

    let q0 = sim.allocate().expect("allocation should succeed");
    sim.h(q0); // |+> = (|0> + |1>) / sqrt(2)

    // Allocate a second qubit. This MUST NOT destroy q0's superposition.
    let q1 = sim.allocate().expect("allocation should succeed");

    // Expected state: (|00> + |10>) / sqrt(2)
    // q0 is in |+>, q1 is in |0>.
    let (state, num_qubits) = sim.get_state().expect("get_state should succeed");
    assert_eq!(num_qubits, 2);
    assert_eq!(state.len(), 2, "should have two non-zero amplitudes");

    let f = std::f64::consts::FRAC_1_SQRT_2;
    for (idx, amp) in &state {
        let i: u64 = idx.try_into().expect("index fits u64");
        // q0 is bit 0, q1 is bit 1. |+0> means bit 0 can be 0 or 1, bit 1 is 0.
        // So indices are 0 (|00>) and 1 (|01> in internal bit ordering = qubit 0 is |1>).
        assert!(i == 0 || i == 1, "unexpected basis state {i}");
        assert!(
            (amp.re - f).abs() < 1e-5 && amp.im.abs() < 1e-5,
            "amplitude mismatch at |{i}>: ({}, {})",
            amp.re,
            amp.im,
        );
    }

    // Further check: the new qubit should be in |0>.
    assert!(
        sim.qubit_is_zero(q1).expect("qubit_is_zero should succeed"),
        "newly allocated qubit should be |0>"
    );
}

/// H-02: Multi-qubit state is preserved across multiple allocations.
#[test]
fn allocate_preserves_entangled_state() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");

    let q0 = sim.allocate().expect("allocation should succeed");
    let q1 = sim.allocate().expect("allocation should succeed");
    sim.h(q0);
    sim.mcx(&[q0], q1); // Bell state: (|00> + |11>) / sqrt(2)

    // Allocate a third qubit.
    let q2 = sim.allocate().expect("allocation should succeed");

    // The Bell pair should still be entangled.
    // Measure q0, q1 should agree.
    let r0 = sim.measure(q0).expect("measurement should succeed");
    let r1 = sim.measure(q1).expect("measurement should succeed");
    assert_eq!(
        r0, r1,
        "Bell pair should remain correlated after allocation"
    );

    // q2 should be |0>.
    assert!(
        sim.qubit_is_zero(q2).expect("qubit_is_zero should succeed"),
        "newly allocated qubit should be |0>"
    );
}

/// H-03: Deterministic measurement always collapses state.
///
/// After measuring a qubit in a deterministic state (P(|1>) ~ 0 or ~ 1),
/// the state vector should be properly projected. Verify by checking that
/// the measured qubit's probability is exactly 0 or 1 after measurement.
#[test]
fn deterministic_measure_collapses_state() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q = sim.allocate().expect("allocation should succeed");

    // |0> state: P(|1>) should be ~0. Measure should return false.
    let result = sim.measure(q).expect("measurement should succeed");
    assert!(!result, "|0> should measure as false");

    // After collapse, state should be exactly |0> (no residual amplitude).
    let (state, _) = sim.get_state().expect("get_state should succeed");
    assert_eq!(state.len(), 1, "should have exactly one basis state");
    assert!(
        (state[0].1.re - 1.0).abs() < 1e-6,
        "amplitude should be exactly 1.0"
    );
}

/// H-03: Deterministic measurement of |1> collapses correctly.
#[test]
fn deterministic_measure_one_collapses() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q = sim.allocate().expect("allocation should succeed");
    sim.x(q); // |1>

    let result = sim.measure(q).expect("measurement should succeed");
    assert!(result, "|1> should measure as true");

    // After collapse, P(|1>) should be exactly 1.0.
    let p = sim
        .joint_probability(&[q])
        .expect("probability computation should succeed");
    assert!(
        (p - 1.0).abs() < 1e-6,
        "P(|1>) should be 1.0 after collapse, got {p}"
    );
}

/// H-01: Rotation gate precision at small angles.
///
/// For very small angles, the old code lost precision by truncating the angle
/// to f32 before computing trig. Verify that Rx(epsilon) applied many times
/// accumulates correctly.
#[test]
fn rotation_precision_small_angle() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q = sim.allocate().expect("allocation should succeed");

    // Apply Rx(pi/1000) 1000 times. Should approximate Rx(pi) = -iX.
    let n = 1000;
    let theta = std::f64::consts::PI / f64::from(n);
    for _ in 0..n {
        sim.rx(theta, q);
    }

    // After Rx(pi), |0> -> cos(pi/2)|0> - i*sin(pi/2)|1> = -i|1>
    // So P(|1>) should be ~1.0.
    let p = sim
        .joint_probability(&[q])
        .expect("probability computation should succeed");
    assert!(
        (p - 1.0).abs() < 0.01,
        "1000x Rx(pi/1000) should approximate Rx(pi), got P(|1>)={p}"
    );
}

/// L-02: First allocate initializes correctly.
#[test]
fn first_allocate_initializes_correctly() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q = sim.allocate().expect("allocation should succeed");
    assert!(
        sim.qubit_is_zero(q).expect("qubit_is_zero should succeed"),
        "first allocated qubit should be |0>"
    );
}

/// H-04: Control-target overlap panics in release builds.
#[test]
#[should_panic(expected = "target qubit must not also be a control")]
fn control_target_overlap_panics() {
    let mut sim = qdk_gpu_sim::GpuQuantumSim::new(Some(42)).expect("GPU sim should init");
    let q0 = sim.allocate().expect("allocation should succeed");
    // Use q0 as both control and target -- should panic.
    sim.mcx(&[q0], q0);
}
