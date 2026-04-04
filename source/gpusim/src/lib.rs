pub mod error;
pub mod gates;
pub mod gpu;
pub mod qubit_map;

use num_bigint::BigUint;
use num_complex::Complex64;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::error::GpuSimError;
use crate::gates::Mat2x2;
use crate::gpu::device::GpuDevice;
use crate::gpu::dispatch;
use crate::gpu::pipeline::PipelineCache;
use crate::gpu::state_buffer::StateBuffer;
use crate::qubit_map::QubitMap;

/// A GPU-accelerated dense state vector quantum simulator.
///
/// Uses wgpu compute shaders to perform gate operations on a state vector
/// that resides entirely on the GPU. The state vector is only copied to the
/// CPU when explicitly requested via [`get_state`](Self::get_state).
///
/// GPU buffers start small and grow dynamically as qubits are allocated.
/// Memory limits are always checked before allocation to prevent crashes.
pub struct GpuQuantumSim {
    gpu: GpuDevice,
    state: StateBuffer,
    pipelines: PipelineCache,
    qubit_map: QubitMap,
    rng: StdRng,
}

impl GpuQuantumSim {
    /// Creates a new GPU quantum simulator.
    ///
    /// Discovers and initializes the best available GPU, allocates a small
    /// initial state buffer, and compiles compute shaders.
    ///
    /// # Arguments
    /// * `seed` - Optional RNG seed for deterministic simulation. If `None`,
    ///   the RNG is seeded from system entropy.
    pub fn new(seed: Option<u64>) -> Result<Self, GpuSimError> {
        let gpu = GpuDevice::new()?;
        let state = StateBuffer::new(&gpu)?;
        let pipelines = PipelineCache::new(gpu.device());
        let qubit_map = QubitMap::new();
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Ok(Self {
            gpu,
            state,
            pipelines,
            qubit_map,
            rng,
        })
    }

    /// Allocates a new qubit in the |0> state and returns its ID.
    ///
    /// The first call initializes the state vector to |0>. Subsequent calls
    /// grow the state vector by one qubit (in the |0> state) by re-initializing
    /// the buffer if the qubit count has increased.
    ///
    /// **Phase 1 limitation**: Allocating a qubit after gates have been applied
    /// will re-initialize the state vector, discarding previous gate operations.
    /// Allocate all qubits before applying any gates.
    pub fn allocate(&mut self) -> usize {
        let id = self.qubit_map.allocate();
        #[allow(clippy::cast_possible_truncation)]
        let required_qubits = (self.qubit_map.max_bit() + 1) as u32;
        if required_qubits > self.state.num_qubits() {
            self.state
                .ensure_capacity(self.gpu.device(), required_qubits)
                .expect("GPU should have capacity for requested qubits");
            self.state.initialize(self.gpu.queue(), required_qubits);
        }
        id
    }

    // ========================================================================
    // Single-qubit gates
    // ========================================================================

    /// Applies the Hadamard gate to the target qubit.
    ///
    /// # Panics
    /// Panics if `target` is not a valid allocated qubit ID.
    pub fn h(&mut self, target: usize) {
        self.dispatch_single_qubit(target, &crate::gates::H);
    }

    /// Applies the Pauli-X gate to the target qubit.
    pub fn x(&mut self, target: usize) {
        self.dispatch_single_qubit(target, &crate::gates::X);
    }

    /// Applies the Pauli-Y gate to the target qubit.
    pub fn y(&mut self, target: usize) {
        self.dispatch_single_qubit(target, &crate::gates::Y);
    }

    /// Applies the Pauli-Z gate to the target qubit.
    pub fn z(&mut self, target: usize) {
        self.dispatch_single_qubit(target, &crate::gates::Z);
    }

    /// Applies the S gate (sqrt(Z)) to the target qubit.
    pub fn s(&mut self, target: usize) {
        self.dispatch_single_qubit(target, &crate::gates::S);
    }

    /// Applies the S-adjoint gate to the target qubit.
    pub fn sadj(&mut self, target: usize) {
        self.dispatch_single_qubit(target, &crate::gates::SADJ);
    }

    /// Applies the T gate (sqrt(S)) to the target qubit.
    pub fn t(&mut self, target: usize) {
        self.dispatch_single_qubit(target, &crate::gates::T);
    }

    /// Applies the T-adjoint gate to the target qubit.
    pub fn tadj(&mut self, target: usize) {
        self.dispatch_single_qubit(target, &crate::gates::TADJ);
    }

    /// Applies the SX (sqrt(X)) gate to the target qubit.
    pub fn sx(&mut self, target: usize) {
        self.dispatch_single_qubit(target, &crate::gates::SX);
    }

    /// Applies the SX-adjoint gate to the target qubit.
    pub fn sxadj(&mut self, target: usize) {
        self.dispatch_single_qubit(target, &crate::gates::SXADJ);
    }

    /// Applies the Rx rotation gate to the target qubit.
    pub fn rx(&mut self, theta: f64, target: usize) {
        self.dispatch_single_qubit(target, &crate::gates::rx(theta));
    }

    /// Applies the Ry rotation gate to the target qubit.
    pub fn ry(&mut self, theta: f64, target: usize) {
        self.dispatch_single_qubit(target, &crate::gates::ry(theta));
    }

    /// Applies the Rz rotation gate to the target qubit.
    pub fn rz(&mut self, theta: f64, target: usize) {
        self.dispatch_single_qubit(target, &crate::gates::rz(theta));
    }

    // ========================================================================
    // Multi-controlled gates
    // ========================================================================

    /// Multi-controlled X gate.
    ///
    /// - 0 controls: X gate
    /// - 1 control: CNOT
    /// - 2 controls: Toffoli
    pub fn mcx(&mut self, ctls: &[usize], target: usize) {
        self.dispatch_mc_gate(ctls, target, &crate::gates::X);
    }

    /// Multi-controlled Y gate.
    pub fn mcy(&mut self, ctls: &[usize], target: usize) {
        self.dispatch_mc_gate(ctls, target, &crate::gates::Y);
    }

    /// Multi-controlled Z gate.
    pub fn mcz(&mut self, ctls: &[usize], target: usize) {
        self.dispatch_mc_gate(ctls, target, &crate::gates::Z);
    }

    /// Multi-controlled Hadamard gate.
    pub fn mch(&mut self, ctls: &[usize], target: usize) {
        self.dispatch_mc_gate(ctls, target, &crate::gates::H);
    }

    /// Multi-controlled S gate.
    pub fn mcs(&mut self, ctls: &[usize], target: usize) {
        self.dispatch_mc_gate(ctls, target, &crate::gates::S);
    }

    /// Multi-controlled S-adjoint gate.
    pub fn mcsadj(&mut self, ctls: &[usize], target: usize) {
        self.dispatch_mc_gate(ctls, target, &crate::gates::SADJ);
    }

    /// Multi-controlled T gate.
    pub fn mct(&mut self, ctls: &[usize], target: usize) {
        self.dispatch_mc_gate(ctls, target, &crate::gates::T);
    }

    /// Multi-controlled T-adjoint gate.
    pub fn mctadj(&mut self, ctls: &[usize], target: usize) {
        self.dispatch_mc_gate(ctls, target, &crate::gates::TADJ);
    }

    /// Multi-controlled Rz gate.
    pub fn mcrz(&mut self, ctls: &[usize], theta: f64, target: usize) {
        self.dispatch_mc_gate(ctls, target, &crate::gates::rz(theta));
    }

    /// Multi-controlled phase gate: diag(1, phase) with controls.
    ///
    /// `phase` is a complex number. The gate applies diag(1, phase) to the
    /// target qubit when all control qubits are |1>.
    pub fn mcphase(&mut self, ctls: &[usize], phase: Complex64, target: usize) {
        #[allow(clippy::cast_possible_truncation)]
        let mat = crate::gates::phase_gate(phase.re as f32, phase.im as f32);
        self.dispatch_mc_gate(ctls, target, &mat);
    }

    // ========================================================================
    // Two-qubit gates (native 4x4 kernel)
    // ========================================================================

    /// SWAP gate using the native two-qubit kernel.
    ///
    /// This is a physical gate that applies the 4x4 SWAP unitary to the state
    /// vector. It is NOT a qubit ID relabeling — see [`swap_qubit_ids`](Self::swap_qubit_ids)
    /// for that.
    pub fn swap(&mut self, q1: usize, q2: usize) {
        let bit_a = self.resolve_bit(q1);
        let bit_b = self.resolve_bit(q2);
        dispatch::dispatch_two_qubit_gate(
            self.gpu.device(),
            self.gpu.queue(),
            &self.pipelines,
            &self.state,
            &crate::gates::SWAP,
            bit_a,
            bit_b,
            self.state.num_qubits(),
        );
    }

    // ========================================================================
    // Qubit ID operations (CPU-only)
    // ========================================================================

    /// Swaps the qubit ID mapping for two qubits.
    ///
    /// This is a CPU-only operation — it relabels which user-facing qubit ID
    /// maps to which internal bit position. No GPU operation or state vector
    /// modification occurs. The state vector is unchanged; only future gate
    /// calls using these qubit IDs will resolve to swapped bit positions.
    ///
    /// This is distinct from [`swap`](Self::swap), which applies the physical
    /// SWAP gate to the state vector on the GPU.
    pub fn swap_qubit_ids(&mut self, qubit1: usize, qubit2: usize) {
        self.qubit_map
            .swap(qubit1, qubit2)
            .expect("both qubits should be allocated");
    }

    // ========================================================================
    // State readout
    // ========================================================================

    /// Reads back the current quantum state from the GPU.
    ///
    /// Returns a sparse representation: a vector of `(basis_state, amplitude)`
    /// pairs where the amplitude magnitude exceeds a threshold (1e-10), along
    /// with the total qubit count.
    ///
    /// The basis state indices use little-endian qubit ordering, matching the
    /// internal state vector layout.
    pub fn get_state(&self) -> Result<(Vec<(BigUint, Complex64)>, usize), GpuSimError> {
        let amplitudes = self.state.readback(self.gpu.device(), self.gpu.queue())?;
        let num_qubits = self.state.num_qubits() as usize;
        let threshold = 1e-10_f64;

        let sparse: Vec<(BigUint, Complex64)> = amplitudes
            .iter()
            .enumerate()
            .filter_map(|(i, &(re, im))| {
                let c = Complex64::new(f64::from(re), f64::from(im));
                if c.norm_sqr() > threshold {
                    Some((BigUint::from(i), c))
                } else {
                    None
                }
            })
            .collect();

        Ok((sparse, num_qubits))
    }

    // ========================================================================
    // Info accessors
    // ========================================================================

    /// Returns information about the GPU adapter in use.
    #[must_use]
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        self.gpu.adapter_info()
    }

    /// Returns the maximum number of qubits that can be simulated at f32 precision.
    #[must_use]
    pub fn max_qubits(&self) -> u32 {
        self.gpu.max_qubits(8) // 8 bytes per f32 complex amplitude
    }

    /// Returns the RNG, primarily for testing.
    #[must_use]
    pub fn rng(&self) -> &StdRng {
        &self.rng
    }

    // ========================================================================
    // Internal dispatch helpers
    // ========================================================================

    /// Resolves a user-facing qubit ID to a GPU bit position.
    fn resolve_bit(&self, id: usize) -> u32 {
        #[allow(clippy::cast_possible_truncation)]
        let bit = self
            .qubit_map
            .bit_position(id)
            .expect("qubit should be allocated") as u32;
        bit
    }

    /// Dispatches a single-qubit gate to the GPU.
    fn dispatch_single_qubit(&mut self, target: usize, matrix: &Mat2x2) {
        let bit = self.resolve_bit(target);
        dispatch::dispatch_single_qubit_gate(
            self.gpu.device(),
            self.gpu.queue(),
            &self.pipelines,
            &self.state,
            matrix,
            bit,
            self.state.num_qubits(),
        );
    }

    /// Builds a control bitmask from a slice of control qubit IDs.
    fn build_control_mask(&self, ctls: &[usize]) -> u32 {
        let mut mask = 0u32;
        for &ctl in ctls {
            mask |= 1 << self.resolve_bit(ctl);
        }
        mask
    }

    /// Dispatch routing for multi-controlled gates.
    ///
    /// - 0 controls: dispatches the single-qubit kernel directly.
    /// - 1+ controls: builds a `control_mask` and dispatches the multi-controlled kernel.
    fn dispatch_mc_gate(&mut self, ctls: &[usize], target: usize, matrix: &Mat2x2) {
        let target_bit = self.resolve_bit(target);
        let n = self.state.num_qubits();

        if ctls.is_empty() {
            dispatch::dispatch_single_qubit_gate(
                self.gpu.device(),
                self.gpu.queue(),
                &self.pipelines,
                &self.state,
                matrix,
                target_bit,
                n,
            );
        } else {
            let control_mask = self.build_control_mask(ctls);
            debug_assert_eq!(
                control_mask & (1 << target_bit),
                0,
                "target qubit must not also be a control"
            );
            dispatch::dispatch_multi_controlled_gate(
                self.gpu.device(),
                self.gpu.queue(),
                &self.pipelines,
                &self.state,
                matrix,
                target_bit,
                control_mask,
                n,
            );
        }
    }
}
