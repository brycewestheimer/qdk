pub mod error;
pub mod gates;
pub mod gpu;
pub mod measurement;
pub mod precision;
#[cfg(feature = "python")]
mod python;
pub mod qubit_map;

use num_bigint::BigUint;
use num_complex::Complex64;
use num_traits::Zero;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::error::GpuSimError;
use crate::gates::Mat2x2;
use crate::gpu::device::GpuDevice;
use crate::gpu::dispatch;
use crate::gpu::pipeline::PipelineCache;
use crate::gpu::state_buffer::StateBuffer;
use crate::measurement::MeasurementEngine;
use crate::precision::Precision;
use crate::qubit_map::QubitMap;

/// Probability threshold for treating a measurement outcome as deterministic.
///
/// If `P(|1>)` is below this threshold, the outcome is treated as certainly |0>.
/// If `P(|1>)` is above `1.0 - DETERMINISTIC_THRESHOLD`, the outcome is treated
/// as certainly |1>. In either case, the collapse shader is still dispatched to
/// ensure the state vector is properly projected and normalized.
const DETERMINISTIC_THRESHOLD: f64 = 1e-10;

/// Probability threshold for `qubit_is_zero()`.
///
/// A qubit is considered to be in |0> if `P(|1>)` is below this threshold.
/// This is intentionally looser than [`DETERMINISTIC_THRESHOLD`] because
/// `qubit_is_zero` is a non-destructive query that does not collapse state.
const QUBIT_ZERO_THRESHOLD: f64 = 1e-6;

/// Amplitude threshold for `get_state()` filtering.
///
/// Amplitudes with `|amplitude|^2 <= AMPLITUDE_FILTER_THRESHOLD` are excluded
/// from the sparse state representation returned by `get_state()`.
const AMPLITUDE_FILTER_THRESHOLD: f64 = 1e-10;

#[cfg(not(feature = "f64_emulation"))]
type ActivePrecision = precision::F32Precision;
#[cfg(feature = "f64_emulation")]
type ActivePrecision = precision::F64EmulatedPrecision;

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
    measurement_engine: MeasurementEngine,
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

        #[cfg(feature = "f64_emulation")]
        verify_fma_is_fused(gpu.device(), gpu.queue())?;

        let state = StateBuffer::new(&gpu)?;
        let pipelines = PipelineCache::new(gpu.device());
        let qubit_map = QubitMap::new();
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let measurement_engine =
            MeasurementEngine::new(gpu.device(), measurement::MAX_MEASUREMENT_WORKGROUPS);
        Ok(Self {
            gpu,
            state,
            pipelines,
            qubit_map,
            rng,
            measurement_engine,
        })
    }

    /// Allocates a new qubit in the |0> state and returns its ID.
    ///
    /// The first call initializes the state vector to |0>. Subsequent calls
    /// grow the state vector by one qubit (in the |0> state), preserving
    /// existing quantum state via GPU buffer copy.
    pub fn allocate(&mut self) -> usize {
        let id = self.qubit_map.allocate();

        match self.qubit_map.max_bit() {
            None => {
                // This shouldn't happen -- we just allocated, so max_bit >= Some(0).
                // But handle defensively.
                id
            }
            Some(max_bit) => {
                #[allow(clippy::cast_possible_truncation)]
                let required_qubits = (max_bit + 1) as u32;
                if self.state.num_qubits() == 0 {
                    // First allocation: initialize to |0...0>.
                    self.state
                        .ensure_capacity(self.gpu.device(), required_qubits)
                        .expect("GPU should have capacity for requested qubits");
                    self.state.initialize(self.gpu.queue(), required_qubits);
                } else if required_qubits > self.state.num_qubits() {
                    // Growth: preserve existing state, tensor in |0> for new qubit.
                    self.state
                        .grow_preserving_state(self.gpu.device(), self.gpu.queue(), required_qubits)
                        .expect("GPU should have capacity for requested qubits");
                }
                // If required_qubits <= num_qubits, a recycled bit position is being
                // reused and no buffer growth is needed.
                id
            }
        }
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
        #[cfg(not(feature = "f64_emulation"))]
        self.dispatch_single_qubit(target, &crate::gates::rx(theta));

        #[cfg(feature = "f64_emulation")]
        {
            let bit = self.resolve_bit(target);
            dispatch::dispatch_single_qubit_gate_f64(
                self.gpu.device(),
                self.gpu.queue(),
                &self.pipelines,
                &self.state,
                &crate::gates::rx_f64(theta),
                bit,
                self.state.num_qubits(),
            );
        }
    }

    /// Applies the Ry rotation gate to the target qubit.
    pub fn ry(&mut self, theta: f64, target: usize) {
        #[cfg(not(feature = "f64_emulation"))]
        self.dispatch_single_qubit(target, &crate::gates::ry(theta));

        #[cfg(feature = "f64_emulation")]
        {
            let bit = self.resolve_bit(target);
            dispatch::dispatch_single_qubit_gate_f64(
                self.gpu.device(),
                self.gpu.queue(),
                &self.pipelines,
                &self.state,
                &crate::gates::ry_f64(theta),
                bit,
                self.state.num_qubits(),
            );
        }
    }

    /// Applies the Rz rotation gate to the target qubit.
    pub fn rz(&mut self, theta: f64, target: usize) {
        #[cfg(not(feature = "f64_emulation"))]
        self.dispatch_single_qubit(target, &crate::gates::rz(theta));

        #[cfg(feature = "f64_emulation")]
        {
            let bit = self.resolve_bit(target);
            dispatch::dispatch_single_qubit_gate_f64(
                self.gpu.device(),
                self.gpu.queue(),
                &self.pipelines,
                &self.state,
                &crate::gates::rz_f64(theta),
                bit,
                self.state.num_qubits(),
            );
        }
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
        #[cfg(not(feature = "f64_emulation"))]
        self.dispatch_mc_gate(ctls, target, &crate::gates::rz(theta));

        #[cfg(feature = "f64_emulation")]
        self.dispatch_mc_gate_f64(ctls, target, &crate::gates::rz_f64(theta));
    }

    /// Multi-controlled phase gate: `diag(1, phase)` with controls.
    ///
    /// `phase` is a complex number. The gate applies `diag(1, phase)` to the
    /// target qubit when all control qubits are |1>.
    pub fn mcphase(&mut self, ctls: &[usize], phase: Complex64, target: usize) {
        #[cfg(not(feature = "f64_emulation"))]
        {
            let mat = crate::gates::phase_gate(phase.re, phase.im);
            self.dispatch_mc_gate(ctls, target, &mat);
        }

        #[cfg(feature = "f64_emulation")]
        {
            let mat = crate::gates::phase_gate_f64(phase.re, phase.im);
            self.dispatch_mc_gate_f64(ctls, target, &mat);
        }
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
    // Measurement
    // ========================================================================

    /// Measures a single qubit in the computational basis.
    ///
    /// Returns `true` if the qubit collapsed to |1>, `false` if |0>.
    /// The state vector is always projected and renormalized in-place on the GPU,
    /// including for deterministic outcomes.
    ///
    /// # Measurement pipeline
    ///
    /// 1. Look up the qubit's internal bit position from the qubit map.
    /// 2. Dispatch the measurement shader to compute `P(qubit = |1>)`.
    /// 3. Determine the outcome:
    ///    - `P < DETERMINISTIC_THRESHOLD`: outcome is |0> (deterministic).
    ///    - `P > 1 - DETERMINISTIC_THRESHOLD`: outcome is |1> (deterministic).
    ///    - Otherwise: sample from Born rule distribution using the RNG.
    /// 4. Dispatch the collapse shader to project and renormalize the state.
    pub fn measure(&mut self, id: usize) -> bool {
        let bit = self
            .qubit_map
            .bit_position(id)
            .expect("qubit should be allocated");
        #[allow(clippy::cast_possible_truncation)]
        let measure_mask = 1u32 << bit;
        let num_qubits = self.state.num_qubits();

        let p1 = self
            .measurement_engine
            .compute_probability(
                self.gpu.device(),
                self.gpu.queue(),
                &self.pipelines,
                &self.state,
                measure_mask,
                num_qubits,
            )
            .expect("probability computation should succeed");

        // Determine outcome. For deterministic cases, we still dispatch collapse
        // to ensure the state vector is properly projected and normalized.
        let result = if p1 < DETERMINISTIC_THRESHOLD {
            false
        } else if p1 > 1.0 - DETERMINISTIC_THRESHOLD {
            true
        } else {
            // Probabilistic case: sample from Born rule distribution.
            let random: f64 = f64::from(self.rng.r#gen::<f32>());
            random < p1
        };

        // Always collapse: zero inconsistent amplitudes, renormalize consistent ones.
        let probability = if result { p1 } else { 1.0 - p1 };
        self.measurement_engine.collapse_state(
            self.gpu.device(),
            self.gpu.queue(),
            &self.pipelines,
            &self.state,
            measure_mask,
            result,
            probability,
            num_qubits,
        );

        result
    }

    /// Performs a joint parity measurement on multiple qubits.
    ///
    /// Returns `true` if the measured parity is odd, `false` if even.
    /// The state vector is always collapsed to the subspace with matching parity,
    /// including for deterministic outcomes.
    ///
    /// For a single qubit, this is equivalent to `measure()`.
    ///
    /// The parity is defined as: for basis state |i>, the parity of the measured
    /// qubits is `countOneBits(i & measure_mask) mod 2`. If odd, it contributes
    /// to `P(odd parity)`.
    pub fn joint_measure(&mut self, ids: &[usize]) -> bool {
        let measure_mask = self.build_measure_mask(ids);
        let num_qubits = self.state.num_qubits();

        let p_odd = self
            .measurement_engine
            .compute_probability(
                self.gpu.device(),
                self.gpu.queue(),
                &self.pipelines,
                &self.state,
                measure_mask,
                num_qubits,
            )
            .expect("probability computation should succeed");

        let result = if p_odd < DETERMINISTIC_THRESHOLD {
            false
        } else if p_odd > 1.0 - DETERMINISTIC_THRESHOLD {
            true
        } else {
            let random: f64 = f64::from(self.rng.r#gen::<f32>());
            random < p_odd
        };

        let probability = if result { p_odd } else { 1.0 - p_odd };
        self.measurement_engine.collapse_state(
            self.gpu.device(),
            self.gpu.queue(),
            &self.pipelines,
            &self.state,
            measure_mask,
            result,
            probability,
            num_qubits,
        );

        result
    }

    /// Computes the probability of a joint measurement yielding odd parity,
    /// WITHOUT collapsing the state.
    ///
    /// Returns `f64` for API compatibility with `QuantumSim`, even though
    /// the internal computation uses f32.
    pub fn joint_probability(&mut self, ids: &[usize]) -> f64 {
        let measure_mask = self.build_measure_mask(ids);
        let num_qubits = self.state.num_qubits();

        self.measurement_engine
            .compute_probability(
                self.gpu.device(),
                self.gpu.queue(),
                &self.pipelines,
                &self.state,
                measure_mask,
                num_qubits,
            )
            .expect("probability computation should succeed")
    }

    /// Returns `true` if the qubit is in the |0> state (within tolerance).
    ///
    /// Computes `P(qubit = |1>)` and checks if it is below
    /// [`QUBIT_ZERO_THRESHOLD`]. Does NOT collapse the state.
    pub fn qubit_is_zero(&mut self, id: usize) -> bool {
        let bit = self
            .qubit_map
            .bit_position(id)
            .expect("qubit should be allocated");
        #[allow(clippy::cast_possible_truncation)]
        let measure_mask = 1u32 << bit;
        let num_qubits = self.state.num_qubits();

        let p1: f64 = self
            .measurement_engine
            .compute_probability(
                self.gpu.device(),
                self.gpu.queue(),
                &self.pipelines,
                &self.state,
                measure_mask,
                num_qubits,
            )
            .expect("probability computation should succeed");

        p1 < QUBIT_ZERO_THRESHOLD
    }

    // ========================================================================
    // State management
    // ========================================================================

    /// Releases a qubit, returning its ID and bit position to the recycling pool.
    ///
    /// The qubit is always measured and, if in |1>, flipped to |0> before release.
    /// This ensures the bit position is in a clean state for reuse.
    ///
    /// Release does NOT shrink the state vector. The dense simulator's state
    /// vector only grows; released bit positions become semantically inactive.
    pub fn release(&mut self, id: usize) {
        // Always measure and reset. This is simpler and more correct than the
        // previous approach of checking qubit_is_zero() first, because:
        // 1. It avoids a semantic inconsistency where |0> qubits skip collapse.
        // 2. The cost of one extra GPU dispatch is negligible for correctness.
        if self.measure(id) {
            self.x(id);
        }
        self.qubit_map
            .release(id)
            .expect("qubit should be allocated for release");
    }

    /// Resets the RNG to a deterministic seed.
    ///
    /// This affects all future measurements. Combined with a fixed initial
    /// state, this enables exact replay of measurement sequences.
    pub fn set_rng_seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }

    // ========================================================================
    // State readout
    // ========================================================================

    /// Reads back the quantum state from the GPU and returns a sparse
    /// representation with qubit-ID-ordered indices.
    ///
    /// # Returns
    ///
    /// `(Vec<(BigUint, Complex64)>, usize)`:
    /// - The vector contains `(basis_state_index, amplitude)` pairs for all
    ///   amplitudes with `|amplitude|^2 > 1e-10`.
    /// - The `usize` is the number of active qubits.
    ///
    /// # Index ordering
    ///
    /// Bit `k` in each returned basis state index corresponds to qubit ID `k`
    /// (NOT internal bit position `k`). The permutation from internal bit
    /// positions to qubit IDs is computed from [`QubitMap::active_mappings`].
    pub fn get_state(&self) -> Result<(Vec<(BigUint, Complex64)>, usize), GpuSimError> {
        let raw_floats = self.state.readback(self.gpu.device(), self.gpu.queue())?;
        let threshold = AMPLITUDE_FILTER_THRESHOLD;
        let floats_per_amp = ActivePrecision::FLOATS_PER_AMPLITUDE as usize;

        // Get the mapping: Vec<(bit_position, qubit_id)> for all active qubits.
        let active_mappings = self.qubit_map.active_mappings();

        let mut result: Vec<(BigUint, Complex64)> = Vec::new();

        for (raw_idx, chunk) in raw_floats.chunks_exact(floats_per_amp).enumerate() {
            let amplitude = ActivePrecision::decode_complex(chunk);
            if amplitude.norm_sqr() <= threshold {
                continue;
            }

            // Remap bits: for each active qubit, check if its internal bit
            // position is set in the raw index, and if so, set the qubit ID
            // bit in the output index.
            let mut permuted_idx = BigUint::zero();
            for &(bit_pos, qubit_id) in &active_mappings {
                if raw_idx & (1 << bit_pos) != 0 {
                    permuted_idx.set_bit(qubit_id as u64, true);
                }
            }

            result.push((permuted_idx, amplitude));
        }

        result.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        Ok((result, self.qubit_map.active_count()))
    }

    /// Returns a human-readable dump of the current quantum state.
    ///
    /// Reads back the state vector from the GPU, filters near-zero amplitudes,
    /// and formats each surviving basis state with its amplitude and probability.
    pub fn dump(&self) -> Result<String, GpuSimError> {
        let (state, num_qubits) = self.get_state()?;
        let mut lines = Vec::new();
        lines.push(format!("STATE ({num_qubits} qubits):"));
        for (idx, amp) in &state {
            let sign = if amp.im >= 0.0 { "+" } else { "-" };
            lines.push(format!(
                "  |{:0width$b}>: {:.6} {sign} {:.6}i  (p = {:.6})",
                idx,
                amp.re,
                amp.im.abs(),
                amp.norm_sqr(),
                width = num_qubits,
            ));
        }
        Ok(lines.join("\n"))
    }

    // ========================================================================
    // Info accessors
    // ========================================================================

    /// Returns information about the GPU adapter in use.
    #[must_use]
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        self.gpu.adapter_info()
    }

    /// Returns the maximum number of qubits that can be simulated.
    ///
    /// In f32 mode this is based on 8 bytes per amplitude. In f64 emulation
    /// mode, each amplitude requires 16 bytes, reducing the maximum by ~1 qubit.
    #[must_use]
    pub fn max_qubits(&self) -> u32 {
        self.gpu.max_qubits(ActivePrecision::BYTES_PER_AMPLITUDE)
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

    /// Builds a measurement bitmask from a slice of qubit IDs.
    ///
    /// Each qubit ID is resolved to its internal bit position, and that bit
    /// is set in the returned mask.
    fn build_measure_mask(&self, ids: &[usize]) -> u32 {
        let mut mask = 0u32;
        for &id in ids {
            mask |= 1u32 << self.resolve_bit(id);
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
            assert_eq!(
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

    /// Dispatch routing for multi-controlled gates with f64 precision matrices.
    ///
    /// Used by rotation and phase gates in the f64-emulation path to avoid
    /// f32 intermediate truncation.
    #[cfg(feature = "f64_emulation")]
    fn dispatch_mc_gate_f64(
        &mut self,
        ctls: &[usize],
        target: usize,
        matrix: &crate::gates::Mat2x2F64,
    ) {
        let target_bit = self.resolve_bit(target);
        let n = self.state.num_qubits();

        if ctls.is_empty() {
            dispatch::dispatch_single_qubit_gate_f64(
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
            assert_eq!(
                control_mask & (1 << target_bit),
                0,
                "target qubit must not also be a control"
            );
            dispatch::dispatch_multi_controlled_gate_f64(
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

/// Runtime FMA self-test: verifies that GPU `fma()` is truly fused.
///
/// DS multiplication relies on `fma(a, b, -a*b)` returning the exact rounding
/// error. If the GPU decomposes `fma` into separate multiply+add, this error
/// term becomes zero and f64 emulation degrades to f32 precision.
///
/// This test is called once during `GpuQuantumSim::new()` when `f64_emulation`
/// is active. It logs a warning if `fma` is not fused but does not hard-fail.
#[cfg(feature = "f64_emulation")]
#[allow(clippy::too_many_lines)]
fn verify_fma_is_fused(device: &wgpu::Device, queue: &wgpu::Queue) -> Result<(), GpuSimError> {
    const FMA_TEST_SHADER: &str = r"
@group(0) @binding(0)
var<storage, read_write> result: f32;

@compute @workgroup_size(1)
fn main() {
    // 1.0 + 2^-23 (smallest value where (a*a) loses the low bit)
    let a: f32 = 1.0 + 0.00000011920928955078125;
    let p: f32 = a * a;
    result = fma(a, a, -p);
}
";

    let result_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fma_test_result"),
        size: 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("fma_test_staging"),
        size: 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("fma_test_shader"),
        source: wgpu::ShaderSource::Wgsl(FMA_TEST_SHADER.into()),
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("fma_test_bgl"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("fma_test_pl"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("fma_test_pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fma_test_bg"),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: result_buffer.as_entire_binding(),
        }],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("fma_test_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fma_test_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging_buffer, 0, 4);
    queue.submit(std::iter::once(encoder.finish()));

    let buffer_slice = staging_buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|_| GpuSimError::BufferMapFailed)?;
    rx.recv()
        .map_err(|_| GpuSimError::BufferMapFailed)?
        .map_err(|e| GpuSimError::DeviceError(format!("FMA test readback failed: {e}")))?;

    let data = buffer_slice.get_mapped_range();
    let result_val = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    drop(data);
    staging_buffer.unmap();

    if result_val == 0.0 {
        log::warn!(
            "GPU fma() does not appear to be a true fused multiply-add. \
             f64 emulation precision may be degraded to f32 levels."
        );
    } else {
        log::info!("GPU fma() verified as true fused multiply-add (result: {result_val:e})");
    }
    Ok(())
}
