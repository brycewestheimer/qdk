pub mod error;
pub mod gates;
pub mod gpu;
pub mod qubit_map;

use num_bigint::BigUint;
use num_complex::Complex64;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::error::GpuSimError;
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

    /// Applies the Hadamard gate to the target qubit.
    ///
    /// # Panics
    /// Panics if `target` is not a valid allocated qubit ID.
    pub fn h(&mut self, target: usize) {
        let bit = self
            .qubit_map
            .bit_position(target)
            .expect("target qubit should be allocated");
        #[allow(clippy::cast_possible_truncation)]
        let bit = bit as u32;
        dispatch::dispatch_single_qubit_gate(
            self.gpu.device(),
            self.gpu.queue(),
            &self.pipelines,
            &self.state,
            &crate::gates::H,
            bit,
            self.state.num_qubits(),
        );
    }

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
}
