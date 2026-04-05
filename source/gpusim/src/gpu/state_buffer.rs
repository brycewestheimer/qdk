use num_complex::Complex64;

use crate::ActivePrecision;
use crate::error::GpuSimError;
use crate::gpu::device::GpuDevice;
use crate::precision::Precision;

/// Default initial capacity: 4 qubits = 16 amplitudes = 128 bytes.
const DEFAULT_INITIAL_QUBITS: u32 = 4;

/// Standard usage flags for the primary state vector buffer.
///
/// `STORAGE`: bind as read-write storage in compute shaders.
/// `COPY_SRC`: source for readback copies to staging buffer.
/// `COPY_DST`: destination for state-preserving growth copies and `clear_buffer`.
const STATE_BUFFER_USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

/// GPU-resident state vector with staging buffer for CPU readback.
///
/// Buffers start small and grow dynamically via [`ensure_capacity`](Self::ensure_capacity)
/// as more qubits are allocated. Growth is capped at the GPU's maximum buffer size.
pub struct StateBuffer {
    /// Primary state vector buffer (read-write storage binding).
    buffer: wgpu::Buffer,
    /// Staging buffer for CPU readback (map-read, copy-dst).
    staging_buffer: wgpu::Buffer,
    /// Current number of active qubits.
    num_qubits: u32,
    /// Current number of amplitudes (`2^num_qubits`).
    num_amplitudes: u64,
    /// Current buffer capacity in amplitudes.
    capacity_amplitudes: u64,
    /// Maximum amplitudes the GPU can support (hard limit from device).
    max_amplitudes: u64,
}

impl StateBuffer {
    /// Allocates GPU buffers with a small initial capacity.
    ///
    /// The primary buffer is created with `STORAGE | COPY_SRC` usage.
    /// The staging buffer is created with `MAP_READ | COPY_DST` usage.
    ///
    /// Buffers start at a default capacity of 2^4 = 16 amplitudes (128 bytes).
    /// Use [`ensure_capacity`](Self::ensure_capacity) to grow before initializing
    /// a larger state vector.
    pub fn new(gpu: &GpuDevice) -> Result<Self, GpuSimError> {
        let max_state_bytes = gpu.max_state_bytes();
        let max_amplitudes = max_state_bytes / ActivePrecision::BYTES_PER_AMPLITUDE;

        let initial_amplitudes = 1u64 << DEFAULT_INITIAL_QUBITS;
        let initial_size = initial_amplitudes * ActivePrecision::BYTES_PER_AMPLITUDE;

        let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("state_vector"),
            size: initial_size,
            usage: STATE_BUFFER_USAGE,
            mapped_at_creation: false,
        });

        let staging_buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("state_vector_staging"),
            size: initial_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            buffer,
            staging_buffer,
            num_qubits: 0,
            num_amplitudes: 1,
            capacity_amplitudes: initial_amplitudes,
            max_amplitudes,
        })
    }

    /// Grows buffers if needed to hold `2^num_qubits` amplitudes.
    ///
    /// Returns `Ok(true)` if buffers were reallocated, `Ok(false)` if the
    /// existing capacity is sufficient. Returns an error if the requested
    /// size exceeds the GPU's maximum buffer size.
    ///
    /// **Existing state is NOT preserved.** The caller must reinitialize the
    /// state vector after a successful growth (i.e., when this returns `Ok(true)`).
    pub fn ensure_capacity(
        &mut self,
        device: &wgpu::Device,
        num_qubits: u32,
    ) -> Result<bool, GpuSimError> {
        // Guard against shift overflow: 1u64 << 64 panics in Rust.
        // For num_qubits >= 64 this is always too large; checked_shl returns None.
        let required_amplitudes = 1u64.checked_shl(num_qubits).unwrap_or(u64::MAX);

        if required_amplitudes > self.max_amplitudes {
            #[allow(clippy::cast_possible_truncation)]
            let max_qubits = if self.max_amplitudes == 0 {
                0u32
            } else {
                63 - self.max_amplitudes.leading_zeros()
            };
            return Err(GpuSimError::TooManyQubits {
                requested: num_qubits,
                max: max_qubits,
            });
        }

        if required_amplitudes <= self.capacity_amplitudes {
            return Ok(false);
        }

        // Doubling strategy, capped at max
        let mut new_capacity = self.capacity_amplitudes;
        while new_capacity < required_amplitudes {
            new_capacity = new_capacity.saturating_mul(2).min(self.max_amplitudes);
        }
        let new_size = new_capacity * ActivePrecision::BYTES_PER_AMPLITUDE;

        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("state_vector"),
            size: new_size,
            usage: STATE_BUFFER_USAGE,
            mapped_at_creation: false,
        });

        self.staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("state_vector_staging"),
            size: new_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.capacity_amplitudes = new_capacity;

        Ok(true)
    }

    /// Grows the state buffer to accommodate `new_num_qubits`, preserving
    /// existing state data.
    ///
    /// The existing `2^old_num_qubits` amplitudes are GPU-copied into the new
    /// buffer. The extension region (`2^old_num_qubits` through `2^new_num_qubits - 1`)
    /// is zero-filled, representing new qubits initialized in |0>.
    ///
    /// This is the correct way to grow the state vector after gates have been
    /// applied. Unlike [`initialize`], which overwrites the buffer with |0...0>,
    /// this method preserves the quantum state and tensors in |0> for the new qubit.
    ///
    /// # Errors
    ///
    /// Returns `GpuSimError::TooManyQubits` if the requested size exceeds GPU limits.
    ///
    /// # Panics
    ///
    /// Panics if `new_num_qubits <= self.num_qubits`.
    pub fn grow_preserving_state(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        new_num_qubits: u32,
    ) -> Result<(), GpuSimError> {
        assert!(
            new_num_qubits > self.num_qubits,
            "grow_preserving_state requires new_num_qubits ({new_num_qubits}) > current ({})",
            self.num_qubits,
        );

        let new_num_amplitudes = 1u64
            .checked_shl(new_num_qubits)
            .expect("new_num_qubits should be within validated range");
        let old_num_amplitudes = self.num_amplitudes;
        let old_active_size = old_num_amplitudes * ActivePrecision::BYTES_PER_AMPLITUDE;

        if new_num_amplitudes > self.max_amplitudes {
            #[allow(clippy::cast_possible_truncation)]
            let max_qubits = if self.max_amplitudes == 0 {
                0u32
            } else {
                63 - self.max_amplitudes.leading_zeros()
            };
            return Err(GpuSimError::TooManyQubits {
                requested: new_num_qubits,
                max: max_qubits,
            });
        }

        if new_num_amplitudes > self.capacity_amplitudes {
            // Need a bigger buffer. Compute new capacity with doubling strategy.
            let mut new_capacity = self.capacity_amplitudes;
            while new_capacity < new_num_amplitudes {
                new_capacity = new_capacity.saturating_mul(2).min(self.max_amplitudes);
            }
            let new_size = new_capacity * ActivePrecision::BYTES_PER_AMPLITUDE;

            let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("state_vector"),
                size: new_size,
                usage: STATE_BUFFER_USAGE,
                mapped_at_creation: false,
            });

            // Clear new buffer to zeros, then copy old state data over the beginning.
            // Commands in a single encoder execute in order: clear first, then copy.
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("grow_state_encoder"),
            });
            encoder.clear_buffer(&new_buffer, 0, None);
            encoder.copy_buffer_to_buffer(&self.buffer, 0, &new_buffer, 0, old_active_size);
            queue.submit(std::iter::once(encoder.finish()));

            self.buffer = new_buffer;
            self.capacity_amplitudes = new_capacity;

            self.staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("state_vector_staging"),
                size: new_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        } else {
            // Capacity sufficient. Clear only the extension region.
            let new_active_size = new_num_amplitudes * ActivePrecision::BYTES_PER_AMPLITUDE;
            let extension_size = new_active_size - old_active_size;

            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("grow_state_clear_extension"),
            });
            encoder.clear_buffer(&self.buffer, old_active_size, Some(extension_size));
            queue.submit(std::iter::once(encoder.finish()));
        }

        self.num_qubits = new_num_qubits;
        self.num_amplitudes = new_num_amplitudes;

        Ok(())
    }

    /// Writes the |0...0> state into the buffer for the given qubit count.
    ///
    /// The first amplitude is set to (1.0, 0.0) and all others to (0.0, 0.0).
    /// Uses `mapped_at_creation` to write the first amplitude directly into a
    /// zero-initialized buffer, avoiding a large CPU-side allocation.
    ///
    /// The caller must ensure the buffer has sufficient capacity via
    /// [`ensure_capacity`](Self::ensure_capacity) before calling this method.
    pub fn initialize(&mut self, device: &wgpu::Device, num_qubits: u32) {
        self.num_qubits = num_qubits;
        self.num_amplitudes = 1u64
            .checked_shl(num_qubits)
            .expect("num_qubits should be within validated range");

        // Create a new buffer with mapped_at_creation = true.
        // The GPU driver provides a zero-initialized mapping. We only need
        // to write the first amplitude (1.0 + 0.0i).
        let active_size = self.num_amplitudes * ActivePrecision::BYTES_PER_AMPLITUDE;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("state_vector"),
            size: active_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });

        // Write just the first amplitude into the mapped memory.
        {
            let mut mapping = buffer.slice(..active_size).get_mapped_range_mut();
            let one_encoded = ActivePrecision::encode_complex(Complex64::new(1.0, 0.0));
            for (i, &val) in one_encoded.iter().enumerate() {
                let start = i * 4;
                mapping[start..start + 4].copy_from_slice(&val.to_le_bytes());
            }
        }
        buffer.unmap();

        self.buffer = buffer;
    }

    /// Copies the state vector from GPU to CPU and returns the raw f32 data.
    ///
    /// In f32 mode, each amplitude is 2 consecutive f32s (re, im).
    /// In f64-emulated mode, each amplitude is 4 consecutive f32s
    /// (`re_hi`, `re_lo`, `im_hi`, `im_lo`).
    ///
    /// Steps:
    /// 1. Copy from primary buffer to staging buffer (GPU-side).
    /// 2. Map the staging buffer for CPU read access.
    /// 3. Read the raw f32 values.
    /// 4. Unmap the staging buffer.
    pub fn readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<f32>, GpuSimError> {
        let active_size = self.num_amplitudes * ActivePrecision::BYTES_PER_AMPLITUDE;

        // Step 1: GPU-side copy from primary to staging buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("readback_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &self.staging_buffer, 0, active_size);
        queue.submit(std::iter::once(encoder.finish()));

        // Step 2: Map the staging buffer
        let buffer_slice = self.staging_buffer.slice(..active_size);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| GpuSimError::DevicePollFailed(format!("{e}")))?;
        receiver
            .recv()
            .map_err(|_| GpuSimError::ChannelDisconnected)?
            .map_err(|e| GpuSimError::BufferMapRejected(format!("{e}")))?;

        // Step 3: Read the data
        let data = buffer_slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let result = floats.to_vec();

        // Step 4: Unmap
        drop(data);
        self.staging_buffer.unmap();

        Ok(result)
    }

    /// Returns a reference to the primary state buffer for bind group creation.
    #[must_use]
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Returns the current number of active qubits.
    #[must_use]
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }

    /// Returns the current number of amplitudes (`2^num_qubits`).
    #[must_use]
    pub fn num_amplitudes(&self) -> u64 {
        self.num_amplitudes
    }

    /// Returns the current buffer capacity in amplitudes.
    #[must_use]
    pub fn capacity_amplitudes(&self) -> u64 {
        self.capacity_amplitudes
    }

    /// Returns the maximum number of amplitudes the GPU can support.
    #[must_use]
    pub fn max_amplitudes(&self) -> u64 {
        self.max_amplitudes
    }
}
