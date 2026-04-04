use crate::error::GpuSimError;
use crate::gpu::device::GpuDevice;

/// Bytes per complex amplitude at f32 precision: 2 x sizeof(f32) = 8.
const BYTES_PER_AMPLITUDE: u64 = 8;

/// Default initial capacity: 4 qubits = 16 amplitudes = 128 bytes.
const DEFAULT_INITIAL_QUBITS: u32 = 4;

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
        let max_buffer_size = gpu.max_buffer_size();
        let max_amplitudes = max_buffer_size / BYTES_PER_AMPLITUDE;

        let initial_amplitudes = 1u64 << DEFAULT_INITIAL_QUBITS;
        let initial_size = initial_amplitudes * BYTES_PER_AMPLITUDE;

        let buffer = gpu.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("state_vector"),
            size: initial_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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
        let required_amplitudes = 1u64 << num_qubits;

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
        let new_size = new_capacity * BYTES_PER_AMPLITUDE;

        self.buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("state_vector"),
            size: new_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
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

    /// Writes the |0...0> state into the buffer for the given qubit count.
    ///
    /// The first amplitude is set to (1.0, 0.0) and all others to (0.0, 0.0).
    /// Uses `queue.write_buffer` for the initial state upload.
    ///
    /// The caller must ensure the buffer has sufficient capacity via
    /// [`ensure_capacity`](Self::ensure_capacity) before calling this method.
    pub fn initialize(&mut self, queue: &wgpu::Queue, num_qubits: u32) {
        self.num_qubits = num_qubits;
        self.num_amplitudes = 1u64 << num_qubits;

        // Build CPU-side state: |0...0> = [(1.0, 0.0), (0.0, 0.0), ...]
        #[allow(clippy::cast_possible_truncation)]
        let byte_count = (self.num_amplitudes * BYTES_PER_AMPLITUDE) as usize;
        let mut data = vec![0u8; byte_count];
        // Write 1.0f32 into the first 4 bytes (real part of amplitude 0)
        data[..4].copy_from_slice(&1.0_f32.to_le_bytes());
        queue.write_buffer(&self.buffer, 0, &data);
    }

    /// Copies the state vector from GPU to CPU and returns the amplitudes
    /// as (re, im) pairs.
    ///
    /// Steps:
    /// 1. Copy from primary buffer to staging buffer (GPU-side).
    /// 2. Map the staging buffer for CPU read access.
    /// 3. Read and convert the raw bytes to `(f32, f32)` pairs.
    /// 4. Unmap the staging buffer.
    pub fn readback(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<Vec<(f32, f32)>, GpuSimError> {
        let active_size = self.num_amplitudes * BYTES_PER_AMPLITUDE;

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
            .map_err(|_| GpuSimError::BufferMapFailed)?;
        receiver
            .recv()
            .map_err(|_| GpuSimError::BufferMapFailed)?
            .map_err(|_| GpuSimError::BufferMapFailed)?;

        // Step 3: Read the data
        let data = buffer_slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let amplitudes: Vec<(f32, f32)> = floats
            .chunks_exact(2)
            .map(|chunk| (chunk[0], chunk[1]))
            .collect();

        // Step 4: Unmap
        drop(data);
        self.staging_buffer.unmap();

        Ok(amplitudes)
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
