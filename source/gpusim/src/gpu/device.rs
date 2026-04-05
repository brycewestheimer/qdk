use wgpu::{Adapter, Device, DeviceType, Queue};

use crate::error::GpuSimError;

/// Maximum number of qubits due to WGSL u32 indexing constraints.
///
/// Shaders index the state vector as `idx * 2u` (f32 mode) or `idx * 4u`
/// (f64 mode). The amplitude index itself is `1u << num_qubits`, which must
/// fit in a u32. Since `1u << 32` overflows u32, the hard cap is 31 qubits.
const MAX_QUBITS_U32_LIMIT: u32 = 31;

/// Holds the wgpu device, queue, and capability metadata.
pub struct GpuDevice {
    device: Device,
    queue: Queue,
    adapter_info: wgpu::AdapterInfo,
    max_buffer_size: u64,
    max_storage_buffer_binding_size: u64,
    max_workgroup_size: u32,
}

impl GpuDevice {
    /// Creates a new GPU device by discovering the best available adapter.
    ///
    /// Uses `futures::executor::block_on` to drive the async wgpu device
    /// request synchronously. Prefers discrete GPUs over integrated, and
    /// Vulkan/Metal over DX12.
    pub fn new() -> Result<Self, GpuSimError> {
        futures::executor::block_on(async { Self::new_async().await })
    }

    async fn new_async() -> Result<Self, GpuSimError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            flags: wgpu::InstanceFlags::from_env_or_default()
                | wgpu::InstanceFlags::ALLOW_UNDERLYING_NONCOMPLIANT_ADAPTER,
            ..Default::default()
        });

        let adapter = Self::select_adapter(&instance)?;
        let adapter_limits = adapter.limits();
        let adapter_info = adapter.get_info();

        log::info!(
            "Selected GPU adapter: {} ({:?}, {:?})",
            adapter_info.name,
            adapter_info.device_type,
            adapter_info.backend,
        );

        let required_limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            max_compute_workgroup_size_x: adapter_limits.max_compute_workgroup_size_x,
            ..Default::default()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("qdk-gpu-sim"),
                required_features: wgpu::Features::empty(),
                required_limits,
                memory_hints: wgpu::MemoryHints::Performance,
                ..Default::default()
            })
            .await?;

        // Register an error handler for uncaptured device errors (e.g., OOM,
        // device lost). These would otherwise be silently dropped.
        device.on_uncaptured_error(std::sync::Arc::new(|error| {
            log::error!("wgpu uncaptured device error: {error}");
        }));

        let device_limits = device.limits();

        Ok(Self {
            max_buffer_size: device_limits.max_buffer_size,
            max_storage_buffer_binding_size: u64::from(
                device_limits.max_storage_buffer_binding_size,
            ),
            max_workgroup_size: device_limits.max_compute_workgroup_size_x,
            adapter_info,
            device,
            queue,
        })
    }

    /// Selects the best available GPU adapter using the same scoring strategy
    /// as the existing QDK GPU simulator (`gpu_resources.rs`).
    ///
    /// Scoring priorities:
    /// 1. Discrete GPU (8) > Integrated GPU (4) > Other (0, filtered out)
    /// 2. Vulkan/Metal (2) > DX12 (1) > Other (0, filtered out)
    /// 3. Higher `max_compute_workgroup_storage_size` breaks ties
    /// 4. Higher `max_storage_buffer_binding_size` breaks further ties
    ///
    /// Requirements:
    /// - Must be a discrete or integrated GPU
    /// - Must support Vulkan, Metal, or DX12
    /// - Must have at least 16 KB compute workgroup storage
    /// - Must have at least 128 MB storage buffer binding size
    #[cfg(not(target_arch = "wasm32"))]
    fn select_adapter(instance: &wgpu::Instance) -> Result<Adapter, GpuSimError> {
        let adapters = instance.enumerate_adapters(wgpu::Backends::PRIMARY);

        let score_adapter = |adapter: &Adapter| -> (u32, u32, u32, u32) {
            let info = adapter.get_info();
            let device_score = match info.device_type {
                DeviceType::DiscreteGpu => 8,
                DeviceType::IntegratedGpu => 4,
                _ => 0,
            };
            let backend_score = match info.backend {
                wgpu::Backend::Vulkan | wgpu::Backend::Metal => 2,
                wgpu::Backend::Dx12 => 1,
                _ => 0,
            };
            let limits = adapter.limits();
            (
                device_score,
                backend_score,
                limits.max_compute_workgroup_storage_size,
                limits.max_storage_buffer_binding_size,
            )
        };

        adapters
            .into_iter()
            .filter(|a| {
                let score = score_adapter(a);
                score.0 > 0 // discrete or integrated
                    && score.1 > 0 // supported backend
                    && score.2 >= (1u32 << 14) // at least 16 KB workgroup storage
                    && score.3 >= (1u32 << 27) // at least 128 MB storage buffers
            })
            .max_by_key(score_adapter)
            .ok_or(GpuSimError::NoAdapter)
    }

    #[cfg(target_arch = "wasm32")]
    fn select_adapter(_instance: &wgpu::Instance) -> Result<Adapter, GpuSimError> {
        // WASM support requires request_adapter (async), not enumerate_adapters.
        // Reserved for future implementation.
        Err(GpuSimError::NoAdapter)
    }

    /// Returns a reference to the wgpu device.
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns a reference to the wgpu queue.
    #[must_use]
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Returns adapter info (name, backend, device type) for diagnostics.
    #[must_use]
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Returns the maximum buffer size supported by the device in bytes.
    #[must_use]
    pub fn max_buffer_size(&self) -> u64 {
        self.max_buffer_size
    }

    /// Returns the maximum state buffer size in bytes, accounting for both the
    /// general buffer size limit and the storage buffer binding size limit.
    ///
    /// The state vector is bound as a storage buffer, so its size is constrained
    /// by `min(max_buffer_size, max_storage_buffer_binding_size)`.
    #[must_use]
    pub fn max_state_bytes(&self) -> u64 {
        self.max_buffer_size
            .min(self.max_storage_buffer_binding_size)
    }

    /// Returns the maximum workgroup size in the X dimension.
    #[must_use]
    pub fn max_workgroup_size(&self) -> u32 {
        self.max_workgroup_size
    }

    /// Computes the maximum number of qubits that can be simulated given the
    /// GPU's state buffer limits and the bytes required per amplitude.
    ///
    /// For f32 complex, `bytes_per_amplitude` is 8 (two f32 values).
    /// For f64-emulated, `bytes_per_amplitude` is 16 (four f32 values).
    ///
    /// Returns `n` such that `2^n * bytes_per_amplitude <= max_state_bytes`.
    /// Also caps at 31 qubits due to WGSL u32 indexing constraints.
    #[must_use]
    pub fn max_qubits(&self, bytes_per_amplitude: u64) -> u32 {
        let max_state = self.max_state_bytes();
        let max_amplitudes = max_state / bytes_per_amplitude;
        if max_amplitudes == 0 {
            return 0;
        }
        // 63 - leading_zeros gives floor(log2(n)) for n > 0
        let from_memory = 63 - max_amplitudes.leading_zeros();
        from_memory.min(MAX_QUBITS_U32_LIMIT)
    }
}
