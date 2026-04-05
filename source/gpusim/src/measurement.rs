use wgpu::util::DeviceExt;

use crate::ActivePrecision;
use crate::error::GpuSimError;
#[cfg(not(feature = "f64_emulation"))]
use crate::gpu::dispatch::CollapseParams;
#[cfg(feature = "f64_emulation")]
use crate::gpu::dispatch::CollapseParamsF64;
use crate::gpu::dispatch::MeasureParams;
use crate::gpu::pipeline::PipelineCache;
use crate::gpu::state_buffer::StateBuffer;
use crate::precision::Precision;

/// Maximum number of workgroups dispatched for the measurement shader.
/// This caps the partial sums buffer size at 1024 * 4 bytes = 4 KB.
/// With 256 threads per workgroup and 1024 workgroups, the shader can
/// process 256 * 1024 = 262,144 amplitudes per grid-stride iteration.
/// Larger state vectors are handled by the grid-stride loop.
pub const MAX_MEASUREMENT_WORKGROUPS: u32 = 1024;

/// Maximum number of workgroups for any single-dimension dispatch.
/// Matches the WebGPU `maxComputeWorkgroupsPerDimension` minimum guarantee.
const MAX_DISPATCH_WORKGROUPS: u32 = 65535;

/// Orchestrates the GPU-CPU-GPU measurement round trip.
///
/// Owns pre-allocated GPU buffers for the measurement pipeline:
/// - `partial_sums_buffer`: workgroup-level partial sums (one f32 per workgroup)
/// - `partial_sums_staging`: staging buffer for CPU readback
///
/// Created once at simulator initialization time and reused for all measurements.
pub struct MeasurementEngine {
    /// GPU storage buffer for workgroup-level partial sums.
    partial_sums_buffer: wgpu::Buffer,
    /// Staging buffer for CPU readback of partial sums.
    partial_sums_staging: wgpu::Buffer,
    /// Number of workgroups to dispatch (capped at `MAX_MEASUREMENT_WORKGROUPS`).
    num_workgroups: u32,
}

impl MeasurementEngine {
    /// Creates a new measurement engine, allocating the partial sum buffers.
    ///
    /// `max_workgroups` is clamped to `MAX_MEASUREMENT_WORKGROUPS`.
    #[must_use]
    pub fn new(device: &wgpu::Device, max_workgroups: u32) -> Self {
        let num_workgroups = max_workgroups.min(MAX_MEASUREMENT_WORKGROUPS);
        // In f64 mode, each workgroup writes a DS pair (2 f32s) instead of 1.
        let floats_per_wg = u64::from(ActivePrecision::FLOATS_PER_PARTIAL_SUM);
        let buffer_size =
            u64::from(num_workgroups) * floats_per_wg * std::mem::size_of::<f32>() as u64;

        let partial_sums_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("measurement_partial_sums"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let partial_sums_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("measurement_partial_sums_staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            partial_sums_buffer,
            partial_sums_staging,
            num_workgroups,
        }
    }

    /// Computes the probability that the measured qubits have odd parity.
    ///
    /// For a single-qubit measurement with `measure_mask = 1 << bit`, this
    /// returns `P(qubit = |1>)`.
    ///
    /// Dynamically scales the workgroup count based on state size:
    /// `min(ceil(2^num_qubits / 256), self.num_workgroups)`. For small state
    /// vectors (< 256 * `num_workgroups` amplitudes), this avoids launching
    /// workgroups that would do no work.
    pub fn compute_probability(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cache: &PipelineCache,
        state_buffer: &StateBuffer,
        measure_mask: u32,
        num_qubits: u32,
    ) -> Result<f64, GpuSimError> {
        // Step 1: Create the uniform buffer with measurement parameters.
        let params = MeasureParams {
            target_bit: measure_mask.trailing_zeros(),
            num_qubits,
            measure_mask,
            _pad: 0,
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("measure_params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Dynamic workgroup count: scale to state size, capped at pre-allocated max.
        let num_amplitudes = 1u64 << num_qubits;
        #[allow(clippy::cast_possible_truncation)]
        let needed_workgroups = num_amplitudes
            .div_ceil(256)
            .min(u64::from(self.num_workgroups)) as u32;

        // Step 2: Create the measurement bind group.
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("measurement_bind_group"),
            layout: cache.measurement_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.partial_sums_buffer.as_entire_binding(),
                },
            ],
        });

        // Step 3-4: Dispatch measurement shader + copy partial sums to staging.
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("measurement_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("measurement_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(cache.measurement_pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(needed_workgroups, 1, 1);
        }
        let floats_per_wg = u64::from(ActivePrecision::FLOATS_PER_PARTIAL_SUM);
        let copy_size =
            u64::from(needed_workgroups) * floats_per_wg * std::mem::size_of::<f32>() as u64;
        encoder.copy_buffer_to_buffer(
            &self.partial_sums_buffer,
            0,
            &self.partial_sums_staging,
            0,
            copy_size,
        );
        queue.submit(std::iter::once(encoder.finish()));

        // Step 5: Map the staging buffer and wait for completion.
        let buffer_slice = self.partial_sums_staging.slice(..copy_size);
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

        // Step 6-7: Read partial sums and accumulate with Kahan summation.
        let data = buffer_slice.get_mapped_range();
        let partial_sums: &[f32] = bytemuck::cast_slice(&data);

        // Kahan compensated summation in f64 for maximum accuracy.
        let mut sum = 0.0_f64;
        let mut compensation = 0.0_f64;

        #[cfg(not(feature = "f64_emulation"))]
        {
            // f32 mode: one f32 per workgroup.
            for &val in partial_sums {
                let y = f64::from(val) - compensation;
                let t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            }
        }

        #[cfg(feature = "f64_emulation")]
        {
            // f64 mode: DS pairs (hi, lo) -- 2 f32s per workgroup.
            for chunk in partial_sums.chunks_exact(2) {
                let val = f64::from(chunk[0]) + f64::from(chunk[1]);
                let y = val - compensation;
                let t = sum + y;
                compensation = (t - sum) - y;
                sum = t;
            }
        }

        // Step 8: Clean up.
        drop(data);
        self.partial_sums_staging.unmap();

        // Step 9: Clamp to valid probability range.
        Ok(sum.clamp(0.0, 1.0))
    }

    /// Dispatches the collapse shader to project the state vector onto the
    /// subspace consistent with the measurement result.
    ///
    /// Uses a grid-stride loop in the shader, with workgroup count capped at
    /// `MAX_DISPATCH_WORKGROUPS` (65535).
    #[allow(clippy::too_many_arguments)]
    pub fn collapse_state(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        cache: &PipelineCache,
        state_buffer: &StateBuffer,
        measure_mask: u32,
        measured_value: bool,
        probability: f64,
        num_qubits: u32,
    ) {
        debug_assert!(probability > 0.0, "collapse_state called with P=0");

        let num_amplitudes = 1u64 << num_qubits;
        #[allow(clippy::cast_possible_truncation)]
        let workgroups = num_amplitudes
            .div_ceil(256)
            .min(u64::from(MAX_DISPATCH_WORKGROUPS)) as u32;

        // Collapse uses the GATE bind group layout (2 bindings: rw storage + uniform).
        #[cfg(not(feature = "f64_emulation"))]
        let uniform_buffer = {
            #[allow(clippy::cast_possible_truncation)]
            let normalization_factor = 1.0_f32 / (probability as f32).sqrt();
            let params = CollapseParams {
                measure_mask,
                measured_value: u32::from(measured_value),
                normalization_factor,
                num_qubits,
                num_workgroups: workgroups,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("collapse_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };

        #[cfg(feature = "f64_emulation")]
        let uniform_buffer = {
            let norm_f64 = 1.0 / probability.sqrt();
            let (norm_hi, norm_lo) = crate::precision::to_ds(norm_f64);
            let params = CollapseParamsF64 {
                measure_mask,
                measured_value: u32::from(measured_value),
                num_qubits,
                num_workgroups: workgroups,
                norm_hi,
                norm_lo,
                _pad2: 0,
                _pad3: 0,
            };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("collapse_params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("collapse_bind_group"),
            layout: cache.bind_group_layout(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("collapse_encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("collapse_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(cache.collapse_pipeline());
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
}
