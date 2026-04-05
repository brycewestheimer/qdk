use crate::gates::{Mat2x2, Mat4x4};
use crate::gpu::pipeline::PipelineCache;
use crate::gpu::state_buffer::StateBuffer;
#[cfg(feature = "f64_emulation")]
use crate::precision;

/// Maximum number of workgroups that can be dispatched in a single dimension.
///
/// Per the WebGPU spec, `maxComputeWorkgroupsPerDimension` is at least 65535.
/// Gate shaders use grid-stride loops to handle work beyond this limit.
const MAX_WORKGROUPS: u32 = 65535;

/// GPU-side gate parameters matching the WGSL `GateParams` struct layout.
///
/// Must be `#[repr(C)]` and `bytemuck::Pod` to be safely cast to raw bytes
/// for upload to a uniform buffer.
///
/// Size: 48 bytes (12 x 4-byte fields), which is a multiple of 16
/// (the minimum uniform buffer alignment).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GateParams {
    /// Matrix element [0,0] real part.
    pub gate_re0: f32,
    /// Matrix element [0,0] imaginary part.
    pub gate_im0: f32,
    /// Matrix element [0,1] real part.
    pub gate_re1: f32,
    /// Matrix element [0,1] imaginary part.
    pub gate_im1: f32,
    /// Matrix element [1,0] real part.
    pub gate_re2: f32,
    /// Matrix element [1,0] imaginary part.
    pub gate_im2: f32,
    /// Matrix element [1,1] real part.
    pub gate_re3: f32,
    /// Matrix element [1,1] imaginary part.
    pub gate_im3: f32,
    /// Bit position of the target qubit (0-indexed from LSB).
    pub target_bit: u32,
    /// Total number of qubits in the state vector.
    pub num_qubits: u32,
    /// Number of workgroups dispatched (for grid-stride loop).
    pub num_workgroups: u32,
    /// Padding for 16-byte alignment.
    _pad: u32,
}

/// GPU-side parameters for the two-qubit gate shader.
///
/// Layout matches `TwoQubitParams` in `two_qubit_gate.wgsl`.
/// The matrix is packed as `array<vec4<f32>, 8>` in WGSL to satisfy the
/// 16-byte uniform array element stride requirement.
///
/// Size: 144 bytes (128 matrix + 16 fields).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TwoQubitGateParams {
    /// 4x4 unitary matrix packed as 8 vec4 chunks.
    /// Logical element `i` is at `mat[i / 4][i % 4]`.
    pub mat: [[f32; 4]; 8],
    /// Bit position of the first qubit (qubit A).
    pub bit_a: u32,
    /// Bit position of the second qubit (qubit B).
    pub bit_b: u32,
    /// Total number of qubits in the system.
    pub num_qubits: u32,
    /// Number of workgroups dispatched (for grid-stride loop).
    pub num_workgroups: u32,
}

/// GPU-side parameters for the multi-controlled gate shader.
///
/// Layout matches `MultiControlledParams` in `multi_controlled_gate.wgsl`.
///
/// Size: 48 bytes (same as `GateParams`).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MultiControlledGateParams {
    /// Matrix element a (row 0, col 0) real part.
    pub mat_a_re: f32,
    /// Matrix element a imaginary part.
    pub mat_a_im: f32,
    /// Matrix element b (row 0, col 1) real part.
    pub mat_b_re: f32,
    /// Matrix element b imaginary part.
    pub mat_b_im: f32,
    /// Matrix element c (row 1, col 0) real part.
    pub mat_c_re: f32,
    /// Matrix element c imaginary part.
    pub mat_c_im: f32,
    /// Matrix element d (row 1, col 1) real part.
    pub mat_d_re: f32,
    /// Matrix element d imaginary part.
    pub mat_d_im: f32,
    /// Bit position of the target qubit.
    pub target_bit: u32,
    /// Total number of qubits.
    pub num_qubits: u32,
    /// Bitmask of control qubit bit positions.
    pub control_mask: u32,
    /// Number of workgroups dispatched (for grid-stride loop).
    pub num_workgroups: u32,
}

/// GPU-side single-qubit gate params for f64 emulation mode.
///
/// Layout matches the f64 WGSL `GateParams` struct: header fields first,
/// then the DS-encoded 2x2 matrix packed as `array<vec4<f32>, 4>`.
///
/// Size: 80 bytes (16 header + 64 matrix).
#[cfg(feature = "f64_emulation")]
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GateParamsF64 {
    pub target_bit: u32,
    pub num_qubits: u32,
    pub num_workgroups: u32,
    _pad: u32,
    /// 2x2 complex matrix in DS format, packed as 4 vec4 chunks.
    pub matrix: [[f32; 4]; 4],
}

/// GPU-side two-qubit gate params for f64 emulation mode.
///
/// Size: 272 bytes (16 header + 256 matrix).
#[cfg(feature = "f64_emulation")]
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TwoQubitGateParamsF64 {
    pub bit_a: u32,
    pub bit_b: u32,
    pub num_qubits: u32,
    pub num_workgroups: u32,
    /// 4x4 complex matrix in DS format, packed as 16 vec4 chunks.
    pub mat: [[f32; 4]; 16],
}

/// GPU-side multi-controlled gate params for f64 emulation mode.
///
/// Size: 80 bytes (16 header + 64 matrix).
#[cfg(feature = "f64_emulation")]
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MultiControlledGateParamsF64 {
    pub target_bit: u32,
    pub num_qubits: u32,
    pub control_mask: u32,
    pub num_workgroups: u32,
    /// 2x2 complex matrix in DS format, packed as 4 vec4 chunks.
    pub matrix: [[f32; 4]; 4],
}

/// GPU-side collapse params for f64 emulation mode.
///
/// Size: 32 bytes.
#[cfg(feature = "f64_emulation")]
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CollapseParamsF64 {
    pub measure_mask: u32,
    pub measured_value: u32,
    pub num_qubits: u32,
    pub num_workgroups: u32,
    /// DS normalization factor (hi, lo).
    pub norm_hi: f32,
    pub norm_lo: f32,
    #[allow(clippy::pub_underscore_fields)]
    pub _pad2: u32,
    #[allow(clippy::pub_underscore_fields)]
    pub _pad3: u32,
}

/// Encode a `Mat2x2` (4 complex entries as `(f32, f32)` tuples) into DS format
/// packed as `[[f32; 4]; 4]` for vec4 uniform upload.
///
/// Each matrix entry becomes 4 f32s: `[re_hi, re_lo, im_hi, im_lo]`.
/// This layout maps directly to `array<vec4<f32>, 4>` in WGSL.
#[cfg(feature = "f64_emulation")]
fn encode_2x2_ds(gate: &Mat2x2) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for (i, &(re, im)) in gate.iter().enumerate() {
        let (re_hi, re_lo) = precision::to_ds(f64::from(re));
        let (im_hi, im_lo) = precision::to_ds(f64::from(im));
        out[i] = [re_hi, re_lo, im_hi, im_lo];
    }
    out
}

/// Encode a `Mat4x4` (16 complex entries) into DS format packed as
/// `[[f32; 4]; 16]` for vec4 uniform upload.
#[cfg(feature = "f64_emulation")]
fn encode_4x4_ds(gate: &Mat4x4) -> [[f32; 4]; 16] {
    let mut out = [[0.0f32; 4]; 16];
    for (i, &(re, im)) in gate.iter().enumerate() {
        let (re_hi, re_lo) = precision::to_ds(f64::from(re));
        let (im_hi, im_lo) = precision::to_ds(f64::from(im));
        out[i] = [re_hi, re_lo, im_hi, im_lo];
    }
    out
}

/// Encode a `Mat2x2F64` (4 complex entries as `(f64, f64)` tuples) into DS format
/// packed as `[[f32; 4]; 4]` for vec4 uniform upload.
///
/// Unlike [`encode_2x2_ds`], this accepts f64 values directly, avoiding the
/// precision loss from an f32 round-trip. Used for rotation gates and phase gates
/// in the f64-emulation path.
#[cfg(feature = "f64_emulation")]
fn encode_2x2_ds_f64(gate: &crate::gates::Mat2x2F64) -> [[f32; 4]; 4] {
    let mut out = [[0.0f32; 4]; 4];
    for (i, &(re, im)) in gate.iter().enumerate() {
        let (re_hi, re_lo) = precision::to_ds(re);
        let (im_hi, im_lo) = precision::to_ds(im);
        out[i] = [re_hi, re_lo, im_hi, im_lo];
    }
    out
}

/// Computes a capped workgroup count for GPU dispatch.
///
/// Returns `min(required.div_ceil(256), MAX_WORKGROUPS)`.
fn capped_workgroup_count(num_items: u64) -> u32 {
    #[allow(clippy::cast_possible_truncation)]
    let needed = num_items.div_ceil(256).min(u64::from(MAX_WORKGROUPS)) as u32;
    needed
}

/// Dispatches a single-qubit gate operation to the GPU.
///
/// This creates a transient uniform buffer for the gate parameters,
/// builds a bind group, records a compute pass, and submits it to the queue.
///
/// The state vector is modified in-place on the GPU. No CPU synchronization
/// occurs -- the operation is fully asynchronous from the CPU's perspective,
/// but wgpu ensures sequential execution of submitted commands.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_single_qubit_gate(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline_cache: &PipelineCache,
    state_buffer: &StateBuffer,
    gate: &Mat2x2,
    target_bit: u32,
    num_qubits: u32,
    param_buffer: &wgpu::Buffer,
) {
    let num_pairs = 1u64 << (num_qubits - 1);
    let workgroup_count = capped_workgroup_count(num_pairs);

    #[cfg(not(feature = "f64_emulation"))]
    {
        let params = GateParams {
            gate_re0: gate[0].0,
            gate_im0: gate[0].1,
            gate_re1: gate[1].0,
            gate_im1: gate[1].1,
            gate_re2: gate[2].0,
            gate_im2: gate[2].1,
            gate_re3: gate[3].0,
            gate_im3: gate[3].1,
            target_bit,
            num_qubits,
            num_workgroups: workgroup_count,
            _pad: 0,
        };
        queue.write_buffer(param_buffer, 0, bytemuck::bytes_of(&params));
    }

    #[cfg(feature = "f64_emulation")]
    {
        let params = GateParamsF64 {
            target_bit,
            num_qubits,
            num_workgroups: workgroup_count,
            _pad: 0,
            matrix: encode_2x2_ds(gate),
        };
        queue.write_buffer(param_buffer, 0, bytemuck::bytes_of(&params));
    }

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gate_bind_group"),
        layout: pipeline_cache.bind_group_layout(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state_buffer.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: param_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gate_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("single_qubit_gate_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline_cache.single_qubit_pipeline());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
}

/// Dispatches a two-qubit gate operation to the GPU.
///
/// Applies an arbitrary 4x4 unitary matrix to the qubit pair at bit
/// positions `bit_a` and `bit_b`. The matrix is provided in the crate's
/// `Mat4x4` format (16 complex tuples) and flattened to `[[f32; 4]; 8]`
/// for GPU upload (vec4-packed to match WGSL uniform layout).
///
/// # Panics
/// Panics if `bit_a == bit_b` or `num_qubits < 2`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_two_qubit_gate(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline_cache: &PipelineCache,
    state_buffer: &StateBuffer,
    gate: &Mat4x4,
    bit_a: u32,
    bit_b: u32,
    num_qubits: u32,
    param_buffer: &wgpu::Buffer,
) {
    assert_ne!(bit_a, bit_b, "two-qubit gate requires distinct qubits");
    assert!(
        num_qubits >= 2,
        "need at least 2 qubits for a two-qubit gate"
    );

    let num_groups = 1u64 << (num_qubits - 2);
    let workgroup_count = capped_workgroup_count(num_groups);

    #[cfg(not(feature = "f64_emulation"))]
    {
        // Flatten Mat4x4 [(f32, f32); 16] to [[f32; 4]; 8] (vec4-packed)
        let mut flat = [0.0f32; 32];
        for (i, &(re, im)) in gate.iter().enumerate() {
            flat[i * 2] = re;
            flat[i * 2 + 1] = im;
        }
        // Reinterpret [f32; 32] as [[f32; 4]; 8]
        let mut mat = [[0.0f32; 4]; 8];
        for (chunk_idx, chunk) in flat.chunks_exact(4).enumerate() {
            mat[chunk_idx] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        }
        let params = TwoQubitGateParams {
            mat,
            bit_a,
            bit_b,
            num_qubits,
            num_workgroups: workgroup_count,
        };
        queue.write_buffer(param_buffer, 0, bytemuck::bytes_of(&params));
    }

    #[cfg(feature = "f64_emulation")]
    {
        let params = TwoQubitGateParamsF64 {
            bit_a,
            bit_b,
            num_qubits,
            num_workgroups: workgroup_count,
            mat: encode_4x4_ds(gate),
        };
        queue.write_buffer(param_buffer, 0, bytemuck::bytes_of(&params));
    }

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("two_qubit_bind_group"),
        layout: pipeline_cache.bind_group_layout(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state_buffer.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: param_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("two_qubit_gate_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("two_qubit_gate_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline_cache.two_qubit_pipeline());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
}

/// Dispatches a multi-controlled gate operation to the GPU.
///
/// Applies a 2x2 unitary to the target qubit only when all control qubits
/// (specified by `control_mask`) are in the |1> state.
///
/// # Panics
/// Panics if `target_bit` is set in `control_mask`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_multi_controlled_gate(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline_cache: &PipelineCache,
    state_buffer: &StateBuffer,
    gate: &Mat2x2,
    target_bit: u32,
    control_mask: u32,
    num_qubits: u32,
    param_buffer: &wgpu::Buffer,
) {
    assert_eq!(
        control_mask & (1 << target_bit),
        0,
        "target_bit must not be set in control_mask"
    );

    let num_pairs = 1u64 << (num_qubits - 1);
    let workgroup_count = capped_workgroup_count(num_pairs);

    #[cfg(not(feature = "f64_emulation"))]
    {
        let params = MultiControlledGateParams {
            mat_a_re: gate[0].0,
            mat_a_im: gate[0].1,
            mat_b_re: gate[1].0,
            mat_b_im: gate[1].1,
            mat_c_re: gate[2].0,
            mat_c_im: gate[2].1,
            mat_d_re: gate[3].0,
            mat_d_im: gate[3].1,
            target_bit,
            num_qubits,
            control_mask,
            num_workgroups: workgroup_count,
        };
        queue.write_buffer(param_buffer, 0, bytemuck::bytes_of(&params));
    }

    #[cfg(feature = "f64_emulation")]
    {
        let params = MultiControlledGateParamsF64 {
            target_bit,
            num_qubits,
            control_mask,
            num_workgroups: workgroup_count,
            matrix: encode_2x2_ds(gate),
        };
        queue.write_buffer(param_buffer, 0, bytemuck::bytes_of(&params));
    }

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("multi_controlled_bind_group"),
        layout: pipeline_cache.bind_group_layout(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state_buffer.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: param_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("multi_controlled_gate_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("multi_controlled_gate_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline_cache.multi_controlled_pipeline());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
}

/// Dispatches a single-qubit gate with f64-precision matrix in DS format.
///
/// This is the f64-emulation counterpart of [`dispatch_single_qubit_gate`].
/// It accepts a `Mat2x2F64` (f64 complex entries) and encodes them directly
/// to DS format via [`encode_2x2_ds_f64`], bypassing f32 intermediates.
#[cfg(feature = "f64_emulation")]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_single_qubit_gate_f64(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline_cache: &PipelineCache,
    state_buffer: &StateBuffer,
    gate: &crate::gates::Mat2x2F64,
    target_bit: u32,
    num_qubits: u32,
    param_buffer: &wgpu::Buffer,
) {
    let num_pairs = 1u64 << (num_qubits - 1);
    let workgroup_count = capped_workgroup_count(num_pairs);

    let params = GateParamsF64 {
        target_bit,
        num_qubits,
        num_workgroups: workgroup_count,
        _pad: 0,
        matrix: encode_2x2_ds_f64(gate),
    };
    queue.write_buffer(param_buffer, 0, bytemuck::bytes_of(&params));

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gate_bind_group"),
        layout: pipeline_cache.bind_group_layout(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state_buffer.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: param_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gate_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("single_qubit_gate_f64_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline_cache.single_qubit_pipeline());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
}

/// Dispatches a multi-controlled gate with f64-precision matrix in DS format.
///
/// This is the f64-emulation counterpart of [`dispatch_multi_controlled_gate`].
/// It accepts a `Mat2x2F64` and encodes directly to DS format.
///
/// # Panics
/// Panics if `target_bit` is set in `control_mask`.
#[cfg(feature = "f64_emulation")]
#[allow(clippy::too_many_arguments)]
pub fn dispatch_multi_controlled_gate_f64(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline_cache: &PipelineCache,
    state_buffer: &StateBuffer,
    gate: &crate::gates::Mat2x2F64,
    target_bit: u32,
    control_mask: u32,
    num_qubits: u32,
    param_buffer: &wgpu::Buffer,
) {
    assert_eq!(
        control_mask & (1 << target_bit),
        0,
        "target_bit must not be set in control_mask"
    );

    let num_pairs = 1u64 << (num_qubits - 1);
    let workgroup_count = capped_workgroup_count(num_pairs);

    let params = MultiControlledGateParamsF64 {
        target_bit,
        num_qubits,
        control_mask,
        num_workgroups: workgroup_count,
        matrix: encode_2x2_ds_f64(gate),
    };
    queue.write_buffer(param_buffer, 0, bytemuck::bytes_of(&params));

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("multi_controlled_bind_group"),
        layout: pipeline_cache.bind_group_layout(),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: state_buffer.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: param_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("multi_controlled_gate_encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("multi_controlled_gate_f64_pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline_cache.multi_controlled_pipeline());
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
}

/// Parameters for the measurement (probability reduction) shader.
///
/// Layout matches `MeasureParams` in `measurement.wgsl` exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeasureParams {
    /// Bit position of the target qubit (informational; shader uses `measure_mask`).
    pub target_bit: u32,
    /// Total number of qubits in the system.
    pub num_qubits: u32,
    /// Bitmask of qubits being measured. Bit `i` is set if the qubit at
    /// internal bit position `i` is part of the measurement.
    pub measure_mask: u32,
    /// Padding for 16-byte alignment.
    #[allow(clippy::pub_underscore_fields)]
    pub _pad: u32,
}

/// Parameters for the state collapse shader.
///
/// Layout matches `CollapseParams` in `collapse.wgsl` exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CollapseParams {
    /// Bitmask of measured qubits (same as in `MeasureParams`).
    pub measure_mask: u32,
    /// Measurement result: 1 if the measured parity was odd, 0 if even.
    pub measured_value: u32,
    /// Normalization factor: `1.0 / sqrt(P(measured_outcome))`.
    pub normalization_factor: f32,
    /// Total number of qubits.
    pub num_qubits: u32,
    /// Number of workgroups dispatched (for grid-stride loop).
    pub num_workgroups: u32,
    /// Padding for 16-byte alignment.
    #[allow(clippy::pub_underscore_fields)]
    pub _pad0: u32,
    #[allow(clippy::pub_underscore_fields)]
    pub _pad1: u32,
    #[allow(clippy::pub_underscore_fields)]
    pub _pad2: u32,
}

const _: () = assert!(std::mem::size_of::<GateParams>() == 48);
const _: () = assert!(std::mem::size_of::<TwoQubitGateParams>() == 144);
const _: () = assert!(std::mem::size_of::<MultiControlledGateParams>() == 48);
const _: () = assert!(std::mem::size_of::<MeasureParams>() == 16);
const _: () = assert!(std::mem::size_of::<CollapseParams>() == 32);

#[cfg(feature = "f64_emulation")]
const _: () = assert!(std::mem::size_of::<GateParamsF64>() == 80);
#[cfg(feature = "f64_emulation")]
const _: () = assert!(std::mem::size_of::<TwoQubitGateParamsF64>() == 272);
#[cfg(feature = "f64_emulation")]
const _: () = assert!(std::mem::size_of::<MultiControlledGateParamsF64>() == 80);
#[cfg(feature = "f64_emulation")]
const _: () = assert!(std::mem::size_of::<CollapseParamsF64>() == 32);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_params_size_is_48_bytes() {
        assert_eq!(std::mem::size_of::<GateParams>(), 48);
    }

    #[test]
    fn gate_params_alignment_is_correct() {
        assert_eq!(std::mem::size_of::<GateParams>() % 16, 0);
    }

    #[test]
    fn two_qubit_gate_params_size_is_144_bytes() {
        assert_eq!(std::mem::size_of::<TwoQubitGateParams>(), 144);
    }

    #[test]
    fn two_qubit_gate_params_alignment_is_correct() {
        assert_eq!(std::mem::size_of::<TwoQubitGateParams>() % 16, 0);
    }

    #[test]
    fn multi_controlled_gate_params_size_is_48_bytes() {
        assert_eq!(std::mem::size_of::<MultiControlledGateParams>(), 48);
    }

    #[test]
    fn multi_controlled_gate_params_alignment_is_correct() {
        assert_eq!(std::mem::size_of::<MultiControlledGateParams>() % 16, 0);
    }

    #[test]
    fn measure_params_size_is_16_bytes() {
        assert_eq!(std::mem::size_of::<MeasureParams>(), 16);
    }

    #[test]
    fn measure_params_alignment_is_correct() {
        assert_eq!(std::mem::size_of::<MeasureParams>() % 16, 0);
    }

    #[test]
    fn collapse_params_size_is_32_bytes() {
        assert_eq!(std::mem::size_of::<CollapseParams>(), 32);
    }

    #[test]
    fn collapse_params_alignment_is_correct() {
        assert_eq!(std::mem::size_of::<CollapseParams>() % 16, 0);
    }

    #[test]
    fn capped_workgroup_count_below_limit() {
        // 20 qubits: 2^19 pairs / 256 = 2048 workgroups (below 65535)
        assert_eq!(capped_workgroup_count(1u64 << 19), 2048);
    }

    #[test]
    fn capped_workgroup_count_at_limit() {
        // 25 qubits: 2^24 pairs / 256 = 65536 > 65535, should cap
        assert_eq!(capped_workgroup_count(1u64 << 24), 65535);
    }

    #[test]
    fn capped_workgroup_count_well_above_limit() {
        // 30 qubits: 2^29 pairs / 256 = 2,097,152 >> 65535, should cap
        assert_eq!(capped_workgroup_count(1u64 << 29), 65535);
    }
}
