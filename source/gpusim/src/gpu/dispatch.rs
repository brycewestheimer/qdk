use crate::gates::{Mat2x2, Mat4x4};
use crate::gpu::pipeline::PipelineCache;
use crate::gpu::state_buffer::StateBuffer;
#[cfg(feature = "f64_emulation")]
use crate::precision;

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
    /// Padding for 16-byte alignment.
    _pad0: u32,
    /// Padding for 16-byte alignment.
    _pad1: u32,
}

/// GPU-side parameters for the two-qubit gate shader.
///
/// Layout matches `TwoQubitParams` in `two_qubit_gate.wgsl`.
///
/// Size: 144 bytes (32 f32 matrix + 4 u32 fields).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TwoQubitGateParams {
    /// 4x4 unitary matrix as 32 f32 values (16 complex numbers, row-major).
    /// `mat[(row * 4 + col) * 2]` = real, `mat[(row * 4 + col) * 2 + 1]` = imag.
    pub mat: [f32; 32],
    /// Bit position of the first qubit (qubit A).
    pub bit_a: u32,
    /// Bit position of the second qubit (qubit B).
    pub bit_b: u32,
    /// Total number of qubits in the system.
    pub num_qubits: u32,
    /// Padding for 16-byte alignment.
    _pad: u32,
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
    /// Padding for 16-byte alignment.
    _pad: u32,
}

/// GPU-side single-qubit gate params for f64 emulation mode.
///
/// Layout matches the f64 WGSL `GateParams` struct: header fields first,
/// then the DS-encoded 2x2 matrix (16 f32s = 4 complex entries * 4 f32s).
///
/// Size: 80 bytes (16 header + 64 matrix).
#[cfg(feature = "f64_emulation")]
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GateParamsF64 {
    pub target_bit: u32,
    pub num_qubits: u32,
    _pad0: u32,
    _pad1: u32,
    /// 2x2 complex matrix in DS format (4 entries, 4 f32s each).
    pub matrix: [f32; 16],
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
    _pad: u32,
    /// 4x4 complex matrix in DS format (16 entries, 4 f32s each).
    pub mat: [f32; 64],
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
    _pad: u32,
    /// 2x2 complex matrix in DS format (4 entries, 4 f32s each).
    pub matrix: [f32; 16],
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
    #[allow(clippy::pub_underscore_fields)]
    pub _pad: u32,
    /// DS normalization factor (hi, lo).
    pub norm_hi: f32,
    pub norm_lo: f32,
    #[allow(clippy::pub_underscore_fields)]
    pub _pad2: u32,
    #[allow(clippy::pub_underscore_fields)]
    pub _pad3: u32,
}

/// Encode a `Mat2x2` (4 complex entries as `(f32, f32)` tuples) into DS format
/// for GPU upload: 4 entries * 4 f32s (`re_hi`, `re_lo`, `im_hi`, `im_lo`) = 16 f32s.
#[cfg(feature = "f64_emulation")]
fn encode_2x2_ds(gate: &Mat2x2) -> [f32; 16] {
    let mut out = [0.0f32; 16];
    for (i, &(re, im)) in gate.iter().enumerate() {
        let (re_hi, re_lo) = precision::to_ds(f64::from(re));
        let (im_hi, im_lo) = precision::to_ds(f64::from(im));
        out[i * 4] = re_hi;
        out[i * 4 + 1] = re_lo;
        out[i * 4 + 2] = im_hi;
        out[i * 4 + 3] = im_lo;
    }
    out
}

/// Encode a `Mat4x4` (16 complex entries) into DS format for GPU upload: 64 f32s.
#[cfg(feature = "f64_emulation")]
fn encode_4x4_ds(gate: &Mat4x4) -> [f32; 64] {
    let mut out = [0.0f32; 64];
    for (i, &(re, im)) in gate.iter().enumerate() {
        let (re_hi, re_lo) = precision::to_ds(f64::from(re));
        let (im_hi, im_lo) = precision::to_ds(f64::from(im));
        out[i * 4] = re_hi;
        out[i * 4 + 1] = re_lo;
        out[i * 4 + 2] = im_hi;
        out[i * 4 + 3] = im_lo;
    }
    out
}

/// Dispatches a single-qubit gate operation to the GPU.
///
/// This creates a transient uniform buffer for the gate parameters,
/// builds a bind group, records a compute pass, and submits it to the queue.
///
/// The state vector is modified in-place on the GPU. No CPU synchronization
/// occurs — the operation is fully asynchronous from the CPU's perspective,
/// but wgpu ensures sequential execution of submitted commands.
pub fn dispatch_single_qubit_gate(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline_cache: &PipelineCache,
    state_buffer: &StateBuffer,
    gate: &Mat2x2,
    target_bit: u32,
    num_qubits: u32,
) {
    #[cfg(not(feature = "f64_emulation"))]
    let param_buffer = {
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
            _pad0: 0,
            _pad1: 0,
        };
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gate_params"),
            size: std::mem::size_of::<GateParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::bytes_of(&params));
        buf
    };

    #[cfg(feature = "f64_emulation")]
    let param_buffer = {
        let params = GateParamsF64 {
            target_bit,
            num_qubits,
            _pad0: 0,
            _pad1: 0,
            matrix: encode_2x2_ds(gate),
        };
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gate_params"),
            size: std::mem::size_of::<GateParamsF64>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::bytes_of(&params));
        buf
    };

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

        // 2^(N-1) amplitude pairs, 256 threads per workgroup
        let num_pairs = 1u64 << (num_qubits - 1);
        #[allow(clippy::cast_possible_truncation)]
        let workgroup_count = num_pairs.div_ceil(256) as u32;
        pass.dispatch_workgroups(workgroup_count, 1, 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
}

/// Dispatches a two-qubit gate operation to the GPU.
///
/// Applies an arbitrary 4x4 unitary matrix to the qubit pair at bit
/// positions `bit_a` and `bit_b`. The matrix is provided in the crate's
/// `Mat4x4` format (16 complex tuples) and flattened to 32 f32 values
/// for GPU upload.
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
) {
    assert_ne!(bit_a, bit_b, "two-qubit gate requires distinct qubits");
    assert!(
        num_qubits >= 2,
        "need at least 2 qubits for a two-qubit gate"
    );

    #[cfg(not(feature = "f64_emulation"))]
    let param_buffer = {
        // Flatten Mat4x4 [(f32, f32); 16] to [f32; 32]
        let mut mat = [0.0f32; 32];
        for (i, &(re, im)) in gate.iter().enumerate() {
            mat[i * 2] = re;
            mat[i * 2 + 1] = im;
        }
        let params = TwoQubitGateParams {
            mat,
            bit_a,
            bit_b,
            num_qubits,
            _pad: 0,
        };
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("two_qubit_params"),
            size: std::mem::size_of::<TwoQubitGateParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::bytes_of(&params));
        buf
    };

    #[cfg(feature = "f64_emulation")]
    let param_buffer = {
        let params = TwoQubitGateParamsF64 {
            bit_a,
            bit_b,
            num_qubits,
            _pad: 0,
            mat: encode_4x4_ds(gate),
        };
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("two_qubit_params"),
            size: std::mem::size_of::<TwoQubitGateParamsF64>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::bytes_of(&params));
        buf
    };

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

        // 2^(N-2) groups of 4 amplitudes, 256 threads per workgroup
        let num_groups = 1u64 << (num_qubits - 2);
        #[allow(clippy::cast_possible_truncation)]
        let workgroup_count = num_groups.div_ceil(256) as u32;
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
/// Debug-asserts that `target_bit` is not set in `control_mask`.
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
) {
    debug_assert_eq!(
        control_mask & (1 << target_bit),
        0,
        "target_bit must not be set in control_mask"
    );

    #[cfg(not(feature = "f64_emulation"))]
    let param_buffer = {
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
            _pad: 0,
        };
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("multi_controlled_params"),
            size: std::mem::size_of::<MultiControlledGateParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::bytes_of(&params));
        buf
    };

    #[cfg(feature = "f64_emulation")]
    let param_buffer = {
        let params = MultiControlledGateParamsF64 {
            target_bit,
            num_qubits,
            control_mask,
            _pad: 0,
            matrix: encode_2x2_ds(gate),
        };
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("multi_controlled_params"),
            size: std::mem::size_of::<MultiControlledGateParamsF64>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::bytes_of(&params));
        buf
    };

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

        // Same dispatch size as single-qubit: 2^(N-1) pairs
        let num_pairs = 1u64 << (num_qubits - 1);
        #[allow(clippy::cast_possible_truncation)]
        let workgroup_count = num_pairs.div_ceil(256) as u32;
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
}

const _: () = assert!(std::mem::size_of::<MeasureParams>() == 16);
const _: () = assert!(std::mem::size_of::<CollapseParams>() == 16);

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
        // Uniform buffers require 16-byte alignment. 48 is a multiple of 16.
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
    fn collapse_params_size_is_16_bytes() {
        assert_eq!(std::mem::size_of::<CollapseParams>(), 16);
    }

    #[test]
    fn collapse_params_alignment_is_correct() {
        assert_eq!(std::mem::size_of::<CollapseParams>() % 16, 0);
    }
}
