use crate::gates::Mat2x2;
use crate::gpu::pipeline::PipelineCache;
use crate::gpu::state_buffer::StateBuffer;

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
    // Step 1: Construct GateParams from the Mat2x2
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

    // Step 2: Create a uniform buffer and write the params
    let param_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("gate_params"),
        size: std::mem::size_of::<GateParams>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&param_buffer, 0, bytemuck::bytes_of(&params));

    // Step 3: Create the bind group
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

    // Step 4: Record and submit the compute pass
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
}
