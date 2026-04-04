// Single-qubit gate kernel.
//
// Applies a 2x2 unitary matrix to the target qubit of an N-qubit state vector.
// Each thread processes one pair of amplitudes: one where the target bit is 0,
// and the corresponding one where the target bit is 1.

struct GateParams {
    // 2x2 unitary matrix as 4 complex numbers (row-major):
    //   [[gate_re0+i*gate_im0, gate_re1+i*gate_im1],
    //    [gate_re2+i*gate_im2, gate_re3+i*gate_im3]]
    gate_re0: f32, gate_im0: f32,
    gate_re1: f32, gate_im1: f32,
    gate_re2: f32, gate_im2: f32,
    gate_re3: f32, gate_im3: f32,
    // Bit position of the target qubit in the state vector index.
    target_bit: u32,
    // Total number of qubits.
    num_qubits: u32,
    // Padding to align to 16 bytes.
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> state: array<f32>;
@group(0) @binding(1) var<uniform> params: GateParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    let num_pairs = 1u << (params.num_qubits - 1u);

    if thread_id >= num_pairs {
        return;
    }

    // Compute the state vector indices for the amplitude pair.
    // idx0 has the target bit = 0, idx1 has the target bit = 1.
    //
    // We split the thread_id bits around the target_bit position:
    //   - bits below target_bit pass through unchanged
    //   - bits at and above target_bit shift up by one to make room
    //   - the target_bit position is set to 0 (idx0) or 1 (idx1)
    let low_mask = (1u << params.target_bit) - 1u;
    let low_bits = thread_id & low_mask;
    let high_bits = (thread_id >> params.target_bit) << (params.target_bit + 1u);
    let idx0 = high_bits | low_bits;
    let idx1 = idx0 | (1u << params.target_bit);

    // Each amplitude is stored as two consecutive f32 values (re, im).
    let off0 = idx0 * 2u;
    let off1 = idx1 * 2u;

    // Load the two amplitudes.
    let a_re = state[off0];
    let a_im = state[off0 + 1u];
    let b_re = state[off1];
    let b_im = state[off1 + 1u];

    // Apply the 2x2 unitary: [a', b'] = U * [a, b]
    //
    // a' = U[0,0]*a + U[0,1]*b
    // b' = U[1,0]*a + U[1,1]*b
    //
    // Complex multiply: (x_re + i*x_im)(y_re + i*y_im)
    //                  = (x_re*y_re - x_im*y_im) + i*(x_re*y_im + x_im*y_re)

    // a' = gate[0]*a + gate[1]*b
    let new_a_re = (params.gate_re0 * a_re - params.gate_im0 * a_im)
                 + (params.gate_re1 * b_re - params.gate_im1 * b_im);
    let new_a_im = (params.gate_re0 * a_im + params.gate_im0 * a_re)
                 + (params.gate_re1 * b_im + params.gate_im1 * b_re);

    // b' = gate[2]*a + gate[3]*b
    let new_b_re = (params.gate_re2 * a_re - params.gate_im2 * a_im)
                 + (params.gate_re3 * b_re - params.gate_im3 * b_im);
    let new_b_im = (params.gate_re2 * a_im + params.gate_im2 * a_re)
                 + (params.gate_re3 * b_im + params.gate_im3 * b_re);

    // Write back in-place.
    state[off0] = new_a_re;
    state[off0 + 1u] = new_a_im;
    state[off1] = new_b_re;
    state[off1 + 1u] = new_b_im;
}
