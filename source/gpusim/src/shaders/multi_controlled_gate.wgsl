// Multi-controlled gate kernel: applies a 2x2 unitary to a target qubit
// only when all control qubits are in the |1> state.
//
// The control_mask is a u32 bitmask where bit i is set if the qubit at
// bit position i is a control qubit. The target qubit's bit must NOT be
// set in the control_mask (enforced on the host).
//
// For 0 controls, the host dispatches the single-qubit kernel instead.
// This shader is only invoked when there is at least 1 control.

struct MultiControlledParams {
    // 2x2 unitary matrix: [[a, b], [c, d]]
    // Stored as [a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im]
    mat_a_re: f32,
    mat_a_im: f32,
    mat_b_re: f32,
    mat_b_im: f32,
    mat_c_re: f32,
    mat_c_im: f32,
    mat_d_re: f32,
    mat_d_im: f32,
    // Target qubit bit position.
    target_bit: u32,
    // Total number of qubits in the system.
    num_qubits: u32,
    // Bitmask of control qubit bit positions.
    control_mask: u32,
    // Number of workgroups dispatched (for grid-stride loop).
    num_workgroups: u32,
};

@group(0) @binding(0) var<storage, read_write> state: array<f32>;
@group(0) @binding(1) var<uniform> params: MultiControlledParams;

fn cmul(a_re: f32, a_im: f32, b_re: f32, b_im: f32) -> vec2<f32> {
    return vec2<f32>(
        a_re * b_re - a_im * b_im,
        a_re * b_im + a_im * b_re
    );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let num_pairs = 1u << (params.num_qubits - 1u);
    let total_threads = 256u * params.num_workgroups;

    var thread_id = gid.x;
    while (thread_id < num_pairs) {
        // Compute the two indices (identical to single-qubit kernel).
        // Insert a 0-bit at position target_bit.
        let low_mask = (1u << params.target_bit) - 1u;
        let low_bits = thread_id & low_mask;
        let high_bits = (thread_id >> params.target_bit) << (params.target_bit + 1u);
        let idx0 = high_bits | low_bits;                    // target_bit is 0
        let idx1 = idx0 | (1u << params.target_bit);        // target_bit is 1

        // Control check: all bits in control_mask must be set in idx0.
        if ((idx0 & params.control_mask) == params.control_mask) {
            // Load the two amplitudes.
            let s0_re = state[idx0 * 2u];
            let s0_im = state[idx0 * 2u + 1u];
            let s1_re = state[idx1 * 2u];
            let s1_im = state[idx1 * 2u + 1u];

            // Apply the 2x2 unitary: [a b; c d] * [s0; s1]
            let new0 = cmul(params.mat_a_re, params.mat_a_im, s0_re, s0_im)
                     + cmul(params.mat_b_re, params.mat_b_im, s1_re, s1_im);
            let new1 = cmul(params.mat_c_re, params.mat_c_im, s0_re, s0_im)
                     + cmul(params.mat_d_re, params.mat_d_im, s1_re, s1_im);

            // Write back.
            state[idx0 * 2u] = new0.x;
            state[idx0 * 2u + 1u] = new0.y;
            state[idx1 * 2u] = new1.x;
            state[idx1 * 2u + 1u] = new1.y;
        }

        thread_id += total_threads;
    }
}
