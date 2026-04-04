// Two-qubit gate kernel: applies an arbitrary 4x4 unitary to a pair of qubits.
//
// The 4x4 matrix operates in the basis |q_a q_b> where q_a is the qubit at
// bit position `bit_a` and q_b is the qubit at bit position `bit_b`.
//
// Dispatch: ceil(2^(num_qubits - 2) / 256) workgroups.

struct TwoQubitParams {
    // 4x4 unitary matrix stored as 16 complex numbers in row-major order.
    // mat[(row * 4 + col) * 2] = real part
    // mat[(row * 4 + col) * 2 + 1] = imaginary part
    mat: array<f32, 32>,
    // Qubit bit positions (not necessarily ordered).
    bit_a: u32,
    bit_b: u32,
    // Total number of qubits in the system.
    num_qubits: u32,
    // Padding for 16-byte alignment.
    _pad: u32,
};

@group(0) @binding(0) var<storage, read_write> state: array<f32>;
@group(0) @binding(1) var<uniform> params: TwoQubitParams;

fn cmul(a_re: f32, a_im: f32, b_re: f32, b_im: f32) -> vec2<f32> {
    return vec2<f32>(
        a_re * b_re - a_im * b_im,
        a_re * b_im + a_im * b_re
    );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_id = gid.x;
    let num_groups = 1u << (params.num_qubits - 2u);

    if (thread_id >= num_groups) {
        return;
    }

    // Determine the lower and higher bit positions.
    let lo = min(params.bit_a, params.bit_b);
    let hi = max(params.bit_a, params.bit_b);

    // --- Double-bit-insertion algorithm ---
    //
    // Map thread_id (a (num_qubits-2)-bit number) to a state vector index
    // where bits at positions `lo` and `hi` are both zero.
    //
    // Step 1: Insert a 0-bit at position `lo`.
    //   Bits below `lo` stay in place; bits at or above `lo` shift up by 1.
    let lo_mask = (1u << lo) - 1u;
    var idx = (thread_id & lo_mask) | ((thread_id & ~lo_mask) << 1u);

    // Step 2: Insert a 0-bit at position `hi`.
    //   After step 1, the bit that was originally at position `hi-1` in
    //   thread_id is now at position `hi` in idx (because the lo insertion
    //   shifted it). We split idx at position `hi` and shift the upper part.
    let hi_mask = (1u << hi) - 1u;
    idx = (idx & hi_mask) | ((idx & ~hi_mask) << 1u);

    // Now idx has 0-bits at both `lo` and `hi`.
    // Build the 4 indices by setting each combination of those two bit positions.
    let idx00 = idx;                              // both bits 0
    let idx_lo = idx | (1u << lo);                // lo bit set
    let idx_hi = idx | (1u << hi);                // hi bit set
    let idx11 = idx | (1u << lo) | (1u << hi);    // both bits set

    // Map to canonical 2-qubit basis ordering |q_a q_b>.
    //
    // |00> = bit_a=0, bit_b=0 -> idx00
    // |01> = bit_a=0, bit_b=1 -> bit_b set
    // |10> = bit_a=1, bit_b=0 -> bit_a set
    // |11> = bit_a=1, bit_b=1 -> idx11
    //
    // When bit_a < bit_b (bit_a == lo, bit_b == hi):
    //   |01> needs bit_b(=hi) set -> idx_hi
    //   |10> needs bit_a(=lo) set -> idx_lo
    //
    // When bit_a > bit_b (bit_a == hi, bit_b == lo):
    //   |01> needs bit_b(=lo) set -> idx_lo
    //   |10> needs bit_a(=hi) set -> idx_hi

    var indices: array<u32, 4>;
    if (params.bit_a < params.bit_b) {
        indices = array<u32, 4>(idx00, idx_hi, idx_lo, idx11);
    } else {
        indices = array<u32, 4>(idx00, idx_lo, idx_hi, idx11);
    }

    // Load the 4 amplitudes (each stored as 2 consecutive f32: real, imag).
    var s: array<vec2<f32>, 4>;
    for (var i = 0u; i < 4u; i++) {
        s[i] = vec2<f32>(state[indices[i] * 2u], state[indices[i] * 2u + 1u]);
    }

    // Apply the 4x4 unitary: result[row] = sum_col mat[row][col] * s[col]
    var result: array<vec2<f32>, 4>;
    for (var row = 0u; row < 4u; row++) {
        result[row] = vec2<f32>(0.0, 0.0);
        for (var col = 0u; col < 4u; col++) {
            let mat_idx = (row * 4u + col) * 2u;
            let mat_re = params.mat[mat_idx];
            let mat_im = params.mat[mat_idx + 1u];
            result[row] += cmul(mat_re, mat_im, s[col].x, s[col].y);
        }
    }

    // Write back.
    for (var i = 0u; i < 4u; i++) {
        state[indices[i] * 2u] = result[i].x;
        state[indices[i] * 2u + 1u] = result[i].y;
    }
}
