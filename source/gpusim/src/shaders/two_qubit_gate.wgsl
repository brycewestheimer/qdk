// Two-qubit gate kernel: applies an arbitrary 4x4 unitary to a pair of qubits.
//
// The 4x4 matrix operates in the basis |q_a q_b> where q_a is the qubit at
// bit position `bit_a` and q_b is the qubit at bit position `bit_b`.

struct TwoQubitParams {
    // 4x4 unitary matrix stored as 16 complex numbers in row-major order,
    // packed as array<vec4<f32>, 8> for uniform buffer layout compliance.
    // Logical element i: mat[i / 4u][i % 4u].
    mat: array<vec4<f32>, 8>,
    // Qubit bit positions (not necessarily ordered).
    bit_a: u32,
    bit_b: u32,
    // Total number of qubits in the system.
    num_qubits: u32,
    // Number of workgroups dispatched (for grid-stride loop).
    num_workgroups: u32,
};

@group(0) @binding(0) var<storage, read_write> state: array<f32>;
@group(0) @binding(1) var<uniform> params: TwoQubitParams;

fn mat_elem(i: u32) -> f32 {
    return params.mat[i / 4u][i % 4u];
}

fn cmul(a_re: f32, a_im: f32, b_re: f32, b_im: f32) -> vec2<f32> {
    return vec2<f32>(
        a_re * b_re - a_im * b_im,
        a_re * b_im + a_im * b_re
    );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let num_groups = 1u << (params.num_qubits - 2u);
    let total_threads = 256u * params.num_workgroups;

    var thread_id = gid.x;
    while (thread_id < num_groups) {
        // Determine the lower and higher bit positions.
        let lo = min(params.bit_a, params.bit_b);
        let hi = max(params.bit_a, params.bit_b);

        // --- Double-bit-insertion algorithm ---
        //
        // Map thread_id (a (num_qubits-2)-bit number) to a state vector index
        // where bits at positions `lo` and `hi` are both zero.
        //
        // Step 1: Insert a 0-bit at position `lo`.
        let lo_mask = (1u << lo) - 1u;
        var idx = (thread_id & lo_mask) | ((thread_id & ~lo_mask) << 1u);

        // Step 2: Insert a 0-bit at position `hi`.
        let hi_mask = (1u << hi) - 1u;
        idx = (idx & hi_mask) | ((idx & ~hi_mask) << 1u);

        // Build the 4 indices by setting each combination of bit positions.
        let idx00 = idx;
        let idx_lo = idx | (1u << lo);
        let idx_hi = idx | (1u << hi);
        let idx11 = idx | (1u << lo) | (1u << hi);

        // --- Basis ordering explanation ---
        //
        // The 4x4 matrix rows/columns are in |q_a q_b> order:
        //   row/col 0 = |00>  (q_a=0, q_b=0)
        //   row/col 1 = |01>  (q_a=0, q_b=1)
        //   row/col 2 = |10>  (q_a=1, q_b=0)
        //   row/col 3 = |11>  (q_a=1, q_b=1)
        //
        // Example: CNOT with bit_a=0 (control), bit_b=2 (target), num_qubits=3
        //   lo = 0, hi = 2
        //   For thread_id = 0: idx = 0
        //     idx00 = 0 (binary 000)
        //     idx_lo = 1 (binary 001, bit 0 set)
        //     idx_hi = 4 (binary 100, bit 2 set)
        //     idx11 = 5 (binary 101, bits 0 and 2 set)
        //
        //   Since bit_a (0) < bit_b (2):
        //     indices = [idx00, idx_hi, idx_lo, idx11] = [0, 4, 1, 5]
        //     |01> means q_a=0, q_b=1 -> bit 2 set -> index 4  ✓
        //     |10> means q_a=1, q_b=0 -> bit 0 set -> index 1  ✓
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
                let m_re = mat_elem(mat_idx);
                let m_im = mat_elem(mat_idx + 1u);
                result[row] += cmul(m_re, m_im, s[col].x, s[col].y);
            }
        }

        // Write back.
        for (var i = 0u; i < 4u; i++) {
            state[indices[i] * 2u] = result[i].x;
            state[indices[i] * 2u + 1u] = result[i].y;
        }

        thread_id += total_threads;
    }
}
