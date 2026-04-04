// Two-qubit gate kernel (f64 emulation variant).
//
// Applies an arbitrary 4x4 unitary to a pair of qubits using double-single
// precision arithmetic. Each amplitude is stored as 4 f32 values.
//
// ds_math.wgsl is prepended at compile time.

struct TwoQubitParams {
    bit_a: u32,
    bit_b: u32,
    num_qubits: u32,
    _pad: u32,
    // 4x4 complex matrix in DS format: 16 entries * 4 f32s each = 64 f32s.
    mat: array<f32, 64>,
};

@group(0) @binding(0) var<storage, read_write> state: array<f32>;
@group(0) @binding(1) var<uniform> params: TwoQubitParams;

fn load_amplitude(idx: u32) -> array<DS, 2> {
    let base = idx * 4u;
    return array<DS, 2>(
        DS(state[base], state[base + 1u]),
        DS(state[base + 2u], state[base + 3u])
    );
}

fn store_amplitude(idx: u32, re: DS, im: DS) {
    let base = idx * 4u;
    state[base] = re.hi;
    state[base + 1u] = re.lo;
    state[base + 2u] = im.hi;
    state[base + 3u] = im.lo;
}

fn load_matrix_entry(entry_idx: u32) -> array<DS, 2> {
    let base = entry_idx * 4u;
    return array<DS, 2>(
        DS(params.mat[base], params.mat[base + 1u]),
        DS(params.mat[base + 2u], params.mat[base + 3u])
    );
}

fn mat_row_dot(row: u32, amps: array<array<DS, 2>, 4>) -> array<DS, 2> {
    var acc_re = DS(0.0, 0.0);
    var acc_im = DS(0.0, 0.0);
    for (var col: u32 = 0u; col < 4u; col++) {
        let m = load_matrix_entry(row * 4u + col);
        let prod = ds_cmul(m[0], m[1], amps[col][0], amps[col][1]);
        acc_re = ds_add(acc_re, prod[0]);
        acc_im = ds_add(acc_im, prod[1]);
    }
    return array<DS, 2>(acc_re, acc_im);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_id = gid.x;
    let num_groups = 1u << (params.num_qubits - 2u);

    if (thread_id >= num_groups) {
        return;
    }

    // Double-bit-insertion: identical to f32 version.
    let lo = min(params.bit_a, params.bit_b);
    let hi = max(params.bit_a, params.bit_b);

    let lo_mask = (1u << lo) - 1u;
    var idx = (thread_id & lo_mask) | ((thread_id & ~lo_mask) << 1u);

    let hi_mask = (1u << hi) - 1u;
    idx = (idx & hi_mask) | ((idx & ~hi_mask) << 1u);

    let idx00 = idx;
    let idx_lo = idx | (1u << lo);
    let idx_hi = idx | (1u << hi);
    let idx11 = idx | (1u << lo) | (1u << hi);

    // Map to canonical 2-qubit basis ordering |q_a q_b>.
    var indices: array<u32, 4>;
    if (params.bit_a < params.bit_b) {
        indices = array<u32, 4>(idx00, idx_hi, idx_lo, idx11);
    } else {
        indices = array<u32, 4>(idx00, idx_lo, idx_hi, idx11);
    }

    // Load the 4 amplitudes.
    var amps: array<array<DS, 2>, 4>;
    for (var i: u32 = 0u; i < 4u; i++) {
        amps[i] = load_amplitude(indices[i]);
    }

    // Apply the 4x4 matrix and store results.
    // Must compute all results before writing to avoid read-after-write hazard.
    var results: array<array<DS, 2>, 4>;
    for (var row: u32 = 0u; row < 4u; row++) {
        results[row] = mat_row_dot(row, amps);
    }
    for (var i: u32 = 0u; i < 4u; i++) {
        store_amplitude(indices[i], results[i][0], results[i][1]);
    }
}
