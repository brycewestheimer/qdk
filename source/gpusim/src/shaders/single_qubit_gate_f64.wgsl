// Single-qubit gate kernel (f64 emulation variant).
//
// Applies a 2x2 unitary matrix to the target qubit using double-single
// precision arithmetic. Each amplitude is stored as 4 f32 values:
// [re_hi, re_lo, im_hi, im_lo].
//
// ds_math.wgsl is prepended at compile time.

struct GateParams {
    target_bit: u32,
    num_qubits: u32,
    _pad0: u32,
    _pad1: u32,
    // 2x2 complex matrix in DS format: 4 entries * 4 f32s each = 16 f32s.
    // Layout: [m00_re_hi, m00_re_lo, m00_im_hi, m00_im_lo,
    //          m01_re_hi, m01_re_lo, m01_im_hi, m01_im_lo,
    //          m10_re_hi, m10_re_lo, m10_im_hi, m10_im_lo,
    //          m11_re_hi, m11_re_lo, m11_im_hi, m11_im_lo]
    matrix: array<f32, 16>,
};

@group(0) @binding(0) var<storage, read_write> state: array<f32>;
@group(0) @binding(1) var<uniform> params: GateParams;

fn load_amplitude(idx: u32) -> array<DS, 2> {
    let base = idx * 4u;
    let re = DS(state[base], state[base + 1u]);
    let im = DS(state[base + 2u], state[base + 3u]);
    return array<DS, 2>(re, im);
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
    let re = DS(params.matrix[base], params.matrix[base + 1u]);
    let im = DS(params.matrix[base + 2u], params.matrix[base + 3u]);
    return array<DS, 2>(re, im);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let thread_id = global_id.x;
    let num_pairs = 1u << (params.num_qubits - 1u);

    if (thread_id >= num_pairs) {
        return;
    }

    // Compute the state vector indices for the amplitude pair.
    let low_mask = (1u << params.target_bit) - 1u;
    let low_bits = thread_id & low_mask;
    let high_bits = (thread_id >> params.target_bit) << (params.target_bit + 1u);
    let idx0 = high_bits | low_bits;
    let idx1 = idx0 | (1u << params.target_bit);

    // Load the two amplitudes.
    let amp0 = load_amplitude(idx0);
    let amp1 = load_amplitude(idx1);

    // Load the 2x2 gate matrix entries.
    let m00 = load_matrix_entry(0u);
    let m01 = load_matrix_entry(1u);
    let m10 = load_matrix_entry(2u);
    let m11 = load_matrix_entry(3u);

    // new0 = m00 * amp0 + m01 * amp1
    let prod00 = ds_cmul(m00[0], m00[1], amp0[0], amp0[1]);
    let prod01 = ds_cmul(m01[0], m01[1], amp1[0], amp1[1]);
    let new0 = ds_cadd(prod00[0], prod00[1], prod01[0], prod01[1]);

    // new1 = m10 * amp0 + m11 * amp1
    let prod10 = ds_cmul(m10[0], m10[1], amp0[0], amp0[1]);
    let prod11 = ds_cmul(m11[0], m11[1], amp1[0], amp1[1]);
    let new1 = ds_cadd(prod10[0], prod10[1], prod11[0], prod11[1]);

    // Store results.
    store_amplitude(idx0, new0[0], new0[1]);
    store_amplitude(idx1, new1[0], new1[1]);
}
