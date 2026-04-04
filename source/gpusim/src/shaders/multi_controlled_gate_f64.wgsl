// Multi-controlled gate kernel (f64 emulation variant).
//
// Applies a 2x2 unitary to a target qubit when all control qubits are |1>,
// using double-single precision arithmetic.
//
// ds_math.wgsl is prepended at compile time.

struct MultiControlledParams {
    target_bit: u32,
    num_qubits: u32,
    control_mask: u32,
    _pad: u32,
    // 2x2 complex matrix in DS format: 4 entries * 4 f32s = 16 f32s.
    matrix: array<f32, 16>,
};

@group(0) @binding(0) var<storage, read_write> state: array<f32>;
@group(0) @binding(1) var<uniform> params: MultiControlledParams;

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
        DS(params.matrix[base], params.matrix[base + 1u]),
        DS(params.matrix[base + 2u], params.matrix[base + 3u])
    );
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let thread_id = gid.x;
    let num_pairs = 1u << (params.num_qubits - 1u);

    if (thread_id >= num_pairs) {
        return;
    }

    // Compute the two indices (identical to single-qubit kernel).
    let low_mask = (1u << params.target_bit) - 1u;
    let low_bits = thread_id & low_mask;
    let high_bits = (thread_id >> params.target_bit) << (params.target_bit + 1u);
    let idx0 = high_bits | low_bits;
    let idx1 = idx0 | (1u << params.target_bit);

    // Control check: all bits in control_mask must be set.
    if ((idx0 & params.control_mask) != params.control_mask) {
        return;
    }

    let amp0 = load_amplitude(idx0);
    let amp1 = load_amplitude(idx1);

    let m00 = load_matrix_entry(0u);
    let m01 = load_matrix_entry(1u);
    let m10 = load_matrix_entry(2u);
    let m11 = load_matrix_entry(3u);

    let prod00 = ds_cmul(m00[0], m00[1], amp0[0], amp0[1]);
    let prod01 = ds_cmul(m01[0], m01[1], amp1[0], amp1[1]);
    let new0 = ds_cadd(prod00[0], prod00[1], prod01[0], prod01[1]);

    let prod10 = ds_cmul(m10[0], m10[1], amp0[0], amp0[1]);
    let prod11 = ds_cmul(m11[0], m11[1], amp1[0], amp1[1]);
    let new1 = ds_cadd(prod10[0], prod10[1], prod11[0], prod11[1]);

    store_amplitude(idx0, new0[0], new0[1]);
    store_amplitude(idx1, new1[0], new1[1]);
}
