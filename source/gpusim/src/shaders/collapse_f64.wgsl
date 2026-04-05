// Post-measurement state collapse kernel (f64 emulation variant).
//
// Zeros amplitudes inconsistent with the measurement result and renormalizes
// surviving amplitudes using double-single precision.
//
// Uses measure_mask + parity check (same as f32 shader) to support both
// single-qubit and joint measurements.
//
// ds_math.wgsl is prepended at compile time.

struct CollapseParams {
    measure_mask: u32,
    measured_value: u32,
    num_qubits: u32,
    num_workgroups: u32,
    // DS normalization factor: 1/sqrt(P(outcome)), stored as (hi, lo).
    norm_hi: f32,
    norm_lo: f32,
    _pad2: u32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read_write> state: array<f32>;
@group(0) @binding(1) var<uniform> params: CollapseParams;

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

fn store_zero(idx: u32) {
    let base = idx * 4u;
    state[base] = 0.0;
    state[base + 1u] = 0.0;
    state[base + 2u] = 0.0;
    state[base + 3u] = 0.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let num_amplitudes = 1u << params.num_qubits;
    let total_threads = 256u * params.num_workgroups;

    var i = gid.x;
    while (i < num_amplitudes) {
        // Parity check: same logic as f32 shader.
        let parity = countOneBits(i & params.measure_mask) & 1u;

        if (parity == params.measured_value) {
            // Renormalize using full DS multiplication.
            let norm = DS(params.norm_hi, params.norm_lo);
            let amp = load_amplitude(i);
            let re_scaled = ds_mul(amp[0], norm);
            let im_scaled = ds_mul(amp[1], norm);
            store_amplitude(i, re_scaled, im_scaled);
        } else {
            store_zero(i);
        }

        i += total_threads;
    }
}
