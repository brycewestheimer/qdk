// Measurement probability reduction kernel (f64 emulation variant).
//
// Computes P(measured qubits have odd parity) using double-single precision
// for both state vector access and probability accumulation.
//
// Uses the same grid-stride + parity check approach as the f32 shader to
// support both single-qubit and joint measurements.
//
// ds_math.wgsl is prepended at compile time.

struct MeasureParams {
    target_bit: u32,
    num_qubits: u32,
    measure_mask: u32,
    _pad: u32,
};

// Shared memory for workgroup reduction: 256 threads * 2 f32s (DS pair).
var<workgroup> shared_sums: array<f32, 512>;

@group(0) @binding(0) var<storage, read> state: array<f32>;
@group(0) @binding(1) var<uniform> params: MeasureParams;
// Output: 2 f32s per workgroup (DS pair: hi, lo).
@group(0) @binding(2) var<storage, read_write> partial_sums: array<f32>;

fn load_amplitude(idx: u32) -> array<DS, 2> {
    let base = idx * 4u;
    return array<DS, 2>(
        DS(state[base], state[base + 1u]),
        DS(state[base + 2u], state[base + 3u])
    );
}

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let thread_id = gid.x;
    let local_id = lid.x;
    let num_amplitudes = 1u << params.num_qubits;

    // Grid-stride loop: total_threads = 256 * num_workgroups.
    // partial_sums stores 2 f32s per workgroup, so num_workgroups = arrayLength / 2.
    let num_workgroups = arrayLength(&partial_sums) / 2u;
    let total_threads = 256u * num_workgroups;
    var local_sum = DS(0.0, 0.0);

    var i = thread_id;
    while (i < num_amplitudes) {
        // Check parity: does this basis state contribute to P(odd)?
        let bits_set = countOneBits(i & params.measure_mask);
        if ((bits_set & 1u) == 1u) {
            let amp = load_amplitude(i);
            let mag2 = ds_cmag2(amp[0], amp[1]);
            local_sum = ds_add(local_sum, mag2);
        }
        i += total_threads;
    }

    // Store this thread's DS partial sum in shared memory.
    shared_sums[local_id * 2u] = local_sum.hi;
    shared_sums[local_id * 2u + 1u] = local_sum.lo;
    workgroupBarrier();

    // Parallel tree reduction using DS addition.
    var stride: u32 = 128u;
    while (stride > 0u) {
        if (local_id < stride) {
            let a = DS(shared_sums[local_id * 2u], shared_sums[local_id * 2u + 1u]);
            let b = DS(shared_sums[(local_id + stride) * 2u], shared_sums[(local_id + stride) * 2u + 1u]);
            let sum = ds_add(a, b);
            shared_sums[local_id * 2u] = sum.hi;
            shared_sums[local_id * 2u + 1u] = sum.lo;
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    // Thread 0 writes this workgroup's DS result.
    if (local_id == 0u) {
        partial_sums[wid.x * 2u] = shared_sums[0];
        partial_sums[wid.x * 2u + 1u] = shared_sums[1];
    }
}
