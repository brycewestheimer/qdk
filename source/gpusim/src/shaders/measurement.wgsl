// measurement.wgsl -- Parallel probability reduction for quantum measurement
//
// Computes P(measured qubits have odd parity) = sum of |alpha_i|^2 for all
// basis states i where countOneBits(i & measure_mask) is odd.
//
// For a single-qubit measurement of qubit at bit position k, the host sets
// measure_mask = 1u << k, and the result is P(qubit k = |1>).
//
// The shader performs a two-level reduction:
//   Level 1 (per-thread): Grid-stride loop accumulates a local sum.
//   Level 2 (per-workgroup): Shared-memory tree reduction produces one partial
//                            sum per workgroup, written to an output buffer.
// The host reads the partial sums buffer (a few hundred f32s) and finishes
// the reduction on the CPU with Kahan compensated summation.

struct MeasureParams {
    // Bit position of the target qubit. Used by the host for potential
    // single-qubit fast paths. The shader always uses measure_mask.
    target_bit: u32,
    // Total number of qubits in the simulation (determines state vector size).
    num_qubits: u32,
    // Bitmask of qubits being measured. For single-qubit measurement,
    // measure_mask = 1u << target_bit. For joint measurement, multiple bits set.
    measure_mask: u32,
    // Padding to 16-byte alignment (required for uniform buffers).
    _pad: u32,
};

// Workgroup-local shared memory for the tree reduction.
// Size matches the workgroup size (256 threads).
var<workgroup> shared_sum: array<f32, 256>;

// Binding 0: State vector (interleaved re/im f32 pairs). READ-ONLY -- the
// measurement shader does not modify the state vector. Declaring as read
// lets the driver overlap this dispatch with prior writes.
@group(0) @binding(0) var<storage, read> state: array<f32>;

// Binding 1: Measurement parameters.
@group(0) @binding(1) var<uniform> params: MeasureParams;

// Binding 2: Output buffer for workgroup-level partial sums. One f32 per
// workgroup. The host reads this back to finish the summation.
@group(0) @binding(2) var<storage, read_write> partial_sums: array<f32>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let thread_id = gid.x;
    let local_id = lid.x;
    let num_amplitudes = 1u << params.num_qubits;

    // --- Level 1: Per-thread accumulation via grid-stride loop ---
    //
    // Total threads across all workgroups = 256 * num_workgroups.
    // num_workgroups = arrayLength(&partial_sums) because we allocate
    // exactly one f32 per workgroup.
    //
    // Each thread handles amplitudes at indices:
    //   thread_id, thread_id + total_threads, thread_id + 2*total_threads, ...
    //
    // This ensures coalesced memory access within each workgroup and handles
    // state vectors larger than the total thread count.
    let total_threads = 256u * arrayLength(&partial_sums);
    var local_sum: f32 = 0.0;

    var i = thread_id;
    while (i < num_amplitudes) {
        // Check if this basis state has odd parity with the measure mask.
        // countOneBits is a built-in that counts set bits (popcount).
        let bits_set = countOneBits(i & params.measure_mask);
        if ((bits_set & 1u) == 1u) {
            // This basis state contributes to P(odd parity).
            // Amplitude is stored as interleaved (re, im) pairs.
            let re = state[i * 2u];
            let im = state[i * 2u + 1u];
            local_sum += re * re + im * im;
        }
        i += total_threads;
    }

    // Store this thread's partial sum in shared memory.
    shared_sum[local_id] = local_sum;
    workgroupBarrier();

    // --- Level 2: Workgroup-level tree reduction in shared memory ---
    //
    // Halve the active thread count at each step. After log2(256) = 8 steps,
    // shared_sum[0] contains the sum for this entire workgroup.
    var stride = 128u;
    while (stride > 0u) {
        if (local_id < stride) {
            shared_sum[local_id] += shared_sum[local_id + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    // The first thread in each workgroup writes the workgroup's partial sum.
    if (local_id == 0u) {
        partial_sums[wid.x] = shared_sum[0];
    }
}
