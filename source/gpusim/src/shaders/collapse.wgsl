// collapse.wgsl -- Post-measurement state collapse and renormalization
//
// After measurement determines an outcome (even or odd parity), this shader:
//   - Zeros all amplitudes whose parity with the measure mask does NOT match
//     the measured value.
//   - Scales all surviving amplitudes by the normalization factor
//     (1.0 / sqrt(probability_of_measured_outcome)) to maintain unit norm.
//
// Each thread processes one or more amplitudes via a grid-stride loop.

struct CollapseParams {
    // Bitmask of qubits that were measured (same as in MeasureParams).
    measure_mask: u32,
    // The measurement result: 1 if the measured parity was odd (|1> for
    // single-qubit), 0 if even (|0> for single-qubit).
    measured_value: u32,
    // Precomputed normalization factor: 1.0 / sqrt(P(measured_outcome)).
    normalization_factor: f32,
    // Total number of qubits (determines state vector size).
    num_qubits: u32,
    // Number of workgroups dispatched (for grid-stride loop).
    num_workgroups: u32,
    // Padding to align to 16 bytes.
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// Binding 0: State vector (read-write). This shader modifies the state in-place.
@group(0) @binding(0) var<storage, read_write> state: array<f32>;

// Binding 1: Collapse parameters.
@group(0) @binding(1) var<uniform> params: CollapseParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let num_amplitudes = 1u << params.num_qubits;
    let total_threads = 256u * params.num_workgroups;

    var i = gid.x;
    while (i < num_amplitudes) {
        // Compute the parity of this basis state with respect to the measured qubits.
        let parity = countOneBits(i & params.measure_mask) & 1u;

        if (parity == params.measured_value) {
            // This amplitude is consistent with the measurement result.
            state[i * 2u] *= params.normalization_factor;
            state[i * 2u + 1u] *= params.normalization_factor;
        } else {
            // This amplitude is inconsistent -- zero it out.
            state[i * 2u] = 0.0;
            state[i * 2u + 1u] = 0.0;
        }

        i += total_threads;
    }
}
