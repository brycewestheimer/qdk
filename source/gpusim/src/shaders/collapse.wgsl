// collapse.wgsl -- Post-measurement state collapse and renormalization
//
// After measurement determines an outcome (even or odd parity), this shader:
//   - Zeros all amplitudes whose parity with the measure mask does NOT match
//     the measured value.
//   - Scales all surviving amplitudes by the normalization factor
//     (1.0 / sqrt(probability_of_measured_outcome)) to maintain unit norm.
//
// Each thread processes exactly one amplitude. The workgroup count is
// ceil(2^num_qubits / 256).

struct CollapseParams {
    // Bitmask of qubits that were measured (same as in MeasureParams).
    measure_mask: u32,
    // The measurement result: 1 if the measured parity was odd (|1> for
    // single-qubit), 0 if even (|0> for single-qubit).
    measured_value: u32,
    // Precomputed normalization factor: 1.0 / sqrt(P(measured_outcome)).
    // Computed on the CPU to avoid per-thread division on the GPU.
    normalization_factor: f32,
    // Total number of qubits (determines state vector size).
    num_qubits: u32,
};

// Binding 0: State vector (read-write). This shader modifies the state in-place.
@group(0) @binding(0) var<storage, read_write> state: array<f32>;

// Binding 1: Collapse parameters.
@group(0) @binding(1) var<uniform> params: CollapseParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let num_amplitudes = 1u << params.num_qubits;

    // Guard: threads beyond the state vector size do nothing.
    if (i >= num_amplitudes) {
        return;
    }

    // Compute the parity of this basis state with respect to the measured qubits.
    // parity = 1 if an odd number of the measured qubits are |1> in state |i>,
    // parity = 0 if even.
    let parity = countOneBits(i & params.measure_mask) & 1u;

    if (parity == params.measured_value) {
        // This amplitude is consistent with the measurement result.
        // Scale by the normalization factor to maintain unit norm.
        state[i * 2u] *= params.normalization_factor;
        state[i * 2u + 1u] *= params.normalization_factor;
    } else {
        // This amplitude is inconsistent with the measurement result.
        // Zero it out (project onto the measured subspace).
        state[i * 2u] = 0.0;
        state[i * 2u + 1u] = 0.0;
    }
}
