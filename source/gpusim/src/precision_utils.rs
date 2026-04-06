//! Shared precision characterization utilities.
//!
//! Used by `tests/precision_characterization.rs` and `examples/precision_report.rs`
//! to avoid code duplication for metrics computation and state vector conversion.

use num_bigint::BigUint;
use num_complex::Complex64;

/// Precision metrics for comparing two quantum state vectors.
pub struct PrecisionMetrics {
    /// Worst-case amplitude deviation: max |gpu[i] - ref[i]|.
    pub max_error: f64,
    /// Root-mean-square amplitude deviation.
    pub rms_error: f64,
    /// State fidelity: |<ref|gpu>|^2 (1.0 = identical).
    pub fidelity: f64,
    /// Trace distance: sqrt(1 - fidelity), max measurement probability difference.
    pub trace_distance: f64,
}

/// Convert a sparse state representation to a dense vector.
///
/// Expands `(BigUint, Complex64)` pairs to a full vector of length `2^num_qubits`,
/// inserting zeros for missing basis states.
#[must_use]
#[allow(clippy::cast_possible_truncation)]
pub fn to_dense(sparse: &[(BigUint, Complex64)], num_qubits: usize) -> Vec<Complex64> {
    let dim = 1usize << num_qubits;
    let mut dense = vec![Complex64::new(0.0, 0.0); dim];
    for (idx, amp) in sparse {
        let i: usize = idx.to_u64_digits().first().copied().unwrap_or(0) as usize;
        if i < dim {
            dense[i] = *amp;
        }
    }
    dense
}

/// Compute precision metrics between a GPU (test) and reference state vector.
///
/// Both vectors must have the same length. The reference vector is typically
/// produced by the sparse simulator (f64 precision).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_metrics(gpu_dense: &[Complex64], ref_dense: &[Complex64]) -> PrecisionMetrics {
    let n = ref_dense.len();
    assert_eq!(gpu_dense.len(), n, "state vector length mismatch");

    let mut max_err: f64 = 0.0;
    let mut sum_sq_err: f64 = 0.0;
    let mut inner_product = Complex64::new(0.0, 0.0);

    for i in 0..n {
        let diff = gpu_dense[i] - ref_dense[i];
        let err = diff.norm();
        max_err = max_err.max(err);
        sum_sq_err += err * err;
        inner_product += ref_dense[i].conj() * gpu_dense[i];
    }

    let rms_error = (sum_sq_err / n as f64).sqrt();
    let fidelity = inner_product.norm_sqr();
    let trace_distance = (1.0 - fidelity).max(0.0).sqrt();

    PrecisionMetrics {
        max_error: max_err,
        rms_error,
        fidelity,
        trace_distance,
    }
}
