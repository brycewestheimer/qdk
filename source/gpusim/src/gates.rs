use std::f32::consts::FRAC_1_SQRT_2;

/// Complex number as (real, imaginary) f32 pair.
pub type C32 = (f32, f32);

/// 2x2 unitary matrix in row-major order: [[a, b], [c, d]] stored as [a, b, c, d].
pub type Mat2x2 = [C32; 4];

/// 4x4 unitary matrix in row-major order, stored as 16 (re, im) pairs.
pub type Mat4x4 = [C32; 16];

/// Identity gate.
pub const I: Mat2x2 = [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0)];

/// Pauli-X gate (bit flip).
///
/// Self-adjoint: X† = X, X² = I.
///
/// ```text
/// X = [[0, 1],
///      [1, 0]]
/// ```
pub const X: Mat2x2 = [(0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)];

/// Pauli-Y gate.
///
/// Self-adjoint: Y† = Y, Y² = I.
///
/// ```text
/// Y = [[0, -i],
///      [i,  0]]
/// ```
pub const Y: Mat2x2 = [(0.0, 0.0), (0.0, -1.0), (0.0, 1.0), (0.0, 0.0)];

/// Pauli-Z gate.
///
/// Self-adjoint: Z† = Z, Z² = I.
///
/// ```text
/// Z = [[1,  0],
///      [0, -1]]
/// ```
pub const Z: Mat2x2 = [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (-1.0, 0.0)];

/// Hadamard gate.
///
/// ```text
/// H = (1/sqrt(2)) * [[1,  1],
///                     [1, -1]]
/// ```
pub const H: Mat2x2 = [
    (FRAC_1_SQRT_2, 0.0),
    (FRAC_1_SQRT_2, 0.0),
    (FRAC_1_SQRT_2, 0.0),
    (-FRAC_1_SQRT_2, 0.0),
];

/// S gate (phase gate, sqrt(Z)).
///
/// ```text
/// S = diag(1, i)
/// ```
pub const S: Mat2x2 = [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 1.0)];

/// S-adjoint gate.
///
/// ```text
/// S_adj = diag(1, -i)
/// ```
pub const SADJ: Mat2x2 = [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, -1.0)];

/// T gate (pi/8 gate, sqrt(S)).
///
/// ```text
/// T = diag(1, e^(i*pi/4)) = diag(1, (1+i)/sqrt(2))
/// ```
///
/// Numerically: `e^(i*pi/4) = cos(pi/4) + i*sin(pi/4) = FRAC_1_SQRT_2 + i*FRAC_1_SQRT_2`
pub const T: Mat2x2 = [
    (1.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (FRAC_1_SQRT_2, FRAC_1_SQRT_2),
];

/// T-adjoint gate.
///
/// ```text
/// T_adj = diag(1, e^(-i*pi/4)) = diag(1, (1-i)/sqrt(2))
/// ```
pub const TADJ: Mat2x2 = [
    (1.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (FRAC_1_SQRT_2, -FRAC_1_SQRT_2),
];

/// SX gate (sqrt(X)).
///
/// ```text
/// SX = (1/2) * [[1+i, 1-i],
///                [1-i, 1+i]]
/// ```
pub const SX: Mat2x2 = [(0.5, 0.5), (0.5, -0.5), (0.5, -0.5), (0.5, 0.5)];

/// SX-adjoint gate.
///
/// ```text
/// SX_adj = (1/2) * [[1-i, 1+i],
///                    [1+i, 1-i]]
/// ```
pub const SXADJ: Mat2x2 = [(0.5, -0.5), (0.5, 0.5), (0.5, 0.5), (0.5, -0.5)];

/// Rx(theta) rotation gate.
///
/// ```text
/// Rx(t) = [[cos(t/2),    -i*sin(t/2)],
///          [-i*sin(t/2),  cos(t/2)   ]]
/// ```
///
/// Trig is computed in f64 for precision, then truncated to f32 for the
/// matrix elements. For the f64-emulation path, use [`rx_f64`] instead.
#[must_use]
pub fn rx(theta: f64) -> Mat2x2 {
    let half = theta / 2.0;
    let cos = half.cos();
    let sin = half.sin();
    #[allow(clippy::cast_possible_truncation)]
    let (cos_f32, sin_f32) = (cos as f32, sin as f32);
    [
        (cos_f32, 0.0),
        (0.0, -sin_f32),
        (0.0, -sin_f32),
        (cos_f32, 0.0),
    ]
}

/// Ry(theta) rotation gate.
///
/// ```text
/// Ry(t) = [[cos(t/2), -sin(t/2)],
///          [sin(t/2),  cos(t/2) ]]
/// ```
///
/// Trig is computed in f64 for precision, then truncated to f32.
/// For the f64-emulation path, use [`ry_f64`] instead.
#[must_use]
pub fn ry(theta: f64) -> Mat2x2 {
    let half = theta / 2.0;
    let cos = half.cos();
    let sin = half.sin();
    #[allow(clippy::cast_possible_truncation)]
    let (cos_f32, sin_f32) = (cos as f32, sin as f32);
    [
        (cos_f32, 0.0),
        (-sin_f32, 0.0),
        (sin_f32, 0.0),
        (cos_f32, 0.0),
    ]
}

/// Rz(theta) rotation gate.
///
/// ```text
/// Rz(t) = diag(e^(-i*t/2), e^(i*t/2))
///       = diag(cos(t/2) - i*sin(t/2), cos(t/2) + i*sin(t/2))
/// ```
///
/// Trig is computed in f64 for precision, then truncated to f32.
/// For the f64-emulation path, use [`rz_f64`] instead.
#[must_use]
pub fn rz(theta: f64) -> Mat2x2 {
    let half = theta / 2.0;
    let cos = half.cos();
    let sin = half.sin();
    #[allow(clippy::cast_possible_truncation)]
    let (cos_f32, sin_f32) = (cos as f32, sin as f32);
    [
        (cos_f32, -sin_f32),
        (0.0, 0.0),
        (0.0, 0.0),
        (cos_f32, sin_f32),
    ]
}

/// Constructs a phase gate: `diag(1, phase)` where `phase` is given as
/// `(re, im)` components. Accepts f64 and truncates to f32.
///
/// For the f64-emulation path, use [`phase_gate_f64`] instead.
#[must_use]
pub fn phase_gate(phase_re: f64, phase_im: f64) -> Mat2x2 {
    #[allow(clippy::cast_possible_truncation)]
    let (re, im) = (phase_re as f32, phase_im as f32);
    [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (re, im)]
}

/// Complex number as (real, imaginary) f64 pair, for f64-emulation path.
#[cfg(feature = "f64_emulation")]
pub type C64Pair = (f64, f64);

/// 2x2 unitary matrix with f64 elements, for f64-emulation DS encoding.
#[cfg(feature = "f64_emulation")]
pub type Mat2x2F64 = [C64Pair; 4];

/// Rx(theta) rotation gate with f64 precision matrix elements.
///
/// Returns the matrix with full f64 trig values for direct DS encoding.
/// Used only in the f64-emulation path.
#[cfg(feature = "f64_emulation")]
#[must_use]
pub fn rx_f64(theta: f64) -> Mat2x2F64 {
    let half = theta / 2.0;
    let cos = half.cos();
    let sin = half.sin();
    [(cos, 0.0), (0.0, -sin), (0.0, -sin), (cos, 0.0)]
}

/// Ry(theta) rotation gate with f64 precision matrix elements.
#[cfg(feature = "f64_emulation")]
#[must_use]
pub fn ry_f64(theta: f64) -> Mat2x2F64 {
    let half = theta / 2.0;
    let cos = half.cos();
    let sin = half.sin();
    [(cos, 0.0), (-sin, 0.0), (sin, 0.0), (cos, 0.0)]
}

/// Rz(theta) rotation gate with f64 precision matrix elements.
#[cfg(feature = "f64_emulation")]
#[must_use]
pub fn rz_f64(theta: f64) -> Mat2x2F64 {
    let half = theta / 2.0;
    let cos = half.cos();
    let sin = half.sin();
    [(cos, -sin), (0.0, 0.0), (0.0, 0.0), (cos, sin)]
}

/// Phase gate with f64 precision: `diag(1, phase_re + i*phase_im)`.
#[cfg(feature = "f64_emulation")]
#[must_use]
pub fn phase_gate_f64(phase_re: f64, phase_im: f64) -> Mat2x2F64 {
    [(1.0, 0.0), (0.0, 0.0), (0.0, 0.0), (phase_re, phase_im)]
}

/// CNOT (CX) gate.
///
/// Control = qubit 0 (row index), target = qubit 1 (column index).
/// In the computational basis {|00>, |01>, |10>, |11>}:
///
/// ```text
/// CNOT = [[1, 0, 0, 0],
///         [0, 1, 0, 0],
///         [0, 0, 0, 1],
///         [0, 0, 1, 0]]
/// ```
pub const CNOT: Mat4x4 = [
    (1.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 0.0),
];

/// SWAP gate.
///
/// ```text
/// SWAP = [[1, 0, 0, 0],
///         [0, 0, 1, 0],
///         [0, 1, 0, 0],
///         [0, 0, 0, 1]]
/// ```
pub const SWAP: Mat4x4 = [
    (1.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (1.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (0.0, 0.0),
    (1.0, 0.0),
];

// ---------------------------------------------------------------------------
// Test helpers and unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Complex multiply: `(a_re + i*a_im) * (b_re + i*b_im)`
    fn cmul(a: C32, b: C32) -> C32 {
        (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
    }

    /// Complex add.
    fn cadd(a: C32, b: C32) -> C32 {
        (a.0 + b.0, a.1 + b.1)
    }

    /// Conjugate transpose of a 2x2 matrix.
    fn adjoint(m: &Mat2x2) -> Mat2x2 {
        [
            (m[0].0, -m[0].1),
            (m[2].0, -m[2].1),
            (m[1].0, -m[1].1),
            (m[3].0, -m[3].1),
        ]
    }

    /// Multiply two 2x2 matrices.
    fn mul_2x2(a: &Mat2x2, b: &Mat2x2) -> Mat2x2 {
        [
            cadd(cmul(a[0], b[0]), cmul(a[1], b[2])),
            cadd(cmul(a[0], b[1]), cmul(a[1], b[3])),
            cadd(cmul(a[2], b[0]), cmul(a[3], b[2])),
            cadd(cmul(a[2], b[1]), cmul(a[3], b[3])),
        ]
    }

    /// Assert that a 2x2 matrix is approximately the identity.
    fn assert_identity(m: &Mat2x2, tolerance: f32) {
        // Diagonal elements should be (1, 0)
        assert!(
            (m[0].0 - 1.0).abs() < tolerance && m[0].1.abs() < tolerance,
            "m[0,0] = ({}, {}) should be (1, 0)",
            m[0].0,
            m[0].1
        );
        assert!(
            (m[3].0 - 1.0).abs() < tolerance && m[3].1.abs() < tolerance,
            "m[1,1] = ({}, {}) should be (1, 0)",
            m[3].0,
            m[3].1
        );
        // Off-diagonal elements should be (0, 0)
        assert!(
            m[1].0.abs() < tolerance && m[1].1.abs() < tolerance,
            "m[0,1] = ({}, {}) should be (0, 0)",
            m[1].0,
            m[1].1
        );
        assert!(
            m[2].0.abs() < tolerance && m[2].1.abs() < tolerance,
            "m[1,0] = ({}, {}) should be (0, 0)",
            m[2].0,
            m[2].1
        );
    }

    /// Assert that a 2x2 matrix is unitary: `U * U_dagger = I`.
    fn assert_unitary(m: &Mat2x2, tolerance: f32) {
        let product = mul_2x2(m, &adjoint(m));
        assert_identity(&product, tolerance);
    }

    const TOL: f32 = 1e-6;

    #[test]
    fn identity_is_unitary() {
        assert_unitary(&I, TOL);
    }

    #[test]
    fn pauli_x_is_unitary() {
        assert_unitary(&X, TOL);
    }

    #[test]
    fn pauli_y_is_unitary() {
        assert_unitary(&Y, TOL);
    }

    #[test]
    fn pauli_z_is_unitary() {
        assert_unitary(&Z, TOL);
    }

    #[test]
    fn hadamard_is_unitary() {
        assert_unitary(&H, TOL);
    }

    #[test]
    fn s_is_unitary() {
        assert_unitary(&S, TOL);
    }

    #[test]
    fn sadj_is_unitary() {
        assert_unitary(&SADJ, TOL);
    }

    #[test]
    fn t_is_unitary() {
        assert_unitary(&T, TOL);
    }

    #[test]
    fn tadj_is_unitary() {
        assert_unitary(&TADJ, TOL);
    }

    #[test]
    fn sx_is_unitary() {
        assert_unitary(&SX, TOL);
    }

    #[test]
    fn sxadj_is_unitary() {
        assert_unitary(&SXADJ, TOL);
    }

    #[test]
    fn rx_at_pi() {
        // Rx(pi) = [[cos(pi/2), -i*sin(pi/2)], [-i*sin(pi/2), cos(pi/2)]]
        //         = [[0, -i], [-i, 0]]
        //         = -i * X (up to global phase)
        let m = rx(std::f64::consts::PI);
        assert!((m[0].0).abs() < TOL, "rx(pi)[0,0].re should be ~0");
        assert!((m[0].1).abs() < TOL, "rx(pi)[0,0].im should be ~0");
        assert!((m[1].0).abs() < TOL, "rx(pi)[0,1].re should be ~0");
        assert!((m[1].1 + 1.0).abs() < TOL, "rx(pi)[0,1].im should be ~-1");
        assert!((m[2].0).abs() < TOL, "rx(pi)[1,0].re should be ~0");
        assert!((m[2].1 + 1.0).abs() < TOL, "rx(pi)[1,0].im should be ~-1");
        assert!((m[3].0).abs() < TOL, "rx(pi)[1,1].re should be ~0");
        assert!((m[3].1).abs() < TOL, "rx(pi)[1,1].im should be ~0");
    }

    #[test]
    fn ry_at_pi() {
        // Ry(pi) = [[cos(pi/2), -sin(pi/2)], [sin(pi/2), cos(pi/2)]]
        //         = [[0, -1], [1, 0]]
        let m = ry(std::f64::consts::PI);
        assert!((m[0].0).abs() < TOL);
        assert!((m[1].0 + 1.0).abs() < TOL);
        assert!((m[2].0 - 1.0).abs() < TOL);
        assert!((m[3].0).abs() < TOL);
        // All imaginary parts should be 0
        for &(_, im) in &m {
            assert!(im.abs() < TOL);
        }
    }

    #[test]
    fn rz_at_pi() {
        // Rz(pi) = diag(e^(-i*pi/2), e^(i*pi/2)) = diag(-i, i)
        let m = rz(std::f64::consts::PI);
        assert!((m[0].0).abs() < TOL, "rz(pi)[0,0].re should be ~0");
        assert!((m[0].1 + 1.0).abs() < TOL, "rz(pi)[0,0].im should be ~-1");
        assert!((m[3].0).abs() < TOL, "rz(pi)[1,1].re should be ~0");
        assert!((m[3].1 - 1.0).abs() < TOL, "rz(pi)[1,1].im should be ~1");
        // Off-diagonal should be 0
        assert!(m[1].0.abs() < TOL && m[1].1.abs() < TOL);
        assert!(m[2].0.abs() < TOL && m[2].1.abs() < TOL);
    }

    #[test]
    fn rx_is_unitary_at_various_angles() {
        for angle in [
            0.0,
            0.5,
            1.0,
            std::f64::consts::PI,
            2.0 * std::f64::consts::PI,
            3.7,
        ] {
            assert_unitary(&rx(angle), TOL);
        }
    }

    #[test]
    fn ry_is_unitary_at_various_angles() {
        for angle in [
            0.0,
            0.5,
            1.0,
            std::f64::consts::PI,
            2.0 * std::f64::consts::PI,
            3.7,
        ] {
            assert_unitary(&ry(angle), TOL);
        }
    }

    #[test]
    fn rz_is_unitary_at_various_angles() {
        for angle in [
            0.0,
            0.5,
            1.0,
            std::f64::consts::PI,
            2.0 * std::f64::consts::PI,
            3.7,
        ] {
            assert_unitary(&rz(angle), TOL);
        }
    }

    #[test]
    fn s_squared_is_z() {
        // S * S = Z
        let ss = mul_2x2(&S, &S);
        for i in 0..4 {
            assert!(
                (ss[i].0 - Z[i].0).abs() < TOL && (ss[i].1 - Z[i].1).abs() < TOL,
                "S*S[{i}] = ({}, {}) != Z[{i}] = ({}, {})",
                ss[i].0,
                ss[i].1,
                Z[i].0,
                Z[i].1
            );
        }
    }

    #[test]
    fn t_squared_is_s() {
        // T * T = S
        let tt = mul_2x2(&T, &T);
        for i in 0..4 {
            assert!(
                (tt[i].0 - S[i].0).abs() < TOL && (tt[i].1 - S[i].1).abs() < TOL,
                "T*T[{i}] = ({}, {}) != S[{i}] = ({}, {})",
                tt[i].0,
                tt[i].1,
                S[i].0,
                S[i].1
            );
        }
    }

    #[cfg(feature = "f64_emulation")]
    #[test]
    fn rx_f64_at_small_angle() {
        // At very small angles, f64 trig is significantly more precise than f32 trig.
        let theta = 1e-8_f64;
        let m = rx_f64(theta);
        let expected_cos = (theta / 2.0).cos();
        let expected_sin = (theta / 2.0).sin();
        assert!(
            (m[0].0 - expected_cos).abs() < 1e-15,
            "f64 rx cos precision"
        );
        assert!(
            (m[1].1 - (-expected_sin)).abs() < 1e-15,
            "f64 rx sin precision"
        );
    }
}
