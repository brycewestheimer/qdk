// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! Compile-time precision abstraction for the GPU simulator.
//!
//! The `Precision` trait parameterizes buffer layouts and shader selection
//! at compile time via Cargo feature flags. This is NOT a runtime dispatch
//! mechanism.

use num_complex::Complex64;

/// Trait abstracting over f32 and f64-emulated precision modes.
///
/// Each precision mode determines:
/// - How many bytes and floats per complex amplitude in the GPU state buffer
/// - Which WGSL shader variants to use
/// - How to encode gate parameters for the GPU
/// - How to decode state amplitudes from the GPU
pub trait Precision {
    /// Bytes per complex amplitude in the GPU state buffer.
    const BYTES_PER_AMPLITUDE: u64;

    /// Number of f32 values per complex amplitude.
    const FLOATS_PER_AMPLITUDE: u32;

    /// Number of f32 values per partial sum in the measurement shader output.
    /// f32 mode: 1 (single f32). f64 mode: 2 (DS pair: hi, lo).
    const FLOATS_PER_PARTIAL_SUM: u32;

    /// Encode a complex value into f32s for GPU upload.
    fn encode_complex(val: Complex64) -> Vec<f32>;

    /// Decode f32s from GPU readback into a complex value.
    fn decode_complex(floats: &[f32]) -> Complex64;

    /// Encode an f64 value into f32s for GPU uniform upload (e.g., theta).
    fn encode_f64(val: f64) -> Vec<f32>;
}

/// Standard f32 precision: 2 floats per amplitude (re, im).
pub struct F32Precision;

impl Precision for F32Precision {
    const BYTES_PER_AMPLITUDE: u64 = 8;
    const FLOATS_PER_AMPLITUDE: u32 = 2;
    const FLOATS_PER_PARTIAL_SUM: u32 = 1;

    #[allow(clippy::cast_possible_truncation)]
    fn encode_complex(val: Complex64) -> Vec<f32> {
        vec![val.re as f32, val.im as f32]
    }

    fn decode_complex(floats: &[f32]) -> Complex64 {
        debug_assert!(
            floats.len() >= 2,
            "f32 decode requires at least 2 floats, got {}",
            floats.len()
        );
        Complex64::new(f64::from(floats[0]), f64::from(floats[1]))
    }

    #[allow(clippy::cast_possible_truncation)]
    fn encode_f64(val: f64) -> Vec<f32> {
        vec![val as f32]
    }
}

/// Double-single emulated f64 precision: 4 floats per amplitude
/// (`re_hi`, `re_lo`, `im_hi`, `im_lo`).
///
/// Each component of the complex number is represented as a pair of f32
/// values where `true_value = hi + lo`. This provides approximately 14
/// decimal digits of precision.
#[cfg(feature = "f64_emulation")]
pub struct F64EmulatedPrecision;

#[cfg(feature = "f64_emulation")]
impl Precision for F64EmulatedPrecision {
    const BYTES_PER_AMPLITUDE: u64 = 16;
    const FLOATS_PER_AMPLITUDE: u32 = 4;
    const FLOATS_PER_PARTIAL_SUM: u32 = 2;

    fn encode_complex(val: Complex64) -> Vec<f32> {
        let (re_hi, re_lo) = to_ds(val.re);
        let (im_hi, im_lo) = to_ds(val.im);
        vec![re_hi, re_lo, im_hi, im_lo]
    }

    fn decode_complex(floats: &[f32]) -> Complex64 {
        debug_assert!(
            floats.len() >= 4,
            "f64 decode requires at least 4 floats, got {}",
            floats.len()
        );
        Complex64::new(
            f64::from(floats[0]) + f64::from(floats[1]),
            f64::from(floats[2]) + f64::from(floats[3]),
        )
    }

    fn encode_f64(val: f64) -> Vec<f32> {
        let (hi, lo) = to_ds(val);
        vec![hi, lo]
    }
}

/// Split an f64 value into a double-single pair (hi, lo) of f32 values
/// such that `f64::from(hi) + f64::from(lo)` approximates the original
/// value with approximately 14 decimal digits of precision.
///
/// The split works by:
/// 1. Casting to f32 to get the high part (rounds to nearest f32).
/// 2. Computing the residual in f64 and casting that to f32.
///
/// The result satisfies: `|val - (f64::from(hi) + f64::from(lo))| < eps^2`
/// where `eps = 2^-24` (f32 machine epsilon).
#[cfg(feature = "f64_emulation")]
#[must_use]
pub fn to_ds(val: f64) -> (f32, f32) {
    #[allow(clippy::cast_possible_truncation)]
    let hi = val as f32;
    #[allow(clippy::cast_possible_truncation)]
    let lo = (val - f64::from(hi)) as f32;
    (hi, lo)
}

/// Reconstruct an f64 from a double-single pair.
#[cfg(feature = "f64_emulation")]
#[must_use]
pub fn from_ds(hi: f32, lo: f32) -> f64 {
    f64::from(hi) + f64::from(lo)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_encode_decode_roundtrip() {
        let val = Complex64::new(0.707_106_781_186_547_5, -0.707_106_781_186_547_5);
        let encoded = F32Precision::encode_complex(val);
        let decoded = F32Precision::decode_complex(&encoded);
        assert!((val.re - decoded.re).abs() < 1e-6);
        assert!((val.im - decoded.im).abs() < 1e-6);
    }

    #[cfg(feature = "f64_emulation")]
    #[test]
    fn ds_roundtrip_pi() {
        let val = std::f64::consts::PI;
        let (hi, lo) = to_ds(val);
        let reconstructed = from_ds(hi, lo);
        assert!(
            (val - reconstructed).abs() < 1e-14,
            "DS roundtrip error too large: {:.2e}",
            (val - reconstructed).abs()
        );
    }

    #[cfg(feature = "f64_emulation")]
    #[test]
    fn ds_roundtrip_small_value() {
        let val = 1.234_567_890_123_456e-10;
        let (hi, lo) = to_ds(val);
        let reconstructed = from_ds(hi, lo);
        assert!(
            (val - reconstructed).abs() < val.abs() * 1e-7,
            "DS roundtrip relative error too large"
        );
    }

    #[cfg(feature = "f64_emulation")]
    #[test]
    fn f64_encode_decode_roundtrip() {
        let val = Complex64::new(std::f64::consts::PI, -std::f64::consts::E);
        let encoded = F64EmulatedPrecision::encode_complex(val);
        let decoded = F64EmulatedPrecision::decode_complex(&encoded);
        assert!(
            (val.re - decoded.re).abs() < 1e-14,
            "Real part error: {:.2e}",
            (val.re - decoded.re).abs()
        );
        assert!(
            (val.im - decoded.im).abs() < 1e-14,
            "Imag part error: {:.2e}",
            (val.im - decoded.im).abs()
        );
    }
}
