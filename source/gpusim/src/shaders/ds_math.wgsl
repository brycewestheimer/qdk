// Double-single arithmetic for f64 emulation.
//
// A DS value represents a number as the sum of two f32 values: hi + lo.
// This provides approximately 14 decimal digits of precision using only
// f32 hardware.
//
// References:
//   Knuth, D.E. (1969). "The Art of Computer Programming", Vol 2, Section 4.2.2
//   Dekker, T.J. (1971). "A floating-point technique for extending the available precision"
//   Shewchuk, J.R. (1997). "Adaptive Precision Floating-Point Arithmetic"

struct DS {
    hi: f32,
    lo: f32,
};

// Construct a DS from a single f32 value.
fn ds_from_f32(v: f32) -> DS {
    return DS(v, 0.0);
}

// Double-single addition using the Knuth two-sum algorithm.
//
// Given a = (a.hi + a.lo) and b = (b.hi + b.lo), computes their sum
// with full double-single precision.
//
// Cost: 10 f32 add/sub operations.
fn ds_add(a: DS, b: DS) -> DS {
    let s = a.hi + b.hi;
    let v = s - a.hi;
    let lo = (a.hi - (s - v)) + (b.hi - v) + a.lo + b.lo;
    return DS(s + lo, lo - ((s + lo) - s));
}

// Double-single subtraction: a - b.
fn ds_sub(a: DS, b: DS) -> DS {
    return ds_add(a, DS(-b.hi, -b.lo));
}

// Double-single negation.
fn ds_neg(a: DS) -> DS {
    return DS(-a.hi, -a.lo);
}

// Double-single multiplication using the Dekker product with FMA.
//
// CRITICAL: fma() must be a true fused multiply-add (IEEE 754 fusedMultiplyAdd).
// If the GPU decomposes fma into separate multiply+add, the error term becomes
// zero and precision degrades to plain f32.
//
// Cost: 3 f32 mul + 1 fma + 4 f32 add/sub = ~8 f32 operations.
fn ds_mul(a: DS, b: DS) -> DS {
    let p = a.hi * b.hi;
    let e = fma(a.hi, b.hi, -p) + a.hi * b.lo + a.lo * b.hi;
    return DS(p + e, e - ((p + e) - p));
}

// Double-single scaling by a plain f32 value.
fn ds_scale(a: DS, s: f32) -> DS {
    let p = a.hi * s;
    let e = fma(a.hi, s, -p) + a.lo * s;
    return DS(p + e, e - ((p + e) - p));
}

// ---- Complex DS operations ----
//
// A complex DS number is represented as a pair of DS values: (re, im).
// We represent this as array<DS, 2> where index 0 is real, index 1 is imaginary.

// Complex DS multiplication:
//   (a_re + i*a_im) * (b_re + i*b_im)
//   = (a_re*b_re - a_im*b_im) + i*(a_re*b_im + a_im*b_re)
fn ds_cmul(a_re: DS, a_im: DS, b_re: DS, b_im: DS) -> array<DS, 2> {
    let re = ds_add(ds_mul(a_re, b_re), ds_neg(ds_mul(a_im, b_im)));
    let im = ds_add(ds_mul(a_re, b_im), ds_mul(a_im, b_re));
    return array<DS, 2>(re, im);
}

// Complex DS addition: component-wise DS addition.
fn ds_cadd(a_re: DS, a_im: DS, b_re: DS, b_im: DS) -> array<DS, 2> {
    return array<DS, 2>(ds_add(a_re, b_re), ds_add(a_im, b_im));
}

// Complex DS magnitude squared: |z|^2 = re*re + im*im.
// Returns a DS value (not a plain f32).
fn ds_cmag2(re: DS, im: DS) -> DS {
    return ds_add(ds_mul(re, re), ds_mul(im, im));
}

// Complex DS scaling by a plain f32 value.
fn ds_cscale(re: DS, im: DS, s: f32) -> array<DS, 2> {
    return array<DS, 2>(ds_scale(re, s), ds_scale(im, s));
}

// Complex DS negation.
fn ds_cneg(re: DS, im: DS) -> array<DS, 2> {
    return array<DS, 2>(ds_neg(re), ds_neg(im));
}

