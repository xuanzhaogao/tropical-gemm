//! AoS `(value, count)` element types for counting kernels (spec F).
//!
//! Layout matches the C-side structs in `kernels/counting_gemm.cu`:
//!
//! ```c
//! struct __align__(8)  PairF32 { float  val; int cnt; };
//! struct __align__(16) PairF64 { double val; int cnt; int _pad; };
//! ```
//!
//! Packing happens once in the host driver before the prime loop, so the
//! per-prime kernel launches read 8-byte (f32) or 16-byte (f64) elements
//! in a single LDG instruction instead of two separate value/count loads.

use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

#[repr(C, align(8))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PairF32 {
    pub val: f32,
    pub cnt: i32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PairF64 {
    pub val: f64,
    pub cnt: i32,
    /// Pad to 16 B so element stride matches the C-side `__align__(16)` struct.
    /// Always zero; never read by the kernel.
    _pad: i32,
}

impl PairF32 {
    pub const fn new(val: f32, cnt: i32) -> Self { Self { val, cnt } }
}

impl PairF64 {
    pub const fn new(val: f64, cnt: i32) -> Self { Self { val, cnt, _pad: 0 } }
}

// Both types are plain repr(C) POD with no pointer fields; safe to copy
// verbatim to/from device memory. Padding bytes (PairF64._pad) are
// initialized to 0 in `new` / `default`.
unsafe impl DeviceRepr for PairF32 {}
unsafe impl DeviceRepr for PairF64 {}
unsafe impl ValidAsZeroBits for PairF32 {}
unsafe impl ValidAsZeroBits for PairF64 {}

/// Pack a row-major `(values, counts)` pair of slices into a `Vec<PairF32>`.
/// Single-pass O(n). Used by the counting driver to convert all-ones-counts
/// inputs to AoS form once before the per-prime kernel launches.
pub fn pack_f32(values: &[f32], counts: &[i32]) -> Vec<PairF32> {
    debug_assert_eq!(values.len(), counts.len());
    values.iter().zip(counts.iter()).map(|(&v, &c)| PairF32::new(v, c)).collect()
}

/// Variant for the all-ones-counts case (the only entry point today).
pub fn pack_f32_ones(values: &[f32]) -> Vec<PairF32> {
    values.iter().map(|&v| PairF32::new(v, 1)).collect()
}

pub fn pack_f64(values: &[f64], counts: &[i32]) -> Vec<PairF64> {
    debug_assert_eq!(values.len(), counts.len());
    values.iter().zip(counts.iter()).map(|(&v, &c)| PairF64::new(v, c)).collect()
}

pub fn pack_f64_ones(values: &[f64]) -> Vec<PairF64> {
    values.iter().map(|&v| PairF64::new(v, 1)).collect()
}

/// Trait abstracting `T -> PairT` packing so the counting driver can be
/// generic over scalar type. Each scalar's pair type implements this.
pub trait PackPair: Copy {
    type Pair: DeviceRepr + ValidAsZeroBits + Default + Clone + Copy + 'static;
    fn pack_ones(values: &[Self]) -> Vec<Self::Pair>;
}

impl PackPair for f32 {
    type Pair = PairF32;
    fn pack_ones(values: &[Self]) -> Vec<Self::Pair> { pack_f32_ones(values) }
}

impl PackPair for f64 {
    type Pair = PairF64;
    fn pack_ones(values: &[Self]) -> Vec<Self::Pair> { pack_f64_ones(values) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pair_f32_layout() {
        // Size and alignment must match the device-side struct.
        assert_eq!(std::mem::size_of::<PairF32>(), 8);
        assert_eq!(std::mem::align_of::<PairF32>(), 8);
    }

    #[test]
    fn pair_f64_layout() {
        assert_eq!(std::mem::size_of::<PairF64>(), 16);
        assert_eq!(std::mem::align_of::<PairF64>(), 16);
    }

    #[test]
    fn pair_f32_default_is_zero() {
        let p = PairF32::default();
        assert_eq!(p.val, 0.0);
        assert_eq!(p.cnt, 0);
    }

    #[test]
    fn pair_f64_default_pad_is_zero() {
        let p = PairF64::default();
        assert_eq!(p.val, 0.0);
        assert_eq!(p.cnt, 0);
        assert_eq!(p._pad, 0);
    }

    #[test]
    fn pack_roundtrip_f32() {
        let v = vec![1.0_f32, 2.0, 3.0, 4.0];
        let c = vec![10_i32, 20, 30, 40];
        let packed = pack_f32(&v, &c);
        for (i, p) in packed.iter().enumerate() {
            assert_eq!(p.val, v[i]);
            assert_eq!(p.cnt, c[i]);
        }
    }

    #[test]
    fn pack_ones_f32() {
        let v = vec![5.0_f32, 7.5, -1.25];
        let packed = pack_f32_ones(&v);
        for (i, p) in packed.iter().enumerate() {
            assert_eq!(p.val, v[i]);
            assert_eq!(p.cnt, 1);
        }
    }

    #[test]
    fn pack_roundtrip_f64() {
        let v = vec![1.0_f64, 2.0, 3.0];
        let c = vec![100_i32, 200, 300];
        let packed = pack_f64(&v, &c);
        for (i, p) in packed.iter().enumerate() {
            assert_eq!(p.val, v[i]);
            assert_eq!(p.cnt, c[i]);
            assert_eq!(p._pad, 0);
        }
    }
}
