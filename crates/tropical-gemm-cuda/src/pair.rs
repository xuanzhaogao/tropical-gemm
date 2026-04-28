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

}
