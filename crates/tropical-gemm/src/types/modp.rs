//! Modular integer scalar `Mod<const P: i32>` for CRT-based counting.
//!
//! `Mod<P>` represents a residue modulo the compile-time prime `P`. It is
//! used as the count field of `CountingTropical<T, Mod<P>, D>` during
//! Chinese Remainder reconstruction of large counts.
//!
//! # Prime size contract
//!
//! `P` must satisfy `(P - 1)² < 2⁶²` so that `scalar_mul`'s `i64` product
//! never overflows. In practice we use 30-bit primes from `CRT_PRIMES`
//! in `crate::crt`, where `(P - 1)² < 2⁶⁰` with room to spare.

use std::fmt;

/// Residue modulo the compile-time prime `P`.
///
/// The inner `i32` is always in `[0, P)` (the normalized representative).
/// Construct via `Mod::new` (which normalizes) or reconstruct from a raw
/// representative via `raw`. See module docs for the size constraint on `P`.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Mod<const P: i32>(i32);

impl<const P: i32> Mod<P> {
    /// Normalizing constructor. Accepts any `i32` and reduces mod `P`.
    #[inline]
    pub const fn new(x: i32) -> Self {
        assert!(P > 1, "Mod<P>: P must be a prime greater than 1");
        Self(x.rem_euclid(P))
    }

    /// Raw inner value, guaranteed to be in `[0, P)`.
    #[inline]
    pub const fn raw(self) -> i32 {
        self.0
    }

    /// Modular addition. Inputs are already in `[0, P)`; sum is in `[0, 2P)`.
    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        let s = self.0 as i64 + rhs.0 as i64;
        Self((s % P as i64) as i32)
    }

    /// Modular multiplication. Product bounded by `(P-1)²`; `i64` is safe
    /// for any `P` satisfying the module contract.
    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        let s = self.0 as i64 * rhs.0 as i64;
        Self((s % P as i64) as i32)
    }
}

impl<const P: i32> Default for Mod<P> {
    #[inline]
    fn default() -> Self {
        Self(0)
    }
}

impl<const P: i32> fmt::Debug for Mod<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mod<{}>({})", P, self.0)
    }
}

impl<const P: i32> fmt::Display for Mod<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: i32 = 7;

    #[test]
    fn new_normalizes_positive() {
        assert_eq!(Mod::<P>::new(10).raw(), 3);
        assert_eq!(Mod::<P>::new(0).raw(), 0);
        assert_eq!(Mod::<P>::new(P).raw(), 0);
        assert_eq!(Mod::<P>::new(P - 1).raw(), P - 1);
    }

    #[test]
    fn new_normalizes_negative() {
        assert_eq!(Mod::<P>::new(-1).raw(), P - 1);
        assert_eq!(Mod::<P>::new(-P).raw(), 0);
        assert_eq!(Mod::<P>::new(-P - 1).raw(), P - 1);
    }

    #[test]
    fn add_wraps() {
        let a = Mod::<P>::new(5);
        let b = Mod::<P>::new(4);
        assert_eq!(a.add(b).raw(), 2); // (5+4) mod 7 = 2
    }

    #[test]
    fn mul_wraps() {
        let a = Mod::<P>::new(5);
        let b = Mod::<P>::new(4);
        assert_eq!(a.mul(b).raw(), 6); // (5*4) mod 7 = 6
    }

    #[test]
    fn add_commutative() {
        let a = Mod::<P>::new(3);
        let b = Mod::<P>::new(6);
        assert_eq!(a.add(b), b.add(a));
    }

    #[test]
    fn mul_commutative() {
        let a = Mod::<P>::new(3);
        let b = Mod::<P>::new(6);
        assert_eq!(a.mul(b), b.mul(a));
    }

    #[test]
    fn distributive() {
        let a = Mod::<P>::new(3);
        let b = Mod::<P>::new(5);
        let c = Mod::<P>::new(6);
        let lhs = a.mul(b.add(c));
        let rhs = a.mul(b).add(a.mul(c));
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn large_prime_product_does_not_overflow() {
        const LP: i32 = 1_073_741_789; // 30-bit prime
        let a = Mod::<LP>::new(LP - 1);
        let b = Mod::<LP>::new(LP - 2);
        let r = a.mul(b);
        // (P-1)(P-2) mod P = (-1)(-2) mod P = 2
        assert_eq!(r.raw(), 2);
    }

    #[test]
    fn default_is_zero() {
        let d = Mod::<P>::default();
        assert_eq!(d.raw(), 0);
    }
}
