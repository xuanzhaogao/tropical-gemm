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

impl<const P: i32> crate::types::scalar::TropicalScalar for Mod<P> {
    #[inline(always)]
    fn scalar_zero() -> Self {
        Self(0)
    }

    #[inline(always)]
    fn scalar_one() -> Self {
        Self(1)
    }

    #[inline(always)]
    fn scalar_add(self, rhs: Self) -> Self {
        Mod::add(self, rhs)
    }

    #[inline(always)]
    fn scalar_mul(self, rhs: Self) -> Self {
        Mod::mul(self, rhs)
    }

    #[inline(always)]
    fn pos_infinity() -> Self {
        unreachable!("Mod<P> is a count scalar, not a tropical value — pos_infinity is undefined")
    }

    #[inline(always)]
    fn neg_infinity() -> Self {
        unreachable!("Mod<P> is a count scalar, not a tropical value — neg_infinity is undefined")
    }

    #[inline(always)]
    fn scalar_max(self, _rhs: Self) -> Self {
        unreachable!("Mod<P> is a count scalar, not a tropical value — scalar_max is undefined")
    }

    #[inline(always)]
    fn scalar_min(self, _rhs: Self) -> Self {
        unreachable!("Mod<P> is a count scalar, not a tropical value — scalar_min is undefined")
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

    #[test]
    fn tropical_scalar_impl_plugs_into_counting_tropical() {
        use super::super::counting::CountingTropical;
        use super::super::direction::Max;
        use super::super::traits::TropicalSemiring;

        // Sanity: we can construct CountingTropical<f32, Mod<7>, Max>.
        let a: CountingTropical<f32, Mod<7>, Max> = CountingTropical::new(3.0, Mod::new(2));
        let b: CountingTropical<f32, Mod<7>, Max> = CountingTropical::new(5.0, Mod::new(4));
        let c = a.tropical_mul(b);
        assert_eq!(c.value, 8.0);            // 3.0 + 5.0
        assert_eq!(c.count.raw(), 1);        // (2 * 4) mod 7 = 1
    }

    #[test]
    fn scalar_add_matches_manual_mod() {
        use super::super::scalar::TropicalScalar;
        let a = Mod::<7>::new(5);
        let b = Mod::<7>::new(4);
        assert_eq!(a.scalar_add(b).raw(), 2);
    }

    #[test]
    fn scalar_mul_matches_manual_mod() {
        use super::super::scalar::TropicalScalar;
        let a = Mod::<7>::new(5);
        let b = Mod::<7>::new(4);
        assert_eq!(a.scalar_mul(b).raw(), 6);
    }

    #[test]
    fn scalar_zero_and_one() {
        use super::super::scalar::TropicalScalar;
        assert_eq!(Mod::<7>::scalar_zero().raw(), 0);
        assert_eq!(Mod::<7>::scalar_one().raw(), 1);
    }
}
