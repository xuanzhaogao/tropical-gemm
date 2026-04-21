//! Chinese Remainder Theorem driver for large-count CountingTropical matmul.
//!
//! Entry point: [`count_ground_states`]. See spec B
//! (`docs/superpowers/specs/2026-04-21-counting-tropical-crt-design.md`)
//! for the math and the `count_upper_bound` contract.

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Zero};

/// 30-bit primes satisfying `(P - 1)² < 2⁶⁰`, useful as `P` in `Mod<P>`
/// for the CRT count reconstruction. All are ≤ `2³⁰` and pairwise coprime
/// (distinct primes). Selected as the 16 largest primes below `2³⁰`.
pub const CRT_PRIMES: [i32; 16] = [
    1_073_741_789, 1_073_741_783, 1_073_741_741, 1_073_741_723,
    1_073_741_719, 1_073_741_717, 1_073_741_689, 1_073_741_671,
    1_073_741_663, 1_073_741_633, 1_073_741_629, 1_073_741_623,
    1_073_741_621, 1_073_741_587, 1_073_741_567, 1_073_741_561,
];

/// Combine a running CRT accumulator `(acc_value, acc_modulus)` with a new
/// `(residue, prime)` pair. Returns `(x, acc_modulus * prime)` where
/// `x ≡ acc_value (mod acc_modulus)` and `x ≡ residue (mod prime)`,
/// `0 <= x < acc_modulus * prime`.
///
/// Preconditions: `gcd(acc_modulus, prime) == 1` (true when `prime` is a
/// fresh element from `CRT_PRIMES`), `0 <= residue < prime`.
pub(crate) fn crt_combine(
    acc_value: &BigInt,
    acc_modulus: &BigInt,
    residue: i32,
    prime: i32,
) -> (BigInt, BigInt) {
    // Standard pairwise CRT:
    //   x = acc_value + acc_modulus * ((residue - acc_value) * acc_modulus^{-1} mod prime)
    let prime_big = BigInt::from(prime);
    let residue_big = BigInt::from(residue);

    // Compute acc_modulus^{-1} mod prime via the extended Euclidean algorithm.
    let ext = acc_modulus.extended_gcd(&prime_big);
    // ext.gcd must be 1 for the inverse to exist.
    debug_assert!(ext.gcd.is_one(), "crt_combine: modulus and prime not coprime");
    let inv = ext.x.mod_floor(&prime_big); // acc_modulus^{-1} mod prime

    let diff = (&residue_big - acc_value).mod_floor(&prime_big);
    let delta = (diff * inv).mod_floor(&prime_big);
    let new_modulus = acc_modulus * &prime_big;
    let new_value = (acc_value + acc_modulus * delta).mod_floor(&new_modulus);
    (new_value, new_modulus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primes_are_distinct() {
        let mut s = std::collections::HashSet::new();
        for p in CRT_PRIMES {
            assert!(s.insert(p), "duplicate prime {}", p);
        }
    }

    #[test]
    fn primes_fit_size_contract() {
        for p in CRT_PRIMES {
            assert!(p > 1, "prime must be > 1");
            let pm1 = (p - 1) as i64;
            let sq = pm1.checked_mul(pm1).expect("squared overflow");
            assert!(sq < 1_i64 << 62, "(P-1)^2 = {} must be < 2^62", sq);
        }
    }

    #[test]
    fn crt_combine_two_small_primes() {
        // x ≡ 2 (mod 3), x ≡ 3 (mod 5) → x = 8 (mod 15)
        let (x, m) = crt_combine(&BigInt::from(2), &BigInt::from(3), 3, 5);
        assert_eq!(x, BigInt::from(8));
        assert_eq!(m, BigInt::from(15));
    }

    #[test]
    fn crt_combine_three_small_primes() {
        // x ≡ 1 (mod 2), x ≡ 2 (mod 3), x ≡ 3 (mod 5) → x = 23 (mod 30)
        let (x1, m1) = crt_combine(&BigInt::from(1), &BigInt::from(2), 2, 3);
        // x1 ≡ 1 (mod 2), x1 ≡ 2 (mod 3) → x1 = 5, m1 = 6
        assert_eq!(x1, BigInt::from(5));
        assert_eq!(m1, BigInt::from(6));

        let (x2, m2) = crt_combine(&x1, &m1, 3, 5);
        assert_eq!(x2, BigInt::from(23));
        assert_eq!(m2, BigInt::from(30));
    }

    #[test]
    fn crt_combine_reconstructs_large_value() {
        // Pick a target that is below the product of two 30-bit primes.
        let p1 = CRT_PRIMES[0] as i64;
        let p2 = CRT_PRIMES[1] as i64;
        let target: i64 = 12_345_678_901_234; // < p1 * p2 ≈ 2^60
        let r1 = (target.rem_euclid(p1)) as i32;
        let r2 = (target.rem_euclid(p2)) as i32;

        let (x, m) = crt_combine(&BigInt::from(r1), &BigInt::from(p1), r2, p2 as i32);
        assert_eq!(x, BigInt::from(target));
        assert_eq!(m, BigInt::from(p1) * BigInt::from(p2));
    }
}
