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
///
/// # Example
///
/// ```
/// use num_bigint::BigInt;
/// use tropical_gemm::crt::crt_combine;
/// // x ≡ 2 (mod 3), x ≡ 3 (mod 5) → x = 8 (mod 15)
/// let (x, m) = crt_combine(&BigInt::from(2), &BigInt::from(3), 3, 5);
/// assert_eq!(x, BigInt::from(8));
/// assert_eq!(m, BigInt::from(15));
/// ```
pub fn crt_combine(
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

use crate::types::{CountingTropical, Mod, TropicalDirection, TropicalScalar};
use crate::tropical_matmul_t;

/// Result of a counting-tropical matmul: one ground-state value per cell
/// plus the exact `BigInt` count of configurations achieving it.
///
/// Layout matches the input: row-major, `nrows * ncols` elements in each
/// `Vec`, cell `(i, j)` at index `i * ncols + j`.
#[derive(Debug, Clone)]
pub struct CountedMat<T: TropicalScalar> {
    pub nrows: usize,
    pub ncols: usize,
    pub values: Vec<T>,
    pub counts: Vec<BigInt>,
}

/// Count optimal configurations per cell of `C = A · B` in the tropical
/// semiring selected by `D`, returning the exact BigInt count even when
/// the count exceeds `u64`.
///
/// `a` is row-major with shape `m × k`. `b` is row-major with shape `k × n`.
/// All per-cell input counts are implicitly `1`.
///
/// # `count_upper_bound` contract
///
/// The caller MUST supply an upper bound on the per-cell count. The driver
/// chooses the smallest number of primes `r` such that `∏ p_i > 2 * bound`,
/// which makes the CRT reconstruction uniquely determined. If the supplied
/// bound is smaller than the true count, the returned counts are the CRT
/// representatives modulo `∏ p_i` and are silently wrong. For a single
/// un-chained matmul with all input counts = 1, a safe bound is the inner
/// dimension `k` (see [`bound_for_single_matmul`]).
///
/// # Panics
///
/// - If `a.len() != m * k` or `b.len() != k * n`.
/// - If the required number of primes exceeds `CRT_PRIMES.len()`
///   (counts exceeding ~2^480). Use fewer, larger primes if this is hit.
/// - If any per-prime matmul produces a different `value` field than the
///   first prime (this is an internal invariant bug — the tropical value
///   field does not depend on the modulus).
pub fn count_ground_states<T, D>(
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
    count_upper_bound: &BigInt,
) -> CountedMat<T>
where
    T: TropicalScalar + Default,
    D: TropicalDirection,
{
    assert_eq!(a.len(), m * k, "A length {} != m*k = {}*{}", a.len(), m, k);
    assert_eq!(b.len(), k * n, "B length {} != k*n = {}*{}", b.len(), k, n);

    let needed_modulus = BigInt::from(2) * count_upper_bound + BigInt::one();
    let (prime_indices, _modulus_product) = choose_primes(&needed_modulus);

    // Run matmul once per selected prime, collecting (value, residue) per cell.
    let ncells = m * n;
    let mut values_ref: Option<Vec<T>> = None;
    let mut residue_streams: Vec<Vec<i32>> = Vec::with_capacity(prime_indices.len());

    for &prime_idx in &prime_indices {
        let residues = match prime_idx {
            0 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[0] }>(a, m, k, b, n),
            1 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[1] }>(a, m, k, b, n),
            2 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[2] }>(a, m, k, b, n),
            3 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[3] }>(a, m, k, b, n),
            4 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[4] }>(a, m, k, b, n),
            5 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[5] }>(a, m, k, b, n),
            6 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[6] }>(a, m, k, b, n),
            7 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[7] }>(a, m, k, b, n),
            8 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[8] }>(a, m, k, b, n),
            9 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[9] }>(a, m, k, b, n),
            10 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[10] }>(a, m, k, b, n),
            11 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[11] }>(a, m, k, b, n),
            12 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[12] }>(a, m, k, b, n),
            13 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[13] }>(a, m, k, b, n),
            14 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[14] }>(a, m, k, b, n),
            15 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[15] }>(a, m, k, b, n),
            _ => unreachable!("choose_primes returned an out-of-range index"),
        };
        assert_eq!(residues.values.len(), ncells);
        assert_eq!(residues.residues.len(), ncells);

        match &values_ref {
            None => values_ref = Some(residues.values),
            Some(v) => assert!(
                v == &residues.values,
                "CRT invariant violated: value field differs across primes"
            ),
        }
        residue_streams.push(residues.residues);
    }

    let values = values_ref.expect("at least one prime ran");

    // Cell-wise CRT reconstruct.
    let mut counts = Vec::with_capacity(ncells);
    for cell in 0..ncells {
        let mut acc_value = BigInt::from(residue_streams[0][cell]);
        let mut acc_modulus = BigInt::from(CRT_PRIMES[prime_indices[0]]);
        for step in 1..prime_indices.len() {
            let p = CRT_PRIMES[prime_indices[step]];
            let (new_value, new_modulus) =
                crt_combine(&acc_value, &acc_modulus, residue_streams[step][cell], p);
            acc_value = new_value;
            acc_modulus = new_modulus;
        }
        counts.push(acc_value);
    }

    CountedMat { nrows: m, ncols: n, values, counts }
}

/// Choose the smallest sequence of `CRT_PRIMES` indices whose product
/// exceeds `needed_modulus`. Returns the index vector and the product.
///
/// Panics if the full table's product does not exceed `needed_modulus`.
///
/// # Example
///
/// ```
/// use num_bigint::BigInt;
/// use tropical_gemm::crt::choose_primes;
/// let (indices, product) = choose_primes(&BigInt::from(5));
/// // Smallest prime in CRT_PRIMES is ~2^30, so one suffices for bound 5.
/// assert_eq!(indices, vec![0]);
/// assert!(product > BigInt::from(5));
/// ```
pub fn choose_primes(needed_modulus: &BigInt) -> (Vec<usize>, BigInt) {
    let mut product = BigInt::one();
    let mut indices = Vec::new();
    for (i, &p) in CRT_PRIMES.iter().enumerate() {
        indices.push(i);
        product *= BigInt::from(p);
        if &product > needed_modulus {
            return (indices, product);
        }
    }
    panic!(
        "count_upper_bound too large: needed modulus {} exceeds product of all {} built-in primes. \
         Extend CRT_PRIMES if required.",
        needed_modulus,
        CRT_PRIMES.len()
    );
}

/// Result of a single-prime matmul: per-cell value field and per-cell residue.
struct SinglePrimeResult<T: TropicalScalar> {
    values: Vec<T>,
    residues: Vec<i32>,
}

/// Run the spec-A matmul once with count type `Mod<P>` and split the result
/// into parallel (value, residue) streams.
fn matmul_one_prime_dispatch<T, D, const P: i32>(
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> SinglePrimeResult<T>
where
    T: TropicalScalar + Default,
    D: TropicalDirection,
{
    // Build CountingTropical<T, Mod<P>, D> inputs with count = Mod(1).
    let one_mod = Mod::<P>::new(1);
    let a_ct: Vec<CountingTropical<T, Mod<P>, D>> =
        a.iter().map(|&v| CountingTropical::new(v, one_mod)).collect();
    let b_ct: Vec<CountingTropical<T, Mod<P>, D>> =
        b.iter().map(|&v| CountingTropical::new(v, one_mod)).collect();

    let c = tropical_matmul_t::<CountingTropical<T, Mod<P>, D>>(&a_ct, m, k, &b_ct, n);

    let mut values = Vec::with_capacity(c.len());
    let mut residues = Vec::with_capacity(c.len());
    for cell in c {
        values.push(cell.value);
        residues.push(cell.count.raw());
    }
    SinglePrimeResult { values, residues }
}

/// Safe count upper bound for a single matmul `C = A · B` when all input
/// counts are `1`: each per-cell count is at most the inner dimension `k`.
pub fn bound_for_single_matmul(k: usize) -> BigInt {
    BigInt::from(k)
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

    #[test]
    fn count_ground_states_trivial_max() {
        use crate::types::Max;

        // 1x1 * 1x1, single k path: count should be 1.
        let a = [3.0_f32];
        let b = [4.0_f32];
        let bound = bound_for_single_matmul(1);
        let r = count_ground_states::<f32, Max>(&a, 1, 1, &b, 1, &bound);
        assert_eq!(r.nrows, 1);
        assert_eq!(r.ncols, 1);
        assert_eq!(r.values, vec![7.0]);
        assert_eq!(r.counts, vec![BigInt::from(1)]);
    }

    #[test]
    fn count_ground_states_ties_max() {
        use crate::types::Max;

        // A = [2, 3], B = [3, 2] (shapes 1x2 and 2x1). Both ks give value 5.
        let a = [2.0_f32, 3.0];
        let b = [3.0_f32, 2.0];
        let bound = bound_for_single_matmul(2);
        let r = count_ground_states::<f32, Max>(&a, 1, 2, &b, 1, &bound);
        assert_eq!(r.values, vec![5.0]);
        assert_eq!(r.counts, vec![BigInt::from(2)]);
    }

    #[test]
    fn count_ground_states_ties_min() {
        use crate::types::Min;

        let a = [2.0_f32, 3.0];
        let b = [3.0_f32, 2.0];
        let bound = bound_for_single_matmul(2);
        let r = count_ground_states::<f32, Min>(&a, 1, 2, &b, 1, &bound);
        assert_eq!(r.values, vec![5.0]);
        assert_eq!(r.counts, vec![BigInt::from(2)]);
    }
}
