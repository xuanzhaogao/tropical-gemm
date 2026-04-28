#![cfg(feature = "testing")]

use num_bigint::BigInt;
use num_traits::One;

use tropical_gemm::{
    bound_for_single_matmul, count_ground_states, CountedMat, Max, Min,
};
use tropical_gemm::testing::bigint_semiring::reference_matmul;

fn random_ish_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    // Simple deterministic LCG; no external dep.
    let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
    (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = (state >> 33) as u32;
            // Restrict to a small integer range so ties happen often → count > 1.
            (x % 7) as f32
        })
        .collect()
}

#[test]
fn crt_matches_reference_max_small() {
    let (m, k, n) = (4, 5, 3);
    let a = random_ish_matrix(m, k, 0x1234);
    let b = random_ish_matrix(k, n, 0x5678);
    let bound = bound_for_single_matmul(k);

    let got: CountedMat<f32> = count_ground_states::<f32, Max>(&a, m, k, &b, n, &bound);
    let (ref_values, ref_counts) = reference_matmul::<f32, Max>(&a, m, k, &b, n);

    assert_eq!(got.values, ref_values, "values must match reference");
    assert_eq!(got.counts, ref_counts, "counts must match reference");
}

#[test]
fn crt_matches_reference_min_small() {
    let (m, k, n) = (3, 6, 4);
    let a = random_ish_matrix(m, k, 0xabcd);
    let b = random_ish_matrix(k, n, 0xef01);
    let bound = bound_for_single_matmul(k);

    let got = count_ground_states::<f32, Min>(&a, m, k, &b, n, &bound);
    let (ref_values, ref_counts) = reference_matmul::<f32, Min>(&a, m, k, &b, n);

    assert_eq!(got.values, ref_values);
    assert_eq!(got.counts, ref_counts);
}

#[test]
fn crt_handles_all_ties() {
    // Every entry is zero, so every k path gives value 0.
    // Count per cell = k. With k = 13 and inner dim, count = 13.
    let (m, k, n) = (2, 13, 2);
    let a = vec![0.0_f32; m * k];
    let b = vec![0.0_f32; k * n];
    let bound = bound_for_single_matmul(k);

    let got = count_ground_states::<f32, Max>(&a, m, k, &b, n, &bound);

    assert_eq!(got.values, vec![0.0; m * n]);
    assert_eq!(got.counts, vec![BigInt::from(k); m * n]);
}

#[test]
fn crt_count_fits_one_prime_still_uses_one_prime() {
    // Bound of 1 means needed modulus = 3; smallest prime in CRT_PRIMES is
    // ≈ 2^30, so one prime suffices. Correctness is the point, not speed.
    let a = [1.0_f32];
    let b = [1.0_f32];
    let bound = BigInt::one();
    let got = count_ground_states::<f32, Max>(&a, 1, 1, &b, 1, &bound);
    assert_eq!(got.counts, vec![BigInt::from(1)]);
}

/// Force the driver to use multiple primes by supplying a bound > u128::MAX.
/// The true count is small (100), so the multi-prime CRT path must still
/// reconstruct it exactly.
#[test]
fn crt_counts_above_u64() {
    let a = vec![0.0_f32; 100];
    let b = vec![0.0_f32; 100];
    let pretend_huge_bound = BigInt::from(u128::MAX);
    let got = count_ground_states::<f32, Max>(&a, 1, 100, &b, 1, &pretend_huge_bound);
    // True count = 100. The bound only affects how many primes we use;
    // correctness is invariant.
    assert_eq!(got.counts, vec![BigInt::from(100)]);
}

/// Documents NaN absorption: known limitation.
///
/// `tropical_add` uses `is_strictly_better(a, b) = a > b` (for Max). Both
/// `NaN > x` and `x > NaN` are false in IEEE 754, so every NaN candidate
/// falls into the tie branch. With the accumulator starting at the
/// tropical zero `-inf`, NaN contributions merge counts onto `-inf`
/// instead of displacing it — the NaN value silently disappears and the
/// accumulator remains `-inf` until a real number arrives.
///
/// For Max direction, the visible effect is: NaN inputs are treated as
/// if they were not there, but their counts leak into the `-inf` count
/// (which is then discarded by the first non-NaN comparison). This is
/// **not** a CRT-layer bug — it's an artifact of how the underlying
/// tropical semiring handles NaN. This test pins the current behavior
/// so that any future change is explicit rather than accidental.
#[test]
fn crt_nan_absorbed_silently_in_max() {
    let a = [f32::NAN, 2.0_f32];
    let b = [1.0_f32, 0.0_f32]; // 1x2 * 2x1
    let bound = bound_for_single_matmul(2);
    let r = count_ground_states::<f32, Max>(&a, 1, 2, &b, 1, &bound);
    // The NaN path is silently dropped; the surviving k=1 path gives 2.
    assert_eq!(r.values, vec![2.0]);
    assert_eq!(r.counts, vec![BigInt::from(1)]);
}
