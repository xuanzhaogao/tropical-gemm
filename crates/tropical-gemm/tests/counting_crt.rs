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
