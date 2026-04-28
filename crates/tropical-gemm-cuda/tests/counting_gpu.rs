//! GPU vs CPU cross-check for count_ground_states (spec C).

use num_bigint::BigInt;
use tropical_gemm::crt::bound_for_single_matmul_u64;
use tropical_gemm::{bound_for_single_matmul, count_ground_states, Max, Min};
use tropical_gemm_cuda::{count_ground_states_gpu, count_ground_states_gpu_u64, CudaContext};

fn random_ish_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
    (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = (state >> 33) as u32;
            (x % 7) as f32
        })
        .collect()
}

#[test]
fn gpu_matches_cpu_max_small() {
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (4, 5, 3);
    let a = random_ish_matrix(m, k, 0x1234);
    let b = random_ish_matrix(k, n, 0x5678);
    let bound = bound_for_single_matmul(k);

    let cpu = count_ground_states::<f32, Max>(&a, m, k, &b, n, &bound);
    let gpu = count_ground_states_gpu::<f32, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();

    assert_eq!(gpu.values, cpu.values);
    assert_eq!(gpu.counts, cpu.counts);
}

#[test]
fn gpu_matches_cpu_min_medium() {
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (8, 16, 8);
    let a = random_ish_matrix(m, k, 0xaaaa);
    let b = random_ish_matrix(k, n, 0xbbbb);
    let bound = bound_for_single_matmul(k);

    let cpu = count_ground_states::<f32, Min>(&a, m, k, &b, n, &bound);
    let gpu = count_ground_states_gpu::<f32, Min>(&ctx, &a, m, k, &b, n, &bound).unwrap();

    assert_eq!(gpu.values, cpu.values);
    assert_eq!(gpu.counts, cpu.counts);
}

#[test]
fn gpu_handles_all_ties() {
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (2, 13, 2);
    let a = vec![0.0_f32; m * k];
    let b = vec![0.0_f32; k * n];
    let bound = bound_for_single_matmul(k);

    let gpu = count_ground_states_gpu::<f32, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();

    assert_eq!(gpu.values, vec![0.0; m * n]);
    assert_eq!(gpu.counts, vec![BigInt::from(k); m * n]);
}

#[test]
fn gpu_multi_prime_large_bound() {
    let ctx = CudaContext::new().unwrap();
    let a = vec![0.0_f32; 100];
    let b = vec![0.0_f32; 100];
    let bound = BigInt::from(u128::MAX);
    let gpu = count_ground_states_gpu::<f32, Max>(&ctx, &a, 1, 100, &b, 1, &bound).unwrap();
    assert_eq!(gpu.counts, vec![BigInt::from(100)]);
}

#[test]
fn gpu_f64_matches_cpu() {
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (3, 4, 2);
    let a: Vec<f64> = (0..m * k).map(|x| (x % 5) as f64).collect();
    let b: Vec<f64> = (0..k * n).map(|x| (x % 5) as f64).collect();
    let bound = bound_for_single_matmul(k);

    let cpu = count_ground_states::<f64, Max>(&a, m, k, &b, n, &bound);
    let gpu = count_ground_states_gpu::<f64, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();

    assert_eq!(gpu.values, cpu.values);
    assert_eq!(gpu.counts, cpu.counts);
}

/// Layout-contract test: asymmetric shapes with hand-chosen values where a
/// row/col swap on either operand would flip the answer. Catches the
/// classic row-major vs column-major bug.
#[test]
fn gpu_layout_contract_asymmetric() {
    let ctx = CudaContext::new().unwrap();

    // A is 2x3 (m=2, k=3), B is 3x2 (k=3, n=2). Row-major.
    //   A = [[1, 2, 3],
    //        [4, 5, 6]]
    //   B = [[1, 2],
    //        [3, 4],
    //        [5, 6]]
    // For Max: C[i,j] = max_k (A[i,k] + B[k,j]).
    //   C[0,0] = max(1+1, 2+3, 3+5) = 8 (unique k=2, count=1)
    //   C[0,1] = max(1+2, 2+4, 3+6) = 9 (unique k=2, count=1)
    //   C[1,0] = max(4+1, 5+3, 6+5) = 11 (unique k=2, count=1)
    //   C[1,1] = max(4+2, 5+4, 6+6) = 12 (unique k=2, count=1)
    //
    // If the kernel confused row-major with column-major on either operand,
    // the answers would differ because the shapes are not symmetric.
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let bound = bound_for_single_matmul(3);
    let gpu = count_ground_states_gpu::<f32, Max>(&ctx, &a, 2, 3, &b, 2, &bound).unwrap();
    assert_eq!(gpu.values, vec![8.0, 9.0, 11.0, 12.0]);
    assert_eq!(
        gpu.counts,
        vec![BigInt::from(1), BigInt::from(1), BigInt::from(1), BigInt::from(1)]
    );
}

/// Large shape. Exercises the tile loop across many block iterations.
#[test]
fn gpu_large_shape_f32() {
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (512, 512, 512);
    let a = random_ish_matrix(m, k, 0xdead);
    let b = random_ish_matrix(k, n, 0xbeef);
    let bound = bound_for_single_matmul(k);

    let cpu = count_ground_states::<f32, Max>(&a, m, k, &b, n, &bound);
    let gpu = count_ground_states_gpu::<f32, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();

    assert_eq!(gpu.values, cpu.values);
    assert_eq!(gpu.counts, cpu.counts);
}

/// Off-block-boundary shape. Every dim is prime / not a multiple of block
/// size, stressing the predicated tile-load bounds checks for all edges.
#[test]
fn gpu_off_boundary_shape() {
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (17, 19, 23);
    let a = random_ish_matrix(m, k, 0x1111);
    let b = random_ish_matrix(k, n, 0x2222);
    let bound = bound_for_single_matmul(k);

    let cpu = count_ground_states::<f32, Max>(&a, m, k, &b, n, &bound);
    let gpu = count_ground_states_gpu::<f32, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();

    assert_eq!(gpu.values, cpu.values);
    assert_eq!(gpu.counts, cpu.counts);
}

/// f64 medium shape. Exercises the f64 tiled macro specifically.
#[test]
fn gpu_f64_medium_shape() {
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (128, 128, 128);
    let mut state = 0xcafef00du64;
    let mut gen = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as u32 % 5) as f64
    };
    let a: Vec<f64> = (0..m * k).map(|_| gen()).collect();
    let b: Vec<f64> = (0..k * n).map(|_| gen()).collect();
    let bound = bound_for_single_matmul(k);

    let cpu = count_ground_states::<f64, Max>(&a, m, k, &b, n, &bound);
    let gpu = count_ground_states_gpu::<f64, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();

    assert_eq!(gpu.values, cpu.values);
    assert_eq!(gpu.counts, cpu.counts);
}

// ------------------------------------------------------------------
// Spec E stage A — warp-K-reduction kernel correctness.
//
// Dispatch threshold inside `launch_counting_gemm` is K >= 64. These tests
// exercise both sides of the threshold and the warpk path's boundary
// handling (M not divisible by ROWS_PER_BLOCK=4, K not a multiple of 32).
// ------------------------------------------------------------------

#[test]
fn warpk_k_threshold_boundary() {
    let ctx = CudaContext::new().unwrap();
    for &k in &[32usize, 63, 64, 65, 95, 128] {
        let (m, n) = (8, 8);
        let a = random_ish_matrix(m, k, 0xaaaa ^ (k as u64));
        let b = random_ish_matrix(k, n, 0xbbbb ^ (k as u64));
        let bound = bound_for_single_matmul(k);
        let cpu = count_ground_states::<f32, Max>(&a, m, k, &b, n, &bound);
        let gpu = count_ground_states_gpu::<f32, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();
        assert_eq!(gpu.values, cpu.values, "values mismatch at K={}", k);
        assert_eq!(gpu.counts, cpu.counts, "counts mismatch at K={}", k);
    }
}

#[test]
fn warpk_non_aligned_dims() {
    // M not divisible by 4 (rows-per-block) -> tail-row predication.
    // K not a multiple of 32 -> tail-step predication on K-stride.
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (37, 131, 29);
    let a = random_ish_matrix(m, k, 0xc0ffee);
    let b = random_ish_matrix(k, n, 0xfeedface);
    let bound = bound_for_single_matmul(k);
    let cpu = count_ground_states::<f32, Max>(&a, m, k, &b, n, &bound);
    let gpu = count_ground_states_gpu::<f32, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();
    assert_eq!(gpu.values, cpu.values);
    assert_eq!(gpu.counts, cpu.counts);
}

#[test]
fn warpk_all_ties_large_k() {
    // K=200 forces warpk path; all inputs equal -> every k contributes the
    // same product. Exercises tropical-add tie-handling under warp shuffle
    // reduction (every reduction step is a tie).
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (5, 200, 7);
    let a = vec![0.0_f32; m * k];
    let b = vec![0.0_f32; k * n];
    let bound = bound_for_single_matmul(k);
    let gpu = count_ground_states_gpu::<f32, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();
    assert_eq!(gpu.values, vec![0.0; m * n]);
    assert_eq!(gpu.counts, vec![BigInt::from(k); m * n]);
}

#[test]
fn warpk_min_direction_f64() {
    // Cover Min + f64 paths for warpk dispatch.
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (6, 80, 9);
    let mut state: u64 = 0xdeadbeef;
    let mut gen = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as u32 % 5) as f64
    };
    let a: Vec<f64> = (0..m * k).map(|_| gen()).collect();
    let b: Vec<f64> = (0..k * n).map(|_| gen()).collect();
    let bound = bound_for_single_matmul(k);
    let cpu = count_ground_states::<f64, Min>(&a, m, k, &b, n, &bound);
    let gpu = count_ground_states_gpu::<f64, Min>(&ctx, &a, m, k, &b, n, &bound).unwrap();
    assert_eq!(gpu.values, cpu.values);
    assert_eq!(gpu.counts, cpu.counts);
}

// ------------------------------------------------------------------
// Spec G — ones-specialized kernel correctness (production path).
//
// `count_ground_states_gpu` now routes through the ones kernels. The
// pre-existing tests above already cover the all-ones case via the driver,
// so these focus on regimes the warpk tests already exercise — naive↔warpk
// boundary, non-aligned dims, and tie-heavy reductions — but specifically
// to catch regressions in the ones path's u32 accumulator and simpler
// shuffle reduction (single shuffle per acc_cnt instead of hi/lo split).
// ------------------------------------------------------------------

#[test]
fn ones_k_threshold_boundary() {
    let ctx = CudaContext::new().unwrap();
    for &k in &[32usize, 63, 64, 65, 95, 128] {
        let (m, n) = (8, 8);
        let a = random_ish_matrix(m, k, 0xa1a1 ^ (k as u64));
        let b = random_ish_matrix(k, n, 0xb2b2 ^ (k as u64));
        let bound = bound_for_single_matmul(k);
        let cpu = count_ground_states::<f32, Max>(&a, m, k, &b, n, &bound);
        let gpu = count_ground_states_gpu::<f32, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();
        assert_eq!(gpu.values, cpu.values, "values mismatch at K={}", k);
        assert_eq!(gpu.counts, cpu.counts, "counts mismatch at K={}", k);
    }
}

#[test]
fn ones_non_aligned_dims_f64() {
    // f64 + tail-row predication + non-multiple-of-32 K.
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (37, 131, 29);
    let mut state: u64 = 0xfacefeed;
    let mut gen = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as u32 % 5) as f64
    };
    let a: Vec<f64> = (0..m * k).map(|_| gen()).collect();
    let b: Vec<f64> = (0..k * n).map(|_| gen()).collect();
    let bound = bound_for_single_matmul(k);
    let cpu = count_ground_states::<f64, Max>(&a, m, k, &b, n, &bound);
    let gpu = count_ground_states_gpu::<f64, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();
    assert_eq!(gpu.values, cpu.values);
    assert_eq!(gpu.counts, cpu.counts);
}

#[test]
fn warpk_transposed_b_layout() {
    // Spec H: warpk dispatch uploads B in transposed (N×K row-major) layout.
    // Use deliberately asymmetric values so the host transpose helper would
    // visibly corrupt the output if rows/cols were swapped wrong.
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (8, 128, 8); // M*N=64 -> warpk; K=128 -> warpk threshold
    // A[i,k] = i*1000 + k; B[k,j] = k*100 + j*7. Distinguishable patterns.
    let a: Vec<f32> = (0..m * k)
        .map(|idx| {
            let i = idx / k;
            let kv = idx % k;
            (i * 1000 + kv) as f32
        })
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|idx| {
            let kv = idx / n;
            let j = idx % n;
            (kv * 100 + j * 7) as f32
        })
        .collect();
    let bound = bound_for_single_matmul(k);
    let cpu = count_ground_states::<f32, Max>(&a, m, k, &b, n, &bound);
    let gpu = count_ground_states_gpu::<f32, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();
    assert_eq!(gpu.values, cpu.values, "transposed-B path corrupted values");
    assert_eq!(gpu.counts, cpu.counts, "transposed-B path corrupted counts");
}

#[test]
fn ones_all_ties_large_k_warpk() {
    // K=200 forces warpk path. All inputs equal -> every k contributes the
    // same partial. Output count should be K mod P = 200 (well below P).
    // Exercises u32 acc_cnt addition under warp shuffle reduction in the
    // ones-specialized kernel.
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (5, 200, 7);
    let a = vec![0.0_f32; m * k];
    let b = vec![0.0_f32; k * n];
    let bound = bound_for_single_matmul(k);
    let gpu = count_ground_states_gpu::<f32, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();
    assert_eq!(gpu.values, vec![0.0; m * n]);
    assert_eq!(gpu.counts, vec![BigInt::from(k); m * n]);
}

// ------------------------------------------------------------------
// Spec I — u64 fast-path correctness.
// ------------------------------------------------------------------

#[test]
fn u64_matches_bigint_single_prime() {
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (16, 24, 12);
    let a = random_ish_matrix(m, k, 0xc0c0);
    let b = random_ish_matrix(k, n, 0xd0d0);
    let bound = bound_for_single_matmul_u64(k);
    let bound_big = bound_for_single_matmul(k);

    let big = count_ground_states_gpu::<f32, Max>(&ctx, &a, m, k, &b, n, &bound_big).unwrap();
    let u64_ = count_ground_states_gpu_u64::<f32, Max>(&ctx, &a, m, k, &b, n, bound).unwrap();

    assert_eq!(u64_.values, big.values);
    for i in 0..m * n {
        assert_eq!(BigInt::from(u64_.counts[i]), big.counts[i],
                   "cell {} mismatch", i);
    }
}

#[test]
fn u64_matches_bigint_two_primes() {
    // Force a 2-prime CRT by passing a bound > 2^30.
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (8, 64, 8);
    let a = random_ish_matrix(m, k, 0xeeee);
    let b = random_ish_matrix(k, n, 0xffff);
    let bound = (1u64 << 31) + 7;
    let bound_big = BigInt::from(bound);

    let big = count_ground_states_gpu::<f32, Max>(&ctx, &a, m, k, &b, n, &bound_big).unwrap();
    let u64_ = count_ground_states_gpu_u64::<f32, Max>(&ctx, &a, m, k, &b, n, bound).unwrap();

    assert_eq!(u64_.values, big.values);
    for i in 0..m * n {
        assert_eq!(BigInt::from(u64_.counts[i]), big.counts[i],
                   "cell {} mismatch", i);
    }
}

#[test]
fn u64_rejects_too_large_bound() {
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (4, 4, 4);
    let a = random_ish_matrix(m, k, 0x1);
    let b = random_ish_matrix(k, n, 0x2);
    let bound = 1u64 << 62;
    let r = count_ground_states_gpu_u64::<f32, Max>(&ctx, &a, m, k, &b, n, bound);
    assert!(r.is_err(), "expected error for too-large u64 bound");
}

#[test]
fn u64_all_ties_large_k_warpk() {
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (5, 200, 7);
    let a = vec![0.0_f32; m * k];
    let b = vec![0.0_f32; k * n];
    let bound = bound_for_single_matmul_u64(k);
    let gpu = count_ground_states_gpu_u64::<f32, Max>(&ctx, &a, m, k, &b, n, bound).unwrap();
    assert_eq!(gpu.values, vec![0.0; m * n]);
    assert_eq!(gpu.counts, vec![k as u64; m * n]);
}
