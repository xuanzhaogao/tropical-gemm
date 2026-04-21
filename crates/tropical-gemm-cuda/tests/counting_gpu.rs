//! GPU vs CPU cross-check for count_ground_states (spec C).

use num_bigint::BigInt;
use tropical_gemm::{bound_for_single_matmul, count_ground_states, Max, Min};
use tropical_gemm_cuda::{count_ground_states_gpu, CudaContext};

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
