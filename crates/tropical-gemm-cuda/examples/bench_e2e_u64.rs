//! End-to-end driver bench: BigInt vs u64 fast-path (Spec I).
//!
//! Times `count_ground_states_gpu` (BigInt) vs `count_ground_states_gpu_u64`
//! at the same shapes, including upload + kernel launches + downloads +
//! reconstruction. Confirms the BigInt-allocation savings.

use std::time::Instant;
use num_bigint::BigInt;
use tropical_gemm::crt::bound_for_single_matmul_u64;
use tropical_gemm::{bound_for_single_matmul, Max};
use tropical_gemm_cuda::{count_ground_states_gpu, count_ground_states_gpu_u64, CudaContext};

fn random_ish(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
    (0..rows * cols)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 33) as u32 % 7) as f32
        })
        .collect()
}

fn bench(ctx: &CudaContext, size: usize) {
    let (m, k, n) = (size, size, size);
    let a = random_ish(m, k, 0xaaaa);
    let b = random_ish(k, n, 0xbbbb);
    let bound_big = bound_for_single_matmul(k);
    let bound = bound_for_single_matmul_u64(k);

    // Warmup both paths.
    let _ = count_ground_states_gpu::<f32, Max>(ctx, &a, m, k, &b, n, &bound_big).unwrap();
    let _ = count_ground_states_gpu_u64::<f32, Max>(ctx, &a, m, k, &b, n, bound).unwrap();

    let iters = if size <= 512 { 5 } else if size <= 1024 { 3 } else { 2 };

    let t0 = Instant::now();
    let mut sink_big: BigInt = BigInt::from(0);
    for _ in 0..iters {
        let r = count_ground_states_gpu::<f32, Max>(ctx, &a, m, k, &b, n, &bound_big).unwrap();
        sink_big += &r.counts[0]; // prevent dead-code elim
    }
    let ms_big = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let t1 = Instant::now();
    let mut sink_u64: u64 = 0;
    for _ in 0..iters {
        let r = count_ground_states_gpu_u64::<f32, Max>(ctx, &a, m, k, &b, n, bound).unwrap();
        sink_u64 = sink_u64.wrapping_add(r.counts[0]);
    }
    let ms_u64 = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    let _ = sink_big; let _ = sink_u64;

    println!(
        "size={:>5}  BigInt {:>9.2} ms   u64 {:>9.2} ms   speedup {:>5.2}x",
        size, ms_big, ms_u64, ms_big / ms_u64
    );
}

fn main() {
    let ctx = CudaContext::new().unwrap();
    println!("End-to-end driver: count_ground_states_gpu (BigInt) vs _u64 (Spec I)");
    println!("{}", "-".repeat(80));
    for &s in &[256usize, 512, 1024, 2048, 4096] {
        bench(&ctx, s);
    }
}
