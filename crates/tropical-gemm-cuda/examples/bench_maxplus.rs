//! Compare MaxPlus GPU throughput to counting GPU throughput to gauge headroom.

use std::time::Instant;
use tropical_gemm::TropicalMaxPlus;
use tropical_gemm_cuda::{tropical_matmul_gpu, CudaContext};

fn rand(size: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9e3779b97f4a7c15);
    (0..size)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as u32 % 7) as f32
        })
        .collect()
}

fn bench(size: usize) {
    let a = rand(size * size, 0xaaaa);
    let b = rand(size * size, 0xbbbb);

    // Warm-up.
    let _ = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, size, size, &b, size).unwrap();

    let iters = if size <= 512 { 10 } else { 5 };
    let t = Instant::now();
    for _ in 0..iters {
        let _ = tropical_matmul_gpu::<TropicalMaxPlus<f32>>(&a, size, size, &b, size).unwrap();
    }
    let ms = t.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let ops = 2.0 * (size as f64).powi(3);
    let gops = ops / (ms * 1e-3) / 1e9;
    println!("size={:>5}  {:>8.2} ms  ({:>6.1} G tropical-ops/s)", size, ms, gops);
}

fn main() {
    let _ctx = CudaContext::new().expect("CUDA init");
    println!("Existing MaxPlus f32 GPU kernel bench (ceiling reference)");
    println!("{}", "-".repeat(60));
    bench(128);
    bench(256);
    bench(512);
    bench(1024);
    bench(2048);
    bench(4096);
}
