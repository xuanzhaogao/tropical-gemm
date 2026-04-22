//! Temporary performance benchmark for count_ground_states_gpu vs CPU.
//! Not a regression test; run with `cargo run --example bench_counting --release`.

use std::time::Instant;
use tropical_gemm::{bound_for_single_matmul, count_ground_states, Max};
use tropical_gemm_cuda::{count_ground_states_gpu, CudaContext};

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

fn bench_one(ctx: &CudaContext, size: usize, cpu: bool) {
    let m = size;
    let k = size;
    let n = size;
    let a = random_ish(m, k, 0xaaaa);
    let b = random_ish(k, n, 0xbbbb);
    let bound = bound_for_single_matmul(k);

    // Warm-up.
    let _ = count_ground_states_gpu::<f32, Max>(ctx, &a, m, k, &b, n, &bound).unwrap();

    // GPU time.
    let iters = if size <= 512 { 5 } else { 3 };
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = count_ground_states_gpu::<f32, Max>(ctx, &a, m, k, &b, n, &bound).unwrap();
    }
    let gpu_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

    // Tropical ops are 2*M*N*K (add + mul per inner-loop step).
    // Counting does 2x: value + count path per step. One prime in this size regime.
    let tropical_ops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let gpu_gops = tropical_ops / (gpu_ms * 1e-3) / 1e9;

    print!(
        "size={:>5}  GPU {:>8.2} ms ({:>6.1} G tropical-ops/s)",
        size, gpu_ms, gpu_gops
    );

    if cpu {
        let t1 = Instant::now();
        let _ = count_ground_states::<f32, Max>(&a, m, k, &b, n, &bound);
        let cpu_ms = t1.elapsed().as_secs_f64() * 1000.0;
        let cpu_gops = tropical_ops / (cpu_ms * 1e-3) / 1e9;
        print!(
            "    CPU {:>9.2} ms ({:>6.1} G)   speedup {:>5.1}x",
            cpu_ms,
            cpu_gops,
            cpu_ms / gpu_ms
        );
    }
    println!();
}

fn main() {
    let ctx = CudaContext::new().expect("CUDA init");
    println!("Tiled CountingTropical GPU kernel bench (f32 Max, 1 prime)");
    println!("{}", "-".repeat(80));
    bench_one(&ctx, 128, true);
    bench_one(&ctx, 256, true);
    bench_one(&ctx, 512, true);
    bench_one(&ctx, 1024, true);
    bench_one(&ctx, 2048, false);
    bench_one(&ctx, 4096, false);
}
