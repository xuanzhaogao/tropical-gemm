//! Single-launch counting kernel timing. Warmup, then ONE timed launch
//! per size (host Instant + sync). Complements bench_kernel_only.rs which
//! amortizes over many launches.

use std::time::Instant;
use cudarc::driver::{LaunchAsync, LaunchConfig};
use tropical_gemm_cuda::CudaContext;

fn bench(ctx: &CudaContext, size: usize) {
    let m = size; let k = size; let n = size;
    let ops = 2.0 * (size as f64).powi(3);

    let a_val: Vec<f32> = (0..m * k).map(|x| (x % 7) as f32).collect();
    let b_val: Vec<f32> = (0..k * n).map(|x| (x % 5) as f32).collect();
    let ones_ak: Vec<i32> = vec![1; m * k];
    let ones_kn: Vec<i32> = vec![1; k * n];

    let d_va = ctx.device().htod_copy(a_val).unwrap();
    let d_ca = ctx.device().htod_copy(ones_ak).unwrap();
    let d_vb = ctx.device().htod_copy(b_val).unwrap();
    let d_cb = ctx.device().htod_copy(ones_kn).unwrap();
    let mut d_vc = ctx.device().alloc_zeros::<f32>(m * n).unwrap();
    let mut d_cc = ctx.device().alloc_zeros::<i32>(m * n).unwrap();

    let p: i32 = 1_073_741_789;
    let mu: u64 = ((1u128 << 64) / p as u128) as u64;

    let kernel = ctx.get_kernel("counting_gemm_f32_max").unwrap();
    let cfg = LaunchConfig {
        grid_dim: CudaContext::counting_grid_dims_f32(m, n),
        block_dim: CudaContext::counting_block_dims_f32(),
        shared_mem_bytes: 0,
    };

    // Warmup.
    unsafe {
        kernel.clone().launch(
            cfg,
            (&d_va, &d_ca, &d_vb, &d_cb, &mut d_vc, &mut d_cc,
             m as i32, n as i32, k as i32, p, mu),
        ).unwrap();
    }
    ctx.device().synchronize().unwrap();

    // Five independent single-launch timings to gauge variance.
    let mut samples = Vec::with_capacity(5);
    for _ in 0..5 {
        let t0 = Instant::now();
        unsafe {
            kernel.clone().launch(
                cfg,
                (&d_va, &d_ca, &d_vb, &d_cb, &mut d_vc, &mut d_cc,
                 m as i32, n as i32, k as i32, p, mu),
            ).unwrap();
        }
        ctx.device().synchronize().unwrap();
        samples.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let min_ms = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ms = samples.iter().cloned().fold(0.0_f64, f64::max);
    let med_ms = { let mut s = samples.clone(); s.sort_by(|a, b| a.partial_cmp(b).unwrap()); s[s.len() / 2] };
    let gops = ops / (med_ms * 1e-3) / 1e9;
    println!("size={:>5}  min={:>7.3} med={:>7.3} max={:>7.3} ms  ({:>6.1} G tropical-ops/s @ med)",
             size, min_ms, med_ms, max_ms, gops);
}

fn main() {
    let ctx = CudaContext::new().unwrap();
    println!("Counting kernel — single-launch (5 independent samples, host timing)");
    println!("{}", "-".repeat(78));
    for &s in &[128, 256, 512, 1024, 2048, 4096] {
        bench(&ctx, s);
    }
}
