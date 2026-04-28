//! Times JUST the counting kernel launch, data already on device.
//! Measures both dispatch paths:
//!   - naive  (large M*N): one thread per output cell.
//!   - warpk  (small M*N, K>=64): 32 threads per cell + warp shuffle reduce.

use std::time::Instant;
use cudarc::driver::{LaunchAsync, LaunchConfig};
use tropical_gemm_cuda::pair::{PairF32, pack_f32_ones};
use tropical_gemm_cuda::CudaContext;

fn bench(ctx: &CudaContext, kernel_name: &'static str,
         grid: (u32, u32, u32), block: (u32, u32, u32),
         m: usize, k: usize, n: usize, label: &str)
{
    let ops = 2.0 * (m as f64) * (n as f64) * (k as f64);

    let a_val: Vec<f32> = (0..m * k).map(|x| (x % 7) as f32).collect();
    let b_val: Vec<f32> = (0..k * n).map(|x| (x % 5) as f32).collect();
    let pa: Vec<PairF32> = pack_f32_ones(&a_val);
    let pb: Vec<PairF32> = pack_f32_ones(&b_val);

    let d_pa = ctx.device().htod_copy(pa).unwrap();
    let d_pb = ctx.device().htod_copy(pb).unwrap();
    let mut d_vc = ctx.device().alloc_zeros::<f32>(m * n).unwrap();
    let mut d_cc = ctx.device().alloc_zeros::<i32>(m * n).unwrap();

    let p: i32 = 1_073_741_789;
    let mu: u64 = ((1u128 << 64) / p as u128) as u64;

    let kernel = ctx.get_kernel(kernel_name).unwrap();
    let cfg = LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: 0 };

    // Warmup.
    unsafe {
        kernel.clone().launch(cfg,
            (&d_pa, &d_pb, &mut d_vc, &mut d_cc,
             m as i32, n as i32, k as i32, p, mu)).unwrap();
    }
    ctx.device().synchronize().unwrap();

    let iters = if m.max(n) <= 256 { 30 } else if m.max(n) <= 1024 { 10 } else { 5 };
    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            kernel.clone().launch(cfg,
                (&d_pa, &d_pb, &mut d_vc, &mut d_cc,
                 m as i32, n as i32, k as i32, p, mu)).unwrap();
        }
    }
    ctx.device().synchronize().unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let gops = ops / (ms * 1e-3) / 1e9;
    println!("{:<8} M={:>5} K={:>5} N={:>5}  {:>8.3} ms  ({:>6.1} G tropical-ops/s)",
             label, m, k, n, ms, gops);
}

fn main() {
    let ctx = CudaContext::new().unwrap();
    println!("Counting kernel — pure launch timing (data already on device, AoS)");
    println!("{}", "-".repeat(85));

    println!("\nNaive path (one thread per output cell):");
    for &s in &[128usize, 256, 512, 1024, 2048, 4096] {
        let grid  = CudaContext::counting_grid_dims_f32(s, s);
        let block = CudaContext::counting_block_dims_f32();
        bench(&ctx, "counting_gemm_f32_max", grid, block, s, s, s, "naive");
    }

    println!("\nWarp-K path (small M*N, large K):");
    for &(m, k, n) in &[(32usize, 4096, 32), (64, 4096, 64), (32, 2048, 32)] {
        let grid  = CudaContext::counting_warpk_grid_dims(m, n);
        let block = CudaContext::counting_warpk_block_dims();
        bench(&ctx, "counting_gemm_f32_max_warpk", grid, block, m, k, n, "warpk");
    }
}
