//! Times JUST the counting kernel launch, data already on device.
//! Measures both dispatch paths:
//!   - naive  (large M*N): one thread per output cell.
//!   - warpk  (small M*N, K>=64): 32 threads per cell + warp shuffle reduce.

use std::time::Instant;
use cudarc::driver::{LaunchAsync, LaunchConfig, CudaSlice};
use tropical_gemm_cuda::pair::{PairF32, pack_f32_ones};
use tropical_gemm_cuda::CudaContext;

fn alloc_aos(ctx: &CudaContext, m: usize, k: usize, n: usize)
    -> (CudaSlice<PairF32>, CudaSlice<PairF32>, CudaSlice<f32>, CudaSlice<i32>)
{
    let a: Vec<f32> = (0..m * k).map(|x| (x % 7) as f32).collect();
    let b: Vec<f32> = (0..k * n).map(|x| (x % 5) as f32).collect();
    (
        ctx.device().htod_copy(pack_f32_ones(&a)).unwrap(),
        ctx.device().htod_copy(pack_f32_ones(&b)).unwrap(),
        ctx.device().alloc_zeros::<f32>(m * n).unwrap(),
        ctx.device().alloc_zeros::<i32>(m * n).unwrap(),
    )
}

fn alloc_ones(ctx: &CudaContext, m: usize, k: usize, n: usize)
    -> (CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>, CudaSlice<i32>)
{
    let a: Vec<f32> = (0..m * k).map(|x| (x % 7) as f32).collect();
    let b: Vec<f32> = (0..k * n).map(|x| (x % 5) as f32).collect();
    (
        ctx.device().htod_copy(a).unwrap(),
        ctx.device().htod_copy(b).unwrap(),
        ctx.device().alloc_zeros::<f32>(m * n).unwrap(),
        ctx.device().alloc_zeros::<i32>(m * n).unwrap(),
    )
}

fn iters_for(size: usize) -> usize {
    if size <= 256 { 30 } else if size <= 1024 { 10 } else { 5 }
}

fn bench_aos(ctx: &CudaContext, kernel_name: &'static str,
             grid: (u32, u32, u32), block: (u32, u32, u32),
             m: usize, k: usize, n: usize) -> f64
{
    let (d_a, d_b, mut d_vc, mut d_cc) = alloc_aos(ctx, m, k, n);
    let kernel = ctx.get_kernel(kernel_name).unwrap();
    let cfg = LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: 0 };
    let p: i32 = 1_073_741_789;
    let mu: u64 = ((1u128 << 64) / p as u128) as u64;
    unsafe { kernel.clone().launch(cfg,
        (&d_a, &d_b, &mut d_vc, &mut d_cc, m as i32, n as i32, k as i32, p, mu)).unwrap(); }
    ctx.device().synchronize().unwrap();
    let iters = iters_for(m.max(n));
    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe { kernel.clone().launch(cfg,
            (&d_a, &d_b, &mut d_vc, &mut d_cc, m as i32, n as i32, k as i32, p, mu)).unwrap(); }
    }
    ctx.device().synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn bench_ones(ctx: &CudaContext, kernel_name: &'static str,
              grid: (u32, u32, u32), block: (u32, u32, u32),
              m: usize, k: usize, n: usize) -> f64
{
    let (d_a, d_b, mut d_vc, mut d_cc) = alloc_ones(ctx, m, k, n);
    let kernel = ctx.get_kernel(kernel_name).unwrap();
    let cfg = LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: 0 };
    let p: i32 = 1_073_741_789;
    let mu: u64 = ((1u128 << 64) / p as u128) as u64;
    unsafe { kernel.clone().launch(cfg,
        (&d_a, &d_b, &mut d_vc, &mut d_cc, m as i32, n as i32, k as i32, p, mu)).unwrap(); }
    ctx.device().synchronize().unwrap();
    let iters = iters_for(m.max(n));
    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe { kernel.clone().launch(cfg,
            (&d_a, &d_b, &mut d_vc, &mut d_cc, m as i32, n as i32, k as i32, p, mu)).unwrap(); }
    }
    ctx.device().synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn report(label: &str, m: usize, k: usize, n: usize, ms_aos: f64, ms_ones: f64) {
    let ops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let g_aos  = ops / (ms_aos  * 1e-3) / 1e9;
    let g_ones = ops / (ms_ones * 1e-3) / 1e9;
    println!(
        "{:<6} M={:>5} K={:>5} N={:>5}  AoS  {:>8.3} ms ({:>6.1} G)   ones {:>8.3} ms ({:>6.1} G)   speedup {:>5.2}x",
        label, m, k, n, ms_aos, g_aos, ms_ones, g_ones, ms_aos / ms_ones
    );
}

fn main() {
    let ctx = CudaContext::new().unwrap();
    println!("Counting kernel — AoS general vs ones-specialized (f32 Max, 1 prime, kernel-only)");
    println!("{}", "-".repeat(115));

    println!("\nNaive path (one thread per output cell):");
    for &s in &[128usize, 256, 512, 1024, 2048, 4096] {
        let grid  = CudaContext::counting_grid_dims_f32(s, s);
        let block = CudaContext::counting_block_dims_f32();
        let aos  = bench_aos (&ctx, "counting_gemm_f32_max",      grid, block, s, s, s);
        let ones = bench_ones(&ctx, "counting_gemm_f32_max_ones", grid, block, s, s, s);
        report("naive", s, s, s, aos, ones);
    }

    println!("\nWarp-K path (small M*N, large K):");
    for &(m, k, n) in &[(32usize, 4096, 32), (64, 4096, 64), (32, 2048, 32)] {
        let grid  = CudaContext::counting_warpk_grid_dims(m, n);
        let block = CudaContext::counting_warpk_block_dims();
        let aos  = bench_aos (&ctx, "counting_gemm_f32_max_warpk",      grid, block, m, k, n);
        let ones = bench_ones(&ctx, "counting_gemm_f32_max_warpk_ones", grid, block, m, k, n);
        report("warpk", m, k, n, aos, ones);
    }
}
