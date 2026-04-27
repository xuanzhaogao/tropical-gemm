//! AoS vs SoA counting kernel head-to-head (spec F).
//!
//! Both variants timed pure kernel-only (data already on device, prime fixed,
//! averaged across iters). Measures the inner-loop LDG savings of packing
//! (value, count) into 8-byte struct vs separate value/count buffers.

use std::time::Instant;
use cudarc::driver::{LaunchAsync, LaunchConfig, CudaSlice};
use tropical_gemm_cuda::pair::{PairF32, pack_f32_ones};
use tropical_gemm_cuda::CudaContext;

struct SoABuffers {
    d_va: CudaSlice<f32>,
    d_ca: CudaSlice<i32>,
    d_vb: CudaSlice<f32>,
    d_cb: CudaSlice<i32>,
    d_vc: CudaSlice<f32>,
    d_cc: CudaSlice<i32>,
}

struct AoSBuffers {
    d_pa: CudaSlice<PairF32>,
    d_pb: CudaSlice<PairF32>,
    d_vc: CudaSlice<f32>,
    d_cc: CudaSlice<i32>,
}

fn alloc_soa(ctx: &CudaContext, m: usize, k: usize, n: usize) -> SoABuffers {
    let a_val: Vec<f32> = (0..m * k).map(|x| (x % 7) as f32).collect();
    let b_val: Vec<f32> = (0..k * n).map(|x| (x % 5) as f32).collect();
    SoABuffers {
        d_va: ctx.device().htod_copy(a_val).unwrap(),
        d_ca: ctx.device().htod_copy(vec![1_i32; m * k]).unwrap(),
        d_vb: ctx.device().htod_copy(b_val).unwrap(),
        d_cb: ctx.device().htod_copy(vec![1_i32; k * n]).unwrap(),
        d_vc: ctx.device().alloc_zeros::<f32>(m * n).unwrap(),
        d_cc: ctx.device().alloc_zeros::<i32>(m * n).unwrap(),
    }
}

fn alloc_aos(ctx: &CudaContext, m: usize, k: usize, n: usize) -> AoSBuffers {
    let a_val: Vec<f32> = (0..m * k).map(|x| (x % 7) as f32).collect();
    let b_val: Vec<f32> = (0..k * n).map(|x| (x % 5) as f32).collect();
    let pa = pack_f32_ones(&a_val);
    let pb = pack_f32_ones(&b_val);
    AoSBuffers {
        d_pa: ctx.device().htod_copy(pa).unwrap(),
        d_pb: ctx.device().htod_copy(pb).unwrap(),
        d_vc: ctx.device().alloc_zeros::<f32>(m * n).unwrap(),
        d_cc: ctx.device().alloc_zeros::<i32>(m * n).unwrap(),
    }
}

fn time_soa(
    ctx: &CudaContext, kernel_name: &'static str,
    grid: (u32, u32, u32), block: (u32, u32, u32),
    bufs: &mut SoABuffers,
    m: usize, n: usize, k: usize, iters: usize,
) -> f64 {
    let kernel = ctx.get_kernel(kernel_name).unwrap();
    let cfg = LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: 0 };
    let p: i32 = 1_073_741_789;
    let mu: u64 = ((1u128 << 64) / p as u128) as u64;
    unsafe {
        kernel.clone().launch(cfg,
            (&bufs.d_va, &bufs.d_ca, &bufs.d_vb, &bufs.d_cb,
             &mut bufs.d_vc, &mut bufs.d_cc, m as i32, n as i32, k as i32, p, mu)).unwrap();
    }
    ctx.device().synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            kernel.clone().launch(cfg,
                (&bufs.d_va, &bufs.d_ca, &bufs.d_vb, &bufs.d_cb,
                 &mut bufs.d_vc, &mut bufs.d_cc, m as i32, n as i32, k as i32, p, mu)).unwrap();
        }
    }
    ctx.device().synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn time_aos(
    ctx: &CudaContext, kernel_name: &'static str,
    grid: (u32, u32, u32), block: (u32, u32, u32),
    bufs: &mut AoSBuffers,
    m: usize, n: usize, k: usize, iters: usize,
) -> f64 {
    let kernel = ctx.get_kernel(kernel_name).unwrap();
    let cfg = LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: 0 };
    let p: i32 = 1_073_741_789;
    let mu: u64 = ((1u128 << 64) / p as u128) as u64;
    unsafe {
        kernel.clone().launch(cfg,
            (&bufs.d_pa, &bufs.d_pb, &mut bufs.d_vc, &mut bufs.d_cc,
             m as i32, n as i32, k as i32, p, mu)).unwrap();
    }
    ctx.device().synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            kernel.clone().launch(cfg,
                (&bufs.d_pa, &bufs.d_pb, &mut bufs.d_vc, &mut bufs.d_cc,
                 m as i32, n as i32, k as i32, p, mu)).unwrap();
        }
    }
    ctx.device().synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn bench_naive(ctx: &CudaContext, m: usize, k: usize, n: usize) {
    let ops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let iters = if m.max(n) <= 512 { 20 } else if m.max(n) <= 2048 { 10 } else { 5 };
    let mut soa = alloc_soa(ctx, m, k, n);
    let mut aos = alloc_aos(ctx, m, k, n);
    let grid  = CudaContext::counting_grid_dims_f32(m, n);
    let block = CudaContext::counting_block_dims_f32();
    let ms_soa = time_soa(ctx, "counting_gemm_f32_max",     grid, block, &mut soa, m, n, k, iters);
    let ms_aos = time_aos(ctx, "counting_gemm_f32_max_aos", grid, block, &mut aos, m, n, k, iters);
    let g_soa = ops / (ms_soa * 1e-3) / 1e9;
    let g_aos = ops / (ms_aos * 1e-3) / 1e9;
    println!(
        "M={:>5} K={:>5} N={:>5}  SoA {:>8.3} ms ({:>6.1} G)  AoS {:>8.3} ms ({:>6.1} G)  speedup {:>5.2}x",
        m, k, n, ms_soa, g_soa, ms_aos, g_aos, ms_soa / ms_aos
    );
}

fn bench_warpk(ctx: &CudaContext, m: usize, k: usize, n: usize) {
    let ops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let iters = 20;
    let mut soa = alloc_soa(ctx, m, k, n);
    let mut aos = alloc_aos(ctx, m, k, n);
    let grid  = CudaContext::counting_warpk_grid_dims(m, n);
    let block = CudaContext::counting_warpk_block_dims();
    let ms_soa = time_soa(ctx, "counting_gemm_f32_max_warpk",     grid, block, &mut soa, m, n, k, iters);
    let ms_aos = time_aos(ctx, "counting_gemm_f32_max_warpk_aos", grid, block, &mut aos, m, n, k, iters);
    let g_soa = ops / (ms_soa * 1e-3) / 1e9;
    let g_aos = ops / (ms_aos * 1e-3) / 1e9;
    println!(
        "M={:>5} K={:>5} N={:>5}  SoA {:>8.3} ms ({:>6.1} G)  AoS {:>8.3} ms ({:>6.1} G)  speedup {:>5.2}x",
        m, k, n, ms_soa, g_soa, ms_aos, g_aos, ms_soa / ms_aos
    );
}

fn main() {
    let ctx = CudaContext::new().unwrap();
    println!("Counting kernel: SoA vs AoS, NAIVE path (f32 Max, 1 prime)");
    println!("{}", "-".repeat(105));
    for &s in &[128usize, 256, 512, 1024, 2048, 4096] {
        bench_naive(&ctx, s, s, s);
    }

    println!("\nCounting kernel: SoA vs AoS, WARPK path (small-shape regime)");
    println!("{}", "-".repeat(105));
    for &(m, k, n) in &[(32usize, 4096, 32), (64, 4096, 64), (32, 2048, 32)] {
        bench_warpk(&ctx, m, k, n);
    }
}
