//! Compare naive vs warp-K-reduction counting kernel head-to-head.
//! Both bench loops time pure kernel launches with data already on device.

use std::time::Instant;
use cudarc::driver::{LaunchAsync, LaunchConfig};
use tropical_gemm_cuda::CudaContext;

struct Buffers {
    d_va: cudarc::driver::CudaSlice<f32>,
    d_ca: cudarc::driver::CudaSlice<i32>,
    d_vb: cudarc::driver::CudaSlice<f32>,
    d_cb: cudarc::driver::CudaSlice<i32>,
    d_vc: cudarc::driver::CudaSlice<f32>,
    d_cc: cudarc::driver::CudaSlice<i32>,
}

fn alloc(ctx: &CudaContext, m: usize, k: usize, n: usize) -> Buffers {
    let a_val: Vec<f32> = (0..m * k).map(|x| (x % 7) as f32).collect();
    let b_val: Vec<f32> = (0..k * n).map(|x| (x % 5) as f32).collect();
    Buffers {
        d_va: ctx.device().htod_copy(a_val).unwrap(),
        d_ca: ctx.device().htod_copy(vec![1_i32; m * k]).unwrap(),
        d_vb: ctx.device().htod_copy(b_val).unwrap(),
        d_cb: ctx.device().htod_copy(vec![1_i32; k * n]).unwrap(),
        d_vc: ctx.device().alloc_zeros::<f32>(m * n).unwrap(),
        d_cc: ctx.device().alloc_zeros::<i32>(m * n).unwrap(),
    }
}

fn time_iters(
    ctx: &CudaContext,
    kernel_name: &'static str,
    grid: (u32, u32, u32),
    block: (u32, u32, u32),
    bufs: &mut Buffers,
    m: usize, n: usize, k: usize,
    iters: usize,
) -> f64 {
    let kernel = ctx.get_kernel(kernel_name).unwrap();
    let cfg = LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: 0 };
    let p: i32 = 1_073_741_789;
    let mu: u64 = ((1u128 << 64) / p as u128) as u64;

    // Warmup.
    unsafe {
        kernel.clone().launch(
            cfg,
            (&bufs.d_va, &bufs.d_ca, &bufs.d_vb, &bufs.d_cb,
             &mut bufs.d_vc, &mut bufs.d_cc,
             m as i32, n as i32, k as i32, p, mu),
        ).unwrap();
    }
    ctx.device().synchronize().unwrap();

    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            kernel.clone().launch(
                cfg,
                (&bufs.d_va, &bufs.d_ca, &bufs.d_vb, &bufs.d_cb,
                 &mut bufs.d_vc, &mut bufs.d_cc,
                 m as i32, n as i32, k as i32, p, mu),
            ).unwrap();
        }
    }
    ctx.device().synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn bench_rect(ctx: &CudaContext, m: usize, k: usize, n: usize) {
    let ops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let iters = 20;

    let mut bufs = alloc(ctx, m, k, n);
    let naive_grid  = CudaContext::counting_grid_dims_f32(m, n);
    let naive_block = CudaContext::counting_block_dims_f32();
    let warpk_grid  = CudaContext::counting_warpk_grid_dims(m, n);
    let warpk_block = CudaContext::counting_warpk_block_dims();

    let ms_naive = time_iters(ctx, "counting_gemm_f32_max",
                              naive_grid, naive_block, &mut bufs, m, n, k, iters);
    let ms_warpk = time_iters(ctx, "counting_gemm_f32_max_warpk",
                              warpk_grid, warpk_block, &mut bufs, m, n, k, iters);
    let g_naive = ops / (ms_naive * 1e-3) / 1e9;
    let g_warpk = ops / (ms_warpk * 1e-3) / 1e9;
    println!(
        "M={:>4} K={:>5} N={:>4}  naive {:>7.3} ms ({:>5.1} G)   warpk {:>7.3} ms ({:>5.1} G)   speedup {:>5.2}x",
        m, k, n, ms_naive, g_naive, ms_warpk, g_warpk, ms_naive / ms_warpk
    );
}

fn bench(ctx: &CudaContext, size: usize) {
    let m = size; let k = size; let n = size;
    let ops = 2.0 * (size as f64).powi(3);
    let iters = if size <= 256 { 30 } else if size <= 1024 { 10 } else { 5 };

    let mut bufs = alloc(ctx, m, k, n);

    let naive_grid  = CudaContext::counting_grid_dims_f32(m, n);
    let naive_block = CudaContext::counting_block_dims_f32();
    let warpk_grid  = CudaContext::counting_warpk_grid_dims(m, n);
    let warpk_block = CudaContext::counting_warpk_block_dims();

    let ms_naive = time_iters(ctx, "counting_gemm_f32_max",
                              naive_grid, naive_block, &mut bufs, m, n, k, iters);
    let ms_warpk = time_iters(ctx, "counting_gemm_f32_max_warpk",
                              warpk_grid, warpk_block, &mut bufs, m, n, k, iters);

    let g_naive = ops / (ms_naive * 1e-3) / 1e9;
    let g_warpk = ops / (ms_warpk * 1e-3) / 1e9;
    println!(
        "size={:>5}  naive {:>8.2} ms ({:>6.1} G)   warpk {:>8.2} ms ({:>6.1} G)   speedup {:>5.2}x",
        size, ms_naive, g_naive, ms_warpk, g_warpk, ms_naive / ms_warpk
    );
}

fn main() {
    let ctx = CudaContext::new().unwrap();
    println!("Counting kernel: naive vs warp-K-reduction (f32 Max, 1 prime)");
    println!("{}", "-".repeat(95));
    println!("\n[square]");
    for &s in &[128usize, 256, 512, 1024, 2048, 4096] {
        bench(&ctx, s);
    }
    // Also check very-small M*N (parallelism-starved) and very-tall-skinny (K>>M*N).
    println!("\n[tall-skinny: small M*N, large K]");
    for &(m, k, n) in &[(32usize, 4096, 32), (64, 4096, 64), (128, 4096, 128)] {
        bench_rect(&ctx, m, k, n);
    }
}
