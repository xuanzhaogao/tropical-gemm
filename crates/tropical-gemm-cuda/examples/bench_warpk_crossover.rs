//! Find where warpk-ones (transposed B) crosses naive-ones, post-Spec H.
//! Both use ones-specialized inner loop; only the parallelization differs.

use std::time::Instant;
use cudarc::driver::{LaunchAsync, LaunchConfig, CudaSlice};
use tropical_gemm_cuda::CudaContext;

fn upload_naive(ctx: &CudaContext, m: usize, k: usize, n: usize)
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

fn upload_warpk(ctx: &CudaContext, m: usize, k: usize, n: usize)
    -> (CudaSlice<f32>, CudaSlice<f32>, CudaSlice<f32>, CudaSlice<i32>)
{
    let a: Vec<f32> = (0..m * k).map(|x| (x % 7) as f32).collect();
    let b: Vec<f32> = (0..k * n).map(|x| (x % 5) as f32).collect();
    // Transpose B from K×N row-major to N×K row-major.
    let mut b_t = vec![0.0_f32; n * k];
    for kv in 0..k {
        for j in 0..n {
            b_t[j * k + kv] = b[kv * n + j];
        }
    }
    (
        ctx.device().htod_copy(a).unwrap(),
        ctx.device().htod_copy(b_t).unwrap(),
        ctx.device().alloc_zeros::<f32>(m * n).unwrap(),
        ctx.device().alloc_zeros::<i32>(m * n).unwrap(),
    )
}

fn time_kernel(ctx: &CudaContext, kernel_name: &'static str,
               grid: (u32, u32, u32), block: (u32, u32, u32),
               d_a: &CudaSlice<f32>, d_b: &CudaSlice<f32>,
               d_vc: &mut CudaSlice<f32>, d_cc: &mut CudaSlice<i32>,
               m: usize, n: usize, k: usize, iters: usize) -> f64
{
    let kernel = ctx.get_kernel(kernel_name).unwrap();
    let cfg = LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: 0 };
    let p: i32 = 1_073_741_789;
    let mu: u64 = ((1u128 << 64) / p as u128) as u64;
    unsafe {
        kernel.clone().launch(cfg, (d_a, d_b, &mut *d_vc, &mut *d_cc,
            m as i32, n as i32, k as i32, p, mu)).unwrap();
    }
    ctx.device().synchronize().unwrap();
    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            kernel.clone().launch(cfg, (d_a, d_b, &mut *d_vc, &mut *d_cc,
                m as i32, n as i32, k as i32, p, mu)).unwrap();
        }
    }
    ctx.device().synchronize().unwrap();
    t0.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn bench(ctx: &CudaContext, m: usize, k: usize, n: usize) {
    let ops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let iters = if m.max(n) <= 256 { 30 } else if m.max(n) <= 1024 { 10 } else { 5 };

    let (d_a_n, d_b_n, mut d_vc_n, mut d_cc_n) = upload_naive(ctx, m, k, n);
    let ms_naive = time_kernel(ctx, "counting_gemm_f32_max_ones",
        CudaContext::counting_grid_dims_f32(m, n),
        CudaContext::counting_block_dims_f32(),
        &d_a_n, &d_b_n, &mut d_vc_n, &mut d_cc_n, m, n, k, iters);

    let (d_a_w, d_b_w, mut d_vc_w, mut d_cc_w) = upload_warpk(ctx, m, k, n);
    let ms_warpk = time_kernel(ctx, "counting_gemm_f32_max_warpk_ones",
        CudaContext::counting_warpk_grid_dims(m, n),
        CudaContext::counting_warpk_block_dims(),
        &d_a_w, &d_b_w, &mut d_vc_w, &mut d_cc_w, m, n, k, iters);

    let g_naive = ops / (ms_naive * 1e-3) / 1e9;
    let g_warpk = ops / (ms_warpk * 1e-3) / 1e9;
    let winner = if ms_warpk < ms_naive { "WARPK" } else { "naive" };
    println!(
        "M={:>5} K={:>5} N={:>5}  M*N={:>7}  naive {:>7.2} ms ({:>6.1} G)  warpk {:>7.2} ms ({:>6.1} G)  speedup {:>5.2}x  -> {}",
        m, k, n, m*n, ms_naive, g_naive, ms_warpk, g_warpk, ms_naive / ms_warpk, winner
    );
}

fn main() {
    let ctx = CudaContext::new().unwrap();
    println!("warpk-ones (transposed B) vs naive-ones — find new crossover");
    println!("{}", "-".repeat(120));
    // Sweep M=N along the diagonal at K=4096.
    for &m in &[16usize, 32, 64, 96, 128, 192, 256, 384, 512, 1024] {
        bench(&ctx, m, 4096, m);
    }
    println!("\nAt K=512 (smaller K reduces warpk amortization):");
    for &m in &[32usize, 64, 128, 256] {
        bench(&ctx, m, 512, m);
    }
    println!("\nNon-square (M small, N varied):");
    for &n in &[32usize, 64, 128, 256, 512] {
        bench(&ctx, 32, 4096, n);
    }
}
