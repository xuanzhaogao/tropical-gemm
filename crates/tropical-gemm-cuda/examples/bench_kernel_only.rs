//! Times JUST the counting kernel launch, data already on device.
//! Strips away CRT driver + upload/download overhead to compare
//! directly against Julia's `CUDA.@sync a * b` timing.

use std::time::Instant;
use tropical_gemm_cuda::{CudaContext};
use cudarc::driver::{DeviceSlice, LaunchAsync, LaunchConfig};

fn bench(ctx: &CudaContext, size: usize) {
    let m = size; let k = size; let n = size;
    let ops = 2.0 * (size as f64).powi(3);

    // Synthesize a = [0..7] cycling, b likewise.
    let a_val: Vec<f32> = (0..m * k).map(|x| (x % 7) as f32).collect();
    let b_val: Vec<f32> = (0..k * n).map(|x| (x % 5) as f32).collect();
    let ones_ak: Vec<i32> = vec![1; m * k];
    let ones_kn: Vec<i32> = vec![1; k * n];

    // Upload once. These stay on device for all bench iters.
    let d_va = ctx.device().htod_copy(a_val).unwrap();
    let d_ca = ctx.device().htod_copy(ones_ak).unwrap();
    let d_vb = ctx.device().htod_copy(b_val).unwrap();
    let d_cb = ctx.device().htod_copy(ones_kn).unwrap();
    let mut d_vc = ctx.device().alloc_zeros::<f32>(m * n).unwrap();
    let mut d_cc = ctx.device().alloc_zeros::<i32>(m * n).unwrap();

    let p: i32 = 1_073_741_789;
    let mu: u64 = ((1u128 << 64) / p as u128) as u64;

    let kernel = ctx.get_kernel("counting_gemm_f32_max").unwrap();
    let (grid, block) = (
        CudaContext::counting_grid_dims_f32(m, n),
        CudaContext::counting_block_dims_f32(),
    );
    let cfg = LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: 0 };

    // Warm.
    unsafe {
        kernel.clone().launch(
            cfg,
            (&d_va, &d_ca, &d_vb, &d_cb, &mut d_vc, &mut d_cc, m as i32, n as i32, k as i32, p, mu),
        ).unwrap();
    }
    ctx.device().synchronize().unwrap();

    let iters = if size <= 512 { 20 } else if size <= 1024 { 10 } else { 5 };
    let t0 = Instant::now();
    for _ in 0..iters {
        unsafe {
            kernel.clone().launch(
                cfg,
                (&d_va, &d_ca, &d_vb, &d_cb, &mut d_vc, &mut d_cc, m as i32, n as i32, k as i32, p, mu),
            ).unwrap();
        }
    }
    ctx.device().synchronize().unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let gops = ops / (ms * 1e-3) / 1e9;
    println!("size={:>5}  kernel-only {:>8.2} ms  ({:>6.1} G tropical-ops/s)", size, ms, gops);
}

fn main() {
    let ctx = CudaContext::new().unwrap();
    println!("Counting kernel — pure launch timing (data already on device)");
    println!("{}", "-".repeat(70));
    bench(&ctx, 128);
    bench(&ctx, 256);
    bench(&ctx, 512);
    bench(&ctx, 1024);
    bench(&ctx, 2048);
    bench(&ctx, 4096);
}
