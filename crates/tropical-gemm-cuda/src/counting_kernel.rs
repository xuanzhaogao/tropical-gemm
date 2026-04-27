//! Launch wrapper for the CountingTropical CUDA kernels (spec C, tiled in spec D).
//!
//! Reuses `GpuMatrix<T>` for values and `GpuMatrix<i32>` for count residues.
//! The kernel expects **row-major** data (A is `m × k` row-major, B is
//! `k × n` row-major, C is `m × n` row-major). `count_ground_states_gpu`
//! uploads the host-supplied row-major slices unchanged, matching the CPU
//! `tropical_matmul_t` / `count_ground_states` convention.

use crate::context::{CudaContext, COUNTING_WARPK_K_THRESHOLD, COUNTING_WARPK_MN_CEILING};
use crate::error::Result;
use crate::memory::GpuMatrix;
use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits};
use tropical_gemm::types::{Max, Min, TropicalDirection};

pub trait CountingCudaKernel<T, D>
where
    T: DeviceRepr + ValidAsZeroBits + Default + Clone + Copy + 'static,
    D: TropicalDirection,
{
    const KERNEL_NAME: &'static str;
    /// Warp-K-reduction variant (spec E stage A).
    const KERNEL_NAME_WARPK: &'static str;

    /// Returns (grid_dim, block_dim) for a launch covering `m × n` output cells.
    fn launch_dims(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32));
    /// Returns (grid_dim, block_dim) for the warpk variant.
    fn launch_dims_warpk(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
        (
            CudaContext::counting_warpk_grid_dims(m, n),
            CudaContext::counting_warpk_block_dims(),
        )
    }

    fn launch_counting_gemm(
        ctx: &CudaContext,
        value_a: &GpuMatrix<T>,
        count_a: &GpuMatrix<i32>,
        value_b: &GpuMatrix<T>,
        count_b: &GpuMatrix<i32>,
        value_c: &mut GpuMatrix<T>,
        count_c: &mut GpuMatrix<i32>,
        modulus: i32,
    ) -> Result<()> {
        let m = value_a.rows();
        let k = value_a.cols();
        let n = value_b.cols();

        assert_eq!(count_a.rows(), m);
        assert_eq!(count_a.cols(), k);
        assert_eq!(value_b.rows(), k);
        assert_eq!(count_b.rows(), k);
        assert_eq!(count_b.cols(), n);
        assert_eq!(value_c.rows(), m);
        assert_eq!(value_c.cols(), n);
        assert_eq!(count_c.rows(), m);
        assert_eq!(count_c.cols(), n);

        // Shape-aware dispatch (measured on A100-80GB):
        //   warpk wins when M*N is small (GPU starved for parallelism) AND
        //   K is large enough to amortize the warp shuffle reduction.
        //   naive wins for square/large M*N: its coalesced-B access beats
        //   warpk's strided-B-by-N access at scale.
        let use_warpk = k >= COUNTING_WARPK_K_THRESHOLD
            && m.saturating_mul(n) <= COUNTING_WARPK_MN_CEILING;
        let kernel_name = if use_warpk {
            Self::KERNEL_NAME_WARPK
        } else {
            Self::KERNEL_NAME
        };
        let kernel = ctx.get_kernel(kernel_name)?;
        let (grid_dim, block_dim) = if use_warpk {
            Self::launch_dims_warpk(m, n)
        } else {
            Self::launch_dims(m, n)
        };
        let cfg = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes: 0,
        };

        // Barrett reciprocal mu = floor(2^64 / P). Host-precomputed so the
        // kernel can replace `% P` with mul-hi + sub + cond-sub.
        // For P >= 1, u128 math is exact. For the special case P == 1 (should
        // not occur with CRT_PRIMES but guard anyway), mu = 0 gives correct
        // r = x which is then reduced by the `if (r >= P) r -= P` correction.
        let mu: u64 = if modulus > 1 {
            ((1u128 << 64) / modulus as u128) as u64
        } else {
            0
        };

        unsafe {
            kernel.launch(
                cfg,
                (
                    value_a.as_slice(),
                    count_a.as_slice(),
                    value_b.as_slice(),
                    count_b.as_slice(),
                    value_c.as_slice_mut(),
                    count_c.as_slice_mut(),
                    m as i32,
                    n as i32,
                    k as i32,
                    modulus,
                    mu,
                ),
            )?;
        }

        ctx.device().synchronize()?;
        Ok(())
    }
}

impl CountingCudaKernel<f32, Max> for (f32, Max) {
    const KERNEL_NAME: &'static str = "counting_gemm_f32_max";
    const KERNEL_NAME_WARPK: &'static str = "counting_gemm_f32_max_warpk";
    fn launch_dims(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
        (CudaContext::counting_grid_dims_f32(m, n), CudaContext::counting_block_dims_f32())
    }
}
impl CountingCudaKernel<f32, Min> for (f32, Min) {
    const KERNEL_NAME: &'static str = "counting_gemm_f32_min";
    const KERNEL_NAME_WARPK: &'static str = "counting_gemm_f32_min_warpk";
    fn launch_dims(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
        (CudaContext::counting_grid_dims_f32(m, n), CudaContext::counting_block_dims_f32())
    }
}
impl CountingCudaKernel<f64, Max> for (f64, Max) {
    const KERNEL_NAME: &'static str = "counting_gemm_f64_max";
    const KERNEL_NAME_WARPK: &'static str = "counting_gemm_f64_max_warpk";
    fn launch_dims(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
        (CudaContext::counting_grid_dims_f64(m, n), CudaContext::counting_block_dims_f64())
    }
}
impl CountingCudaKernel<f64, Min> for (f64, Min) {
    const KERNEL_NAME: &'static str = "counting_gemm_f64_min";
    const KERNEL_NAME_WARPK: &'static str = "counting_gemm_f64_min_warpk";
    fn launch_dims(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
        (CudaContext::counting_grid_dims_f64(m, n), CudaContext::counting_block_dims_f64())
    }
}

pub fn launch_counting_gemm<T, D>(
    ctx: &CudaContext,
    value_a: &GpuMatrix<T>,
    count_a: &GpuMatrix<i32>,
    value_b: &GpuMatrix<T>,
    count_b: &GpuMatrix<i32>,
    value_c: &mut GpuMatrix<T>,
    count_c: &mut GpuMatrix<i32>,
    modulus: i32,
) -> Result<()>
where
    T: DeviceRepr + ValidAsZeroBits + Default + Clone + Copy + 'static,
    D: TropicalDirection,
    (T, D): CountingCudaKernel<T, D>,
{
    <(T, D) as CountingCudaKernel<T, D>>::launch_counting_gemm(
        ctx, value_a, count_a, value_b, count_b, value_c, count_c, modulus,
    )
}
