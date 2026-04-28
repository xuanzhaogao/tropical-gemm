//! Launch wrapper for the CountingTropical CUDA kernels.
//!
//! Element layout is AoS (`PairF32` / `PairF64` packed `(value, count)`).
//! The kernel expects **row-major** data (A is `m × k` row-major, B is
//! `k × n` row-major, C is `m × n` row-major). `count_ground_states_gpu`
//! packs the host-supplied row-major slices into `Vec<PairT>` once and
//! uploads the result, matching the CPU `tropical_matmul_t` /
//! `count_ground_states` convention.
//!
//! Two parallelization strategies, dispatched by shape:
//!   - naive: one thread per output cell. Wins for square / large M·N.
//!   - warpk: 32 threads cooperate per cell with K-stride loop and a
//!     `__shfl_xor_sync` tree reduction. Wins for small M·N + large K.

use crate::context::{CudaContext, COUNTING_WARPK_K_THRESHOLD, COUNTING_WARPK_MN_CEILING};
use crate::error::Result;
use crate::memory::GpuMatrix;
use crate::pair::PackPair;
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits};
use tropical_gemm::types::{Max, Min, TropicalDirection};

/// Wraps a raw `CUdeviceptr` so it can be passed to `kernel.launch(...)` as a
/// kernel parameter. Used by the kernel-only entry points whose callers
/// (e.g. CUDA.jl on the Julia side) own the device buffers themselves —
/// there is no `CudaSlice` to borrow from.
///
/// Layout matches what the kernel expects for any pointer-typed argument:
/// `as_kernel_param` returns a pointer to the inner `CUdeviceptr`, so the
/// driver dereferences it and pushes the actual device address as the
/// kernel arg. This is the same pattern `&CudaSlice<T>` uses internally.
#[derive(Copy, Clone, Debug)]
pub struct DevPtr(pub CUdeviceptr);

unsafe impl DeviceRepr for DevPtr {
    #[inline(always)]
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (&self.0) as *const CUdeviceptr as *mut std::ffi::c_void
    }
}

pub trait CountingCudaKernel<T, D>
where
    T: DeviceRepr + ValidAsZeroBits + Default + Clone + Copy + PackPair + 'static,
    D: TropicalDirection,
{
    /// Naive AoS kernel name (one thread per output cell, packed inputs).
    const KERNEL_NAME: &'static str;
    /// Warp-K-reduction AoS kernel name (32 threads/cell, packed inputs).
    const KERNEL_NAME_WARPK: &'static str;
    /// Naive ones-specialized kernel name (value-only inputs, count=1).
    const KERNEL_NAME_ONES: &'static str;
    /// Warp-K-reduction ones-specialized kernel name.
    const KERNEL_NAME_WARPK_ONES: &'static str;

    /// Returns (grid_dim, block_dim) for the naive variant.
    fn launch_dims(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32));
    /// Returns (grid_dim, block_dim) for the warpk variant.
    fn launch_dims_warpk(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
        (
            CudaContext::counting_warpk_grid_dims(m, n),
            CudaContext::counting_warpk_block_dims(),
        )
    }

    /// Launch the counting kernel. Output is SoA (separate `value_c` and
    /// `count_c` buffers); inputs are AoS pair buffers.
    fn launch_counting_gemm(
        ctx: &CudaContext,
        pair_a: &GpuMatrix<<T as PackPair>::Pair>,
        pair_b: &GpuMatrix<<T as PackPair>::Pair>,
        value_c: &mut GpuMatrix<T>,
        count_c: &mut GpuMatrix<i32>,
        modulus: i32,
    ) -> Result<()> {
        let m = pair_a.rows();
        let k = pair_a.cols();
        let n = pair_b.cols();

        assert_eq!(pair_b.rows(), k);
        assert_eq!(value_c.rows(), m);
        assert_eq!(value_c.cols(), n);
        assert_eq!(count_c.rows(), m);
        assert_eq!(count_c.cols(), n);

        // Shape-aware dispatch (measured on A100-80GB):
        //   warpk wins when M*N is small (GPU starved for parallelism) AND
        //   K is large enough to amortize the warp shuffle reduction.
        //   naive wins for square / large M*N: its coalesced-B access beats
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
        let mu: u64 = if modulus > 1 {
            ((1u128 << 64) / modulus as u128) as u64
        } else {
            0
        };

        unsafe {
            kernel.launch(
                cfg,
                (
                    pair_a.as_slice(),
                    pair_b.as_slice(),
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

    /// Kernel-only launch. Identical kernel + dispatch to
    /// `launch_counting_gemm`, but takes raw device pointers (`DevPtr`) for
    /// the four data buffers instead of `&GpuMatrix<...>`. Used by callers
    /// who own their device memory (e.g. CUDA.jl); Rust performs no
    /// allocation, no upload, and no download.
    ///
    /// # Stream coordination
    ///
    /// Calls `ctx.device().synchronize()` *before* the kernel launch so that
    /// any prior work on other streams (e.g. CUDA.jl's task-local stream
    /// just used to upload `pair_a` / `pair_b`) is fully complete. Required
    /// because cudarc launches on its own stream — without this barrier,
    /// the kernel can race uploads issued from the Julia side.
    ///
    /// # Safety contract
    ///
    /// Caller is responsible for asserting that:
    /// - `pair_a_dev` points to at least `m * k` `<T as PackPair>::Pair` elements,
    /// - `pair_b_dev` points to at least `k * n` `<T as PackPair>::Pair` elements,
    /// - `out_val_dev` and `out_cnt_dev` each point to at least `m * n` elements
    ///   of `T` and `i32` respectively,
    /// - all four pointers are valid in the primary CUDA context.
    ///
    /// Buffer-length validation is impossible from raw pointers alone.
    fn launch_counting_gemm_dev_ptr(
        ctx: &CudaContext,
        pair_a_dev: DevPtr,
        m: usize,
        k: usize,
        pair_b_dev: DevPtr,
        n: usize,
        out_val_dev: DevPtr,
        out_cnt_dev: DevPtr,
        modulus: i32,
    ) -> Result<()> {
        // Sync *before* the launch: cudarc launches on its own stream, while
        // the caller (CUDA.jl) may have just enqueued uploads on its
        // task-local stream. A device-wide barrier here ensures those
        // uploads are visible to the kernel.
        ctx.device().synchronize()?;

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

        let mu: u64 = if modulus > 1 {
            ((1u128 << 64) / modulus as u128) as u64
        } else {
            0
        };

        unsafe {
            kernel.launch(
                cfg,
                (
                    pair_a_dev,
                    pair_b_dev,
                    out_val_dev,
                    out_cnt_dev,
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

    /// Ones-specialized launch (value-only inputs; counts are uniformly 1).
    /// This is the entry point used by `count_ground_states_gpu`. Output is
    /// SoA, same as the AoS path. Same shape-aware naive-vs-warpk dispatch.
    ///
    /// `value_b` layout depends on dispatch (caller's responsibility):
    ///   - naive (`M*N > 64*64`): K × N row-major (B as `B[k,j]`).
    ///   - warpk (`M*N <= 64*64` && `K >= 64`): N × K row-major (B^T,
    ///     so lanes load coalesced K-strides).
    /// `value_a` is always M × K row-major. M, K, N are passed explicitly
    /// because the `GpuMatrix` shape of `value_b` is layout-dependent.
    fn launch_counting_gemm_ones(
        ctx: &CudaContext,
        value_a: &GpuMatrix<T>,
        value_b: &GpuMatrix<T>,
        value_c: &mut GpuMatrix<T>,
        count_c: &mut GpuMatrix<i32>,
        m: usize,
        k: usize,
        n: usize,
        modulus: i32,
    ) -> Result<()> {
        debug_assert_eq!(value_a.rows(), m);
        debug_assert_eq!(value_a.cols(), k);
        debug_assert_eq!(value_a.rows() * value_a.cols(), m * k);
        debug_assert_eq!(value_b.rows() * value_b.cols(), k * n);
        debug_assert_eq!(value_c.rows(), m);
        debug_assert_eq!(value_c.cols(), n);
        debug_assert_eq!(count_c.rows(), m);
        debug_assert_eq!(count_c.cols(), n);

        let use_warpk = k >= COUNTING_WARPK_K_THRESHOLD
            && m.saturating_mul(n) <= COUNTING_WARPK_MN_CEILING;
        let kernel_name = if use_warpk {
            Self::KERNEL_NAME_WARPK_ONES
        } else {
            Self::KERNEL_NAME_ONES
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
                    value_b.as_slice(),
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
    const KERNEL_NAME_ONES: &'static str = "counting_gemm_f32_max_ones";
    const KERNEL_NAME_WARPK_ONES: &'static str = "counting_gemm_f32_max_warpk_ones";
    fn launch_dims(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
        (CudaContext::counting_grid_dims_f32(m, n), CudaContext::counting_block_dims_f32())
    }
}
impl CountingCudaKernel<f32, Min> for (f32, Min) {
    const KERNEL_NAME: &'static str = "counting_gemm_f32_min";
    const KERNEL_NAME_WARPK: &'static str = "counting_gemm_f32_min_warpk";
    const KERNEL_NAME_ONES: &'static str = "counting_gemm_f32_min_ones";
    const KERNEL_NAME_WARPK_ONES: &'static str = "counting_gemm_f32_min_warpk_ones";
    fn launch_dims(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
        (CudaContext::counting_grid_dims_f32(m, n), CudaContext::counting_block_dims_f32())
    }
}
impl CountingCudaKernel<f64, Max> for (f64, Max) {
    const KERNEL_NAME: &'static str = "counting_gemm_f64_max";
    const KERNEL_NAME_WARPK: &'static str = "counting_gemm_f64_max_warpk";
    const KERNEL_NAME_ONES: &'static str = "counting_gemm_f64_max_ones";
    const KERNEL_NAME_WARPK_ONES: &'static str = "counting_gemm_f64_max_warpk_ones";
    fn launch_dims(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
        (CudaContext::counting_grid_dims_f64(m, n), CudaContext::counting_block_dims_f64())
    }
}
impl CountingCudaKernel<f64, Min> for (f64, Min) {
    const KERNEL_NAME: &'static str = "counting_gemm_f64_min";
    const KERNEL_NAME_WARPK: &'static str = "counting_gemm_f64_min_warpk";
    const KERNEL_NAME_ONES: &'static str = "counting_gemm_f64_min_ones";
    const KERNEL_NAME_WARPK_ONES: &'static str = "counting_gemm_f64_min_warpk_ones";
    fn launch_dims(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
        (CudaContext::counting_grid_dims_f64(m, n), CudaContext::counting_block_dims_f64())
    }
}

pub fn launch_counting_gemm<T, D>(
    ctx: &CudaContext,
    pair_a: &GpuMatrix<<T as PackPair>::Pair>,
    pair_b: &GpuMatrix<<T as PackPair>::Pair>,
    value_c: &mut GpuMatrix<T>,
    count_c: &mut GpuMatrix<i32>,
    modulus: i32,
) -> Result<()>
where
    T: DeviceRepr + ValidAsZeroBits + Default + Clone + Copy + PackPair + 'static,
    D: TropicalDirection,
    (T, D): CountingCudaKernel<T, D>,
{
    <(T, D) as CountingCudaKernel<T, D>>::launch_counting_gemm(
        ctx, pair_a, pair_b, value_c, count_c, modulus,
    )
}

pub fn launch_counting_gemm_dev_ptr<T, D>(
    ctx: &CudaContext,
    pair_a_dev: DevPtr,
    m: usize,
    k: usize,
    pair_b_dev: DevPtr,
    n: usize,
    out_val_dev: DevPtr,
    out_cnt_dev: DevPtr,
    modulus: i32,
) -> Result<()>
where
    T: DeviceRepr + ValidAsZeroBits + Default + Clone + Copy + PackPair + 'static,
    D: TropicalDirection,
    (T, D): CountingCudaKernel<T, D>,
{
    <(T, D) as CountingCudaKernel<T, D>>::launch_counting_gemm_dev_ptr(
        ctx, pair_a_dev, m, k, pair_b_dev, n, out_val_dev, out_cnt_dev, modulus,
    )
}

pub fn launch_counting_gemm_ones<T, D>(
    ctx: &CudaContext,
    value_a: &GpuMatrix<T>,
    value_b: &GpuMatrix<T>,
    value_c: &mut GpuMatrix<T>,
    count_c: &mut GpuMatrix<i32>,
    m: usize,
    k: usize,
    n: usize,
    modulus: i32,
) -> Result<()>
where
    T: DeviceRepr + ValidAsZeroBits + Default + Clone + Copy + PackPair + 'static,
    D: TropicalDirection,
    (T, D): CountingCudaKernel<T, D>,
{
    <(T, D) as CountingCudaKernel<T, D>>::launch_counting_gemm_ones(
        ctx, value_a, value_b, value_c, count_c, m, k, n, modulus,
    )
}
