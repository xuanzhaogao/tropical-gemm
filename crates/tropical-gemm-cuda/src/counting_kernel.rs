//! Spec M: column-major counting tropical GEMM launch.
//!
//! `launch_tropical_matmul` selects the right `tropical_matmul_<T>_<dir>_<NN|NT|TN|TT>`
//! kernel based on the (transA, transB) flags and launches it on caller-owned
//! device buffers (raw `CUdeviceptr` wrapped in `DevPtr`).

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::DeviceRepr;

use crate::context::CudaContext;
use crate::error::Result;

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

/// Spec M: column-major NN/NT/TN/TT counting matmul launch.
/// Operates on raw device pointers (caller-owned, e.g. CUDA.jl).
/// `ctx.device().synchronize()` is called before launch to coordinate
/// with CUDA.jl's stream.
pub fn launch_tropical_matmul<T, D>(
    ctx: &CudaContext,
    tA: char,
    tB: char,
    m: usize,
    k: usize,
    n: usize,
    a_dev_ptr: u64,
    b_dev_ptr: u64,
    p: i32,
    out_dev_ptr: u64,
) -> Result<()>
where
    T: tropical_gemm::types::TropicalScalar
        + cudarc::driver::DeviceRepr
        + cudarc::driver::ValidAsZeroBits
        + Default + Clone + Copy
        + crate::pair::PackPair
        + 'static,
    D: tropical_gemm::types::TropicalDirection,
    (T, D): TropicalMatmulKernelName<T, D>,
{
    let suffix = match (tA, tB) {
        ('N', 'N') => "NN",
        ('N', 'T') => "NT",
        ('T', 'N') => "TN",
        ('T', 'T') => "TT",
        _ => return Err(crate::error::CudaError::InvalidState(format!(
            "tA/tB must be in {{'N','T'}}, got tA={}, tB={}", tA, tB
        ))),
    };
    let kernel_name_owned: String = format!(
        "{}_{}",
        <(T, D) as TropicalMatmulKernelName<T, D>>::BASE_NAME,
        suffix
    );
    // cudarc requires &'static str for kernel lookup. We leak the formatted
    // name once per (T, D, tA, tB) at runtime — 16 leaks total, bounded.
    let kernel_name: &'static str = Box::leak(kernel_name_owned.into_boxed_str());
    let kernel = ctx.get_kernel(kernel_name)?;

    // Block 16x16, grid covers M, N (column-major: blockIdx.y is row, blockIdx.x is col).
    let block: (u32, u32, u32) = (16, 16, 1);
    let grid: (u32, u32, u32) = (
        ((n + 15) / 16) as u32,
        ((m + 15) / 16) as u32,
        1,
    );
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: 0,
    };

    // Barrett mu for the kernel.
    let mu: u64 = if p > 1 {
        ((1u128 << 64) / p as u128) as u64
    } else {
        0
    };

    // Pre-launch sync: wait for any pending CUDA.jl uploads on caller's stream.
    ctx.device().synchronize()?;

    // Pass raw device pointers via DevPtr (impls DeviceRepr).
    let a_dp = DevPtr(a_dev_ptr);
    let b_dp = DevPtr(b_dev_ptr);
    let out_dp = DevPtr(out_dev_ptr);

    unsafe {
        use cudarc::driver::LaunchAsync;
        kernel.launch(
            cfg,
            (
                a_dp, b_dp, out_dp,
                m as i32, n as i32, k as i32, p, mu,
            ),
        )?;
    }

    ctx.device().synchronize()?;
    Ok(())
}

/// Trait providing the base kernel name for each (T, D) combo. The runtime
/// dispatch to NN/NT/TN/TT appends a suffix.
pub trait TropicalMatmulKernelName<T, D> {
    const BASE_NAME: &'static str;
}
impl TropicalMatmulKernelName<f32, tropical_gemm::types::Max> for (f32, tropical_gemm::types::Max) {
    const BASE_NAME: &'static str = "tropical_matmul_f32_max";
}
impl TropicalMatmulKernelName<f32, tropical_gemm::types::Min> for (f32, tropical_gemm::types::Min) {
    const BASE_NAME: &'static str = "tropical_matmul_f32_min";
}
impl TropicalMatmulKernelName<f64, tropical_gemm::types::Max> for (f64, tropical_gemm::types::Max) {
    const BASE_NAME: &'static str = "tropical_matmul_f64_max";
}
impl TropicalMatmulKernelName<f64, tropical_gemm::types::Min> for (f64, tropical_gemm::types::Min) {
    const BASE_NAME: &'static str = "tropical_matmul_f64_min";
}
