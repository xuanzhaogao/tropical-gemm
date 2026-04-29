//! Spec M: column-major counting tropical GEMM launch.
//!
//! `launch_tropical_matmul` selects the right `tropical_matmul_<T>_<dir>_<NN|NT|TN|TT>`
//! kernel based on the (transA, transB) flags and launches it on caller-owned
//! device buffers (raw `CUdeviceptr` wrapped in `DevPtr`).

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::DeviceRepr;

use crate::context::CudaContext;
use crate::error::Result;

/// True iff the device backing `ctx` has compute capability ≥ 8.0
/// (Ampere+). The pipelined `_pl` kernels target sm_80+; on older
/// devices we route to the sync kernels.
fn device_is_sm80_plus(ctx: &CudaContext) -> bool {
    use cudarc::driver::sys::CUdevice_attribute::*;
    ctx.device()
        .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map(|major| major >= 8)
        .unwrap_or(false)
}

/// Worst-case raw cnt accumulator value over the K-reduction:
/// `K * (P-1)^2`. Used by both u32 and u64 defer-mod gates.
fn defer_mod_bound(p: i32, k: usize) -> u128 {
    if p <= 1 {
        return 0;
    }
    let pm1 = (p as i64 - 1) as u128;
    pm1.saturating_mul(pm1).saturating_mul(k as u128)
}

/// Spec R u32 fast path: gate `K·(P-1)² < 2^32`. Strictly faster than
/// u64 when applicable (~1.7× on A100); requires sm_80+.
fn defer_mod_u32_safe(p: i32, k: usize) -> bool {
    defer_mod_bound(p, k) < (1u128 << 32)
}

/// Spec Q u64 fast path: gate `K·(P-1)² < 2^63`. Falls back to sync
/// when violated (full Barrett-in-loop, always correct).
fn defer_mod_u64_safe(p: i32, k: usize) -> bool {
    defer_mod_bound(p, k) < (1u128 << 63)
}

/// Pick the kernel suffix variant. Three-way preference:
/// 1. `_plu32` if sm_80+ and `K·(P-1)² < 2^32` (Spec R, fastest).
/// 2. `_pl` if sm_80+ and `K·(P-1)² < 2^63` (Spec Q, defer-mod u64).
/// 3. `""` (sync, Barrett-in-loop, always correct) otherwise.
fn pick_variant_suffix(ctx: &CudaContext, p: i32, k: usize) -> &'static str {
    if device_is_sm80_plus(ctx) {
        if defer_mod_u32_safe(p, k) {
            return "_plu32";
        }
        if defer_mod_u64_safe(p, k) {
            return "_pl";
        }
    }
    ""
}

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

/// Spec N: per-dtype tile dimensions. Block dim = (BN/TN, BM/TM, 1).
/// Grid dim = (ceil(N/BN), ceil(M/BM), 1).
pub trait TileDims {
    const BM: usize;
    const BN: usize;
    const TM: usize;
    const TN: usize;
}
impl TileDims for f32 {
    const BM: usize = 64; const BN: usize = 64; const TM: usize = 4; const TN: usize = 4;
}
impl TileDims for f64 {
    const BM: usize = 32; const BN: usize = 32; const TM: usize = 2; const TN: usize = 4;
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
        + TileDims
        + Default + Clone + Copy
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
    let suffix_with_variant = format!("{}{}", suffix, pick_variant_suffix(ctx, p, k));
    let kernel_name_owned: String = format!(
        "{}_{}",
        <(T, D) as TropicalMatmulKernelName<T, D>>::BASE_NAME,
        suffix_with_variant
    );
    // cudarc requires &'static str for kernel lookup. We leak the formatted
    // name once per (T, D, tA, tB) at runtime — 16 leaks total, bounded.
    let kernel_name: &'static str = Box::leak(kernel_name_owned.into_boxed_str());
    let kernel = ctx.get_kernel(kernel_name)?;

    // Spec N: dtype-specific tile dims. Block = (BN/TN, BM/TM); grid covers M, N.
    let block: (u32, u32, u32) = ((T::BN / T::TN) as u32, (T::BM / T::TM) as u32, 1);
    let grid: (u32, u32, u32) = (
        ((n + T::BN - 1) / T::BN) as u32,
        ((m + T::BM - 1) / T::BM) as u32,
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

#[cfg(test)]
mod gate_tests {
    use super::{defer_mod_u32_safe, defer_mod_u64_safe};

    #[test]
    fn small_p_picks_u32() {
        assert!(defer_mod_u32_safe(7, 4096));
        assert!(defer_mod_u32_safe(7, 1 << 25));   // 7·36 = 252 << 2^32
        assert!(defer_mod_u64_safe(7, 1 << 25));
    }

    #[test]
    fn medium_p_picks_u64_only() {
        // 22-bit prime: (P-1)^2 ≈ 2^44, so any K ≥ 1 trips u32 gate.
        let p = 2_965_819i32;
        assert!(!defer_mod_u32_safe(p, 1));
        assert!(defer_mod_u64_safe(p, 1 << 18));   // (P-1)²≈2^43; 2^43·2^18 = 2^61
        assert!(!defer_mod_u64_safe(p, 1 << 21));  // 2^43·2^21 = 2^64 ≥ 2^63
    }

    #[test]
    fn large_p_falls_through_to_sync() {
        // P near 2^31 trips both gates beyond very tiny K.
        let p = i32::MAX;
        assert!(!defer_mod_u32_safe(p, 16));
        assert!(!defer_mod_u64_safe(p, 16));
    }

    #[test]
    fn p_at_or_below_one_is_safe_for_both() {
        assert!(defer_mod_u32_safe(0, 1 << 30));
        assert!(defer_mod_u32_safe(1, 1 << 30));
        assert!(defer_mod_u64_safe(0, 1 << 30));
    }
}
