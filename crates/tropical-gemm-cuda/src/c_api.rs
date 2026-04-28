//! C ABI for the counting tropical matmul.
//!
//! Exposes four `extern "C"` entry points (one per `(T, D)` combo) plus
//! `tg_api_version` and `tg_last_error_message` for callers that want a
//! human-readable error message after a non-zero return. Used by
//! `CountingTropicalGEMM.jl` (Julia binding); other languages can `dlopen`
//! the shared library and call these directly.
//!
//! ## Error codes
//!
//! - `0` — success
//! - `1` — invalid input (null pointer, dimension mismatch)
//! - `3` — CUDA error (kernel launch / memory / context)
//! - `4` — Rust panic or other internal error
//!
//! ## Thread safety
//!
//! The lazy global `CudaContext` is mutex-guarded internally, so multiple
//! threads can call these functions concurrently. Each thread has its own
//! last-error TLS slot.

use std::ffi::{c_char, CString};
use std::os::raw::c_int;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::cell::RefCell;

use tropical_gemm::types::{Max, Min};

use crate::get_global_context;
use crate::matmul_mod::tropical_matmul_kernel;

/// Bumped on any source-incompatible C ABI change. The Julia binding (or
/// any other ABI consumer) is expected to assert this at load time.
pub const TG_API_VERSION: i32 = 1;

const OK: i32 = 0;
const ERR_INVALID_INPUT: i32 = 1;
const ERR_CUDA: i32 = 3;
const ERR_INTERNAL: i32 = 4;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn store_error(msg: impl Into<String>) {
    let s = msg.into();
    let c = CString::new(s).unwrap_or_else(|_| CString::new("(non-utf8 error)").unwrap());
    LAST_ERROR.with(|cell| *cell.borrow_mut() = Some(c));
}

/// API version. Lets ABI consumers (Julia package) assert compatibility.
#[no_mangle]
pub extern "C" fn tg_api_version() -> c_int {
    TG_API_VERSION
}

/// Pointer to the last error message on this thread, or NULL if no error
/// has been stored. The pointer is valid until the next call on this
/// thread.
#[no_mangle]
pub extern "C" fn tg_last_error_message() -> *const c_char {
    LAST_ERROR.with(|cell| {
        cell.borrow()
            .as_ref()
            .map(|s| s.as_ptr())
            .unwrap_or(ptr::null())
    })
}

// ---------------------------------------------------------------------------
// Spec M: column-major BLAS-style mod-P counting tropical matmul.
// Caller-owned device buffers; flags select the (transA, transB) kernel.
// ---------------------------------------------------------------------------

fn run_tropical_matmul<T, D>(
    tA: i8, tB: i8,
    m: usize, k: usize, n: usize,
    a_dev: u64, b_dev: u64,
    p: i32,
    out_dev: u64,
) -> i32
where
    T: tropical_gemm::types::TropicalScalar
        + cudarc::driver::DeviceRepr
        + cudarc::driver::ValidAsZeroBits
        + crate::counting_kernel::TileDims
        + Default + Clone + Copy
        + 'static,
    D: tropical_gemm::types::TropicalDirection,
    (T, D): crate::counting_kernel::TropicalMatmulKernelName<T, D>,
{
    let tA_char = tA as u8 as char;
    let tB_char = tB as u8 as char;
    if tA_char != 'N' && tA_char != 'T' {
        store_error(format!("tA must be 'N' or 'T', got {:?}", tA_char));
        return ERR_INVALID_INPUT;
    }
    if tB_char != 'N' && tB_char != 'T' {
        store_error(format!("tB must be 'N' or 'T', got {:?}", tB_char));
        return ERR_INVALID_INPUT;
    }
    if p < 2 {
        store_error(format!("modulus must be >= 2, got {}", p));
        return ERR_INVALID_INPUT;
    }
    if m == 0 || k == 0 || n == 0 {
        store_error("dimensions must be non-zero");
        return ERR_INVALID_INPUT;
    }
    if a_dev == 0 || b_dev == 0 || out_dev == 0 {
        store_error("null device pointer");
        return ERR_INVALID_INPUT;
    }

    let ctx = match get_global_context() {
        Ok(c) => c,
        Err(e) => {
            store_error(format!("CUDA context init failed: {}", e));
            return ERR_CUDA;
        }
    };

    match tropical_matmul_kernel::<T, D>(
        ctx, tA_char, tB_char, m, k, n, a_dev, b_dev, p, out_dev,
    ) {
        Ok(()) => OK,
        Err(e) => { store_error(format!("{}", e)); ERR_CUDA }
    }
}

macro_rules! cabi_tropical_matmul {
    ($name:ident, $T:ty, $D:ty) => {
        #[no_mangle]
        pub extern "C" fn $name(
            tA: i8, tB: i8,
            m: usize, k: usize, n: usize,
            a_dev: u64, b_dev: u64,
            p: i32,
            out_dev: u64,
        ) -> c_int {
            let res = catch_unwind(AssertUnwindSafe(|| {
                run_tropical_matmul::<$T, $D>(tA, tB, m, k, n, a_dev, b_dev, p, out_dev)
            }));
            match res {
                Ok(code) => code,
                Err(_) => { store_error("Rust panic across FFI boundary"); ERR_INTERNAL }
            }
        }
    };
}

cabi_tropical_matmul!(tg_tropical_matmul_f32_max, f32, Max);
cabi_tropical_matmul!(tg_tropical_matmul_f32_min, f32, Min);
cabi_tropical_matmul!(tg_tropical_matmul_f64_max, f64, Max);
cabi_tropical_matmul!(tg_tropical_matmul_f64_min, f64, Min);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_one() {
        assert_eq!(tg_api_version(), 1);
    }

    #[test]
    fn tg_tropical_matmul_f32_max_nn_smoke() {
        use crate::pair::PairF32;
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let pair_a_host = vec![
            PairF32::new(1.0, 1), PairF32::new(3.0, 1),
            PairF32::new(2.0, 1), PairF32::new(4.0, 1),
        ];
        let pair_b_host = vec![
            PairF32::new(5.0, 1), PairF32::new(7.0, 1),
            PairF32::new(6.0, 1), PairF32::new(8.0, 1),
        ];
        let pair_a_dev = ctx.device().htod_copy(pair_a_host).unwrap();
        let pair_b_dev = ctx.device().htod_copy(pair_b_host).unwrap();
        let out_dev = ctx.device().alloc_zeros::<PairF32>(4).unwrap();

        use cudarc::driver::DevicePtr;
        let code = tg_tropical_matmul_f32_max(
            b'N' as i8, b'N' as i8,
            2, 2, 2,
            *pair_a_dev.device_ptr(),
            *pair_b_dev.device_ptr(),
            7,
            *out_dev.device_ptr(),
        );
        assert_eq!(code, OK);
        let out = ctx.device().dtoh_sync_copy(&out_dev).unwrap();
        assert_eq!(out[0].val, 9.0);
        assert_eq!(out[1].val, 11.0);
        assert_eq!(out[2].val, 10.0);
        assert_eq!(out[3].val, 12.0);
    }

    #[test]
    fn tg_tropical_matmul_rejects_bad_flag() {
        let code = tg_tropical_matmul_f32_max(
            b'X' as i8, b'N' as i8, 2, 2, 2, 0, 0, 7, 0,
        );
        assert_eq!(code, ERR_INVALID_INPUT);
    }
}
