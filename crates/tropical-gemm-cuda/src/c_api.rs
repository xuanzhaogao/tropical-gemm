//! C ABI for `count_ground_states_gpu_u64`.
//!
//! Exposes four `extern "C"` entry points (one per `(T, D)` combo) plus
//! `tg_last_error_message` for callers that want a human-readable error
//! message after a non-zero return. Used by `CountingTropicalGEMM.jl`
//! (Julia binding); other languages can `dlopen` the shared library and
//! call these directly.
//!
//! ## Error codes
//!
//! - `0` — success
//! - `1` — invalid input (null pointer, dimension mismatch)
//! - `2` — bound exceeds u64 envelope (≥ 3 primes needed); fall back to
//!   the BigInt entry point in the Rust crate
//! - `3` — CUDA error (kernel launch / memory / context)
//! - `4` — Rust panic or other internal error
//!
//! ## Inputs / outputs
//!
//! All matrices are row-major. Inputs are read-only; output buffers are
//! caller-allocated, length `m * n` each (values: `T*`, counts: `u64*`).
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

use crate::crt::count_ground_states_gpu_u64;
use crate::error::CudaError;
use crate::get_global_context;

/// Bumped on any source-incompatible C ABI change. The Julia binding (or
/// any other ABI consumer) is expected to assert this at load time.
pub const TG_API_VERSION: i32 = 1;

const OK: i32 = 0;
const ERR_INVALID_INPUT: i32 = 1;
const ERR_BOUND_TOO_LARGE: i32 = 2;
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

fn classify_cuda_error(e: &CudaError) -> i32 {
    let msg = format!("{}", e);
    // The bound-too-large case is reported as InvalidState with a specific
    // marker message in `count_ground_states_gpu_u64`. Detect it here so
    // callers can distinguish "use BigInt path" from generic CUDA failures.
    if msg.contains("u63-safe envelope") || msg.contains("u64 fast-path") {
        ERR_BOUND_TOO_LARGE
    } else {
        ERR_CUDA
    }
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
// Generic dispatch helper. The four extern fns below differ only in the
// concrete (T, D) types they hand to this body.
// ---------------------------------------------------------------------------

fn run_u64<T, D>(
    a: *const T,
    m: usize,
    k: usize,
    b: *const T,
    n: usize,
    bound: u64,
    out_values: *mut T,
    out_counts: *mut u64,
) -> i32
where
    T: tropical_gemm::types::TropicalScalar
        + cudarc::driver::DeviceRepr
        + cudarc::driver::ValidAsZeroBits
        + Default
        + Clone
        + Copy
        + crate::pair::PackPair
        + 'static,
    D: tropical_gemm::types::TropicalDirection,
    (T, D): crate::counting_kernel::CountingCudaKernel<T, D>,
{
    if a.is_null() || b.is_null() || out_values.is_null() || out_counts.is_null() {
        store_error("null pointer");
        return ERR_INVALID_INPUT;
    }
    if m == 0 || k == 0 || n == 0 {
        store_error("dimensions must be non-zero");
        return ERR_INVALID_INPUT;
    }

    // Build slices from the raw pointers. Caller's responsibility that
    // these buffers have the documented length and alignment.
    let a_slice = unsafe { std::slice::from_raw_parts(a, m * k) };
    let b_slice = unsafe { std::slice::from_raw_parts(b, k * n) };

    let ctx = match get_global_context() {
        Ok(c) => c,
        Err(e) => {
            store_error(format!("CUDA context init failed: {}", e));
            return ERR_CUDA;
        }
    };

    let result = count_ground_states_gpu_u64::<T, D>(ctx, a_slice, m, k, b_slice, n, bound);
    match result {
        Ok(mat) => {
            // Copy into caller-provided buffers. Lengths checked above.
            let v_out = unsafe { std::slice::from_raw_parts_mut(out_values, m * n) };
            let c_out = unsafe { std::slice::from_raw_parts_mut(out_counts, m * n) };
            v_out.copy_from_slice(&mat.values);
            c_out.copy_from_slice(&mat.counts);
            OK
        }
        Err(e) => {
            let code = classify_cuda_error(&e);
            store_error(format!("{}", e));
            code
        }
    }
}

// ---------------------------------------------------------------------------
// Four extern fns: Max/Min × f32/f64.
// ---------------------------------------------------------------------------

macro_rules! cabi_count_ground_states_u64 {
    ($name:ident, $T:ty, $D:ty) => {
        #[no_mangle]
        pub extern "C" fn $name(
            a: *const $T,
            m: usize,
            k: usize,
            b: *const $T,
            n: usize,
            bound: u64,
            out_values: *mut $T,
            out_counts: *mut u64,
        ) -> c_int {
            let res = catch_unwind(AssertUnwindSafe(|| {
                run_u64::<$T, $D>(a, m, k, b, n, bound, out_values, out_counts)
            }));
            match res {
                Ok(code) => code,
                Err(_) => {
                    store_error("Rust panic across FFI boundary");
                    ERR_INTERNAL
                }
            }
        }
    };
}

cabi_count_ground_states_u64!(tg_count_ground_states_gpu_u64_f32_max, f32, Max);
cabi_count_ground_states_u64!(tg_count_ground_states_gpu_u64_f32_min, f32, Min);
cabi_count_ground_states_u64!(tg_count_ground_states_gpu_u64_f64_max, f64, Max);
cabi_count_ground_states_u64!(tg_count_ground_states_gpu_u64_f64_min, f64, Min);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_one() {
        assert_eq!(tg_api_version(), 1);
    }

    #[test]
    fn null_input_returns_invalid_input() {
        let mut vals = vec![0.0_f32; 4];
        let mut cnts = vec![0_u64; 4];
        let code = tg_count_ground_states_gpu_u64_f32_max(
            ptr::null(), 2, 2, ptr::null(), 2, 1u64,
            vals.as_mut_ptr(), cnts.as_mut_ptr(),
        );
        assert_eq!(code, ERR_INVALID_INPUT);
        let msg_ptr = tg_last_error_message();
        assert!(!msg_ptr.is_null());
    }

    #[test]
    fn zero_dim_returns_invalid_input() {
        let a = vec![0.0_f32];
        let b = vec![0.0_f32];
        let mut vals = vec![0.0_f32; 1];
        let mut cnts = vec![0_u64; 1];
        let code = tg_count_ground_states_gpu_u64_f32_max(
            a.as_ptr(), 0, 1, b.as_ptr(), 1, 1u64,
            vals.as_mut_ptr(), cnts.as_mut_ptr(),
        );
        assert_eq!(code, ERR_INVALID_INPUT);
    }
}
