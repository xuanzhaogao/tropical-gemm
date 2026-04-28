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
use std::time::Instant;

use tropical_gemm::crt::{choose_primes_u64, CRT_PRIMES};
use tropical_gemm::types::{Max, Min};

use crate::context::{COUNTING_WARPK_K_THRESHOLD, COUNTING_WARPK_MN_CEILING};
use crate::counting_kernel::{launch_counting_gemm_ones, CountingCudaKernel};
use crate::crt::count_ground_states_gpu_u64;
use crate::error::CudaError;
use crate::get_global_context;
use crate::memory::GpuMatrix;
use crate::pair::PackPair;
use crate::matmul_mod::matmul_mod_p;

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

// ---------------------------------------------------------------------------
// Spec K: single-prime mod-P matmul. Slow-path C ABI (split val/cnt buffers).
// ---------------------------------------------------------------------------

fn run_matmul_mod_p<T, D>(
    a_val: *const T, a_cnt: *const i32,
    m: usize, k: usize,
    b_val: *const T, b_cnt: *const i32,
    n: usize,
    p: i32,
    out_val: *mut T, out_cnt: *mut i32,
) -> i32
where
    T: tropical_gemm::types::TropicalScalar
        + cudarc::driver::DeviceRepr
        + cudarc::driver::ValidAsZeroBits
        + Default + Clone + Copy
        + crate::pair::PackPair
        + 'static,
    D: tropical_gemm::types::TropicalDirection,
    (T, D): crate::counting_kernel::CountingCudaKernel<T, D>,
{
    if a_val.is_null() || a_cnt.is_null()
        || b_val.is_null() || b_cnt.is_null()
        || out_val.is_null() || out_cnt.is_null()
    {
        store_error("null pointer");
        return ERR_INVALID_INPUT;
    }
    if m == 0 || k == 0 || n == 0 {
        store_error("dimensions must be non-zero");
        return ERR_INVALID_INPUT;
    }
    if p < 2 {
        store_error(format!("modulus must be >= 2, got {}", p));
        return ERR_INVALID_INPUT;
    }

    let a_val_s = unsafe { std::slice::from_raw_parts(a_val, m * k) };
    let a_cnt_s = unsafe { std::slice::from_raw_parts(a_cnt, m * k) };
    let b_val_s = unsafe { std::slice::from_raw_parts(b_val, k * n) };
    let b_cnt_s = unsafe { std::slice::from_raw_parts(b_cnt, k * n) };
    let out_val_s = unsafe { std::slice::from_raw_parts_mut(out_val, m * n) };
    let out_cnt_s = unsafe { std::slice::from_raw_parts_mut(out_cnt, m * n) };

    let ctx = match get_global_context() {
        Ok(c) => c,
        Err(e) => {
            store_error(format!("CUDA context init failed: {}", e));
            return ERR_CUDA;
        }
    };

    match matmul_mod_p::<T, D>(
        ctx, a_val_s, a_cnt_s, m, k, b_val_s, b_cnt_s, n, p,
        out_val_s, out_cnt_s,
    ) {
        Ok(()) => OK,
        Err(e) => { store_error(format!("{}", e)); ERR_CUDA }
    }
}

macro_rules! cabi_matmul_mod_p {
    ($name:ident, $T:ty, $D:ty) => {
        #[no_mangle]
        pub extern "C" fn $name(
            a_val: *const $T, a_cnt: *const i32,
            m: usize, k: usize,
            b_val: *const $T, b_cnt: *const i32,
            n: usize,
            p: i32,
            out_val: *mut $T, out_cnt: *mut i32,
        ) -> c_int {
            let res = catch_unwind(AssertUnwindSafe(|| {
                run_matmul_mod_p::<$T, $D>(
                    a_val, a_cnt, m, k, b_val, b_cnt, n, p, out_val, out_cnt,
                )
            }));
            match res {
                Ok(code) => code,
                Err(_) => { store_error("Rust panic across FFI boundary"); ERR_INTERNAL }
            }
        }
    };
}

cabi_matmul_mod_p!(tg_matmul_mod_p_f32_max, f32, Max);
cabi_matmul_mod_p!(tg_matmul_mod_p_f32_min, f32, Min);
cabi_matmul_mod_p!(tg_matmul_mod_p_f64_max, f64, Max);
cabi_matmul_mod_p!(tg_matmul_mod_p_f64_min, f64, Min);

// ---------------------------------------------------------------------------
// Kernel-only timing: upload data once, run the kernel `iters` times, and
// return the average per-launch wall time in milliseconds. Bypasses CRT
// combine and BigInt/u64 reconstruction entirely. For perf measurement
// from non-Rust callers (Julia bench, etc.).
// ---------------------------------------------------------------------------

fn bench_kernel_only_impl<T, D>(
    a: *const T,
    m: usize,
    k: usize,
    b: *const T,
    n: usize,
    bound: u64,
    iters: u32,
    out_avg_ms: *mut f64,
) -> i32
where
    T: tropical_gemm::types::TropicalScalar
        + cudarc::driver::DeviceRepr
        + cudarc::driver::ValidAsZeroBits
        + Default
        + Clone
        + Copy
        + PackPair
        + 'static,
    D: tropical_gemm::types::TropicalDirection,
    (T, D): CountingCudaKernel<T, D>,
{
    if a.is_null() || b.is_null() || out_avg_ms.is_null() {
        store_error("null pointer");
        return ERR_INVALID_INPUT;
    }
    if m == 0 || k == 0 || n == 0 || iters == 0 {
        store_error("dimensions and iters must be non-zero");
        return ERR_INVALID_INPUT;
    }

    let ctx = match get_global_context() {
        Ok(c) => c,
        Err(e) => {
            store_error(format!("CUDA context init failed: {}", e));
            return ERR_CUDA;
        }
    };

    // Pick any single prime — kernel work is the same per prime; we just
    // need a valid P. Prefer one chosen by the same heuristic the production
    // path uses, so timing reflects what users actually see.
    let needed = match bound.checked_mul(2).and_then(|x| x.checked_add(1)) {
        Some(v) => v,
        None => {
            store_error("count_upper_bound too large for u64 fast-path");
            return ERR_BOUND_TOO_LARGE;
        }
    };
    let prime_indices = match choose_primes_u64(needed) {
        Some((idx, _)) => idx,
        None => {
            store_error("count_upper_bound exceeds u63 envelope");
            return ERR_BOUND_TOO_LARGE;
        }
    };
    let p = CRT_PRIMES[prime_indices[0]];

    let a_slice = unsafe { std::slice::from_raw_parts(a, m * k) };
    let b_slice = unsafe { std::slice::from_raw_parts(b, k * n) };

    // Same layout choice as the production driver (Spec H).
    let use_warpk = k >= COUNTING_WARPK_K_THRESHOLD
        && m.saturating_mul(n) <= COUNTING_WARPK_MN_CEILING;

    let result: Result<f64, CudaError> = (|| {
        let value_a_dev = GpuMatrix::<T>::from_host(ctx, a_slice, m, k)?;
        let value_b_dev = if use_warpk {
            // Host transpose K×N → N×K row-major.
            let mut b_t: Vec<T> = vec![T::default(); n * k];
            for kv in 0..k {
                for j in 0..n {
                    b_t[j * k + kv] = b_slice[kv * n + j];
                }
            }
            GpuMatrix::<T>::from_host(ctx, &b_t, n, k)?
        } else {
            GpuMatrix::<T>::from_host(ctx, b_slice, k, n)?
        };

        let mut value_c = GpuMatrix::<T>::alloc(ctx, m, n)?;
        let mut count_c = GpuMatrix::<i32>::alloc(ctx, m, n)?;

        // Warmup launch (also pays NVRTC cost on first call).
        launch_counting_gemm_ones::<T, D>(
            ctx,
            &value_a_dev,
            &value_b_dev,
            &mut value_c,
            &mut count_c,
            m, k, n,
            p,
        )?;

        // Timed loop. launch_counting_gemm_ones synchronizes after each
        // launch, so wall time wraps the kernel + sync per iter.
        let t0 = Instant::now();
        for _ in 0..iters {
            launch_counting_gemm_ones::<T, D>(
                ctx,
                &value_a_dev,
                &value_b_dev,
                &mut value_c,
                &mut count_c,
                m, k, n,
                p,
            )?;
        }
        let avg_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        Ok(avg_ms)
    })();

    match result {
        Ok(avg_ms) => {
            unsafe { *out_avg_ms = avg_ms };
            OK
        }
        Err(e) => {
            store_error(format!("{}", e));
            ERR_CUDA
        }
    }
}

macro_rules! cabi_bench_kernel_only {
    ($name:ident, $T:ty, $D:ty) => {
        #[no_mangle]
        pub extern "C" fn $name(
            a: *const $T,
            m: usize,
            k: usize,
            b: *const $T,
            n: usize,
            bound: u64,
            iters: u32,
            out_avg_ms: *mut f64,
        ) -> c_int {
            let res = catch_unwind(AssertUnwindSafe(|| {
                bench_kernel_only_impl::<$T, $D>(a, m, k, b, n, bound, iters, out_avg_ms)
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

cabi_bench_kernel_only!(tg_bench_kernel_only_u64_f32_max, f32, Max);
cabi_bench_kernel_only!(tg_bench_kernel_only_u64_f32_min, f32, Min);
cabi_bench_kernel_only!(tg_bench_kernel_only_u64_f64_max, f64, Max);
cabi_bench_kernel_only!(tg_bench_kernel_only_u64_f64_min, f64, Min);

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

    #[test]
    fn matmul_mod_p_f32_max_smoke() {
        let a_val = vec![1.0_f32, 2.0, 3.0, 4.0];
        let a_cnt = vec![1_i32, 1, 1, 1];
        let b_val = vec![5.0_f32, 6.0, 7.0, 8.0];
        let b_cnt = vec![1_i32, 1, 1, 1];
        let mut out_val = vec![0.0_f32; 4];
        let mut out_cnt = vec![0_i32; 4];

        let code = tg_matmul_mod_p_f32_max(
            a_val.as_ptr(), a_cnt.as_ptr(), 2, 2,
            b_val.as_ptr(), b_cnt.as_ptr(), 2,
            7,
            out_val.as_mut_ptr(), out_cnt.as_mut_ptr(),
        );
        assert_eq!(code, OK);
        assert_eq!(out_val, vec![9.0_f32, 10.0, 11.0, 12.0]);
        assert_eq!(out_cnt, vec![1_i32, 1, 1, 1]);
    }

    #[test]
    fn matmul_mod_p_invalid_p_returns_invalid() {
        let a_val = vec![1.0_f32; 4]; let a_cnt = vec![1_i32; 4];
        let b_val = vec![1.0_f32; 4]; let b_cnt = vec![1_i32; 4];
        let mut out_val = vec![0.0_f32; 4];
        let mut out_cnt = vec![0_i32; 4];
        let code = tg_matmul_mod_p_f32_max(
            a_val.as_ptr(), a_cnt.as_ptr(), 2, 2,
            b_val.as_ptr(), b_cnt.as_ptr(), 2,
            1,    // p=1 invalid
            out_val.as_mut_ptr(), out_cnt.as_mut_ptr(),
        );
        assert_eq!(code, ERR_INVALID_INPUT);
    }
}
