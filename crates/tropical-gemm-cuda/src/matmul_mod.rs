//! Single-prime mod-P counting tropical matmul (Spec K).
//!
//! Wraps the AoS general counting kernel `launch_counting_gemm` for callers
//! who want raw per-prime residues — no CRT, no BigInt. Intended for the
//! Julia GEMM API operating on `Matrix{CountingTropical{T, Mod{P}}}`.

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

use tropical_gemm::types::TropicalDirection;

use crate::context::CudaContext;
use crate::counting_kernel::{
    launch_counting_gemm, launch_counting_gemm_dev_ptr, CountingCudaKernel, DevPtr,
};
use crate::error::{CudaError, Result};
use crate::memory::GpuMatrix;
use crate::pair::PackPair;

/// Minimum and maximum allowed modulus. The kernel takes `i32` modulus, so
/// `p` must fit in positive `i32`. `p == 1` collapses every count to zero
/// (degenerate); `p == 0` is invalid.
const P_MIN: i32 = 2;

/// Shared device-side core of `matmul_mod_p` and `matmul_mod_p_pair`.
/// Caller has already validated shapes and produced row-major
/// `pair_a` (M × K) and `pair_b` (K × N). Uploads, launches the kernel,
/// and copies SoA outputs back into the caller-provided slices.
fn run_packed<T, D>(
    ctx: &CudaContext,
    pair_a: &[<T as PackPair>::Pair],
    m: usize,
    k: usize,
    pair_b: &[<T as PackPair>::Pair],
    n: usize,
    p: i32,
    out_val: &mut [T],
    out_cnt: &mut [i32],
) -> Result<()>
where
    T: tropical_gemm::types::TropicalScalar
        + DeviceRepr
        + ValidAsZeroBits
        + Default
        + Clone
        + Copy
        + PackPair
        + 'static,
    D: TropicalDirection,
    (T, D): CountingCudaKernel<T, D>,
{
    let pair_a_dev = GpuMatrix::<<T as PackPair>::Pair>::from_host(ctx, pair_a, m, k)?;
    let pair_b_dev = GpuMatrix::<<T as PackPair>::Pair>::from_host(ctx, pair_b, k, n)?;

    let mut value_c = GpuMatrix::<T>::alloc(ctx, m, n)?;
    let mut count_c = GpuMatrix::<i32>::alloc(ctx, m, n)?;

    launch_counting_gemm::<T, D>(
        ctx, &pair_a_dev, &pair_b_dev,
        &mut value_c, &mut count_c, p,
    )?;

    let host_val = value_c.to_host(ctx)?;
    let host_cnt = count_c.to_host(ctx)?;
    out_val.copy_from_slice(&host_val);
    out_cnt.copy_from_slice(&host_cnt);
    Ok(())
}

/// Slow path: caller provides separate value and count arrays. Pack happens
/// host-side before upload. Used when the caller's count storage is not
/// `i32` (e.g. Julia's `Mod{P, Int}` defaults to Int64 on 64-bit hosts).
pub fn matmul_mod_p<T, D>(
    ctx: &CudaContext,
    a_val: &[T],
    a_cnt: &[i32],
    m: usize,
    k: usize,
    b_val: &[T],
    b_cnt: &[i32],
    n: usize,
    p: i32,
    out_val: &mut [T],
    out_cnt: &mut [i32],
) -> Result<()>
where
    T: tropical_gemm::types::TropicalScalar
        + DeviceRepr
        + ValidAsZeroBits
        + Default
        + Clone
        + Copy
        + PackPair
        + 'static,
    D: TropicalDirection,
    (T, D): CountingCudaKernel<T, D>,
{
    // Validate P range.
    if p < P_MIN {
        return Err(CudaError::InvalidState(format!(
            "modulus must satisfy {} <= p < 2^31, got {}",
            P_MIN, p
        )));
    }

    // Validate buffer lengths against shape.
    if a_val.len() != m * k
        || a_cnt.len() != m * k
        || b_val.len() != k * n
        || b_cnt.len() != k * n
        || out_val.len() != m * n
        || out_cnt.len() != m * n
    {
        return Err(CudaError::InvalidState(format!(
            "buffer length mismatch: m={}, k={}, n={}, but \
             a_val={} a_cnt={} b_val={} b_cnt={} out_val={} out_cnt={}",
            m, k, n,
            a_val.len(), a_cnt.len(), b_val.len(), b_cnt.len(),
            out_val.len(), out_cnt.len()
        )));
    }
    if m == 0 || k == 0 || n == 0 {
        return Err(CudaError::InvalidState(
            "dimensions must be non-zero".into(),
        ));
    }

    // Host-side pack: zip (val, cnt) into Pair.
    let pair_a_host: Vec<<T as PackPair>::Pair> = T::pack_pair(a_val, a_cnt);
    let pair_b_host: Vec<<T as PackPair>::Pair> = T::pack_pair(b_val, b_cnt);

    run_packed::<T, D>(ctx, &pair_a_host, m, k, &pair_b_host, n, p, out_val, out_cnt)
}

/// Fast path: caller has already packed into the device-compatible
/// `PairT` layout. Used by Julia callers whose host-side
/// `Matrix{CountingTropical{T, Mod{P, Int32}}}` is byte-compatible with
/// `PairT` and can be reinterpreted with no per-element split.
///
/// `pair_a` is M × K row-major; `pair_b` is K × N row-major.
pub fn matmul_mod_p_pair<T, D>(
    ctx: &CudaContext,
    pair_a: &[<T as PackPair>::Pair],
    m: usize,
    k: usize,
    pair_b: &[<T as PackPair>::Pair],
    n: usize,
    p: i32,
    out_val: &mut [T],
    out_cnt: &mut [i32],
) -> Result<()>
where
    T: tropical_gemm::types::TropicalScalar
        + DeviceRepr
        + ValidAsZeroBits
        + Default
        + Clone
        + Copy
        + PackPair
        + 'static,
    D: TropicalDirection,
    (T, D): CountingCudaKernel<T, D>,
{
    if p < P_MIN {
        return Err(CudaError::InvalidState(format!(
            "modulus must satisfy {} <= p < 2^31, got {}",
            P_MIN, p
        )));
    }
    if pair_a.len() != m * k
        || pair_b.len() != k * n
        || out_val.len() != m * n
        || out_cnt.len() != m * n
    {
        return Err(CudaError::InvalidState(format!(
            "buffer length mismatch: m={}, k={}, n={}, but \
             pair_a={} pair_b={} out_val={} out_cnt={}",
            m, k, n,
            pair_a.len(), pair_b.len(), out_val.len(), out_cnt.len()
        )));
    }
    if m == 0 || k == 0 || n == 0 {
        return Err(CudaError::InvalidState(
            "dimensions must be non-zero".into(),
        ));
    }

    run_packed::<T, D>(ctx, pair_a, m, k, pair_b, n, p, out_val, out_cnt)
}

/// Spec L: kernel-only entry point. Caller already owns device buffers
/// (e.g. CUDA.jl uploaded the pair inputs and pre-allocated the SoA outputs)
/// and wants Rust to do nothing but launch the kernel.
///
/// `pair_a_dev_ptr` and `pair_b_dev_ptr` are raw `CUdeviceptr`s for the
/// row-major M×K and K×N pair buffers. `out_val_dev_ptr` and
/// `out_cnt_dev_ptr` are raw `CUdeviceptr`s for the M×N value and count
/// SoA outputs (T and i32 respectively). All four point into the primary
/// CUDA context, which both this crate (via cudarc `CudaContext::new`) and
/// CUDA.jl share.
///
/// Validates `2 <= p < 2^31` and `m, k, n > 0`. **Cannot** check buffer
/// length — caller is responsible for asserting that each pointer
/// references at least the documented number of elements.
///
/// Synchronizes the device before the kernel launch (defensive against
/// uploads in flight on the caller's stream); see
/// `CountingCudaKernel::launch_counting_gemm_dev_ptr` for details.
pub fn matmul_mod_p_kernel_only<T, D>(
    ctx: &CudaContext,
    pair_a_dev_ptr: CUdeviceptr,
    m: usize,
    k: usize,
    pair_b_dev_ptr: CUdeviceptr,
    n: usize,
    p: i32,
    out_val_dev_ptr: CUdeviceptr,
    out_cnt_dev_ptr: CUdeviceptr,
) -> Result<()>
where
    T: tropical_gemm::types::TropicalScalar
        + DeviceRepr
        + ValidAsZeroBits
        + Default
        + Clone
        + Copy
        + PackPair
        + 'static,
    D: TropicalDirection,
    (T, D): CountingCudaKernel<T, D>,
{
    if p < P_MIN {
        return Err(CudaError::InvalidState(format!(
            "modulus must satisfy {} <= p < 2^31, got {}",
            P_MIN, p
        )));
    }
    if m == 0 || k == 0 || n == 0 {
        return Err(CudaError::InvalidState(
            "dimensions must be non-zero".into(),
        ));
    }

    launch_counting_gemm_dev_ptr::<T, D>(
        ctx,
        DevPtr(pair_a_dev_ptr),
        m,
        k,
        DevPtr(pair_b_dev_ptr),
        n,
        DevPtr(out_val_dev_ptr),
        DevPtr(out_cnt_dev_ptr),
        p,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tropical_gemm::types::Max;

    #[test]
    fn matmul_mod_p_2x2_max_p7() {
        // C[i,j] = max_k (A[i,k] + B[k,j]); count = number of k attaining max,
        // multiplied by input counts pairwise, summed mod P.
        // A = [[1,2],[3,4]], B = [[5,6],[7,8]]; all input counts = 1.
        // C[0,0] = max(1+5, 2+7) = 9 (k=1), count = 1.
        // C[0,1] = max(1+6, 2+8) = 10 (k=1), count = 1.
        // C[1,0] = max(3+5, 4+7) = 11 (k=1), count = 1.
        // C[1,1] = max(3+6, 4+8) = 12 (k=1), count = 1.
        // mod P=7 leaves counts unchanged (all = 1).
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let a_val = vec![1.0_f32, 2.0, 3.0, 4.0];
        let a_cnt = vec![1_i32, 1, 1, 1];
        let b_val = vec![5.0_f32, 6.0, 7.0, 8.0];
        let b_cnt = vec![1_i32, 1, 1, 1];
        let mut out_val = vec![0.0_f32; 4];
        let mut out_cnt = vec![0_i32; 4];

        matmul_mod_p::<f32, Max>(
            ctx, &a_val, &a_cnt, 2, 2, &b_val, &b_cnt, 2, 7,
            &mut out_val, &mut out_cnt,
        ).expect("driver ok");

        assert_eq!(out_val, vec![9.0_f32, 10.0, 11.0, 12.0]);
        assert_eq!(out_cnt, vec![1_i32, 1, 1, 1]);
    }

    #[test]
    fn matmul_mod_p_observable_reduction() {
        // Construct an all-tie matmul where each output cell sums K input
        // count products. Pick K=5, all values = 0.0, all input counts = 2,
        // P = 3. Each cell: sum of 5 ties, each 2*2 = 4. Total = 20.
        // 20 mod 3 = 2. So output count should be 2 everywhere.
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let m = 3usize; let k = 5usize; let n = 4usize;
        let a_val = vec![0.0_f32; m * k];
        let a_cnt = vec![2_i32; m * k];
        let b_val = vec![0.0_f32; k * n];
        let b_cnt = vec![2_i32; k * n];
        let mut out_val = vec![1.0_f32; m * n];  // sentinel
        let mut out_cnt = vec![99_i32; m * n];   // sentinel

        matmul_mod_p::<f32, Max>(
            ctx, &a_val, &a_cnt, m, k, &b_val, &b_cnt, n, 3,
            &mut out_val, &mut out_cnt,
        ).expect("driver ok");

        assert!(out_val.iter().all(|&v| v == 0.0_f32));
        // 20 mod 3 = 2 — verifies the modulus actually threads through.
        assert!(out_cnt.iter().all(|&c| c == 2_i32),
            "expected all counts = 2, got {:?}", out_cnt);
    }

    #[test]
    fn matmul_mod_p_rejects_p_one() {
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let a_val = vec![1.0_f32; 4]; let a_cnt = vec![1_i32; 4];
        let b_val = vec![1.0_f32; 4]; let b_cnt = vec![1_i32; 4];
        let mut out_val = vec![0.0_f32; 4];
        let mut out_cnt = vec![0_i32; 4];

        let err = matmul_mod_p::<f32, Max>(
            ctx, &a_val, &a_cnt, 2, 2, &b_val, &b_cnt, 2, 1,
            &mut out_val, &mut out_cnt,
        ).expect_err("p=1 must be rejected");
        let msg = format!("{}", err);
        assert!(msg.contains("modulus") || msg.contains("p"),
            "expected modulus error, got: {}", msg);
    }

    #[test]
    fn matmul_mod_p_pair_2x2_max_p7() {
        // Same setup as the slow-path test, but caller pre-packs into PairF32.
        use crate::pair::PairF32;
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let pair_a = vec![
            PairF32::new(1.0, 1), PairF32::new(2.0, 1),
            PairF32::new(3.0, 1), PairF32::new(4.0, 1),
        ];
        let pair_b = vec![
            PairF32::new(5.0, 1), PairF32::new(6.0, 1),
            PairF32::new(7.0, 1), PairF32::new(8.0, 1),
        ];
        let mut out_val = vec![0.0_f32; 4];
        let mut out_cnt = vec![0_i32; 4];

        matmul_mod_p_pair::<f32, Max>(
            ctx, &pair_a, 2, 2, &pair_b, 2, 7,
            &mut out_val, &mut out_cnt,
        ).expect("driver ok");

        assert_eq!(out_val, vec![9.0_f32, 10.0, 11.0, 12.0]);
        assert_eq!(out_cnt, vec![1_i32, 1, 1, 1]);
    }

    #[test]
    fn matmul_mod_p_4x4_min_random_p11() {
        use tropical_gemm::types::Min;
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let m = 4usize; let k = 6usize; let n = 4usize;
        // Discrete inputs to force ties.
        let a_val: Vec<f64> = (0..m*k).map(|i| (i % 5) as f64).collect();
        let a_cnt: Vec<i32> = (0..m*k).map(|i| ((i + 1) % 11) as i32).collect();
        let b_val: Vec<f64> = (0..k*n).map(|i| (i % 4) as f64).collect();
        let b_cnt: Vec<i32> = (0..k*n).map(|i| ((i + 2) % 11) as i32).collect();

        let mut out_val = vec![0.0_f64; m*n];
        let mut out_cnt = vec![0_i32; m*n];
        matmul_mod_p::<f64, Min>(
            ctx, &a_val, &a_cnt, m, k, &b_val, &b_cnt, n, 11,
            &mut out_val, &mut out_cnt,
        ).expect("driver ok");

        // Reference.
        for i in 0..m {
            for j in 0..n {
                let mut best = f64::INFINITY;
                let mut acc: i64 = 0;
                for kk in 0..k {
                    let v = a_val[i*k + kk] + b_val[kk*n + j];
                    if v < best {
                        best = v;
                        acc = (a_cnt[i*k + kk] as i64) * (b_cnt[kk*n + j] as i64) % 11;
                    } else if v == best {
                        acc = (acc + (a_cnt[i*k + kk] as i64) * (b_cnt[kk*n + j] as i64)) % 11;
                    }
                }
                assert_eq!(out_val[i*n + j], best, "value mismatch at ({},{})", i, j);
                assert_eq!(out_cnt[i*n + j] as i64, acc, "count mismatch at ({},{})", i, j);
            }
        }
    }

    #[test]
    fn matmul_mod_p_kernel_only_round_trip_max_p7() {
        // Spec L: caller-owned device pointers (simulating CUDA.jl). Uploads
        // via cudarc, extracts raw CUdeviceptr, runs kernel-only path.
        use cudarc::driver::DevicePtr;
        use crate::pair::PairF32;
        let ctx = crate::get_global_context().expect("CUDA ctx");

        let pair_a_host = vec![
            PairF32::new(1.0, 1), PairF32::new(2.0, 1),
            PairF32::new(3.0, 1), PairF32::new(4.0, 1),
        ];
        let pair_b_host = vec![
            PairF32::new(5.0, 1), PairF32::new(6.0, 1),
            PairF32::new(7.0, 1), PairF32::new(8.0, 1),
        ];

        let pair_a_dev = ctx.device().htod_copy(pair_a_host.clone()).expect("upload A");
        let pair_b_dev = ctx.device().htod_copy(pair_b_host.clone()).expect("upload B");
        let out_val_dev = ctx.device().alloc_zeros::<f32>(4).expect("alloc out_val");
        let out_cnt_dev = ctx.device().alloc_zeros::<i32>(4).expect("alloc out_cnt");

        let a_ptr = *pair_a_dev.device_ptr();
        let b_ptr = *pair_b_dev.device_ptr();
        let out_v_ptr = *out_val_dev.device_ptr();
        let out_c_ptr = *out_cnt_dev.device_ptr();

        matmul_mod_p_kernel_only::<f32, Max>(
            ctx, a_ptr, 2, 2, b_ptr, 2, 7, out_v_ptr, out_c_ptr,
        ).expect("kernel-only ok");

        let out_val = ctx.device().dtoh_sync_copy(&out_val_dev).expect("download v");
        let out_cnt = ctx.device().dtoh_sync_copy(&out_cnt_dev).expect("download c");
        assert_eq!(out_val, vec![9.0_f32, 10.0, 11.0, 12.0]);
        assert_eq!(out_cnt, vec![1_i32, 1, 1, 1]);
    }

    #[test]
    fn matmul_mod_p_kernel_only_round_trip_min_f64_p11() {
        use cudarc::driver::DevicePtr;
        use crate::pair::PairF64;
        use tropical_gemm::types::Min;
        let ctx = crate::get_global_context().expect("CUDA ctx");

        let m = 4usize; let k = 6usize; let n = 4usize;
        let a_val: Vec<f64> = (0..m*k).map(|i| (i % 5) as f64).collect();
        let a_cnt: Vec<i32> = (0..m*k).map(|i| ((i + 1) % 11) as i32).collect();
        let b_val: Vec<f64> = (0..k*n).map(|i| (i % 4) as f64).collect();
        let b_cnt: Vec<i32> = (0..k*n).map(|i| ((i + 2) % 11) as i32).collect();

        let pair_a_host: Vec<PairF64> = a_val.iter().zip(a_cnt.iter())
            .map(|(&v, &c)| PairF64::new(v, c)).collect();
        let pair_b_host: Vec<PairF64> = b_val.iter().zip(b_cnt.iter())
            .map(|(&v, &c)| PairF64::new(v, c)).collect();

        let pair_a_dev = ctx.device().htod_copy(pair_a_host).expect("upload A");
        let pair_b_dev = ctx.device().htod_copy(pair_b_host).expect("upload B");
        let out_val_dev = ctx.device().alloc_zeros::<f64>(m * n).expect("alloc out_val");
        let out_cnt_dev = ctx.device().alloc_zeros::<i32>(m * n).expect("alloc out_cnt");

        matmul_mod_p_kernel_only::<f64, Min>(
            ctx,
            *pair_a_dev.device_ptr(), m, k,
            *pair_b_dev.device_ptr(), n,
            11,
            *out_val_dev.device_ptr(),
            *out_cnt_dev.device_ptr(),
        ).expect("kernel-only ok");

        let out_val = ctx.device().dtoh_sync_copy(&out_val_dev).expect("download v");
        let out_cnt = ctx.device().dtoh_sync_copy(&out_cnt_dev).expect("download c");

        // Reference (same as matmul_mod_p_4x4_min_random_p11).
        for i in 0..m {
            for j in 0..n {
                let mut best = f64::INFINITY;
                let mut acc: i64 = 0;
                for kk in 0..k {
                    let v = a_val[i*k + kk] + b_val[kk*n + j];
                    if v < best {
                        best = v;
                        acc = (a_cnt[i*k + kk] as i64) * (b_cnt[kk*n + j] as i64) % 11;
                    } else if v == best {
                        acc = (acc + (a_cnt[i*k + kk] as i64) * (b_cnt[kk*n + j] as i64)) % 11;
                    }
                }
                assert_eq!(out_val[i*n + j], best, "value mismatch at ({},{})", i, j);
                assert_eq!(out_cnt[i*n + j] as i64, acc, "count mismatch at ({},{})", i, j);
            }
        }
    }

    #[test]
    fn matmul_mod_p_kernel_only_rejects_p_one() {
        let ctx = crate::get_global_context().expect("CUDA ctx");
        // Use bogus pointers — validation must happen before we touch them.
        let err = matmul_mod_p_kernel_only::<f32, Max>(
            ctx, 0, 2, 2, 0, 2, 1, 0, 0,
        ).expect_err("p=1 must be rejected");
        let msg = format!("{}", err);
        assert!(msg.contains("modulus") || msg.contains("p"),
            "expected modulus error, got: {}", msg);
    }

    #[test]
    fn matmul_mod_p_pair_4x4_min_random_p11() {
        use tropical_gemm::types::Min;
        use crate::pair::PairF64;
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let m = 4usize; let k = 6usize; let n = 4usize;

        // Same input pattern as matmul_mod_p_4x4_min_random_p11 — pack into PairF64
        // up front and route through the fast path. Result must match.
        let a_val: Vec<f64> = (0..m*k).map(|i| (i % 5) as f64).collect();
        let a_cnt: Vec<i32> = (0..m*k).map(|i| ((i + 1) % 11) as i32).collect();
        let b_val: Vec<f64> = (0..k*n).map(|i| (i % 4) as f64).collect();
        let b_cnt: Vec<i32> = (0..k*n).map(|i| ((i + 2) % 11) as i32).collect();

        let pair_a: Vec<PairF64> = a_val.iter().zip(a_cnt.iter())
            .map(|(&v, &c)| PairF64::new(v, c)).collect();
        let pair_b: Vec<PairF64> = b_val.iter().zip(b_cnt.iter())
            .map(|(&v, &c)| PairF64::new(v, c)).collect();

        let mut out_val = vec![0.0_f64; m*n];
        let mut out_cnt = vec![0_i32; m*n];
        matmul_mod_p_pair::<f64, Min>(
            ctx, &pair_a, m, k, &pair_b, n, 11,
            &mut out_val, &mut out_cnt,
        ).expect("fast-path driver ok");

        // Reference (same as in the slow-path test).
        for i in 0..m {
            for j in 0..n {
                let mut best = f64::INFINITY;
                let mut acc: i64 = 0;
                for kk in 0..k {
                    let v = a_val[i*k + kk] + b_val[kk*n + j];
                    if v < best {
                        best = v;
                        acc = (a_cnt[i*k + kk] as i64) * (b_cnt[kk*n + j] as i64) % 11;
                    } else if v == best {
                        acc = (acc + (a_cnt[i*k + kk] as i64) * (b_cnt[kk*n + j] as i64)) % 11;
                    }
                }
                assert_eq!(out_val[i*n + j], best, "value mismatch at ({},{})", i, j);
                assert_eq!(out_cnt[i*n + j] as i64, acc, "count mismatch at ({},{})", i, j);
            }
        }
    }
}
