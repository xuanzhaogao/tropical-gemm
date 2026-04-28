//! Single-prime mod-P counting tropical matmul (Spec K).
//!
//! Wraps the AoS general counting kernel `launch_counting_gemm` for callers
//! who want raw per-prime residues — no CRT, no BigInt. Intended for the
//! Julia GEMM API operating on `Matrix{CountingTropical{T, Mod{P}}}`.

use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

use tropical_gemm::types::TropicalDirection;

use crate::context::CudaContext;
use crate::counting_kernel::{launch_counting_gemm, CountingCudaKernel};
use crate::error::{CudaError, Result};
use crate::memory::GpuMatrix;
use crate::pair::PackPair;

/// Minimum and maximum allowed modulus. The kernel takes `i32` modulus, so
/// `p` must fit in positive `i32`. `p == 1` collapses every count to zero
/// (degenerate); `p == 0` is invalid.
const P_MIN: i32 = 2;

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
    if p < P_MIN || p < 0 {
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

    // Upload.
    let pair_a_dev = GpuMatrix::<<T as PackPair>::Pair>::from_host(ctx, &pair_a_host, m, k)?;
    let pair_b_dev = GpuMatrix::<<T as PackPair>::Pair>::from_host(ctx, &pair_b_host, k, n)?;

    let mut value_c = GpuMatrix::<T>::alloc(ctx, m, n)?;
    let mut count_c = GpuMatrix::<i32>::alloc(ctx, m, n)?;

    // Launch general AoS kernel with the user's prime as modulus.
    launch_counting_gemm::<T, D>(
        ctx,
        &pair_a_dev,
        &pair_b_dev,
        &mut value_c,
        &mut count_c,
        p,
    )?;

    // Download.
    let host_val = value_c.to_host(ctx)?;
    let host_cnt = count_c.to_host(ctx)?;
    out_val.copy_from_slice(&host_val);
    out_cnt.copy_from_slice(&host_cnt);

    Ok(())
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
}
