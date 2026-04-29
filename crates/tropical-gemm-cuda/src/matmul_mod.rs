//! Spec M: column-major counting tropical matmul driver.
//!
//! `tropical_matmul_kernel` validates inputs and launches the right
//! `tropical_matmul_<T>_<dir>_<NN|NT|TN|TT>` CUDA kernel for caller-owned
//! device buffers (e.g. CUDA.jl on the Julia side).

use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

use tropical_gemm::types::TropicalDirection;

use crate::context::CudaContext;
use crate::error::{CudaError, Result};

/// Minimum allowed modulus. The kernel takes `i32` modulus, so `p` must fit
/// in positive `i32`. `p == 1` collapses every count to zero (degenerate);
/// `p == 0` is invalid.
const P_MIN: i32 = 2;

/// Spec M: column-major counting tropical matmul.
/// Caller owns device buffers (e.g. allocated via CUDA.jl).
/// Validates flags, P, dims; launches the right kernel for (T, D, tA, tB).
pub fn tropical_matmul_kernel<T, D>(
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
        + DeviceRepr
        + ValidAsZeroBits
        + crate::counting_kernel::TileDims
        + Default + Clone + Copy
        + 'static,
    D: TropicalDirection,
    (T, D): crate::counting_kernel::TropicalMatmulKernelName<T, D>,
{
    if tA != 'N' && tA != 'T' {
        return Err(CudaError::InvalidState(format!(
            "tA must be 'N' or 'T', got {:?}", tA
        )));
    }
    if tB != 'N' && tB != 'T' {
        return Err(CudaError::InvalidState(format!(
            "tB must be 'N' or 'T', got {:?}", tB
        )));
    }
    if p < P_MIN {
        return Err(CudaError::InvalidState(format!(
            "modulus must satisfy {} <= p < 2^31, got {}", P_MIN, p
        )));
    }
    if m == 0 || k == 0 || n == 0 {
        return Err(CudaError::InvalidState(
            "dimensions must be non-zero".into(),
        ));
    }
    crate::counting_kernel::launch_tropical_matmul::<T, D>(
        ctx, tA, tB, m, k, n, a_dev_ptr, b_dev_ptr, p, out_dev_ptr,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvrtc_compiles_pipelined_kernel() {
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let _k = ctx.get_kernel("tropical_matmul_f32_max_NN_pl")
            .expect("pipelined kernel must compile under NVRTC");
    }

    #[test]
    fn tropical_matmul_nn_2x2_max_p7() {
        // Column-major A (2×2): A[0,0]=1, A[1,0]=3, A[0,1]=2, A[1,1]=4.
        // Bytes flat (col-major): [1, 3, 2, 4].
        // Column-major B (2×2): B[0,0]=5, B[1,0]=7, B[0,1]=6, B[1,1]=8.
        // Bytes flat (col-major): [5, 7, 6, 8].
        // Max-plus C = A * B. Counts all 1.
        // C[0,0] = max(1+5, 2+7) = 9   (k=1 wins)
        // C[1,0] = max(3+5, 4+7) = 11
        // C[0,1] = max(1+6, 2+8) = 10
        // C[1,1] = max(3+6, 4+8) = 12
        // Output column-major: [9, 11, 10, 12].
        use crate::pair::PairF32;
        use tropical_gemm::types::Max;
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
        let a_ptr = *pair_a_dev.device_ptr();
        let b_ptr = *pair_b_dev.device_ptr();
        let out_ptr = *out_dev.device_ptr();

        tropical_matmul_kernel::<f32, Max>(
            ctx, 'N', 'N', 2, 2, 2, a_ptr, b_ptr, 7, out_ptr,
        ).expect("kernel ok");

        let out = ctx.device().dtoh_sync_copy(&out_dev).unwrap();
        assert_eq!(out[0].val, 9.0);  assert_eq!(out[0].cnt, 1);
        assert_eq!(out[1].val, 11.0); assert_eq!(out[1].cnt, 1);
        assert_eq!(out[2].val, 10.0); assert_eq!(out[2].cnt, 1);
        assert_eq!(out[3].val, 12.0); assert_eq!(out[3].cnt, 1);
    }

    #[test]
    fn tropical_matmul_tt_2x2_max_p7() {
        // 'T','T' on the same logical A and B as the NN test.
        // A 'T' storage (K×M=2×2 col-major): bytes [A(0,0),A(0,1),A(1,0),A(1,1)] = [1,2,3,4].
        // B 'T' storage (N×K=2×2 col-major): bytes [B(0,0),B(0,1),B(1,0),B(1,1)] = [5,6,7,8].
        use crate::pair::PairF32;
        use tropical_gemm::types::Max;
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let pair_a_host = vec![
            PairF32::new(1.0, 1), PairF32::new(2.0, 1),
            PairF32::new(3.0, 1), PairF32::new(4.0, 1),
        ];
        let pair_b_host = vec![
            PairF32::new(5.0, 1), PairF32::new(6.0, 1),
            PairF32::new(7.0, 1), PairF32::new(8.0, 1),
        ];

        let pair_a_dev = ctx.device().htod_copy(pair_a_host).unwrap();
        let pair_b_dev = ctx.device().htod_copy(pair_b_host).unwrap();
        let out_dev = ctx.device().alloc_zeros::<PairF32>(4).unwrap();

        use cudarc::driver::DevicePtr;
        tropical_matmul_kernel::<f32, Max>(
            ctx, 'T', 'T', 2, 2, 2,
            *pair_a_dev.device_ptr(),
            *pair_b_dev.device_ptr(),
            7,
            *out_dev.device_ptr(),
        ).expect("kernel ok");

        let out = ctx.device().dtoh_sync_copy(&out_dev).unwrap();
        assert_eq!(out[0].val, 9.0);
        assert_eq!(out[1].val, 11.0);
        assert_eq!(out[2].val, 10.0);
        assert_eq!(out[3].val, 12.0);
    }

    #[test]
    fn tropical_matmul_nn_4x4_min_random_p11() {
        use crate::pair::PairF64;
        use tropical_gemm::types::Min;
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let m = 4usize; let k = 6usize; let n = 4usize;
        let p = 11i32;
        let a_val: Vec<f64> = (0..m*k).map(|i| (i % 5) as f64).collect();
        let a_cnt: Vec<i32> = (0..m*k).map(|i| ((i + 1) % 11) as i32).collect();
        let b_val: Vec<f64> = (0..k*n).map(|i| (i % 4) as f64).collect();
        let b_cnt: Vec<i32> = (0..k*n).map(|i| ((i + 2) % 11) as i32).collect();

        // Pack to PairF64 in column-major order.
        let mut pair_a_host = vec![PairF64::default(); m * k];
        for i in 0..m { for j in 0..k {
            pair_a_host[i + j * m] = PairF64::new(a_val[i * k + j], a_cnt[i * k + j]);
        }}
        let mut pair_b_host = vec![PairF64::default(); k * n];
        for i in 0..k { for j in 0..n {
            pair_b_host[i + j * k] = PairF64::new(b_val[i * n + j], b_cnt[i * n + j]);
        }}

        let pair_a_dev = ctx.device().htod_copy(pair_a_host).unwrap();
        let pair_b_dev = ctx.device().htod_copy(pair_b_host).unwrap();
        let out_dev = ctx.device().alloc_zeros::<PairF64>(m * n).unwrap();

        use cudarc::driver::DevicePtr;
        tropical_matmul_kernel::<f64, Min>(
            ctx, 'N', 'N', m, k, n,
            *pair_a_dev.device_ptr(),
            *pair_b_dev.device_ptr(),
            p,
            *out_dev.device_ptr(),
        ).expect("kernel ok");

        let out = ctx.device().dtoh_sync_copy(&out_dev).unwrap();

        for i in 0..m {
            for j in 0..n {
                let mut best = f64::INFINITY;
                let mut acc: i64 = 0;
                for kk in 0..k {
                    let v = a_val[i*k + kk] + b_val[kk*n + j];
                    let c = (a_cnt[i*k + kk] as i64) * (b_cnt[kk*n + j] as i64) % (p as i64);
                    if v < best { best = v; acc = c; }
                    else if v == best { acc = (acc + c) % (p as i64); }
                }
                let cell = out[i + j * m];
                assert_eq!(cell.val, best, "value mismatch at ({},{})", i, j);
                assert_eq!(cell.cnt as i64, acc, "count mismatch at ({},{})", i, j);
            }
        }
    }

    #[test]
    fn tropical_matmul_rejects_bad_flag() {
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let err = tropical_matmul_kernel::<f32, tropical_gemm::types::Max>(
            ctx, 'X', 'N', 2, 2, 2, 0, 0, 7, 0,
        );
        assert!(err.is_err(), "bad flag must be rejected");
        let msg = format!("{:?}", err.unwrap_err());
        assert!(msg.contains("tA must be"), "got: {}", msg);
    }

    #[test]
    fn tropical_matmul_rejects_p_one() {
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let err = tropical_matmul_kernel::<f32, tropical_gemm::types::Max>(
            ctx, 'N', 'N', 2, 2, 2, 0, 0, 1, 0,
        );
        assert!(err.is_err(), "p=1 must be rejected");
    }

    // ---- Spec N: tile-edge and ragged-K correctness tests ----

    #[allow(non_snake_case)]
    fn cpu_ref_max_f32(
        tA: char, tB: char, m: usize, k: usize, n: usize,
        a: &[crate::pair::PairF32], b: &[crate::pair::PairF32], p: i32,
    ) -> Vec<crate::pair::PairF32> {
        let mut out = vec![crate::pair::PairF32::new(f32::NEG_INFINITY, 0); m * n];
        let p_u = p as u64;
        for j in 0..n {
            for i in 0..m {
                let mut acc_v = f32::NEG_INFINITY;
                let mut acc_c: u64 = 0;
                for kk in 0..k {
                    let av = if tA == 'N' { a[i + kk * m] } else { a[kk + i * k] };
                    let bv = if tB == 'N' { b[kk + j * k] } else { b[j + kk * n] };
                    let pv = av.val + bv.val;
                    let pc = ((av.cnt as u64) * (bv.cnt as u64)) % p_u;
                    if pv > acc_v { acc_v = pv; acc_c = pc; }
                    else if pv == acc_v { acc_c = (acc_c + pc) % p_u; }
                }
                out[i + j * m] = crate::pair::PairF32::new(acc_v, (acc_c % p_u) as i32);
            }
        }
        out
    }

    fn rand_pairs_f32(n: usize, p: i32) -> Vec<crate::pair::PairF32> {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(42);
        (0..n).map(|_| crate::pair::PairF32::new(
            rng.gen_range(0.0..4.0),
            rng.gen_range(1..p),
        )).collect()
    }

    #[allow(non_snake_case)]
    fn run_and_check_f32_max(tA: char, tB: char, m: usize, k: usize, n: usize, p: i32) {
        use cudarc::driver::DevicePtr;
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let a_len = m * k; let b_len = k * n;
        let a_host = rand_pairs_f32(a_len, p);
        let b_host = rand_pairs_f32(b_len, p);
        let expect = cpu_ref_max_f32(tA, tB, m, k, n, &a_host, &b_host, p);
        let a_dev = ctx.device().htod_copy(a_host).unwrap();
        let b_dev = ctx.device().htod_copy(b_host).unwrap();
        let out_dev = ctx.device().alloc_zeros::<crate::pair::PairF32>(m * n).unwrap();
        crate::matmul_mod::tropical_matmul_kernel::<f32, tropical_gemm::types::Max>(
            ctx, tA, tB, m, k, n,
            *a_dev.device_ptr(), *b_dev.device_ptr(), p, *out_dev.device_ptr(),
        ).unwrap();
        let got = ctx.device().dtoh_sync_copy(&out_dev).unwrap();
        for idx in 0..(m*n) {
            assert_eq!(got[idx].val, expect[idx].val,
                "val mismatch at {} (tA={},tB={},M={},K={},N={})", idx, tA, tB, m, k, n);
            assert_eq!(got[idx].cnt, expect[idx].cnt,
                "cnt mismatch at {} (tA={},tB={},M={},K={},N={})", idx, tA, tB, m, k, n);
        }
    }

    #[test] fn tile_edge_nn_65_65_65_f32() { run_and_check_f32_max('N','N', 65, 65, 65, 7); }
    #[test] fn tile_edge_tt_100_33_77_f32() { run_and_check_f32_max('T','T', 100, 33, 77, 11); }
    #[test] fn tile_exact_nt_128_128_128_f32() { run_and_check_f32_max('N','T', 128, 128, 128, 13); }

    #[test] fn ragged_k_bk_plus_1_nn_f32() { run_and_check_f32_max('N','N', 8, 9, 8, 7); }
    #[test] fn ragged_k_bk_plus_1_nt_f32() { run_and_check_f32_max('N','T', 8, 9, 8, 7); }
    #[test] fn ragged_k_bk_plus_1_tn_f32() { run_and_check_f32_max('T','N', 8, 9, 8, 7); }
    #[test] fn ragged_k_bk_plus_1_tt_f32() { run_and_check_f32_max('T','T', 8, 9, 8, 7); }

    #[test] fn ragged_k_2bk_minus_1_nn_f32() { run_and_check_f32_max('N','N', 8, 15, 8, 7); }
    #[test] fn ragged_k_1_tt_f32() { run_and_check_f32_max('T','T', 8, 1, 8, 7); }
}
