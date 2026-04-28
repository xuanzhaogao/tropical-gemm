//! GPU CRT driver for `count_ground_states`.
//!
//! Two entry points:
//!   - [`count_ground_states_gpu`] — general path returning `Vec<BigInt>`.
//!     Use for `count_upper_bound` ≥ 2⁶⁰ or chained-matmul callers with
//!     non-trivial input counts (future work).
//!   - [`count_ground_states_gpu_u64`] — u64 fast-path (Spec I) returning
//!     `Vec<u64>`. Use whenever the bound fits in u63; eliminates the
//!     per-cell BigInt heap allocation that dominates e2e time on the
//!     general path.
//!
//! Both share [`run_kernels_per_prime`] for the device side: AoS-vs-naive
//! layout, B-transpose-or-not, kernel launches, value/residue downloads.
//! They differ only in the host-side reconstruction of per-cell counts.

use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use num_bigint::BigInt;
use num_traits::One;

use tropical_gemm::crt::{
    choose_primes, choose_primes_u64, crt_combine, crt_combine_u64, CRT_PRIMES,
};
use tropical_gemm::types::TropicalDirection;
use tropical_gemm::CountedMat;

use crate::context::{CudaContext, COUNTING_WARPK_K_THRESHOLD, COUNTING_WARPK_MN_CEILING};
use crate::counting_kernel::{launch_counting_gemm_ones, CountingCudaKernel};
use crate::error::{CudaError, Result};
use crate::memory::GpuMatrix;
use crate::pair::PackPair;

/// Result of `count_ground_states_gpu_u64`. Counts are `u64` instead of
/// `BigInt`, eliminating per-cell heap allocation. Caller is responsible for
/// ensuring `count_upper_bound` fits in the u64-eligible regime.
#[derive(Debug, Clone)]
pub struct CountedMatU64<T> {
    pub nrows: usize,
    pub ncols: usize,
    pub values: Vec<T>,
    pub counts: Vec<u64>,
}

/// Transpose a row-major matrix `(rows × cols)` to row-major `(cols × rows)`.
fn transpose_row_major<T: Copy + Default>(src: &[T], rows: usize, cols: usize) -> Vec<T> {
    debug_assert_eq!(src.len(), rows * cols);
    let mut out = vec![T::default(); rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = src[r * cols + c];
        }
    }
    out
}

/// Run the per-prime kernel loop and return the value field plus per-prime
/// residue streams. Encapsulates the shared device-side work between the
/// BigInt and u64 entry points.
///
/// Returns `(values, residue_streams)` where `residue_streams[i]` is the
/// per-cell residue mod `CRT_PRIMES[prime_indices[i]]`.
fn run_kernels_per_prime<T, D>(
    ctx: &CudaContext,
    a_values: &[T],
    m: usize,
    k: usize,
    b_values: &[T],
    n: usize,
    prime_indices: &[usize],
) -> Result<(Vec<T>, Vec<Vec<i32>>)>
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
    // Layout choice (Spec H): warpk wants B^T (N × K row-major) for coalesced
    // lane loads; naive wants the original K × N row-major.
    let use_warpk = k >= COUNTING_WARPK_K_THRESHOLD
        && m.saturating_mul(n) <= COUNTING_WARPK_MN_CEILING;
    let value_a_dev = GpuMatrix::<T>::from_host(ctx, a_values, m, k)?;
    let value_b_dev = if use_warpk {
        let b_t = transpose_row_major::<T>(b_values, k, n);
        GpuMatrix::<T>::from_host(ctx, &b_t, n, k)?
    } else {
        GpuMatrix::<T>::from_host(ctx, b_values, k, n)?
    };

    let mut value_c = GpuMatrix::<T>::alloc(ctx, m, n)?;
    let mut count_c = GpuMatrix::<i32>::alloc(ctx, m, n)?;

    let mut residue_streams: Vec<Vec<i32>> = Vec::with_capacity(prime_indices.len());
    let mut values_ref: Option<Vec<T>> = None;

    for (i, &prime_idx) in prime_indices.iter().enumerate() {
        let p = CRT_PRIMES[prime_idx];

        launch_counting_gemm_ones::<T, D>(
            ctx,
            &value_a_dev,
            &value_b_dev,
            &mut value_c,
            &mut count_c,
            m,
            k,
            n,
            p,
        )?;

        // Download value_c only on the first prime. Skip on subsequent —
        // the tropical value field is invariant of P. In debug builds,
        // sanity-check on prime #2.
        if i == 0 {
            values_ref = Some(value_c.to_host(ctx)?);
        } else if cfg!(debug_assertions) && i == 1 {
            let v2 = value_c.to_host(ctx)?;
            if values_ref.as_ref().unwrap() != &v2 {
                return Err(CudaError::InvalidState(
                    "CRT invariant violated: value field differs across primes".into(),
                ));
            }
        }

        residue_streams.push(count_c.to_host(ctx)?);
    }

    let values = values_ref.expect("at least one prime");
    Ok((values, residue_streams))
}

/// GPU version of `tropical_gemm::count_ground_states`. Same semantics,
/// same caller contract on `count_upper_bound`. Returns `Vec<BigInt>`
/// counts. For bounds that fit in u63, prefer
/// [`count_ground_states_gpu_u64`] — ~5× faster end-to-end by eliminating
/// per-cell BigInt heap allocations.
pub fn count_ground_states_gpu<T, D>(
    ctx: &CudaContext,
    a_values: &[T],
    m: usize,
    k: usize,
    b_values: &[T],
    n: usize,
    count_upper_bound: &BigInt,
) -> Result<CountedMat<T>>
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
    assert_eq!(a_values.len(), m * k);
    assert_eq!(b_values.len(), k * n);

    let needed = BigInt::from(2) * count_upper_bound + BigInt::one();
    let (prime_indices, _product) = choose_primes(&needed);
    let num_primes = prime_indices.len();

    let (values, residue_streams) =
        run_kernels_per_prime::<T, D>(ctx, a_values, m, k, b_values, n, &prime_indices)?;

    let ncells = m * n;
    let counts: Vec<BigInt> = if num_primes == 1 {
        residue_streams[0].iter().map(|&r| BigInt::from(r)).collect()
    } else {
        let mut out = Vec::with_capacity(ncells);
        for cell in 0..ncells {
            let mut acc_value = BigInt::from(residue_streams[0][cell]);
            let mut acc_modulus = BigInt::from(CRT_PRIMES[prime_indices[0]]);
            for step in 1..num_primes {
                let p = CRT_PRIMES[prime_indices[step]];
                let (new_value, new_modulus) = crt_combine(
                    &acc_value,
                    &acc_modulus,
                    residue_streams[step][cell],
                    p,
                );
                acc_value = new_value;
                acc_modulus = new_modulus;
            }
            out.push(acc_value);
        }
        out
    };

    Ok(CountedMat {
        nrows: m,
        ncols: n,
        values,
        counts,
    })
}

/// u64 fast-path. Same kernel work as `count_ground_states_gpu`, but
/// reconstructs counts as `Vec<u64>` directly — no per-cell BigInt heap
/// allocation. Eligible when `2 * count_upper_bound + 1` is achievable with
/// a prefix of `CRT_PRIMES` whose product fits in u63 (≤ 2 of the 30-bit
/// primes; covers `count_upper_bound < 2^60`).
///
/// Returns `CudaError::InvalidState` if the bound exceeds the u63-safe
/// envelope; the caller should fall back to [`count_ground_states_gpu`].
pub fn count_ground_states_gpu_u64<T, D>(
    ctx: &CudaContext,
    a_values: &[T],
    m: usize,
    k: usize,
    b_values: &[T],
    n: usize,
    count_upper_bound: u64,
) -> Result<CountedMatU64<T>>
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
    assert_eq!(a_values.len(), m * k);
    assert_eq!(b_values.len(), k * n);

    // needed = 2 * bound + 1, with overflow check.
    let needed = count_upper_bound
        .checked_mul(2)
        .and_then(|x| x.checked_add(1))
        .ok_or_else(|| CudaError::InvalidState(
            "count_upper_bound too large for u64 fast-path; use count_ground_states_gpu (BigInt)".into(),
        ))?;

    let (prime_indices, _product) = choose_primes_u64(needed).ok_or_else(|| {
        CudaError::InvalidState(
            "count_upper_bound exceeds u63-safe envelope (>= 3 primes needed); \
             use count_ground_states_gpu (BigInt) instead"
                .into(),
        )
    })?;
    let num_primes = prime_indices.len();

    let (values, residue_streams) =
        run_kernels_per_prime::<T, D>(ctx, a_values, m, k, b_values, n, &prime_indices)?;

    let ncells = m * n;
    let counts: Vec<u64> = if num_primes == 1 {
        // Single-prime: the residue stream is the answer mod p_0. No CRT.
        residue_streams[0].iter().map(|&r| r as u64).collect()
    } else {
        // 2+ primes (in practice ≤ 2 here; choose_primes_u64 enforces).
        let mut out = Vec::with_capacity(ncells);
        for cell in 0..ncells {
            let mut acc_value = residue_streams[0][cell] as u64;
            let mut acc_modulus = CRT_PRIMES[prime_indices[0]] as u64;
            for step in 1..num_primes {
                let p = CRT_PRIMES[prime_indices[step]];
                let (new_value, new_modulus) =
                    crt_combine_u64(acc_value, acc_modulus, residue_streams[step][cell], p);
                acc_value = new_value;
                acc_modulus = new_modulus;
            }
            out.push(acc_value);
        }
        out
    };

    Ok(CountedMatU64 {
        nrows: m,
        ncols: n,
        values,
        counts,
    })
}
