//! GPU CRT driver for `count_ground_states` (spec C).
//!
//! Mirrors the CPU spec-B driver in `tropical_gemm::crt`, dispatching the
//! per-prime matmul to the CUDA kernel. Uploads value matrices once
//! (shared across all prime runs), reuses count matrices fresh per prime.

use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use num_bigint::BigInt;
use num_traits::One;

use tropical_gemm::crt::{choose_primes, crt_combine, CRT_PRIMES};
use tropical_gemm::types::TropicalDirection;
use tropical_gemm::CountedMat;

use crate::context::CudaContext;
use crate::counting_kernel::{launch_counting_gemm, CountingCudaKernel};
use crate::error::{CudaError, Result};
use crate::memory::GpuMatrix;

/// GPU version of `tropical_gemm::count_ground_states`. Same semantics,
/// same caller contract on `count_upper_bound`. See the CPU docstring for
/// details.
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
        + 'static,
    D: TropicalDirection,
    (T, D): CountingCudaKernel<T, D>,
{
    assert_eq!(a_values.len(), m * k);
    assert_eq!(b_values.len(), k * n);

    // Upload value matrices once. Reused across all prime runs.
    let value_a = GpuMatrix::<T>::from_host(ctx, a_values, m, k)?;
    let value_b = GpuMatrix::<T>::from_host(ctx, b_values, k, n)?;

    let needed = BigInt::from(2) * count_upper_bound + BigInt::one();
    let (prime_indices, _product) = choose_primes(&needed);

    let ncells = m * n;
    let mut values_ref: Option<Vec<T>> = None;
    let mut residue_streams: Vec<Vec<i32>> = Vec::with_capacity(prime_indices.len());

    // Reusable "ones" buffers for count inputs.
    let ones_a_host = vec![1_i32; m * k];
    let ones_b_host = vec![1_i32; k * n];

    for &prime_idx in &prime_indices {
        let p = CRT_PRIMES[prime_idx];

        // Fresh count inputs per prime (the kernel reads these, so we could
        // also reuse a single pair — but re-uploading is cheap and keeps
        // the loop straightforward).
        let count_a = GpuMatrix::<i32>::from_host(ctx, &ones_a_host, m, k)?;
        let count_b = GpuMatrix::<i32>::from_host(ctx, &ones_b_host, k, n)?;

        // Output buffers. Initial values don't matter — kernel overwrites.
        let zeros_v = vec![T::default(); ncells];
        let zeros_c = vec![0_i32; ncells];
        let mut value_c = GpuMatrix::<T>::from_host(ctx, &zeros_v, m, n)?;
        let mut count_c = GpuMatrix::<i32>::from_host(ctx, &zeros_c, m, n)?;

        launch_counting_gemm::<T, D>(
            ctx,
            &value_a,
            &count_a,
            &value_b,
            &count_b,
            &mut value_c,
            &mut count_c,
            p,
        )?;

        let host_values = value_c.to_host(ctx)?;
        let host_counts = count_c.to_host(ctx)?;

        match &values_ref {
            None => values_ref = Some(host_values),
            Some(v) => {
                if v != &host_values {
                    return Err(CudaError::InvalidState(
                        "CRT invariant violated: value field differs across primes".into(),
                    ));
                }
            }
        }
        residue_streams.push(host_counts);
    }

    let values = values_ref.expect("at least one prime");

    // CRT reconstruct per cell.
    let mut counts = Vec::with_capacity(ncells);
    for cell in 0..ncells {
        let mut acc_value = BigInt::from(residue_streams[0][cell]);
        let mut acc_modulus = BigInt::from(CRT_PRIMES[prime_indices[0]]);
        for step in 1..prime_indices.len() {
            let p = CRT_PRIMES[prime_indices[step]];
            let (new_value, new_modulus) =
                crt_combine(&acc_value, &acc_modulus, residue_streams[step][cell], p);
            acc_value = new_value;
            acc_modulus = new_modulus;
        }
        counts.push(acc_value);
    }

    Ok(CountedMat {
        nrows: m,
        ncols: n,
        values,
        counts,
    })
}
