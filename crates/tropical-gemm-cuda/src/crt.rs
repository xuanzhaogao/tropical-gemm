//! GPU CRT driver for `count_ground_states` (spec C, optimized).
//!
//! Mirrors the CPU spec-B driver in `tropical_gemm::crt`, dispatching the
//! per-prime matmul to the CUDA kernel. All device buffers (inputs,
//! outputs, count buffers) are allocated ONCE and reused across prime
//! runs. The fast single-prime path skips CRT combine entirely.

use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use num_bigint::BigInt;
use num_traits::One;

use tropical_gemm::crt::{choose_primes, crt_combine, CRT_PRIMES};
use tropical_gemm::types::TropicalDirection;
use tropical_gemm::CountedMat;

use crate::context::CudaContext;
use crate::counting_kernel::{launch_counting_gemm_ones, CountingCudaKernel};
use crate::error::{CudaError, Result};
use crate::memory::GpuMatrix;
use crate::pair::PackPair;

/// GPU version of `tropical_gemm::count_ground_states`. Same semantics,
/// same caller contract on `count_upper_bound`.
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

    // Counts at the entry point are uniformly 1. Use the ones-specialized
    // kernel: value-only inputs uploaded verbatim (no AoS pack), count
    // multiply and per-step Barrett dropped (~1.5-2x speedup over the AoS
    // general kernel; see Spec G).
    let value_a_dev = GpuMatrix::<T>::from_host(ctx, a_values, m, k)?;
    let value_b_dev = GpuMatrix::<T>::from_host(ctx, b_values, k, n)?;

    // Output buffers reused across primes. GPU-side zero-init (no host→device
    // upload of zero buffers).
    let ncells = m * n;
    let mut value_c = GpuMatrix::<T>::alloc(ctx, m, n)?;
    let mut count_c = GpuMatrix::<i32>::alloc(ctx, m, n)?;

    let needed = BigInt::from(2) * count_upper_bound + BigInt::one();
    let (prime_indices, _product) = choose_primes(&needed);
    let num_primes = prime_indices.len();

    // Residue downloads stay on host across iterations for the final CRT step.
    let mut residue_streams: Vec<Vec<i32>> = Vec::with_capacity(num_primes);
    let mut values_ref: Option<Vec<T>> = None;

    for (i, &prime_idx) in prime_indices.iter().enumerate() {
        let p = CRT_PRIMES[prime_idx];

        launch_counting_gemm_ones::<T, D>(
            ctx,
            &value_a_dev,
            &value_b_dev,
            &mut value_c,
            &mut count_c,
            p,
        )?;

        // Download value_c only on the first prime. Skip on subsequent primes
        // — the tropical value field is invariant of P by construction.
        // In debug builds, verify on prime #2 as a sanity check.
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

    // CRT reconstruct per cell. Fast path when only one prime was used:
    // no CRT combine needed; counts are already the final residues mod p_0.
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
