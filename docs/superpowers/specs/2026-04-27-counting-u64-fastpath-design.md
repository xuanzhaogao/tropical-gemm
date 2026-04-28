# Spec I — `count_ground_states_gpu_u64` fast-path

**Date:** 2026-04-27
**Status:** design (ready to implement)
**Branch:** `counting-tropical`
**Depends on:** specs C, F, G.
**Scope:** add a parallel entry point `count_ground_states_gpu_u64` returning `Vec<u64>` counts instead of `Vec<BigInt>`. Eliminates the per-cell `BigInt` heap allocation that dominates end-to-end time for the common case where the count product fits in u64 (≤ 2 primes from `CRT_PRIMES`).

## Goal

The current `count_ground_states_gpu` end-to-end profile at 2048² spends ~65% of its time (~275 ms of ~459 ms) constructing `Vec<BigInt>` — 4M `BigInt::from(i32)` heap allocations. The kernel itself is now <5% of e2e (1946 G/s ones-kernel @ 4096² runs in ~70 ms). The dominant cost is the output format.

For `count_upper_bound < 2^60`, the CRT product fits in u64 (one or two 30-bit primes). At that point the BigInt machinery is overkill — counts could be returned as a flat `Vec<u64>` with zero per-cell allocation.

**Expected:** ~5× e2e speedup at 2048² (459 ms → ~80–100 ms) for the u64-eligible case. Bench will confirm.

## Non-goals

- Removing the BigInt path. Stays for callers needing the general case (chained matmul with growing counts, or `count_upper_bound > 2^60`).
- Python-binding update. Cover in a follow-up.

## Eligibility

CRT primes are 30-bit (~2^30):
- 1 prime: product < 2^30. Covers `bound_for_single_matmul(k) < 2^30` ≈ K < 1 billion.
- 2 primes: product < 2^60. Covers K < 2^60.
- ≥3 primes: product > 2^90, overflows u64. **Reject.**

So the u64 fast-path covers *every* single-matmul case the existing `bound_for_single_matmul(k)` would emit (since `k <= 2^60` for any realistic problem). For chained-matmul callers with growing counts, eligibility may fail and they fall back to BigInt.

## Public API

```rust
#[derive(Debug, Clone)]
pub struct CountedMatU64<T: TropicalScalar> {
    pub nrows: usize,
    pub ncols: usize,
    pub values: Vec<T>,
    pub counts: Vec<u64>,
}

pub fn count_ground_states_gpu_u64<T, D>(
    ctx: &CudaContext,
    a_values: &[T], m: usize, k: usize,
    b_values: &[T], n: usize,
    count_upper_bound: u64,  // u64 instead of BigInt
) -> Result<CountedMatU64<T>>
```

If `2 * count_upper_bound + 1 > 2^60`, returns `CudaError::InvalidState` with a message directing the caller to `count_ground_states_gpu` (BigInt path).

## CPU-side helpers (in `tropical_gemm::crt`)

Add three small helpers next to the existing BigInt versions:

```rust
/// u64 CRT combine. Preconditions:
///   - 0 <= acc_value < acc_modulus
///   - 0 <= residue   < prime
///   - gcd(acc_modulus, prime) == 1
///   - acc_modulus * prime fits in u64
pub fn crt_combine_u64(
    acc_value: u64, acc_modulus: u64,
    residue: i32, prime: i32,
) -> (u64, u64);

/// Choose primes for u64-bounded reconstruction. Returns Some((indices, product))
/// when found (product > needed AND product fits in u63). Returns None when no
/// valid prefix of CRT_PRIMES satisfies both.
pub fn choose_primes_u64(needed: u64) -> Option<(Vec<usize>, u64)>;

pub fn bound_for_single_matmul_u64(k: usize) -> u64 { k as u64 }
```

The combine uses standard pairwise CRT with extended-Euclidean inverse mod `prime` in i64:

```rust
pub fn crt_combine_u64(acc_value: u64, acc_modulus: u64, residue: i32, prime: i32) -> (u64, u64) {
    let p = prime as u64;
    let r = residue as u64;
    let diff = (r + p - (acc_value % p)) % p;
    let inv  = modinv_u64(acc_modulus % p, p);
    let delta = (diff * inv) % p;
    let new_modulus = acc_modulus.checked_mul(p).expect("u64 CRT overflow");
    let new_value   = acc_value + acc_modulus * delta;
    debug_assert!(new_value < new_modulus);
    (new_value, new_modulus)
}
```

`modinv_u64` is a standard extended-Euclidean implementation in i64 arithmetic (CRT_PRIMES are < 2^31 so no overflow).

## GPU driver (`tropical_gemm_cuda::crt`)

Refactor the existing `count_ground_states_gpu` to extract a private helper:

```rust
fn run_kernels_per_prime<T, D>(
    ctx, a, m, k, b, n, prime_indices,
) -> Result<(Vec<T>, Vec<Vec<i32>>)>  // (values, residue_streams)
```

This encapsulates: AoS-or-ones layout choice, B-transpose-or-not, kernel launch loop, value download, residue downloads. Both BigInt and u64 entry points reuse it.

`count_ground_states_gpu_u64` then:
1. Compute `needed = 2 * count_upper_bound + 1` (in u64 with overflow check).
2. Call `choose_primes_u64(needed)` — return error if `None`.
3. Call `run_kernels_per_prime` to get values + residue streams.
4. Reconstruct counts as `Vec<u64>`:
   - 1 prime: `residues[0].iter().map(|&r| r as u64).collect()`. **Zero CRT cost.**
   - 2 primes: pairwise `crt_combine_u64` per cell.

`count_ground_states_gpu` (BigInt) keeps its existing reconstruction logic but uses `run_kernels_per_prime` for the kernel portion.

## Tests

In `tests/counting_gpu.rs`:

1. `u64_matches_bigint_single_prime` — small case, K such that 1 prime suffices, assert `gpu_u64.counts[i] as BigInt == gpu_bigint.counts[i]` for all cells. Both paths same kernel; tests reconstruction parity.
2. `u64_matches_bigint_two_primes` — pick K where bound forces 2 primes. Same parity assertion.
3. `u64_rejects_too_large_bound` — pass `count_upper_bound = 2^61`, assert error.
4. `u64_all_ties_large_k_warpk` — K=200 all-zeros, assert counts == K.

In `tropical_gemm` lib:

5. `crt_combine_u64_matches_bigint` — randomized inputs (10K samples), assert `crt_combine_u64` output mod each prime equals the BigInt version.
6. `choose_primes_u64_correctness` — for `needed ∈ {0, 1, 2^30, 2^31, 2^60}`, assert chosen product > needed AND ≤ 2^63.
7. `choose_primes_u64_rejects_overflow` — `needed = 2^61`, assert `None`.

## Bench

Add `examples/bench_e2e.rs` (or extend `bench_counting`) to time the **driver** (not just kernel) at 2048², comparing BigInt vs u64. Confirm ≥4× e2e speedup.

## Roll-out

1. Land CPU helpers + their tests.
2. Land GPU driver refactor (`run_kernels_per_prime`) — preserve existing BigInt path semantics; tests stay green.
3. Land `count_ground_states_gpu_u64` + `CountedMatU64`. Add 4 new integration tests.
4. Run e2e bench. Verify expected speedup.
5. Update memory + commit.

## Risks

- **Refactor regression on BigInt path.** Mitigated: existing 17 integration tests cover the BigInt path; they must stay green after the refactor.
- **Overflow in `crt_combine_u64`.** `acc_modulus * delta` is bounded by `(2^30) * (2^30) = 2^60` — safe. New modulus is bounded by 2^60 — safe. Single Barrett-style overflow check on `checked_mul` for the new modulus catches any future prime expansion.
- **Eligibility false negatives.** A caller with `count_upper_bound < 2^60` but coming from a path where the BigInt was already constructed wouldn't benefit. That's fine — they can opt in via the new entry point.
- **API surface growth.** Two parallel entry points now (`_u64` and BigInt). The u64 path is the recommended one; BigInt remains for general counts. Documented in module-level docs.

## Non-goals (deferred)

- Python-binding for u64 path. Separate work.
- Auto-routing inside `count_ground_states_gpu` to detect u64 eligibility and skip BigInt construction internally. Possible but adds an output-type-erasure complication; cleaner to give callers two entry points.
- Removing BigInt path. Stays for general use.
