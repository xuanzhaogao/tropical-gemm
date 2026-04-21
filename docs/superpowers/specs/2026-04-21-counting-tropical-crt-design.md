# Spec B — CRT driver + `BigInt` counts + Python surface

**Date:** 2026-04-21
**Status:** design
**Depends on:** spec A (`2026-04-21-counting-tropical-compose-design.md`).
**Scope:** layer CRT-based exact `BigInt` counting on top of a working
  `CountingTropical` GEMM. CPU only. CUDA is a later follow-up.

## Goal

Count ground-state (or optimal) configurations whose multiplicity exceeds
`u64` range, by running the `CountingTropical` matmul once per prime with
count type `Mod<P>` and Chinese-Remainder-reconstructing the count field
to `BigInt` host-side. Mirrors `big_integer_solve` in
GenericTensorNetworks.jl.

## Preconditions from spec A

- `Mat<CountingTropical<T, C, D>>` can be multiplied end to end for any
  `C: TropicalScalar`.
- `MatRef` / kernel / packing thread `T` not `T::Scalar`.
- Direction marker `D: TropicalDirection` selects Max or Min.

## Design

### 1. `Mod<P>` — modular count scalar

New module `crates/tropical-gemm/src/types/modp.rs`.

```rust
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Mod<const P: i32>(i32);          // field private; invariant: 0 <= .0 < P

impl<const P: i32> Mod<P> {
    pub const fn new(x: i32) -> Self {      // normalizing constructor
        let r = x.rem_euclid(P);
        Self(r)
    }
    pub const fn raw(self) -> i32 { self.0 }
}
```

`scalar_add` / `scalar_mul` use `i64` intermediates with a compile-time
prime:

```rust
fn scalar_add(self, rhs: Self) -> Self {
    let s = self.0 as i64 + rhs.0 as i64;      // bounded by 2*(P-1) < 2^31
    Self((s % P as i64) as i32)
}
fn scalar_mul(self, rhs: Self) -> Self {
    let s = self.0 as i64 * rhs.0 as i64;      // bounded by (P-1)^2
    Self((s % P as i64) as i32)
}
```

**Prime size constraint (must be stated in code doc):** `P` must satisfy
`(P-1)^2 < 2^62` so the `i64` product never overflows. In practice we
use 30-bit primes where `(P-1)^2 < 2^60`. A const table of such primes:

```rust
pub const CRT_PRIMES: [i32; 16] = [/* prevprime walk from 1 << 30 */];
```

Concrete prime values chosen at implementation time via a small
build-time script or by pre-computation. All values go in the source.

`pos_infinity` / `neg_infinity` on `TropicalScalar`: implemented as
`unreachable!("Mod<P> is a count scalar, not a tropical value")`. `Mod`
never participates in the tropical value field, only the count field.

### 2. CRT driver

New module `crates/tropical-gemm/src/crt.rs`.

**Public API:**

```rust
pub struct CountedMat<T: TropicalScalar> {
    pub values: Mat<T>,                     // ground-state value per cell
    pub counts: Vec<num_bigint::BigInt>,    // column-major, len = nrows*ncols
}

pub fn count_ground_states<T, D>(
    a_values: &[T], a_rows: usize, a_cols: usize,
    b_values: &[T], b_rows: usize, b_cols: usize,
    count_upper_bound: &num_bigint::BigInt,
) -> CountedMat<T>
where T: TropicalScalar, D: TropicalDirection;
```

Callers pass raw tropical value slices (column-major) plus dimensions.
The direction is chosen at the type level by `D`. The driver constructs
`Mat<CountingTropical<T, Mod<p_i>, D>>` internally with count = `Mod(1)`
on every input cell and runs the spec-A matmul.

**Algorithm.**

1. Let `B = count_upper_bound` (required caller-supplied bound on the
   maximum possible count in any cell; see §3).
2. Choose the smallest `k` such that `M_k = p_1 * p_2 * ... * p_k > 2 * B`.
   Fail fast with a clear error if this exceeds the built-in prime table.
3. For `i in 1..=k`:
   - Build `Mat<CountingTropical<T, Mod<p_i>, D>>` from the input value
     matrices with count = `Mod(1)` everywhere.
   - Run matmul via spec-A pipeline.
   - Record `(value_i: Mat<T>, residues_i: Vec<Mod<p_i>>)`.
4. Assert `value_i == value_1` for all `i`. Any divergence is a bug,
   fail loudly (this is not a convergence criterion, it is an invariant
   — the tropical computation does not depend on `P`).
5. CRT-combine `(residues_1, p_1), …, (residues_k, p_k)` cell-wise to
   yield `BigInt` counts in `[0, M_k)`. Because `M_k > 2 * B ≥ 2 * count`,
   the CRT reconstruction is unique and equals the true count.
6. Return `CountedMat { values: value_1, counts }`.

Dependencies: `num-bigint`, `num-integer`.

### 3. Where `count_upper_bound` comes from

**This is the key correctness requirement that the original design got
wrong.** CRT without a bound is silently non-unique. Options:

- **A. Caller-supplied.** Caller computes `B` from problem structure.
  For a single matmul `C = A · B` where all input counts are 1, the per-cell
  count is at most `K` (the inner dimension), so `B = K` is trivially
  correct. For chained products, the bound grows multiplicatively.
- **B. Library-computed worst-case.** Derive `B` automatically from
  `nrows`, `ncols`, `inner dim`, and the range of tropical values. For
  non-degenerate inputs this is conservative but safe.
- **C. Iterate-until-stable, with declared tolerance for being wrong.**
  Cheap but wrong. Rejected. This is the trap the earlier draft fell
  into.

**Decision: A with a sensible default.** Public API takes
`count_upper_bound: &BigInt`. A helper `bound_for_single_matmul(k: usize)`
returns `BigInt::from(k)`. Users who chain products compute their own
bound. Docstring explains the contract clearly. If we later want
convenience, add a second API `count_ground_states_auto` that uses option
B; keep the explicit-bound API as the primitive.

### 4. Float `T` and exact-equality across primes

The `value_i == value_1` check in step 4 uses `PartialEq` on the value
field. For integer `T`, this is exact. For float `T`, it is exact iff
the sequence of floating-point operations is deterministic across runs
with different `C`. It is, because:

- The summation order in the micro-kernel is fixed by loop structure,
  not data-dependent.
- `scalar_add` on `Mod<P>` is independent of the value-field compare.

We document this invariant in `crt.rs` with a pointer to the kernel
code. If a future SIMD path introduces data-dependent reduction order,
this contract must be revisited.

### 5. Python surface

Location: `crates/tropical-gemm-python/src/lib.rs`.

```python
tropical_gemm.count_ground_states(
    a: np.ndarray,                   # 2D, f32 or f64
    b: np.ndarray,                   # 2D, same dtype
    direction: Literal['max', 'min'] = 'min',
    count_upper_bound: int,
) -> tuple[np.ndarray, np.ndarray]   # (values, counts as object array of Python int)
```

The counts array has `dtype=object` so each element is a Python `int`
(unbounded). No loss of precision. Users can convert to list or reshape
trivially.

## Testing

- **Unit — `Mod<P>`.** Ring axioms for a small `P`: commutativity,
  associativity, distributivity, `scalar_one` is multiplicative identity,
  normalizing constructor rejects out-of-range inputs correctly.
- **Unit — CRT math.** Two primes, hand-constructed residues, verify
  reconstruction matches known BigInt. Include a case where the residue
  of an "unreconstructible" number (one that would alias) differs,
  confirming we detect it via the bound check.
- **Correctness — small graphs.** 2×N MIS ladder, Ising chain
  degeneracy; values known analytically. Compare CRT result to a reference
  BigInt matmul implemented directly on `CountingTropical<T, BigInt, D>`
  (slow but correct — built as a test-only semiring).
- **Correctness — large count.** Construct an input where the true count
  provably exceeds `2^64`. Assert CRT returns the exact BigInt and that
  a `u64` version would have wrapped (sanity on the motivation).
- **Bound under-specification.** When the caller supplies a bound too
  small for the true count, the driver's behavior is documented: it
  returns the CRT-unique representative mod `M_k`, which may be wrong.
  Add a debug-only assertion that reconstructs with `k+1` primes and
  compares; if the user opts in, the driver can fail instead of silently
  returning. Decide default at implementation time.
- **Direction.** Same tests with `direction='min'` and `'max'`.
- **Python.** Round-trip a small case: `np.ndarray` in, list of Python
  ints out, value matches Rust-side test.

## Out of scope (revisit later)

- CUDA kernel for `CountingTropical<T, Mod<P>, D>`. Once spec A + B land,
  porting to CUDA is straightforward: the prime loop is host-side, one
  kernel launch per prime, no change to CRT driver.
- SIMD (AVX2/AVX-512/NEON) for the counting inner kernel.
- Argmax composition with counting.
- An `auto`-bound API (option B in §3).

## Risks

- **Silent wrong answers if `count_upper_bound` is under-stated.** This
  is the load-bearing contract. Docstrings, examples, and the debug-only
  double-check assertion mitigate. If we later add an auto-bound path,
  make it the default for Python.
- **Prime table exhaustion.** `2 * B > M_k` for `k = 16` means counts
  exceeding `~2^480`, which is extreme. If a real use case hits this,
  extend the table; do not raise silently.
- **Float value stability across `C` types.** See §4. If a future SIMD
  kernel re-orders reductions in a data-dependent way, this invariant
  breaks. Documented.
