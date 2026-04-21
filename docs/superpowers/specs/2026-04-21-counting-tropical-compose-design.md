# Spec A — Make `CountingTropical` compose with the GEMM pipeline

**Date:** 2026-04-21
**Status:** design, revised after codex review
**Scope:** structural only. No CRT, no `BigInt`, no Python bindings.
**Successor:** spec B — `2026-04-21-counting-tropical-crt-design.md`.

## Why this spec exists separately

The original single-spec design bundled (i) making `CountingTropical` work
through the existing kernels with (ii) a CRT/BigInt driver and Python API.
A codex review flagged that (i) is itself a non-trivial ABI change, because
the current pipeline threads `T::Scalar` (not `T`) end to end and relies on
`#[repr(transparent)]` newtype wrappers:

- `MatRef<'a, S>` holds `&[S::Scalar]` and `MatRef::from_mat` transmutes
  `&[S]` to `&[S::Scalar]` assuming `S` is `repr(transparent)` over
  `Scalar` — see `crates/tropical-gemm/src/mat/ref_.rs:28` and `:79`,
  `crates/tropical-gemm/src/mat/owned.rs:184`.
- `Microkernel::execute` takes `a: *const T::Scalar, b: *const T::Scalar`
  — `crates/tropical-gemm/src/core/kernel.rs:24`.
- Packing allocates `Vec<T::Scalar>` and copies `T::Scalar` —
  `crates/tropical-gemm/src/core/gemm.rs:78`, `core/packing.rs:46`.

`CountingTropical<T, C>` is `#[repr(C)]` with two fields
(`src/types/counting.rs:20`). The transmute is unsound for it. Until the
pipeline is widened, no CRT work can land.

## Goal of spec A

Make `Mat<CountingTropical<T, C>> * Mat<CountingTropical<T, C>>` work end
to end via the existing `matmul` entry points, with `C: TropicalScalar`
being any plain integer or float. The direction (Max or Min) is
parameterized by a marker trait.

Concretely: after spec A lands, the following compiles and passes tests:

```rust
let a: Mat<CountingTropical<f32, u64, Max>> = /* ... */;
let b: Mat<CountingTropical<f32, u64, Max>> = /* ... */;
let c = &a * &b;   // semiring matmul, counts in u64
```

No `BigInt`. No CRT. No `Mod<P>`. Those live in spec B.

## Design

### 1. ABI change: thread `T` end to end instead of `T::Scalar`

This is the central decision. The `T::Scalar` threading is an optimization
that only works when `T` is a single-scalar newtype. Compound elements
break it. Options considered:

- **A. Drop the `Scalar` threading.** `MatRef<'a, S>` holds `&[S]`; packing
  allocates `Vec<S>`; `Microkernel::execute` takes `*const S`. For
  existing single-scalar types (`MaxPlus<f32>`, etc.), `S` is
  `repr(transparent)` over its scalar, so runtime layout is unchanged and
  no perf regression is expected. All unsafe transmutes in `mat/*` go
  away.
- **B. Add a parallel "compound element" API next to the scalar one.**
  Duplicate kernel/packing paths. Rejected: doubles the surface area we
  have to maintain and keep in SIMD/CUDA parity.
- **C. Make `CountingTropical` itself `repr(transparent)` via a packed
  integer representation.** Rejected: only works when `T` and `C` can be
  packed into one primitive; not general; fights the compiler.

**Decision: A.** The `unsafe` transmutes in `mat/ref_.rs` and
`mat/owned.rs` are replaced with plain `&[S]` / `&mut [S]` views. Every
internal call site that currently passes `T::Scalar` slices is updated to
pass `T` slices.

The `TropicalScalar` associated type on `TropicalSemiring` stays — it is
still the correct "user-facing numeric type" for `from_scalar`, `value()`,
and public constructors. It just stops being the buffer element type.

### 2. Direction marker

New module `crates/tropical-gemm/src/types/direction.rs`:

```rust
pub trait TropicalDirection: Copy + Clone + Default + Send + Sync + 'static {
    fn zero_value<T: TropicalScalar>() -> T;   // −∞ for Max, +∞ for Min
    fn is_strictly_better<T: TropicalScalar>(candidate: T, incumbent: T) -> bool;
}

#[derive(Copy, Clone, Default, Debug)]
pub struct Max;
#[derive(Copy, Clone, Default, Debug)]
pub struct Min;
```

`CountingTropical<T, C>` becomes `CountingTropical<T, C, D: TropicalDirection = Max>`.
The default preserves source compatibility with existing tests.

`tropical_zero`, `tropical_add`, and `tropical_add_argmax` consult `D`:

```rust
fn tropical_add(self, rhs: Self) -> Self {
    use std::cmp::Ordering::*;
    if D::is_strictly_better(self.value, rhs.value) { self }
    else if D::is_strictly_better(rhs.value, self.value) { rhs }
    else { Self { value: self.value, count: self.count.scalar_add(rhs.count) } }
}
```

### 3. Kernel / packing changes

- `Microkernel::execute` becomes `fn execute(a: *const T, b: *const T, ...)`.
- `pack_a` / `pack_b` in `core/packing.rs` allocate and write `T`, not
  `T::Scalar`. For single-scalar semirings, `T` and `T::Scalar` have the
  same layout, so the generated code is identical.
- A new inner-loop path for `CountingTropical<T, C, D>` in
  `core/kernel.rs`: scalar implementation only (no SIMD yet), since the
  existing `SimdTropical` lanes all assume scalar element type. The
  `SIMD_WIDTH = 8` claim currently set in `types/counting.rs:123` is
  removed — we replace it with `SIMD_AVAILABLE = false` for v1.
- `simd/dispatch.rs` routes `CountingTropical<_, _, _>` to the scalar
  kernel. AVX/NEON lanes for counting are explicit future work (noted
  below, implemented in a later spec).

### 4. Nothing else changes

- No `Mod<P>`.
- No CRT driver.
- No Python API changes.
- No CUDA changes. `tropical-gemm-cuda` keeps its current
  `T::Scalar`-based kernels; those still work because the scalar-threading
  remains valid internally for single-scalar types used by CUDA.

## Testing

- **Unit.** Existing `CountingTropical` tests pass unchanged (default
  `D = Max`). New tests for `CountingTropical<f64, u64, Min>` covering
  semiring identity, addition with different values, addition with equal
  values (count merge), and multiplication.
- **Direction marker.** `Max::is_strictly_better(5, 3) == true`,
  `Min::is_strictly_better(5, 3) == false`, and vice versa.
- **ABI change.** All existing `MaxPlus`/`MinPlus`/`MaxMul`/`AndOr`
  tests continue to pass after the `T::Scalar` → `T` threading
  refactor. This is the main risk of the spec and the primary thing CI
  needs to verify.
- **End-to-end matmul on `CountingTropical<f32, u64, Max>`.** Small
  hand-checked matrix where both value and count are known.
- **End-to-end matmul on `CountingTropical<f32, u64, Min>`.** Same shape,
  minimization direction, hand-checked.

## Explicit non-goals

- Performance parity with `MaxPlus<f32>` on counting inputs. Scalar-only
  kernel is fine for v1.
- `BigInt` counts. Use spec B.
- Exact counts when `u64` overflows. Use spec B.
- Python surface for counting matmul. Use spec B.
- CUDA support for counting. Use spec B + later follow-up.

## Risks

- The `T::Scalar` → `T` refactor touches many files. Regressions in the
  existing semirings would block the project. Mitigation: the refactor is
  mechanical (type substitution); existing tests catch regressions.
- `#[repr(transparent)]` on single-scalar semirings must still hold so
  their `T` slices remain layout-compatible with `T::Scalar` slices for
  any external FFI call sites. Audit `tropical-gemm-cuda` and
  `tropical-gemm-python` for places that rely on this.
