# Spec A — CountingTropical composes with GEMM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Mat<CountingTropical<T, C, D>> * Mat<_>` work end-to-end through the existing GEMM pipeline. No CRT, no BigInt, no Python.

**Architecture:** (1) Add a `TropicalDirection` marker trait so `CountingTropical` can represent both max- and min-tropical counting. (2) Widen the internal GEMM ABI from `*const T::Scalar` to `*const T` so compound elements flow through packing and the micro-kernel — existing single-scalar semirings are unaffected at runtime because they are `#[repr(transparent)]`. (3) Keep the public `tropical_matmul<S>(&[S::Scalar], …)` API source-compatible via a `ReprTransparentTropical` bridge, and add a parallel `tropical_matmul_t<S>(&[S], …)` for compound elements.

**Tech Stack:** Rust 1.x, `cargo test`, no new deps.

**Spec:** `docs/superpowers/specs/2026-04-21-counting-tropical-compose-design.md`

---

## File map

- **Create:** `crates/tropical-gemm/src/types/direction.rs` — `TropicalDirection` trait, `Max`/`Min` marker structs.
- **Modify:** `crates/tropical-gemm/src/types/mod.rs` — re-export direction types.
- **Modify:** `crates/tropical-gemm/src/types/counting.rs` — add `D: TropicalDirection` generic; route `tropical_zero`/`tropical_add`/`tropical_add_argmax` through `D`; fix `SIMD_AVAILABLE`.
- **Modify:** `crates/tropical-gemm/src/types/traits.rs` — add `ReprTransparentTropical` marker trait with safety contract.
- **Modify:** `crates/tropical-gemm/src/types/max_plus.rs`, `min_plus.rs`, `max_mul.rs`, `and_or.rs` — impl `ReprTransparentTropical` (no behavior change).
- **Modify:** `crates/tropical-gemm/src/core/kernel.rs` — change `Microkernel::execute` to take `*const T` (not `*const T::Scalar`); update `PortableMicrokernel`.
- **Modify:** `crates/tropical-gemm/src/core/packing.rs` — relax `pack_a`/`pack_b` bound to `T: Copy + Default`.
- **Modify:** `crates/tropical-gemm/src/core/gemm.rs` — thread `*const T` internally; callers can still pass `*const T::Scalar` via the bridge.
- **Modify:** `crates/tropical-gemm/src/simd/dispatch.rs` — compound elements route to `PortableMicrokernel` unconditionally; existing SIMD paths use the bridge.
- **Modify:** `crates/tropical-gemm/src/simd/kernels/portable.rs` — signature update to match new kernel trait.
- **Modify:** `crates/tropical-gemm/src/mat/ref_.rs` — `MatRef` stores `&[S]` not `&[S::Scalar]`; drop the unsafe transmute in `from_mat`; safe `from_slice(&[S::Scalar])` now gated on `S: ReprTransparentTropical`.
- **Modify:** `crates/tropical-gemm/src/mat/owned.rs` — `Mat::as_ref` uses the safe bridge.
- **Modify:** `crates/tropical-gemm/src/api.rs` — public `tropical_matmul` stays source-compatible via the bridge; add `tropical_matmul_t` for compound elements.
- **Create:** `crates/tropical-gemm/tests/counting_compose.rs` — integration tests for `CountingTropical<T, u64, Max>` and `Min`.

---

## Phase 1 — Direction marker + `CountingTropical` generics

Purely additive type work. No GEMM pipeline changes yet.

### Task 1: Add `TropicalDirection` trait

**Files:**
- Create: `crates/tropical-gemm/src/types/direction.rs`
- Modify: `crates/tropical-gemm/src/types/mod.rs`

- [ ] **Step 1: Write the failing test**

Create `crates/tropical-gemm/src/types/direction.rs`:

```rust
//! Direction marker for CountingTropical: selects Max or Min tropical semantics.

use super::scalar::TropicalScalar;

/// Marker trait selecting tropical direction (max or min).
pub trait TropicalDirection:
    Copy + Clone + Default + std::fmt::Debug + PartialEq + Send + Sync + 'static
{
    /// The tropical zero (additive identity) for this direction in scalar `T`.
    fn zero_value<T: TropicalScalar>() -> T;

    /// True iff `candidate` is strictly better than `incumbent` for this direction.
    fn is_strictly_better<T: TropicalScalar>(candidate: T, incumbent: T) -> bool;
}

/// Maximization direction: zero = -inf, larger is better.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct Max;

/// Minimization direction: zero = +inf, smaller is better.
#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct Min;

impl TropicalDirection for Max {
    #[inline(always)]
    fn zero_value<T: TropicalScalar>() -> T { T::neg_infinity() }
    #[inline(always)]
    fn is_strictly_better<T: TropicalScalar>(c: T, i: T) -> bool { c > i }
}

impl TropicalDirection for Min {
    #[inline(always)]
    fn zero_value<T: TropicalScalar>() -> T { T::pos_infinity() }
    #[inline(always)]
    fn is_strictly_better<T: TropicalScalar>(c: T, i: T) -> bool { c < i }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_prefers_larger() {
        assert!(Max::is_strictly_better::<f64>(5.0, 3.0));
        assert!(!Max::is_strictly_better::<f64>(3.0, 5.0));
        assert!(!Max::is_strictly_better::<f64>(3.0, 3.0));
    }

    #[test]
    fn min_prefers_smaller() {
        assert!(Min::is_strictly_better::<f64>(3.0, 5.0));
        assert!(!Min::is_strictly_better::<f64>(5.0, 3.0));
        assert!(!Min::is_strictly_better::<f64>(3.0, 3.0));
    }

    #[test]
    fn zero_values() {
        assert!(Max::zero_value::<f64>().is_infinite() && Max::zero_value::<f64>() < 0.0);
        assert!(Min::zero_value::<f64>().is_infinite() && Min::zero_value::<f64>() > 0.0);
    }
}
```

Add to `crates/tropical-gemm/src/types/mod.rs`:

```rust
pub mod direction;
pub use direction::{Max, Min, TropicalDirection};
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test -p tropical-gemm types::direction -- --nocapture`
Expected: 3 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm/src/types/direction.rs crates/tropical-gemm/src/types/mod.rs
git commit -m "Add TropicalDirection marker trait with Max/Min impls"
```

---

### Task 2: Parameterize `CountingTropical` by direction

**Files:**
- Modify: `crates/tropical-gemm/src/types/counting.rs`

- [ ] **Step 1: Write the failing test**

Append to the test module in `crates/tropical-gemm/src/types/counting.rs`:

```rust
#[test]
fn counting_min_prefers_smaller_value() {
    use super::super::direction::Min;
    let a = CountingTropical::<f64, f64, Min>::new(3.0, 2.0);
    let b = CountingTropical::<f64, f64, Min>::new(5.0, 7.0);
    let r = a.tropical_add(b);
    assert_eq!(r.value, 3.0);
    assert_eq!(r.count, 2.0);
}

#[test]
fn counting_min_merges_on_tie() {
    use super::super::direction::Min;
    let a = CountingTropical::<f64, f64, Min>::new(3.0, 2.0);
    let b = CountingTropical::<f64, f64, Min>::new(3.0, 5.0);
    let r = a.tropical_add(b);
    assert_eq!(r.value, 3.0);
    assert_eq!(r.count, 7.0);
}

#[test]
fn counting_min_zero_is_pos_infinity() {
    use super::super::direction::Min;
    let z = CountingTropical::<f64, f64, Min>::tropical_zero();
    assert!(z.value.is_infinite() && z.value > 0.0);
    assert_eq!(z.count, 0.0);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test -p tropical-gemm counting_min`
Expected: compile error — `CountingTropical` does not take 3 generics.

- [ ] **Step 3: Edit `CountingTropical` to add `D: TropicalDirection = Max`**

In `crates/tropical-gemm/src/types/counting.rs`:

Replace the struct declaration (currently lines 19-26) with:

```rust
use super::direction::{Max, TropicalDirection};

#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
pub struct CountingTropical<
    T: TropicalScalar,
    C: TropicalScalar = T,
    D: TropicalDirection = Max,
> {
    pub value: T,
    pub count: C,
    _dir: std::marker::PhantomData<D>,
}
```

Update the three `impl` blocks to carry the `D: TropicalDirection` bound (currently unbounded `<T, C>` becomes `<T, C, D: TropicalDirection>`), and add `_dir: PhantomData` to every `Self { … }` literal.

Update `tropical_zero` (currently lines 49-54):

```rust
fn tropical_zero() -> Self {
    Self { value: D::zero_value::<T>(), count: C::scalar_zero(), _dir: PhantomData }
}
```

Update `tropical_add` (currently lines 64-77) to use `D::is_strictly_better`:

```rust
fn tropical_add(self, rhs: Self) -> Self {
    if D::is_strictly_better(self.value, rhs.value) { self }
    else if D::is_strictly_better(rhs.value, self.value) { rhs }
    else {
        Self {
            value: self.value,
            count: self.count.scalar_add(rhs.count),
            _dir: PhantomData,
        }
    }
}
```

Update `tropical_add_argmax` similarly.

Update `from_value`, `new`, `From<T>`, `Default` impls to include `_dir: PhantomData<D>` field — for these constructors where `D` is not otherwise fixed, use `std::marker::PhantomData`. Add `use std::marker::PhantomData;` at the top of the file.

- [ ] **Step 4: Verify all existing tests still pass (default `D = Max`)**

Run: `cargo test -p tropical-gemm counting`
Expected: all existing tests pass (they use the `Max` default) plus the three new `counting_min_*` tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/tropical-gemm/src/types/counting.rs
git commit -m "Parameterize CountingTropical by TropicalDirection (default Max)"
```

---

### Task 3: Fix the `SIMD_AVAILABLE` claim

**Files:**
- Modify: `crates/tropical-gemm/src/types/counting.rs:120-124`

- [ ] **Step 1: Update the `SimdTropical` impl for `CountingTropical`**

In `crates/tropical-gemm/src/types/counting.rs`, replace:

```rust
impl<T: TropicalScalar, C: TropicalScalar, D: TropicalDirection> SimdTropical for CountingTropical<T, C, D> {
    const SIMD_AVAILABLE: bool = false;
    const SIMD_WIDTH: usize = 1;
}
```

(The old `SIMD_AVAILABLE = true`, `SIMD_WIDTH = 8` was aspirational and incorrect — there is no SIMD kernel for this type yet.)

Update the `test_simd_tropical` test accordingly:

```rust
#[test]
fn test_simd_tropical() {
    assert!(!CountingTropical::<f64>::SIMD_AVAILABLE);
    assert_eq!(CountingTropical::<f64>::SIMD_WIDTH, 1);
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p tropical-gemm counting::tests::test_simd_tropical`
Expected: pass.

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm/src/types/counting.rs
git commit -m "Correct SIMD_AVAILABLE=false for CountingTropical (no SIMD kernel exists)"
```

---

## Phase 2 — Widen the internal GEMM ABI

The core pipeline currently threads `*const T::Scalar` and transmutes under a `#[repr(transparent)]` assumption. For compound elements (`CountingTropical`) this is unsound. We widen the internals to thread `*const T` directly. Single-scalar semirings continue to work at runtime because the compiler sees the same layout; we expose the layout contract explicitly via a new `ReprTransparentTropical` trait.

**This phase is the biggest edit.** Multiple files must land together. The test oracle is the full existing test suite — no new tests are added until Phase 3. If any existing test breaks after Phase 2, the refactor has a bug.

### Task 4: Add `ReprTransparentTropical` bridge trait

**Files:**
- Modify: `crates/tropical-gemm/src/types/traits.rs`
- Modify: `crates/tropical-gemm/src/types/max_plus.rs`
- Modify: `crates/tropical-gemm/src/types/min_plus.rs`
- Modify: `crates/tropical-gemm/src/types/max_mul.rs`
- Modify: `crates/tropical-gemm/src/types/and_or.rs`
- Modify: `crates/tropical-gemm/src/types/mod.rs`

- [ ] **Step 1: Define the trait**

Append to `crates/tropical-gemm/src/types/traits.rs`:

```rust
/// Marker trait: `Self` has identical memory layout to `Self::Scalar`.
///
/// # Safety
/// Implementors must be `#[repr(transparent)]` newtype wrappers over
/// exactly one field of type `Self::Scalar`. This allows safe
/// reinterpretation of `&[Scalar]` as `&[Self]` and vice versa.
pub unsafe trait ReprTransparentTropical: TropicalSemiring {}
```

- [ ] **Step 2: Implement for all single-scalar semirings**

In each of `max_plus.rs`, `min_plus.rs`, `max_mul.rs`, `and_or.rs` — after the existing `impl TropicalSemiring for …` block, add:

```rust
// Safety: this type is #[repr(transparent)] over its scalar field.
unsafe impl crate::types::traits::ReprTransparentTropical for TropicalMaxPlus<T> {}
```

(substitute the correct type name per file). Verify each type is already `#[repr(transparent)]` in its definition; if not, add it.

Export from `types/mod.rs`:

```rust
pub use traits::{ReprTransparentTropical, SimdTropical, TropicalSemiring, TropicalWithArgmax};
```

- [ ] **Step 3: Verify compilation**

Run: `cargo build -p tropical-gemm`
Expected: clean build.

- [ ] **Step 4: Commit**

```bash
git add crates/tropical-gemm/src/types/
git commit -m "Add ReprTransparentTropical marker for single-scalar semirings"
```

---

### Task 5: Widen `Microkernel::execute` to `*const T`

**Files:**
- Modify: `crates/tropical-gemm/src/core/kernel.rs`

- [ ] **Step 1: Change the trait signature and `PortableMicrokernel` impl**

In `crates/tropical-gemm/src/core/kernel.rs`:

Replace the `Microkernel` trait declaration (lines 7-34) so `a` and `b` are `*const T` instead of `*const T::Scalar`. Update the `PortableMicrokernel` impl (lines 72-116) to read `T` directly:

```rust
pub trait Microkernel<T: TropicalSemiring> {
    const MR: usize;
    const NR: usize;

    /// # Safety
    /// - `a` points to at least `mr * k` elements of `T` (packed column-major in MR chunks)
    /// - `b` points to at least `k * nr` elements of `T` (packed row-major in NR chunks)
    /// - `c` points to at least `mr * ldc` elements of `T`
    unsafe fn execute(
        &self,
        mr: usize, nr: usize, k: usize,
        a: *const T, b: *const T,
        c: *mut T, ldc: usize,
    );
}

impl<T: TropicalSemiring> Microkernel<T> for PortableMicrokernel {
    const MR: usize = 4;
    const NR: usize = 4;

    unsafe fn execute(
        &self,
        mr: usize, nr: usize, k: usize,
        a: *const T, b: *const T,
        c: *mut T, ldc: usize,
    ) {
        const MR: usize = 4;
        const NR: usize = 4;

        let mut acc = [[T::tropical_zero(); NR]; MR];
        for i in 0..mr {
            for j in 0..nr {
                acc[i][j] = *c.add(i * ldc + j);
            }
        }

        for p in 0..k {
            for i in 0..mr {
                let a_val: T = *a.add(p * MR + i);
                for j in 0..nr {
                    let b_val: T = *b.add(p * NR + j);
                    acc[i][j] = acc[i][j].tropical_add(a_val.tropical_mul(b_val));
                }
            }
        }

        for i in 0..mr {
            for j in 0..nr {
                *c.add(i * ldc + j) = acc[i][j];
            }
        }
    }
}
```

Do the same for `MicrokernelWithArgmax::execute_with_argmax`: `a: *const T`, `b: *const T` instead of `T::Scalar`; drop the `T::from_scalar` wrapping in the impl.

- [ ] **Step 2: Update the unit tests in this file**

Existing tests (`test_portable_kernel` and friends) pass `[f64; 12]` arrays and cast `.as_ptr()` to `*const f64`. Change them to build `[TropicalMaxPlus<f64>; 12]` arrays instead — e.g. `[TropicalMaxPlus(1.0), TropicalMaxPlus(4.0), …]` — and pass `*const TropicalMaxPlus<f64>`.

For example, replace (currently lines 183-191):

```rust
let a: [f64; 12] = [1.0, 4.0, 0.0, 0.0, 2.0, 5.0, 0.0, 0.0, 3.0, 6.0, 0.0, 0.0];
let b: [f64; 12] = [1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0];
```

with:

```rust
let a = [1.0_f64, 4.0, 0.0, 0.0, 2.0, 5.0, 0.0, 0.0, 3.0, 6.0, 0.0, 0.0]
    .map(TropicalMaxPlus);
let b = [1.0_f64, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0]
    .map(TropicalMaxPlus);
```

(and analogous for the MinPlus, MaxMul, argmax tests). Ensure the expected results stay the same (`c[0].0 == 8.0` etc. — `.0` reads the scalar field of the newtype).

- [ ] **Step 3: Verify**

Run: `cargo test -p tropical-gemm core::kernel`
Expected: all kernel tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/tropical-gemm/src/core/kernel.rs
git commit -m "Widen Microkernel::execute from *const T::Scalar to *const T"
```

---

### Task 6: Relax `pack_a` / `pack_b` element bound

**Files:**
- Modify: `crates/tropical-gemm/src/core/packing.rs`

- [ ] **Step 1: Change bounds**

In `pack_a` (currently line 46) and `pack_b` (line 115), change the generic bound from `T: TropicalScalar` to `T: Copy + Default`. Replace the `T::scalar_zero()` padding (line 56) with `T::default()`.

All internal calls `get_element` stay unchanged — it's pure pointer arithmetic over `T`.

- [ ] **Step 2: Verify**

Run: `cargo build -p tropical-gemm`
Expected: compile error at call sites in `core/gemm.rs` (those will be fixed in Task 7). The `pack_a`/`pack_b` functions themselves should compile.

Run `cargo test -p tropical-gemm core::packing` — packing-internal unit tests, if any, should pass.

- [ ] **Step 3: Commit (will not fully build yet — gemm.rs still uses the old signature)**

Do not commit until Task 7 lands. Move to Task 7.

---

### Task 7: Thread `*const T` through `tropical_gemm_inner`

**Files:**
- Modify: `crates/tropical-gemm/src/core/gemm.rs`

- [ ] **Step 1: Change `tropical_gemm_portable` / `tropical_gemm_inner` signatures**

Change `a: *const T::Scalar, b: *const T::Scalar` to `a: *const T, b: *const T` throughout `tropical_gemm_portable` (line 34, 37) and `tropical_gemm_inner` (line 58, 61). The `T: TropicalSemiring` bound covers `T: Copy` via the trait's supertrait; ensure `Default` is satisfied by adding a `T: Default` bound where `pack_a`/`pack_b` are called (or change those functions to take a zero-filling closure — simpler: add `T: Default` bound; every `TropicalSemiring` impl has `Default` via `tropical_zero` but that's a method, not an impl — so add `Default` as a supertrait of `TropicalSemiring` OR require it locally).

**Decision:** require `T: Default` locally at the `tropical_gemm_inner` call site (the tighter bound is closer to the need and avoids widening the public trait). Add it to the function signature: `pub unsafe fn tropical_gemm_inner<T: TropicalSemiring + Default, K: Microkernel<T>>`.

Change `packed_a: Vec<T::Scalar>` to `packed_a: Vec<T>` (line 78); initializer uses `T::tropical_zero()` (semantically the additive identity, semantics-appropriate for padding). Same for `packed_b` (line 79).

Call sites (`pack_b::<T::Scalar>` at line 87, `pack_a::<T::Scalar>` at line 101) become `pack_b::<T>` and `pack_a::<T>`. The `b_panel_ptr` / `a_panel_ptr` helpers (used at lines 90, 104) need their signatures changed similarly: they should return `*const T` and accept `*const T`. Find them in the same file and update.

- [ ] **Step 2: Update `Default` impls**

Each of `TropicalMaxPlus`, `TropicalMinPlus`, `TropicalMaxMul`, `TropicalAndOr` already implements `Default` via `derive`, returning the scalar default. Confirm this matches `tropical_zero` for each type — for `TropicalMaxPlus<f32>`, scalar default is `0.0` but `tropical_zero` is `-inf`. This is a semantic mismatch for padding, but as noted in the design, padded cells are never read by the kernel, so `Default::default() = 0.0` is functionally fine (just not semantically crisp).

Adjust the packed buffer init (line 78-79) to use `Default::default()` instead of `T::tropical_zero()` to match the `T: Default` bound and keep the code simple:

```rust
let mut packed_a = vec![T::default(); packed_a_size(params.mc, params.kc, K::MR)];
let mut packed_b = vec![T::default(); packed_b_size(params.kc, params.nc, K::NR)];
```

- [ ] **Step 3: Verify**

Run: `cargo build -p tropical-gemm`
Expected: further errors at `api.rs` and `simd/dispatch.rs` call sites. Those are fixed in Tasks 8–10.

- [ ] **Step 4: Do not commit — proceed to Task 8**

---

### Task 8: Update SIMD dispatch and portable SIMD kernel wrapper

**Files:**
- Modify: `crates/tropical-gemm/src/simd/dispatch.rs`
- Modify: `crates/tropical-gemm/src/simd/kernels/portable.rs`

- [ ] **Step 1: Widen signatures**

In `simd/dispatch.rs`, the public `tropical_gemm_dispatch` function takes `*const T::Scalar` pointers (mirroring `tropical_gemm_portable`). Widen it to `*const T`. Propagate to the `KernelDispatch` trait.

In `simd/kernels/portable.rs`, the SIMD-portable kernel is just a re-wrapper around `PortableMicrokernel`. Update its `Microkernel` trait impl signature to match Task 5.

- [ ] **Step 2: Verify**

Run: `cargo build -p tropical-gemm`
Expected: errors narrow to `api.rs` and `mat/*.rs`.

- [ ] **Step 3: Do not commit — proceed to Task 9**

---

### Task 9: Update `MatRef` and `Mat::as_ref`

**Files:**
- Modify: `crates/tropical-gemm/src/mat/ref_.rs`
- Modify: `crates/tropical-gemm/src/mat/owned.rs`
- Modify: `crates/tropical-gemm/src/mat/ops.rs`

- [ ] **Step 1: Change the `MatRef` field type**

In `crates/tropical-gemm/src/mat/ref_.rs`:

Replace line 29 (`data: &'a [S::Scalar]`) with `data: &'a [S]`.

Replace the `from_slice` constructor (line 47) signature: takes `data: &'a [S]` directly. Add a second constructor `from_scalar_slice` gated on `S: ReprTransparentTropical` that accepts `&'a [S::Scalar]` and casts under the layout guarantee:

```rust
impl<'a, S: ReprTransparentTropical> MatRef<'a, S> {
    pub fn from_scalar_slice(data: &'a [S::Scalar], nrows: usize, ncols: usize) -> Self {
        assert_eq!(data.len(), nrows * ncols);
        // Safety: S: ReprTransparentTropical guarantees same layout as S::Scalar.
        let view = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const S, data.len())
        };
        Self { data: view, nrows, ncols, _phantom: PhantomData }
    }
}
```

Delete `MatRef::from_mat` (the old unsafe transmute at lines 67-88). Replace its sole caller (`Mat::as_ref`) with a safe construction from `&[S]`:

In `mat/owned.rs`, update `as_ref` (lines 184-193):

```rust
pub fn as_ref(&self) -> MatRef<'_, S> {
    MatRef::from_direct_slice(&self.data, self.nrows, self.ncols)
}
```

And add to `MatRef`:

```rust
pub(crate) fn from_direct_slice(data: &'a [S], nrows: usize, ncols: usize) -> Self {
    Self { data, nrows, ncols, _phantom: PhantomData }
}
```

- [ ] **Step 2: Update `MatRef` accessors**

Any method that returns `S::Scalar` from `self.data` (e.g. `.get(i, j)`) should now return `S` (or `S::value()` if scalar is still wanted). Grep `mat/ref_.rs` for `data[` access patterns and update. If the public API returned `S::Scalar`, add a convenience `pub fn get_value(&self, i, j) -> S::Scalar { self.get(i, j).value() }` to preserve source-compat for call sites.

Same for `MatMut` in `mat/ops.rs` — the mut view follows the same shape.

- [ ] **Step 3: Update matmul dispatch in `mat/ops.rs`**

Any place that passes `MatRef::data.as_ptr()` to `tropical_gemm_dispatch` now passes `*const S` (matches Task 8). No cast needed.

- [ ] **Step 4: Verify**

Run: `cargo build -p tropical-gemm`
Expected: errors remain only in `api.rs`.

- [ ] **Step 5: Do not commit — proceed to Task 10**

---

### Task 10: Update `api.rs` public entry points

**Files:**
- Modify: `crates/tropical-gemm/src/api.rs`

- [ ] **Step 1: Keep existing scalar-slice API via the bridge; add `tropical_matmul_t` for compound elements**

For `tropical_matmul` (line 33), keep the signature `fn tropical_matmul<T: TropicalSemiring + KernelDispatch>(a: &[T::Scalar], …) -> Vec<T>` but add a `T: ReprTransparentTropical` bound. Cast inside:

```rust
pub fn tropical_matmul<T>(a: &[T::Scalar], m: usize, k: usize, b: &[T::Scalar], n: usize) -> Vec<T>
where
    T: TropicalSemiring + KernelDispatch + ReprTransparentTropical + Default,
{
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![T::tropical_zero(); m * n];
    // Safety: T: ReprTransparentTropical means &[T::Scalar] and &[T] have identical layout.
    unsafe {
        let a_t = a.as_ptr() as *const T;
        let b_t = b.as_ptr() as *const T;
        tropical_gemm_dispatch::<T>(m, n, k, a_t, k, Transpose::NoTrans,
                                    b_t, n, Transpose::NoTrans, c.as_mut_ptr(), n);
    }
    c
}
```

Add a new function `tropical_matmul_t` that takes `&[T]` directly (no bridge required — works for CountingTropical too):

```rust
pub fn tropical_matmul_t<T>(a: &[T], m: usize, k: usize, b: &[T], n: usize) -> Vec<T>
where
    T: TropicalSemiring + KernelDispatch + Default,
{
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![T::tropical_zero(); m * n];
    unsafe {
        tropical_gemm_dispatch::<T>(m, n, k, a.as_ptr(), k, Transpose::NoTrans,
                                    b.as_ptr(), n, Transpose::NoTrans, c.as_mut_ptr(), n);
    }
    c
}
```

Do the same for `tropical_matmul_with_argmax`, `tropical_matmul_batched`, `TropicalGemm::execute`, etc. — add the `ReprTransparentTropical` bound on the existing scalar-slice APIs.

Re-export `tropical_matmul_t` from `lib.rs`.

- [ ] **Step 2: Verify full build**

Run: `cargo build -p tropical-gemm`
Expected: clean build.

- [ ] **Step 3: Run the full test suite — this is the core regression gate**

Run: `cargo test -p tropical-gemm`
Expected: every existing test passes. Any failure here means the ABI refactor introduced a bug. Common failure modes: forgot to update a `*const T::Scalar` cast; `Default::default()` mismatch for a type where that matters; PhantomData missing somewhere.

- [ ] **Step 4: Commit the whole Phase 2 refactor**

```bash
git add crates/tropical-gemm
git commit -m "Thread *const T (not *const T::Scalar) through GEMM pipeline

Widens the internal kernel/packing/dispatch ABI so compound element
types (CountingTropical) can flow end-to-end. Single-scalar semirings
preserve their scalar-slice public API via the new
ReprTransparentTropical bridge. No runtime behavior change for
existing types."
```

---

## Phase 3 — Wire `CountingTropical` end-to-end

### Task 11: SIMD dispatch routes CountingTropical to PortableMicrokernel

**Files:**
- Modify: `crates/tropical-gemm/src/simd/dispatch.rs`

- [ ] **Step 1: Ensure `KernelDispatch` has a blanket impl that falls back to `PortableMicrokernel`**

`KernelDispatch` is already implemented for `TropicalMaxPlus`, etc. Add an impl for `CountingTropical`:

```rust
impl<T, C, D> KernelDispatch for CountingTropical<T, C, D>
where
    T: TropicalScalar, C: TropicalScalar, D: TropicalDirection,
{
    // Same signature shape as existing impls; route to portable.
    unsafe fn tropical_gemm_dispatch(
        m: usize, n: usize, k: usize,
        a: *const Self, lda: usize, trans_a: Transpose,
        b: *const Self, ldb: usize, trans_b: Transpose,
        c: *mut Self, ldc: usize,
    ) {
        crate::core::tropical_gemm_portable::<Self>(
            m, n, k, a, lda, trans_a, b, ldb, trans_b, c, ldc,
        );
    }
}
```

(The exact trait shape mirrors what's in `simd/dispatch.rs` after Task 8 — match its signature exactly. Import paths per the actual file.)

- [ ] **Step 2: Verify**

Run: `cargo build -p tropical-gemm`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm/src/simd/dispatch.rs
git commit -m "Dispatch CountingTropical matmul to PortableMicrokernel"
```

---

### Task 12: End-to-end integration test — `Max` direction

**Files:**
- Create: `crates/tropical-gemm/tests/counting_compose.rs`

- [ ] **Step 1: Write a failing integration test**

Create `crates/tropical-gemm/tests/counting_compose.rs`:

```rust
use tropical_gemm::{tropical_matmul_t, CountingTropical, Max, Min, TropicalSemiring};

fn ct_max(v: f32, c: u64) -> CountingTropical<f32, u64, Max> {
    CountingTropical::new(v, c)
}

#[test]
fn counting_tropical_max_small_matmul() {
    // A is 2x3, B is 3x2, row-major
    // All input counts = 1.
    let a = [
        ct_max(1.0, 1), ct_max(2.0, 1), ct_max(3.0, 1),
        ct_max(4.0, 1), ct_max(5.0, 1), ct_max(6.0, 1),
    ];
    let b = [
        ct_max(1.0, 1), ct_max(2.0, 1),
        ct_max(3.0, 1), ct_max(4.0, 1),
        ct_max(5.0, 1), ct_max(6.0, 1),
    ];

    let c = tropical_matmul_t::<CountingTropical<f32, u64, Max>>(&a, 2, 3, &b, 2);

    // C[0,0] = max_k(A[0,k] + B[k,0]) = max(1+1, 2+3, 3+5) = 8, unique maximizer k=2 → count=1
    assert_eq!(c[0].value, 8.0);
    assert_eq!(c[0].count, 1);

    // C[0,1] = max(1+2, 2+4, 3+6) = 9, unique k=2 → count=1
    assert_eq!(c[1].value, 9.0);
    assert_eq!(c[1].count, 1);

    // C[1,0] = max(4+1, 5+3, 6+5) = 11, unique k=2 → count=1
    assert_eq!(c[2].value, 11.0);
    assert_eq!(c[2].count, 1);

    // C[1,1] = max(4+2, 5+4, 6+6) = 12, unique k=2 → count=1
    assert_eq!(c[3].value, 12.0);
    assert_eq!(c[3].count, 1);
}

#[test]
fn counting_tropical_max_merges_ties() {
    // Construct so that C[0,0] has two optima with count-sum > 1.
    // A = [2, 3]; B = [3, 2] → A[0,0]+B[0,0]=5, A[0,1]+B[1,0]=5. Both tie at 5.
    let a = [ct_max(2.0, 1), ct_max(3.0, 1)];
    let b = [ct_max(3.0, 1), ct_max(2.0, 1)];

    let c = tropical_matmul_t::<CountingTropical<f32, u64, Max>>(&a, 1, 2, &b, 1);

    assert_eq!(c[0].value, 5.0);
    assert_eq!(c[0].count, 2);  // both k contribute
}

#[test]
fn counting_tropical_max_multiplies_counts() {
    // Single-k case: C = A*B where input counts multiply.
    // A = [(3.0, 2)], B = [(4.0, 5)] → C = [(7.0, 10)]
    let a = [CountingTropical::<f32, u64, Max>::new(3.0, 2)];
    let b = [CountingTropical::<f32, u64, Max>::new(4.0, 5)];

    let c = tropical_matmul_t::<CountingTropical<f32, u64, Max>>(&a, 1, 1, &b, 1);
    assert_eq!(c[0].value, 7.0);
    assert_eq!(c[0].count, 10);
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p tropical-gemm --test counting_compose`
Expected: all three pass.

If they do not pass, the most likely culprits are (a) `Default` not returning `tropical_zero` for CountingTropical (which would bias accumulator initialization — check the kernel's accumulator setup which reads from C, and C was initialized in `tropical_matmul_t` with `T::tropical_zero()`, so this should be fine), (b) a missed pointer-type update in Phase 2.

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm/tests/counting_compose.rs
git commit -m "Add CountingTropical<_, u64, Max> end-to-end matmul tests"
```

---

### Task 13: End-to-end integration test — `Min` direction

**Files:**
- Modify: `crates/tropical-gemm/tests/counting_compose.rs`

- [ ] **Step 1: Append Min tests**

Append to `crates/tropical-gemm/tests/counting_compose.rs`:

```rust
fn ct_min(v: f32, c: u64) -> CountingTropical<f32, u64, Min> {
    CountingTropical::new(v, c)
}

#[test]
fn counting_tropical_min_small_matmul() {
    // Same A, B as max test; expected min instead of max along k.
    let a = [
        ct_min(1.0, 1), ct_min(2.0, 1), ct_min(3.0, 1),
        ct_min(4.0, 1), ct_min(5.0, 1), ct_min(6.0, 1),
    ];
    let b = [
        ct_min(1.0, 1), ct_min(2.0, 1),
        ct_min(3.0, 1), ct_min(4.0, 1),
        ct_min(5.0, 1), ct_min(6.0, 1),
    ];

    let c = tropical_matmul_t::<CountingTropical<f32, u64, Min>>(&a, 2, 3, &b, 2);

    // C[0,0] = min(1+1, 2+3, 3+5) = 2, unique minimizer k=0 → count=1
    assert_eq!(c[0].value, 2.0);
    assert_eq!(c[0].count, 1);
    // C[0,1] = min(1+2, 2+4, 3+6) = 3, unique k=0 → count=1
    assert_eq!(c[1].value, 3.0);
    assert_eq!(c[1].count, 1);
    // C[1,0] = min(4+1, 5+3, 6+5) = 5, unique k=0 → count=1
    assert_eq!(c[2].value, 5.0);
    assert_eq!(c[2].count, 1);
    // C[1,1] = min(4+2, 5+4, 6+6) = 6, unique k=0 → count=1
    assert_eq!(c[3].value, 6.0);
    assert_eq!(c[3].count, 1);
}

#[test]
fn counting_tropical_min_merges_ties() {
    // A = [2, 3]; B = [3, 2] → A[0,0]+B[0,0]=5, A[0,1]+B[1,0]=5. Both tie at 5 (still min).
    let a = [ct_min(2.0, 1), ct_min(3.0, 1)];
    let b = [ct_min(3.0, 1), ct_min(2.0, 1)];
    let c = tropical_matmul_t::<CountingTropical<f32, u64, Min>>(&a, 1, 2, &b, 1);
    assert_eq!(c[0].value, 5.0);
    assert_eq!(c[0].count, 2);
}
```

- [ ] **Step 2: Run**

Run: `cargo test -p tropical-gemm --test counting_compose`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm/tests/counting_compose.rs
git commit -m "Add CountingTropical<_, u64, Min> end-to-end matmul tests"
```

---

### Task 14: Run the full workspace test suite as a final gate

- [ ] **Step 1: Run everything**

Run: `cargo test --workspace`
Expected: every test in every crate passes. `tropical-gemm-cuda` and `tropical-gemm-python` are expected to compile unchanged (they use the scalar-slice public API, which retained source compatibility via the `ReprTransparentTropical` bridge).

If any test fails, investigate and fix before calling spec A done.

- [ ] **Step 2: No commit needed unless fixes required**

---

## Out of scope for this plan

- `Mod<const P: i32>` scalar type, CRT driver, BigInt counts — spec B.
- Python bindings for counting — spec B.
- CUDA support for CountingTropical — later follow-up.
- AVX/NEON SIMD lanes for the counting kernel — later follow-up.

---

## Self-review notes

- **Spec coverage:** direction marker (Task 1), CountingTropical<D> parameterization (Task 2), SIMD lie correction (Task 3), ABI widening (Tasks 4-10), dispatch wiring (Task 11), end-to-end tests both directions (Tasks 12-13), regression gate (Task 14). Spec A §3 "kernel/packing changes" → Tasks 5-8. Spec A §4 "nothing else changes" (no CRT, Python, CUDA) → listed in "Out of scope".
- **Placeholder scan:** no TBDs. Task 9 contains judgment calls about how to preserve `MatRef`'s external API; the plan picks a concrete shape (add `from_scalar_slice` gated on `ReprTransparentTropical`, rename old `from_slice` to `from_direct_slice` for the `&[S]` case).
- **Type consistency:** `tropical_matmul_t` is used consistently (Tasks 10, 12, 13). `ReprTransparentTropical` bound appears in Tasks 4, 9, 10. `TropicalDirection` and `Max`/`Min` used consistently (Tasks 1, 2, 11, 12, 13).
- **Risk callouts:** Phase 2 is a multi-file atomic refactor; Tasks 6-10 cannot be committed individually (the codebase won't build between them). The plan groups them into a single Phase 2 commit at Task 10 step 4 after the whole refactor compiles and existing tests pass.
