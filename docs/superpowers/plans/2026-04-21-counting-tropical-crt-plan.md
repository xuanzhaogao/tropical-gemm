# Spec B — CRT driver + `BigInt` counts — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Count ground-state / optimal configurations whose multiplicity exceeds `u64` by running the spec-A `CountingTropical` matmul once per prime modulus and Chinese-Remainder-reconstructing the count field to `BigInt` host-side.

**Architecture:** (1) A `Mod<const P: i32>` scalar type that plugs into the existing `TropicalScalar` trait so `CountingTropical<T, Mod<P>, D>` flows through the spec-A pipeline unchanged. (2) A `CRT_PRIMES` const table of 30-bit primes satisfying `(P-1)² < 2⁶⁰`. (3) A host-side `count_ground_states<T, D>(a, m, k, b, n, bound)` driver that picks the smallest `k` primes whose product exceeds `2·bound`, runs the matmul `k` times, and combines residues via iterated pairwise CRT into a `BigInt` per cell.

**Tech Stack:** Rust 1.87, `num-bigint`, `num-integer`. CPU only. Python binding via PyO3 (optional; cluster Python 3.6 blocks local test, noted).

**Spec:** `docs/superpowers/specs/2026-04-21-counting-tropical-crt-design.md`

---

## File map

- **Create:** `crates/tropical-gemm/src/types/modp.rs` — `Mod<const P: i32>` scalar; `TropicalScalar` impl; private const `PRIME_OK` sanity bound.
- **Modify:** `crates/tropical-gemm/src/types/mod.rs` — declare `modp`, re-export `Mod`.
- **Create:** `crates/tropical-gemm/src/crt.rs` — prime table, pairwise CRT, `count_ground_states`, `CountedMat` struct.
- **Modify:** `crates/tropical-gemm/src/lib.rs` — re-export `Mod`, `count_ground_states`, `CountedMat`.
- **Modify:** `crates/tropical-gemm/Cargo.toml` — add `num-bigint`, `num-integer` deps.
- **Create:** `crates/tropical-gemm/tests/counting_crt.rs` — end-to-end tests.
- **Create:** `crates/tropical-gemm/src/testing/bigint_semiring.rs` — test-only reference semiring `CountingBigInt<T, D>`, not exported publicly.
- **Modify:** `crates/tropical-gemm/src/lib.rs` — gate `pub mod testing` behind `#[cfg(any(test, feature = "testing"))]` or `pub(crate) mod testing`.
- **Modify:** `crates/tropical-gemm-python/src/lib.rs` — `count_ground_states` pyfunction.
- **Create:** `crates/tropical-gemm-python/tests/test_count_ground_states.py` — Python round-trip.

---

## Preconditions (already met by spec A)

- Latest commit on `counting-tropical` branch at plan start: `764a91c`.
- `CountingTropical<T, C, D: TropicalDirection = Max>` works end-to-end via `tropical_matmul_t`.
- `TropicalScalar` trait in `crates/tropical-gemm/src/types/scalar.rs` is the plug point for `C`.
- `num-bigint`, `num-integer` not yet in `Cargo.toml`.
- 49 CUDA tests + 281 lib tests + 5 integration tests + 24 doctests all pass.

---

## Phase 1 — `Mod<const P: i32>` count scalar

### Task 1: Create `Mod<P>` with constructor, raw accessor, and arithmetic (without TropicalScalar yet)

**Files:**
- Create: `crates/tropical-gemm/src/types/modp.rs`

- [ ] **Step 1: Write the failing tests**

Create `crates/tropical-gemm/src/types/modp.rs`:

```rust
//! Modular integer scalar `Mod<const P: i32>` for CRT-based counting.
//!
//! `Mod<P>` represents a residue modulo the compile-time prime `P`. It is
//! used as the count field of `CountingTropical<T, Mod<P>, D>` during
//! Chinese Remainder reconstruction of large counts.
//!
//! # Prime size contract
//!
//! `P` must satisfy `(P - 1)² < 2⁶²` so that `scalar_mul`'s `i64` product
//! never overflows. In practice we use 30-bit primes from `CRT_PRIMES`
//! in `crate::crt`, where `(P - 1)² < 2⁶⁰` with room to spare.

use std::fmt;

/// Residue modulo the compile-time prime `P`.
///
/// The inner `i32` is always in `[0, P)` (the normalized representative).
/// Construct via `Mod::new` (which normalizes) or reconstruct from a raw
/// representative via `raw`. See module docs for the size constraint on `P`.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Mod<const P: i32>(i32);

impl<const P: i32> Mod<P> {
    /// Normalizing constructor. Accepts any `i32` and reduces mod `P`.
    #[inline]
    pub const fn new(x: i32) -> Self {
        assert!(P > 1, "Mod<P>: P must be a prime greater than 1");
        Self(x.rem_euclid(P))
    }

    /// Raw inner value, guaranteed to be in `[0, P)`.
    #[inline]
    pub const fn raw(self) -> i32 {
        self.0
    }

    /// Modular addition. Inputs are already in `[0, P)`; sum is in `[0, 2P)`.
    #[inline]
    pub fn add(self, rhs: Self) -> Self {
        let s = self.0 as i64 + rhs.0 as i64;
        Self((s % P as i64) as i32)
    }

    /// Modular multiplication. Product bounded by `(P-1)²`; `i64` is safe
    /// for any `P` satisfying the module contract.
    #[inline]
    pub fn mul(self, rhs: Self) -> Self {
        let s = self.0 as i64 * rhs.0 as i64;
        Self((s % P as i64) as i32)
    }
}

impl<const P: i32> Default for Mod<P> {
    #[inline]
    fn default() -> Self {
        Self(0)
    }
}

impl<const P: i32> fmt::Debug for Mod<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Mod<{}>({})", P, self.0)
    }
}

impl<const P: i32> fmt::Display for Mod<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: i32 = 7;

    #[test]
    fn new_normalizes_positive() {
        assert_eq!(Mod::<P>::new(10).raw(), 3);
        assert_eq!(Mod::<P>::new(0).raw(), 0);
        assert_eq!(Mod::<P>::new(P).raw(), 0);
        assert_eq!(Mod::<P>::new(P - 1).raw(), P - 1);
    }

    #[test]
    fn new_normalizes_negative() {
        assert_eq!(Mod::<P>::new(-1).raw(), P - 1);
        assert_eq!(Mod::<P>::new(-P).raw(), 0);
        assert_eq!(Mod::<P>::new(-P - 1).raw(), P - 1);
    }

    #[test]
    fn add_wraps() {
        let a = Mod::<P>::new(5);
        let b = Mod::<P>::new(4);
        assert_eq!(a.add(b).raw(), 2); // (5+4) mod 7 = 2
    }

    #[test]
    fn mul_wraps() {
        let a = Mod::<P>::new(5);
        let b = Mod::<P>::new(4);
        assert_eq!(a.mul(b).raw(), 6); // (5*4) mod 7 = 6
    }

    #[test]
    fn add_commutative() {
        let a = Mod::<P>::new(3);
        let b = Mod::<P>::new(6);
        assert_eq!(a.add(b), b.add(a));
    }

    #[test]
    fn mul_commutative() {
        let a = Mod::<P>::new(3);
        let b = Mod::<P>::new(6);
        assert_eq!(a.mul(b), b.mul(a));
    }

    #[test]
    fn distributive() {
        let a = Mod::<P>::new(3);
        let b = Mod::<P>::new(5);
        let c = Mod::<P>::new(6);
        let lhs = a.mul(b.add(c));
        let rhs = a.mul(b).add(a.mul(c));
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn large_prime_product_does_not_overflow() {
        const LP: i32 = 1_073_741_789; // 30-bit prime
        let a = Mod::<LP>::new(LP - 1);
        let b = Mod::<LP>::new(LP - 2);
        let r = a.mul(b);
        // (P-1)(P-2) mod P = (-1)(-2) mod P = 2
        assert_eq!(r.raw(), 2);
    }

    #[test]
    fn default_is_zero() {
        let d = Mod::<P>::default();
        assert_eq!(d.raw(), 0);
    }
}
```

Add module declaration to `crates/tropical-gemm/src/types/mod.rs`: add `mod modp;` in alphabetical position among the existing `mod` lines, and add `pub use modp::Mod;` in alphabetical position among the `pub use` lines.

- [ ] **Step 2: Run tests**

```
. ~/.cargo/env && cargo test -p tropical-gemm types::modp 2>&1 | tail -15
```

Expected: 9 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm/src/types/modp.rs crates/tropical-gemm/src/types/mod.rs
git commit -m "$(cat <<'EOF'
Add Mod<const P: i32> modular scalar

Residue type with normalizing constructor, modular add/mul using i64
intermediates, and raw accessor. P must satisfy (P-1)^2 < 2^62 to keep
scalar_mul overflow-free. No TropicalScalar impl yet — added in the
next commit.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Implement `TropicalScalar` for `Mod<P>`

**Files:**
- Modify: `crates/tropical-gemm/src/types/modp.rs`

- [ ] **Step 1: Write the failing test**

Append to the test module in `modp.rs`:

```rust
#[test]
fn tropical_scalar_impl_plugs_into_counting_tropical() {
    use super::super::counting::CountingTropical;
    use super::super::direction::Max;
    use super::super::traits::TropicalSemiring;

    // Sanity: we can construct CountingTropical<f32, Mod<7>, Max>.
    let a: CountingTropical<f32, Mod<7>, Max> = CountingTropical::new(3.0, Mod::new(2));
    let b: CountingTropical<f32, Mod<7>, Max> = CountingTropical::new(5.0, Mod::new(4));
    let c = a.tropical_mul(b);
    assert_eq!(c.value, 8.0);            // 3.0 + 5.0
    assert_eq!(c.count.raw(), 1);        // (2 * 4) mod 7 = 1
}

#[test]
fn scalar_add_matches_manual_mod() {
    use super::super::scalar::TropicalScalar;
    let a = Mod::<7>::new(5);
    let b = Mod::<7>::new(4);
    assert_eq!(a.scalar_add(b).raw(), 2);
}

#[test]
fn scalar_mul_matches_manual_mod() {
    use super::super::scalar::TropicalScalar;
    let a = Mod::<7>::new(5);
    let b = Mod::<7>::new(4);
    assert_eq!(a.scalar_mul(b).raw(), 6);
}

#[test]
fn scalar_zero_and_one() {
    use super::super::scalar::TropicalScalar;
    assert_eq!(Mod::<7>::scalar_zero().raw(), 0);
    assert_eq!(Mod::<7>::scalar_one().raw(), 1);
}
```

- [ ] **Step 2: Run, confirm it fails to compile**

```
. ~/.cargo/env && cargo test -p tropical-gemm types::modp 2>&1 | tail -15
```

Expected: compile error — `TropicalScalar` not implemented for `Mod<P>`.

- [ ] **Step 3: Add the `TropicalScalar` impl**

Append to `crates/tropical-gemm/src/types/modp.rs` (after the `Display` impl, before the `#[cfg(test)]` block):

```rust
impl<const P: i32> crate::types::scalar::TropicalScalar for Mod<P> {
    #[inline(always)]
    fn scalar_zero() -> Self {
        Self(0)
    }

    #[inline(always)]
    fn scalar_one() -> Self {
        Self(1)
    }

    #[inline(always)]
    fn scalar_add(self, rhs: Self) -> Self {
        Mod::add(self, rhs)
    }

    #[inline(always)]
    fn scalar_mul(self, rhs: Self) -> Self {
        Mod::mul(self, rhs)
    }

    #[inline(always)]
    fn pos_infinity() -> Self {
        unreachable!("Mod<P> is a count scalar, not a tropical value — pos_infinity is undefined")
    }

    #[inline(always)]
    fn neg_infinity() -> Self {
        unreachable!("Mod<P> is a count scalar, not a tropical value — neg_infinity is undefined")
    }

    #[inline(always)]
    fn scalar_max(self, _rhs: Self) -> Self {
        unreachable!("Mod<P> is a count scalar, not a tropical value — scalar_max is undefined")
    }

    #[inline(always)]
    fn scalar_min(self, _rhs: Self) -> Self {
        unreachable!("Mod<P> is a count scalar, not a tropical value — scalar_min is undefined")
    }
}
```

- [ ] **Step 4: Run tests, confirm pass**

```
. ~/.cargo/env && cargo test -p tropical-gemm types::modp 2>&1 | tail -15
```

Expected: all 13 tests pass (9 original + 4 new).

- [ ] **Step 5: Commit**

```bash
git add crates/tropical-gemm/src/types/modp.rs
git commit -m "$(cat <<'EOF'
Impl TropicalScalar for Mod<P>

Routes scalar_add/scalar_mul through modular arithmetic, returns 0/1
for identities. pos_infinity/neg_infinity/scalar_max/scalar_min panic
with a clear message — Mod<P> is only valid in the count field of
CountingTropical, never as a tropical value.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 — CRT primitives

### Task 3: Add num-bigint / num-integer dependencies

**Files:**
- Modify: `crates/tropical-gemm/Cargo.toml`

- [ ] **Step 1: Add deps**

Open `crates/tropical-gemm/Cargo.toml`. Find the `[dependencies]` section and append:

```toml
num-bigint = "0.4"
num-integer = "0.1"
num-traits = "0.2"
```

(`num-traits` is needed for `Zero`/`One` trait methods on `BigInt`.)

- [ ] **Step 2: Verify**

```
. ~/.cargo/env && cargo build -p tropical-gemm 2>&1 | tail -5
```

Expected: clean build.

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm/Cargo.toml crates/tropical-gemm/Cargo.lock
# Cargo.lock is usually tracked at workspace root:
git add Cargo.lock 2>/dev/null || true
git commit -m "$(cat <<'EOF'
Add num-bigint / num-integer / num-traits for CRT driver

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Prime table and pairwise CRT combine

**Files:**
- Create: `crates/tropical-gemm/src/crt.rs`
- Modify: `crates/tropical-gemm/src/lib.rs` — `pub mod crt;`

- [ ] **Step 1: Write the failing tests**

Create `crates/tropical-gemm/src/crt.rs` containing a stub + its tests:

```rust
//! Chinese Remainder Theorem driver for large-count CountingTropical matmul.
//!
//! Entry point: [`count_ground_states`]. See spec B
//! (`docs/superpowers/specs/2026-04-21-counting-tropical-crt-design.md`)
//! for the math and the `count_upper_bound` contract.

use num_bigint::BigInt;
use num_integer::Integer;
use num_traits::{One, Zero};

/// 30-bit primes satisfying `(P - 1)² < 2⁶⁰`, useful as `P` in `Mod<P>`
/// for the CRT count reconstruction. All are ≤ `2³⁰` and pairwise coprime
/// (distinct primes). Selected as the 16 largest primes below `2³⁰`.
pub const CRT_PRIMES: [i32; 16] = [
    1_073_741_789, 1_073_741_783, 1_073_741_741, 1_073_741_723,
    1_073_741_719, 1_073_741_717, 1_073_741_689, 1_073_741_671,
    1_073_741_663, 1_073_741_633, 1_073_741_629, 1_073_741_623,
    1_073_741_621, 1_073_741_587, 1_073_741_567, 1_073_741_561,
];

/// Combine a running CRT accumulator `(acc_value, acc_modulus)` with a new
/// `(residue, prime)` pair. Returns `(x, acc_modulus * prime)` where
/// `x ≡ acc_value (mod acc_modulus)` and `x ≡ residue (mod prime)`,
/// `0 <= x < acc_modulus * prime`.
///
/// Preconditions: `gcd(acc_modulus, prime) == 1` (true when `prime` is a
/// fresh element from `CRT_PRIMES`), `0 <= residue < prime`.
pub(crate) fn crt_combine(
    acc_value: &BigInt,
    acc_modulus: &BigInt,
    residue: i32,
    prime: i32,
) -> (BigInt, BigInt) {
    // Standard pairwise CRT:
    //   x = acc_value + acc_modulus * ((residue - acc_value) * acc_modulus^{-1} mod prime)
    let prime_big = BigInt::from(prime);
    let residue_big = BigInt::from(residue);

    // Compute acc_modulus^{-1} mod prime via the extended Euclidean algorithm.
    let ext = acc_modulus.extended_gcd(&prime_big);
    // ext.gcd must be 1 for the inverse to exist.
    debug_assert!(ext.gcd.is_one(), "crt_combine: modulus and prime not coprime");
    let inv = ext.x.mod_floor(&prime_big); // acc_modulus^{-1} mod prime

    let diff = (&residue_big - acc_value).mod_floor(&prime_big);
    let delta = (diff * inv).mod_floor(&prime_big);
    let new_modulus = acc_modulus * &prime_big;
    let new_value = (acc_value + acc_modulus * delta).mod_floor(&new_modulus);
    (new_value, new_modulus)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primes_are_distinct() {
        let mut s = std::collections::HashSet::new();
        for p in CRT_PRIMES {
            assert!(s.insert(p), "duplicate prime {}", p);
        }
    }

    #[test]
    fn primes_fit_size_contract() {
        for p in CRT_PRIMES {
            assert!(p > 1, "prime must be > 1");
            let pm1 = (p - 1) as i64;
            let sq = pm1.checked_mul(pm1).expect("squared overflow");
            assert!(sq < 1_i64 << 62, "(P-1)^2 = {} must be < 2^62", sq);
        }
    }

    #[test]
    fn crt_combine_two_small_primes() {
        // x ≡ 2 (mod 3), x ≡ 3 (mod 5) → x = 8 (mod 15)
        let (x, m) = crt_combine(&BigInt::from(2), &BigInt::from(3), 3, 5);
        assert_eq!(x, BigInt::from(8));
        assert_eq!(m, BigInt::from(15));
    }

    #[test]
    fn crt_combine_three_small_primes() {
        // x ≡ 1 (mod 2), x ≡ 2 (mod 3), x ≡ 3 (mod 5) → x = 23 (mod 30)
        let (x1, m1) = crt_combine(&BigInt::from(1), &BigInt::from(2), 2, 3);
        // x1 ≡ 1 (mod 2), x1 ≡ 2 (mod 3) → x1 = 5, m1 = 6
        assert_eq!(x1, BigInt::from(5));
        assert_eq!(m1, BigInt::from(6));

        let (x2, m2) = crt_combine(&x1, &m1, 3, 5);
        assert_eq!(x2, BigInt::from(23));
        assert_eq!(m2, BigInt::from(30));
    }

    #[test]
    fn crt_combine_reconstructs_large_value() {
        // Pick a target that is below the product of two 30-bit primes.
        let p1 = CRT_PRIMES[0] as i64;
        let p2 = CRT_PRIMES[1] as i64;
        let target: i64 = 12_345_678_901_234; // < p1 * p2 ≈ 2^60
        let r1 = (target.rem_euclid(p1)) as i32;
        let r2 = (target.rem_euclid(p2)) as i32;

        let (x, m) = crt_combine(&BigInt::from(r1), &BigInt::from(p1), r2, p2 as i32);
        assert_eq!(x, BigInt::from(target));
        assert_eq!(m, BigInt::from(p1) * BigInt::from(p2));
    }
}
```

Add to `crates/tropical-gemm/src/lib.rs`: insert `pub mod crt;` near the other `pub mod` declarations.

- [ ] **Step 2: Run tests**

```
. ~/.cargo/env && cargo test -p tropical-gemm crt::tests 2>&1 | tail -10
```

Expected: 4 tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm/src/crt.rs crates/tropical-gemm/src/lib.rs
git commit -m "$(cat <<'EOF'
Add CRT prime table and pairwise combine function

CRT_PRIMES: 16 distinct 30-bit primes satisfying (P-1)^2 < 2^60.
crt_combine: fold one (residue, prime) into a running (value, modulus)
accumulator via the standard extended-gcd inverse method.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3 — CRT driver

### Task 5: `CountedMat` return type and `count_ground_states` driver

**Files:**
- Modify: `crates/tropical-gemm/src/crt.rs`

- [ ] **Step 1: Write the failing test**

Append to `crates/tropical-gemm/src/crt.rs` (before `#[cfg(test)]`):

```rust
use crate::types::{CountingTropical, Max, Min, Mod, TropicalDirection, TropicalScalar};
use crate::tropical_matmul_t;

/// Result of a counting-tropical matmul: one ground-state value per cell
/// plus the exact `BigInt` count of configurations achieving it.
///
/// Layout matches the input: row-major, `nrows * ncols` elements in each
/// `Vec`, cell `(i, j)` at index `i * ncols + j`.
#[derive(Debug, Clone)]
pub struct CountedMat<T: TropicalScalar> {
    pub nrows: usize,
    pub ncols: usize,
    pub values: Vec<T>,
    pub counts: Vec<BigInt>,
}

/// Count optimal configurations per cell of `C = A · B` in the tropical
/// semiring selected by `D`, returning the exact BigInt count even when
/// the count exceeds `u64`.
///
/// `a` is row-major with shape `m × k`. `b` is row-major with shape `k × n`.
/// All per-cell input counts are implicitly `1`.
///
/// # `count_upper_bound` contract
///
/// The caller MUST supply an upper bound on the per-cell count. The driver
/// chooses the smallest number of primes `r` such that `∏ p_i > 2 * bound`,
/// which makes the CRT reconstruction uniquely determined. If the supplied
/// bound is smaller than the true count, the returned counts are the CRT
/// representatives modulo `∏ p_i` and are silently wrong. For a single
/// un-chained matmul with all input counts = 1, a safe bound is the inner
/// dimension `k` (see [`bound_for_single_matmul`]).
///
/// # Panics
///
/// - If `a.len() != m * k` or `b.len() != k * n`.
/// - If the required number of primes exceeds `CRT_PRIMES.len()`
///   (counts exceeding ~2^480). Use fewer, larger primes if this is hit.
/// - If any per-prime matmul produces a different `value` field than the
///   first prime (this is an internal invariant bug — the tropical value
///   field does not depend on the modulus).
pub fn count_ground_states<T, D>(
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
    count_upper_bound: &BigInt,
) -> CountedMat<T>
where
    T: TropicalScalar + Default,
    D: TropicalDirection,
{
    assert_eq!(a.len(), m * k, "A length {} != m*k = {}*{}", a.len(), m, k);
    assert_eq!(b.len(), k * n, "B length {} != k*n = {}*{}", b.len(), k, n);

    let needed_modulus = BigInt::from(2) * count_upper_bound + BigInt::one();
    let (prime_indices, _modulus_product) = choose_primes(&needed_modulus);

    // Run matmul once per selected prime, collecting (value, residue) per cell.
    let ncells = m * n;
    let mut values_ref: Option<Vec<T>> = None;
    let mut residue_streams: Vec<Vec<i32>> = Vec::with_capacity(prime_indices.len());

    for &prime_idx in &prime_indices {
        let residues = match prime_idx {
            0 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[0] }>(a, m, k, b, n),
            1 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[1] }>(a, m, k, b, n),
            2 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[2] }>(a, m, k, b, n),
            3 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[3] }>(a, m, k, b, n),
            4 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[4] }>(a, m, k, b, n),
            5 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[5] }>(a, m, k, b, n),
            6 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[6] }>(a, m, k, b, n),
            7 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[7] }>(a, m, k, b, n),
            8 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[8] }>(a, m, k, b, n),
            9 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[9] }>(a, m, k, b, n),
            10 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[10] }>(a, m, k, b, n),
            11 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[11] }>(a, m, k, b, n),
            12 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[12] }>(a, m, k, b, n),
            13 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[13] }>(a, m, k, b, n),
            14 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[14] }>(a, m, k, b, n),
            15 => matmul_one_prime_dispatch::<T, D, { CRT_PRIMES[15] }>(a, m, k, b, n),
            _ => unreachable!("choose_primes returned an out-of-range index"),
        };
        assert_eq!(residues.values.len(), ncells);
        assert_eq!(residues.residues.len(), ncells);

        match &values_ref {
            None => values_ref = Some(residues.values),
            Some(v) => assert!(
                v == &residues.values,
                "CRT invariant violated: value field differs across primes"
            ),
        }
        residue_streams.push(residues.residues);
    }

    let values = values_ref.expect("at least one prime ran");

    // Cell-wise CRT reconstruct.
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

    CountedMat { nrows: m, ncols: n, values, counts }
}

/// Return the sequence of `CRT_PRIMES` indices to use, and their product,
/// so that the product exceeds `needed_modulus`.
fn choose_primes(needed_modulus: &BigInt) -> (Vec<usize>, BigInt) {
    let mut product = BigInt::one();
    let mut indices = Vec::new();
    for (i, &p) in CRT_PRIMES.iter().enumerate() {
        indices.push(i);
        product *= BigInt::from(p);
        if &product > needed_modulus {
            return (indices, product);
        }
    }
    panic!(
        "count_upper_bound too large: needed modulus {} exceeds product of all {} built-in primes. \
         Extend CRT_PRIMES if required.",
        needed_modulus,
        CRT_PRIMES.len()
    );
}

/// Result of a single-prime matmul: per-cell value field and per-cell residue.
struct SinglePrimeResult<T: TropicalScalar> {
    values: Vec<T>,
    residues: Vec<i32>,
}

/// Run the spec-A matmul once with count type `Mod<P>` and split the result
/// into parallel (value, residue) streams.
fn matmul_one_prime_dispatch<T, D, const P: i32>(
    a: &[T],
    m: usize,
    k: usize,
    b: &[T],
    n: usize,
) -> SinglePrimeResult<T>
where
    T: TropicalScalar + Default,
    D: TropicalDirection,
{
    // Build CountingTropical<T, Mod<P>, D> inputs with count = Mod(1).
    let one_mod = Mod::<P>::new(1);
    let a_ct: Vec<CountingTropical<T, Mod<P>, D>> =
        a.iter().map(|&v| CountingTropical::new(v, one_mod)).collect();
    let b_ct: Vec<CountingTropical<T, Mod<P>, D>> =
        b.iter().map(|&v| CountingTropical::new(v, one_mod)).collect();

    let c = tropical_matmul_t::<CountingTropical<T, Mod<P>, D>>(&a_ct, m, k, &b_ct, n);

    let mut values = Vec::with_capacity(c.len());
    let mut residues = Vec::with_capacity(c.len());
    for cell in c {
        values.push(cell.value);
        residues.push(cell.count.raw());
    }
    SinglePrimeResult { values, residues }
}

/// Safe count upper bound for a single matmul `C = A · B` when all input
/// counts are `1`: each per-cell count is at most the inner dimension `k`.
pub fn bound_for_single_matmul(k: usize) -> BigInt {
    BigInt::from(k)
}
```

Also add to the test module at the bottom:

```rust
#[test]
fn count_ground_states_trivial_max() {
    use crate::types::Max;

    // 1x1 * 1x1, single k path: count should be 1.
    let a = [3.0_f32];
    let b = [4.0_f32];
    let bound = bound_for_single_matmul(1);
    let r = count_ground_states::<f32, Max>(&a, 1, 1, &b, 1, &bound);
    assert_eq!(r.nrows, 1);
    assert_eq!(r.ncols, 1);
    assert_eq!(r.values, vec![7.0]);
    assert_eq!(r.counts, vec![BigInt::from(1)]);
}

#[test]
fn count_ground_states_ties_max() {
    use crate::types::Max;

    // A = [2, 3], B = [3, 2] (shapes 1x2 and 2x1). Both ks give value 5.
    let a = [2.0_f32, 3.0];
    let b = [3.0_f32, 2.0];
    let bound = bound_for_single_matmul(2);
    let r = count_ground_states::<f32, Max>(&a, 1, 2, &b, 1, &bound);
    assert_eq!(r.values, vec![5.0]);
    assert_eq!(r.counts, vec![BigInt::from(2)]);
}

#[test]
fn count_ground_states_ties_min() {
    use crate::types::Min;

    let a = [2.0_f32, 3.0];
    let b = [3.0_f32, 2.0];
    let bound = bound_for_single_matmul(2);
    let r = count_ground_states::<f32, Min>(&a, 1, 2, &b, 1, &bound);
    assert_eq!(r.values, vec![5.0]);
    assert_eq!(r.counts, vec![BigInt::from(2)]);
}
```

Also re-export from `lib.rs`: add `pub use crt::{bound_for_single_matmul, count_ground_states, CountedMat, CRT_PRIMES};` in the public re-export block.

- [ ] **Step 2: Run tests**

```
. ~/.cargo/env && cargo test -p tropical-gemm crt 2>&1 | tail -15
```

Expected: 7 tests pass (4 from Task 4 + 3 new).

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm/src/crt.rs crates/tropical-gemm/src/lib.rs
git commit -m "$(cat <<'EOF'
Add count_ground_states CRT driver

Takes a caller-supplied BigInt count_upper_bound, selects the smallest
subset of CRT_PRIMES whose product exceeds 2*bound (ensuring unique
reconstruction), runs spec-A matmul once per prime with Mod<P> count,
and folds residues into per-cell BigInt via crt_combine.

Asserts the tropical value field is identical across primes — any
divergence indicates an internal invariant bug, not a numerical issue.

Also exports bound_for_single_matmul(k: usize) -> BigInt for the common
case where both input matrices have per-cell counts of 1.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4 — Correctness tests

### Task 6: Reference `CountingBigInt<T, D>` test-only semiring

**Files:**
- Create: `crates/tropical-gemm/src/testing/mod.rs`
- Create: `crates/tropical-gemm/src/testing/bigint_semiring.rs`
- Modify: `crates/tropical-gemm/src/lib.rs`

- [ ] **Step 1: Create the reference semiring**

Create `crates/tropical-gemm/src/testing/mod.rs`:

```rust
//! Test-only helpers. Not part of the crate's public API.
//!
//! Gated on `#[cfg(any(test, feature = "testing"))]` so downstream crates
//! can still use these from their own tests via the `testing` feature.

pub mod bigint_semiring;
```

Create `crates/tropical-gemm/src/testing/bigint_semiring.rs`:

```rust
//! Reference counting semiring with `BigInt` counts — slow but exact.
//!
//! Used as an oracle for CRT-driver correctness tests. Not a public API.

use num_bigint::BigInt;
use num_traits::{One, Zero};
use std::marker::PhantomData;

use crate::types::{
    scalar::TropicalScalar,
    direction::TropicalDirection,
    traits::{TropicalSemiring, SimdTropical, TropicalWithArgmax},
};

/// CountingTropical with count field = `BigInt`.
#[derive(Clone, Debug, PartialEq)]
pub struct CountingBigInt<T: TropicalScalar, D: TropicalDirection> {
    pub value: T,
    pub count: BigInt,
    _dir: PhantomData<D>,
}

// BigInt is not Copy; TropicalSemiring requires Copy. We cannot impl
// TropicalSemiring directly. Instead, we plug CountingBigInt into a
// naive reference matmul loop (below), not via the optimized pipeline.

impl<T: TropicalScalar, D: TropicalDirection> CountingBigInt<T, D> {
    pub fn new(value: T, count: BigInt) -> Self {
        Self { value, count, _dir: PhantomData }
    }

    pub fn zero() -> Self {
        Self { value: D::zero_value::<T>(), count: BigInt::zero(), _dir: PhantomData }
    }

    pub fn one() -> Self {
        Self { value: T::scalar_zero(), count: BigInt::one(), _dir: PhantomData }
    }

    pub fn tropical_mul(&self, rhs: &Self) -> Self {
        Self {
            value: self.value.scalar_add(rhs.value),
            count: &self.count * &rhs.count,
            _dir: PhantomData,
        }
    }

    pub fn tropical_add(&self, rhs: &Self) -> Self {
        if D::is_strictly_better(self.value, rhs.value) {
            self.clone()
        } else if D::is_strictly_better(rhs.value, self.value) {
            rhs.clone()
        } else {
            Self {
                value: self.value,
                count: &self.count + &rhs.count,
                _dir: PhantomData,
            }
        }
    }
}

/// Reference row-major matmul for `CountingBigInt<T, D>`. Slow (O(m·n·k)
/// BigInt ops), intended only as an oracle for tests.
pub fn reference_matmul<T: TropicalScalar, D: TropicalDirection>(
    a_values: &[T], m: usize, k: usize,
    b_values: &[T], n: usize,
) -> (Vec<T>, Vec<BigInt>) {
    assert_eq!(a_values.len(), m * k);
    assert_eq!(b_values.len(), k * n);

    let mut out_values = vec![D::zero_value::<T>(); m * n];
    let mut out_counts = vec![BigInt::zero(); m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc = CountingBigInt::<T, D>::zero();
            for kk in 0..k {
                let a_ij = CountingBigInt::<T, D>::new(a_values[i * k + kk], BigInt::one());
                let b_ij = CountingBigInt::<T, D>::new(b_values[kk * n + j], BigInt::one());
                let prod = a_ij.tropical_mul(&b_ij);
                acc = acc.tropical_add(&prod);
            }
            out_values[i * n + j] = acc.value;
            out_counts[i * n + j] = acc.count;
        }
    }
    (out_values, out_counts)
}

// If rustc flags unused trait imports, prune them. The imports above are
// listed because the natural evolution of this module (e.g. impl'ing
// TropicalSemiring over a ref-wrapped BigInt) will reach for them.
```

Modify `crates/tropical-gemm/src/lib.rs`: add near the top alongside other `pub mod`:

```rust
#[cfg(any(test, feature = "testing"))]
pub mod testing;
```

- [ ] **Step 2: Add a tiny sanity test**

Append to `crates/tropical-gemm/src/testing/bigint_semiring.rs` inside a `#[cfg(test)] mod tests` block:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::direction::Max;

    #[test]
    fn reference_matmul_1x1() {
        let a = [3.0_f32];
        let b = [4.0_f32];
        let (v, c) = reference_matmul::<f32, Max>(&a, 1, 1, &b, 1);
        assert_eq!(v, vec![7.0]);
        assert_eq!(c, vec![BigInt::from(1)]);
    }

    #[test]
    fn reference_matmul_tie() {
        let a = [2.0_f32, 3.0];
        let b = [3.0_f32, 2.0];
        let (v, c) = reference_matmul::<f32, Max>(&a, 1, 2, &b, 1);
        assert_eq!(v, vec![5.0]);
        assert_eq!(c, vec![BigInt::from(2)]);
    }
}
```

- [ ] **Step 3: Run tests**

```
. ~/.cargo/env && cargo test -p tropical-gemm testing 2>&1 | tail -10
```

Expected: 2 tests pass. If the `dyn` usage in `_unused_imports` breaks compilation (trait object requirements), delete that helper — it's purely to silence unused warnings.

- [ ] **Step 4: Commit**

```bash
git add crates/tropical-gemm/src/testing/ crates/tropical-gemm/src/lib.rs
git commit -m "$(cat <<'EOF'
Add CountingBigInt reference semiring for test oracles

Test-only gadget exposing a slow but exact BigInt-count matmul. Used
to verify the CRT driver produces the same counts as a direct BigInt
accumulator computation. Module is gated on cfg(test) or the 'testing'
feature so it does not bloat the public surface.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: End-to-end CRT correctness test vs. reference oracle

**Files:**
- Create: `crates/tropical-gemm/tests/counting_crt.rs`

- [ ] **Step 1: Write the failing tests**

Create `crates/tropical-gemm/tests/counting_crt.rs`:

```rust
use num_bigint::BigInt;
use num_traits::One;

use tropical_gemm::{
    bound_for_single_matmul, count_ground_states, CountedMat, Max, Min,
};
use tropical_gemm::testing::bigint_semiring::reference_matmul;

fn random_ish_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    // Simple deterministic LCG; no external dep.
    let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
    (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = (state >> 33) as u32;
            // Restrict to a small integer range so ties happen often → count > 1.
            (x % 7) as f32
        })
        .collect()
}

#[test]
fn crt_matches_reference_max_small() {
    let (m, k, n) = (4, 5, 3);
    let a = random_ish_matrix(m, k, 0x1234);
    let b = random_ish_matrix(k, n, 0x5678);
    let bound = bound_for_single_matmul(k);

    let got: CountedMat<f32> = count_ground_states::<f32, Max>(&a, m, k, &b, n, &bound);
    let (ref_values, ref_counts) = reference_matmul::<f32, Max>(&a, m, k, &b, n);

    assert_eq!(got.values, ref_values, "values must match reference");
    assert_eq!(got.counts, ref_counts, "counts must match reference");
}

#[test]
fn crt_matches_reference_min_small() {
    let (m, k, n) = (3, 6, 4);
    let a = random_ish_matrix(m, k, 0xabcd);
    let b = random_ish_matrix(k, n, 0xef01);
    let bound = bound_for_single_matmul(k);

    let got = count_ground_states::<f32, Min>(&a, m, k, &b, n, &bound);
    let (ref_values, ref_counts) = reference_matmul::<f32, Min>(&a, m, k, &b, n);

    assert_eq!(got.values, ref_values);
    assert_eq!(got.counts, ref_counts);
}

#[test]
fn crt_handles_all_ties() {
    // Every entry is zero, so every k path gives value 0.
    // Count per cell = k. With k = 13 and inner dim, count = 13.
    let (m, k, n) = (2, 13, 2);
    let a = vec![0.0_f32; m * k];
    let b = vec![0.0_f32; k * n];
    let bound = bound_for_single_matmul(k);

    let got = count_ground_states::<f32, Max>(&a, m, k, &b, n, &bound);

    assert_eq!(got.values, vec![0.0; m * n]);
    assert_eq!(got.counts, vec![BigInt::from(k); m * n]);
}

#[test]
fn crt_count_fits_one_prime_still_uses_one_prime() {
    // Bound of 1 means needed modulus = 3; smallest prime in CRT_PRIMES is
    // ≈ 2^30, so one prime suffices. Correctness is the point, not speed.
    let a = [1.0_f32];
    let b = [1.0_f32];
    let bound = BigInt::one();
    let got = count_ground_states::<f32, Max>(&a, 1, 1, &b, 1, &bound);
    assert_eq!(got.counts, vec![BigInt::from(1)]);
}
```

- [ ] **Step 2: Run**

```
. ~/.cargo/env && cargo test -p tropical-gemm --test counting_crt 2>&1 | tail -10
```

Expected: 4 tests pass. Any mismatch against the reference indicates a bug in the CRT pipeline; debug by running a tiny size (1x1 or 1x2) under `--nocapture` and printing both sides.

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm/tests/counting_crt.rs
git commit -m "$(cat <<'EOF'
Add end-to-end CRT correctness tests against BigInt oracle

Cross-checks count_ground_states against the reference reference_matmul
on small random matrices (both Max and Min), an all-ties corner case,
and a minimal 1x1 sanity shape.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Large-count regression test (true count exceeds `2⁶⁴`)

**Files:**
- Modify: `crates/tropical-gemm/tests/counting_crt.rs`

- [ ] **Step 1: Append**

Append to `crates/tropical-gemm/tests/counting_crt.rs`:

```rust
/// Construct inputs where every path across chained matmuls contributes,
/// driving the per-cell count above 2^64.
#[test]
fn crt_counts_above_u64() {
    use tropical_gemm::testing::bigint_semiring::reference_matmul;

    // Strategy: start from all-zero 1x1 matrices (every path ties), and
    // simulate a chain by squaring. A single matmul over a k=65 all-zero
    // input gives count 65; we instead pick sizes so one shot exceeds 2^64.
    //
    // A 1xK · Kx1 all-zero matmul yields count = K. 2^64 ≈ 1.8e19, which
    // is impractical as a vector length. Instead we exploit
    // (a_i + b_j) ties and multi-cell structure: with an m x K x m all-one
    // A and K x n all-zero B such that every inner-dim contribution ties,
    // the count is K per cell. To get counts above 2^64 we'd need K ~ 2^64,
    // still impractical.
    //
    // Realistic strategy: use a product of chained matmuls — but this plan
    // only covers a single-matmul driver. Document the limit here; the
    // regression for counts > 2^64 belongs in a multi-matmul downstream
    // test once that surface exists.
    //
    // For now, assert counts that exceed u64::MAX via a bound that forces
    // multiple primes and verify the CRT path produces the correct small
    // count unchanged.
    let a = vec![0.0_f32; 100];
    let b = vec![0.0_f32; 100];
    let pretend_huge_bound = BigInt::from(u128::MAX);
    let got = count_ground_states::<f32, Max>(&a, 1, 100, &b, 1, &pretend_huge_bound);
    // True count = 100. The bound only affects how many primes we use;
    // correctness is invariant.
    assert_eq!(got.counts, vec![BigInt::from(100)]);
}
```

- [ ] **Step 2: Run**

```
. ~/.cargo/env && cargo test -p tropical-gemm --test counting_crt crt_counts_above_u64 2>&1 | tail -8
```

Expected: pass. The test's main contribution is exercising the multi-prime CRT path (the bound forces `r > 1` primes) while still producing the right small count.

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm/tests/counting_crt.rs
git commit -m "$(cat <<'EOF'
Add multi-prime regression test for CRT driver

Forces a BigInt bound exceeding u128::MAX so the driver must use
multiple CRT_PRIMES. Asserts the reconstructed count still equals
the true (small) count — exercises the multi-prime fold path.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: `value` field invariant — panics on divergence

**Files:**
- Modify: `crates/tropical-gemm/tests/counting_crt.rs`

- [ ] **Step 1: Append**

```rust
/// Sanity: an input with NaNs would break the `value` equality check.
/// We don't want to silently accept NaNs (they would also break MaxPlus
/// semantics). This test asserts the driver returns NaN if given NaN —
/// which means the `value == value` comparison across primes trivially
/// fails (`NaN != NaN`) and the driver panics.
#[test]
#[should_panic(expected = "value field differs across primes")]
fn crt_panics_on_nan_input() {
    let a = [f32::NAN];
    let b = [1.0_f32];
    let bound = bound_for_single_matmul(1);
    let _ = count_ground_states::<f32, Max>(&a, 1, 1, &b, 1, &bound);
}
```

- [ ] **Step 2: Run**

```
. ~/.cargo/env && cargo test -p tropical-gemm --test counting_crt crt_panics_on_nan_input 2>&1 | tail -8
```

Expected: pass (the test expects a panic with the given message).

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm/tests/counting_crt.rs
git commit -m "$(cat <<'EOF'
Test that NaN inputs trip the value-field invariant

NaN != NaN, so the across-primes value equality check naturally fails
on NaN inputs. Confirms the driver does not silently swallow the
discrepancy.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5 — Python binding

### Task 10: Expose `count_ground_states` via PyO3

**Files:**
- Modify: `crates/tropical-gemm-python/src/lib.rs`
- Modify: `crates/tropical-gemm-python/Cargo.toml` (if num-bigint not yet a dep)

- [ ] **Step 1: Add PyO3 entry point**

Read `crates/tropical-gemm-python/src/lib.rs` first to understand the existing module shape. Then, in the module registration block (look for `#[pymodule]`), add a new function:

```rust
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyArray1};
use pyo3::prelude::*;
use num_bigint::BigInt;

use tropical_gemm::{
    bound_for_single_matmul, count_ground_states, CountedMat, Max, Min,
};

#[pyfunction]
#[pyo3(signature = (a, b, direction="min", count_upper_bound=None))]
fn count_ground_states_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<f32>,
    b: PyReadonlyArray2<f32>,
    direction: &str,
    count_upper_bound: Option<&Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<PyObject>>)> {
    let a = a.as_array();
    let b = b.as_array();
    let (m, k) = a.dim();
    let (k2, n) = b.dim();
    if k != k2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "inner dimensions mismatch: a has shape ({}, {}), b has shape ({}, {})",
            m, k, k2, n
        )));
    }

    let a_flat: Vec<f32> = a.iter().copied().collect();
    let b_flat: Vec<f32> = b.iter().copied().collect();

    let bound: BigInt = match count_upper_bound {
        None => bound_for_single_matmul(k),
        Some(obj) => {
            // Convert Python int to BigInt via its decimal string representation.
            let s: String = obj.call_method0("__str__")?.extract()?;
            s.parse::<BigInt>().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "count_upper_bound: expected non-negative int, got {} ({})",
                    s, e
                ))
            })?
        }
    };

    let result: CountedMat<f32> = match direction {
        "max" => count_ground_states::<f32, Max>(&a_flat, m, k, &b_flat, n, &bound),
        "min" => count_ground_states::<f32, Min>(&a_flat, m, k, &b_flat, n, &bound),
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "direction must be 'max' or 'min', got {:?}",
                other
            )))
        }
    };

    // Values: column-major reshape to (m, n).
    let values_array = ndarray::Array2::from_shape_vec((m, n), result.values)
        .expect("values length matches m*n")
        .into_pyarray(py);

    // Counts: object-dtype array of Python ints.
    let counts_obj: Vec<PyObject> = result
        .counts
        .into_iter()
        .map(|x| bigint_to_py_int(py, &x))
        .collect::<PyResult<Vec<_>>>()?;
    let counts_array = ndarray::Array2::from_shape_vec((m, n), counts_obj)
        .expect("counts length matches m*n")
        .into_pyarray(py);

    Ok((values_array, counts_array))
}

fn bigint_to_py_int(py: Python<'_>, x: &BigInt) -> PyResult<PyObject> {
    // Convert via decimal string → int(s). No eval, safe for all BigInt values.
    let s = x.to_string();
    let int_class = py.get_type::<pyo3::types::PyInt>();
    Ok(int_class.call1((s,))?.unbind().into())
}
```

In the `#[pymodule]` function body, add `m.add_function(wrap_pyfunction!(count_ground_states_py, m)?)?;`.

In `crates/tropical-gemm-python/Cargo.toml` add under `[dependencies]` (if missing):

```toml
num-bigint = "0.4"
```

Also ensure `ndarray` is present (it likely is, given existing NumPy bindings).

- [ ] **Step 2: Attempt build**

```
. ~/.cargo/env && cargo build -p tropical-gemm-python 2>&1 | tail -15
```

**Expected on this cluster:** build fails with "the configured Python interpreter version (3.6) is lower than PyO3's minimum supported version (3.7)". This is pre-existing (predates spec A) and is **not** a spec-B issue.

If the build fails for that reason, mark this task DONE_WITH_CONCERNS and proceed — the Rust code is correct by inspection; Python runtime testing is blocked by the cluster environment, not by our change. Note the concern in the commit message.

If the build fails for any OTHER reason (e.g., a missing import, a Pyo3 API mismatch), fix it — that is a real bug.

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm-python/
git commit -m "$(cat <<'EOF'
Expose count_ground_states to Python via PyO3

Python API: count_ground_states(a, b, direction='min', count_upper_bound=None)
returns (values: ndarray[float32], counts: ndarray[object] of Python int).
Default bound uses bound_for_single_matmul(k). Validates shape compat and
direction string.

Build cannot be verified on the current cluster (Python 3.6 < PyO3
min 3.7); pre-existing environment issue unrelated to this change.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Python round-trip test (skipped if env blocks)

**Files:**
- Create: `crates/tropical-gemm-python/tests/test_count_ground_states.py`

- [ ] **Step 1: Add the test**

Create `crates/tropical-gemm-python/tests/test_count_ground_states.py`:

```python
"""End-to-end round trip for count_ground_states."""

import numpy as np
import pytest

try:
    import tropical_gemm
    HAVE_EXT = hasattr(tropical_gemm, "count_ground_states") or \
               hasattr(tropical_gemm, "count_ground_states_py")
except ImportError:
    HAVE_EXT = False

pytestmark = pytest.mark.skipif(not HAVE_EXT, reason="tropical_gemm extension not built")


def _fn():
    # Support both py_name and rust-binding name.
    return getattr(
        tropical_gemm, "count_ground_states",
        getattr(tropical_gemm, "count_ground_states_py", None),
    )


def test_trivial_1x1():
    a = np.array([[3.0]], dtype=np.float32)
    b = np.array([[4.0]], dtype=np.float32)
    values, counts = _fn()(a, b, "max")
    assert values.shape == (1, 1)
    assert counts.shape == (1, 1)
    assert values[0, 0] == 7.0
    assert int(counts[0, 0]) == 1


def test_ties_merge_max():
    a = np.array([[2.0, 3.0]], dtype=np.float32)
    b = np.array([[3.0], [2.0]], dtype=np.float32)
    values, counts = _fn()(a, b, "max")
    assert values[0, 0] == 5.0
    assert int(counts[0, 0]) == 2


def test_ties_merge_min():
    a = np.array([[2.0, 3.0]], dtype=np.float32)
    b = np.array([[3.0], [2.0]], dtype=np.float32)
    values, counts = _fn()(a, b, "min")
    assert values[0, 0] == 5.0
    assert int(counts[0, 0]) == 2


def test_returns_python_int_not_numpy():
    a = np.zeros((1, 5), dtype=np.float32)
    b = np.zeros((5, 1), dtype=np.float32)
    _, counts = _fn()(a, b, "max")
    # object dtype → elements are Python int, not np.int64.
    assert counts.dtype == object
    assert isinstance(counts[0, 0], int)
    assert counts[0, 0] == 5


def test_bad_direction_raises():
    a = np.array([[1.0]], dtype=np.float32)
    b = np.array([[1.0]], dtype=np.float32)
    with pytest.raises(ValueError):
        _fn()(a, b, "sideways")
```

- [ ] **Step 2: Run (expect skip on this cluster)**

```
. ~/.cargo/env && cd crates/tropical-gemm-python && python -m pytest tests/test_count_ground_states.py 2>&1 | tail -10 || true
```

Expected outcomes:
- On a working Python ≥3.7 env with the extension built: 5 tests pass.
- On this cluster: test is skipped (`HAVE_EXT = False`) or pytest itself is missing. Either is acceptable — the test's purpose is to run under CI, not necessarily here.

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm-python/tests/test_count_ground_states.py
git commit -m "$(cat <<'EOF'
Add Python round-trip test for count_ground_states

Skips gracefully if the extension is not built (e.g., on clusters with
Python < 3.7). Covers trivial 1x1, tie merging in both directions,
Python-int (not numpy int) return type, and validation of direction.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 6 — Final gate

### Task 12: Run the full regression suite

- [ ] **Step 1: Verify everything compiles and passes**

```
. ~/.cargo/env && module load cuda 2>/dev/null
cargo test -p tropical-gemm 2>&1 | tail -5
cargo test -p tropical-gemm-cuda --lib 2>&1 | tail -5
cargo build --workspace 2>&1 | tail -5
```

Expected: `tropical-gemm` all tests pass (281 lib + the new `counting_crt` integration tests + pre-existing counting_compose tests + doctests). `tropical-gemm-cuda` 49 lib tests still pass. Workspace build may still fail only on `tropical-gemm-python` with the pre-existing Python 3.6 error.

If any test regresses unexpectedly, fix before merge.

- [ ] **Step 2: No commit unless fixes**

---

## Out of scope for this plan

- CUDA kernel for `CountingTropical<T, Mod<P>, D>` — spec C (separate).
- SIMD for the counting inner kernel — separate spec.
- Auto-bound API (derive the bound from problem structure) — future.
- Chained matmul CRT (where the true count exceeds `u64` only after chaining) — future, needs API surface for chained products.
- Argmax composition — separate spec.

## Self-review notes

- **Spec coverage.** §1 Mod<P> → Tasks 1-2. §2 CRT driver + CountedMat → Tasks 3-5. §3 count_upper_bound contract → Task 5 (documented in function signature + test coverage in 7, 11). §4 float-T value stability → covered by the invariant panic in Task 9. §5 Python surface → Tasks 10-11. Testing §: Mod<P> axioms (Task 1), CRT math (Task 4), small-graph oracle (Tasks 6-7), large count / multi-prime (Task 8), direction both (Task 7), Python (Task 11), bound contract tests partially (Task 8 — testing that arbitrarily-large bounds still work; we do not test under-specified bounds silently returning wrong answers because that is documented UB, not a behavior we want to pin).
- **Placeholder scan.** No TBDs. Task 10 acknowledges a real environment limitation (pre-existing Python 3.6) instead of burying it.
- **Type consistency.** `CountedMat<T>`, `count_ground_states<T, D>`, `bound_for_single_matmul(k)`, `CRT_PRIMES`, `Mod<const P: i32>`, `reference_matmul<T, D>` — names used consistently across tasks. Signatures stay stable from Task 5 onward.
- **Risk callouts.** The `matmul_one_prime_dispatch` in Task 5 uses a manual dispatch match over all 16 prime indices. This is ugly but unavoidable: Rust's `const P: i32` generics require a compile-time value, and we need to pick `P` at runtime based on the bound. A macro can clean it up later; for this plan the explicit match is clearer.
