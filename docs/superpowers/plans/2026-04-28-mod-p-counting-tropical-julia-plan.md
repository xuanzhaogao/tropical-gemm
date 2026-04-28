# Mod-P CountingTropical Julia GEMM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose the existing AoS counting kernel as a standard Julia matmul over `Matrix{CountingTropical{T, Mod{P, U}}}` with `T ∈ {Float32, Float64}` and both Max-plus + Min-plus directions, no CRT.

**Architecture:** Add a new Rust host driver `matmul_mod_p` that wraps `launch_counting_gemm` (general AoS kernel), expose two C ABI families (slow path with split val/cnt buffers + fast path with packed Pair buffers), and dispatch in the Julia wrapper based on `Mod{P, U}`'s storage type `U` (zero-copy fast path when `U == Int32`, else slow path). Defines a local `CountingTropicalMin` type for the min-plus direction.

**Tech Stack:** Rust + cudarc (existing host driver pattern), CUDA C++ kernel (unchanged, reused), Julia 1.11 + TropicalNumbers.jl + Mods.jl + LinearAlgebra (stdlib).

---

## File Structure

| File | Role | Status |
|---|---|---|
| `crates/tropical-gemm-cuda/src/matmul_mod.rs` | Host driver: pack + upload + launch + download. Two functions (slow + fast). | NEW |
| `crates/tropical-gemm-cuda/src/lib.rs` | Module declaration | EDIT |
| `crates/tropical-gemm-cuda/src/c_api.rs` | 8 new C ABI entries (4 slow + 4 fast) via macros | EDIT |
| `CountingTropicalGEMM.jl/Project.toml` | Add `TropicalNumbers`, `Mods` deps | EDIT |
| `CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl` | New Julia API: types, dispatch, FFI calls, mul! overloads | EDIT |
| `CountingTropicalGEMM.jl/test/runtests.jl` | New testsets for slow + fast paths, errors | EDIT |

All edits live in those six files. No changes to existing kernels, existing C ABI entries, or existing Julia entry points.

---

## Task 1: Rust driver — slow path (`matmul_mod_p`)

**Files:**
- Create: `crates/tropical-gemm-cuda/src/matmul_mod.rs`
- Modify: `crates/tropical-gemm-cuda/src/lib.rs` (add `pub mod matmul_mod;`)

- [ ] **Step 1: Write the failing test**

Create `crates/tropical-gemm-cuda/src/matmul_mod.rs` with this initial content (test only — function not yet defined):

```rust
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
    todo!("implement in next step")
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
```

In `crates/tropical-gemm-cuda/src/lib.rs`, after the existing `pub mod c_api;` line, add:

```rust
pub mod matmul_mod;
```

- [ ] **Step 2: Run test to verify it fails (compile-fail OK)**

Run:
```bash
cargo test -p tropical-gemm-cuda --features cuda matmul_mod_p_2x2_max_p7 -- --nocapture
```
Expected: compile success but `panic at "implement in next step"` OR `not yet implemented`.

- [ ] **Step 3: Implement the driver**

Replace the `todo!` body in `matmul_mod_p` with:

```rust
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
```

- [ ] **Step 4: Add the `pack_pair` trait method to `PackPair`**

Modify `crates/tropical-gemm-cuda/src/pair.rs`:

In the `PackPair` trait definition, add a second method:

```rust
pub trait PackPair: Copy {
    type Pair: DeviceRepr + ValidAsZeroBits + Default + Clone + Copy + 'static;
    fn pack_ones(values: &[Self]) -> Vec<Self::Pair>;
    fn pack_pair(values: &[Self], counts: &[i32]) -> Vec<Self::Pair>;
}
```

Update both impls:

```rust
impl PackPair for f32 {
    type Pair = PairF32;
    fn pack_ones(values: &[Self]) -> Vec<Self::Pair> { pack_f32_ones(values) }
    fn pack_pair(values: &[Self], counts: &[i32]) -> Vec<Self::Pair> {
        pack_f32(values, counts)
    }
}

impl PackPair for f64 {
    type Pair = PairF64;
    fn pack_ones(values: &[Self]) -> Vec<Self::Pair> { pack_f64_ones(values) }
    fn pack_pair(values: &[Self], counts: &[i32]) -> Vec<Self::Pair> {
        pack_f64(values, counts)
    }
}
```

(`pack_f32` and `pack_f64` already exist at lines 52 and 62.)

- [ ] **Step 5: Run the test to verify it passes**

Run:
```bash
cargo test -p tropical-gemm-cuda --features cuda matmul_mod_p_2x2_max_p7 -- --nocapture
```
Expected: PASS.

- [ ] **Step 6: Verify nothing else broke**

Run:
```bash
cargo test -p tropical-gemm-cuda --features cuda 2>&1 | tail -10
```
Expected: all existing tests pass; new test passes.

- [ ] **Step 7: Commit**

```bash
git add crates/tropical-gemm-cuda/src/matmul_mod.rs \
        crates/tropical-gemm-cuda/src/lib.rs \
        crates/tropical-gemm-cuda/src/pair.rs
git commit -m "$(cat <<'EOF'
Spec K: add matmul_mod_p Rust driver (slow path)

Single-prime general AoS counting matmul for callers who want raw
mod-P residues without CRT or BigInt. Packs (val, cnt) host-side via
new PackPair::pack_pair trait method, uploads as Pair, launches the
existing launch_counting_gemm general kernel, downloads results.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Rust driver — fast path (`matmul_mod_p_pair`)

**Files:**
- Modify: `crates/tropical-gemm-cuda/src/matmul_mod.rs` (add second function)

- [ ] **Step 1: Write the failing test**

Append to the `tests` module in `matmul_mod.rs`:

```rust
    #[test]
    fn matmul_mod_p_pair_2x2_max_p7() {
        // Same setup as the slow-path test, but caller pre-packs into PairF32.
        use crate::pair::PairF32;
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let pair_a = vec![
            PairF32::new(1.0, 1), PairF32::new(2.0, 1),
            PairF32::new(3.0, 1), PairF32::new(4.0, 1),
        ];
        let pair_b = vec![
            PairF32::new(5.0, 1), PairF32::new(6.0, 1),
            PairF32::new(7.0, 1), PairF32::new(8.0, 1),
        ];
        let mut out_val = vec![0.0_f32; 4];
        let mut out_cnt = vec![0_i32; 4];

        matmul_mod_p_pair::<f32, Max>(
            ctx, &pair_a, 2, 2, &pair_b, 2, 7,
            &mut out_val, &mut out_cnt,
        ).expect("driver ok");

        assert_eq!(out_val, vec![9.0_f32, 10.0, 11.0, 12.0]);
        assert_eq!(out_cnt, vec![1_i32, 1, 1, 1]);
    }
```

- [ ] **Step 2: Run to verify it fails to compile**

Run:
```bash
cargo test -p tropical-gemm-cuda --features cuda matmul_mod_p_pair_2x2_max_p7 -- --nocapture
```
Expected: compile error — `matmul_mod_p_pair` is not defined.

- [ ] **Step 3: Add `matmul_mod_p_pair` function**

In `crates/tropical-gemm-cuda/src/matmul_mod.rs`, add after `matmul_mod_p`:

```rust
/// Fast path: caller has already packed into the device-compatible
/// `PairT` layout. Used by Julia callers whose host-side
/// `Matrix{CountingTropical{T, Mod{P, Int32}}}` is byte-compatible with
/// `PairT` and can be reinterpreted with no per-element split.
///
/// `pair_a` is M × K row-major; `pair_b` is K × N row-major.
pub fn matmul_mod_p_pair<T, D>(
    ctx: &CudaContext,
    pair_a: &[<T as PackPair>::Pair],
    m: usize,
    k: usize,
    pair_b: &[<T as PackPair>::Pair],
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
    if p < P_MIN || p < 0 {
        return Err(CudaError::InvalidState(format!(
            "modulus must satisfy {} <= p < 2^31, got {}",
            P_MIN, p
        )));
    }
    if pair_a.len() != m * k
        || pair_b.len() != k * n
        || out_val.len() != m * n
        || out_cnt.len() != m * n
    {
        return Err(CudaError::InvalidState(format!(
            "buffer length mismatch: m={}, k={}, n={}, but \
             pair_a={} pair_b={} out_val={} out_cnt={}",
            m, k, n,
            pair_a.len(), pair_b.len(), out_val.len(), out_cnt.len()
        )));
    }
    if m == 0 || k == 0 || n == 0 {
        return Err(CudaError::InvalidState(
            "dimensions must be non-zero".into(),
        ));
    }

    // Upload directly — no host-side pack.
    let pair_a_dev = GpuMatrix::<<T as PackPair>::Pair>::from_host(ctx, pair_a, m, k)?;
    let pair_b_dev = GpuMatrix::<<T as PackPair>::Pair>::from_host(ctx, pair_b, k, n)?;

    let mut value_c = GpuMatrix::<T>::alloc(ctx, m, n)?;
    let mut count_c = GpuMatrix::<i32>::alloc(ctx, m, n)?;

    launch_counting_gemm::<T, D>(
        ctx,
        &pair_a_dev,
        &pair_b_dev,
        &mut value_c,
        &mut count_c,
        p,
    )?;

    let host_val = value_c.to_host(ctx)?;
    let host_cnt = count_c.to_host(ctx)?;
    out_val.copy_from_slice(&host_val);
    out_cnt.copy_from_slice(&host_cnt);

    Ok(())
}
```

- [ ] **Step 4: Run test to verify pass**

Run:
```bash
cargo test -p tropical-gemm-cuda --features cuda matmul_mod_p_pair_2x2_max_p7 -- --nocapture
```
Expected: PASS.

- [ ] **Step 5: Add a Min-direction smoke test**

Append to the `tests` module:

```rust
    #[test]
    fn matmul_mod_p_4x4_min_random_p11() {
        use tropical_gemm::types::Min;
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let m = 4usize; let k = 6usize; let n = 4usize;
        // Discrete inputs to force ties.
        let a_val: Vec<f64> = (0..m*k).map(|i| (i % 5) as f64).collect();
        let a_cnt: Vec<i32> = (0..m*k).map(|i| ((i + 1) % 11) as i32).collect();
        let b_val: Vec<f64> = (0..k*n).map(|i| (i % 4) as f64).collect();
        let b_cnt: Vec<i32> = (0..k*n).map(|i| ((i + 2) % 11) as i32).collect();

        let mut out_val = vec![0.0_f64; m*n];
        let mut out_cnt = vec![0_i32; m*n];
        matmul_mod_p::<f64, Min>(
            ctx, &a_val, &a_cnt, m, k, &b_val, &b_cnt, n, 11,
            &mut out_val, &mut out_cnt,
        ).expect("driver ok");

        // Reference.
        for i in 0..m {
            for j in 0..n {
                let mut best = f64::INFINITY;
                let mut acc: i64 = 0;
                for kk in 0..k {
                    let v = a_val[i*k + kk] + b_val[kk*n + j];
                    if v < best {
                        best = v;
                        acc = (a_cnt[i*k + kk] as i64) * (b_cnt[kk*n + j] as i64) % 11;
                    } else if v == best {
                        acc = (acc + (a_cnt[i*k + kk] as i64) * (b_cnt[kk*n + j] as i64)) % 11;
                    }
                }
                assert_eq!(out_val[i*n + j], best, "value mismatch at ({},{})", i, j);
                assert_eq!(out_cnt[i*n + j] as i64, acc, "count mismatch at ({},{})", i, j);
            }
        }
    }
```

- [ ] **Step 6: Run all matmul_mod tests**

Run:
```bash
cargo test -p tropical-gemm-cuda --features cuda matmul_mod -- --nocapture
```
Expected: 3/3 PASS.

- [ ] **Step 7: Commit**

```bash
git add crates/tropical-gemm-cuda/src/matmul_mod.rs
git commit -m "$(cat <<'EOF'
Spec K: add matmul_mod_p_pair fast-path Rust driver

Variant of matmul_mod_p taking pre-packed PairT input slices. Skips
the host-side (val, cnt) zip — used by Julia callers whose
CountingTropical matrix layout already matches PairT. Adds a Min
direction f64 randomized cross-check test.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: C ABI — slow path entries

**Files:**
- Modify: `crates/tropical-gemm-cuda/src/c_api.rs`

- [ ] **Step 1: Write the failing test**

In `c_api.rs`, append to the existing `tests` module:

```rust
    #[test]
    fn matmul_mod_p_f32_max_smoke() {
        let a_val = vec![1.0_f32, 2.0, 3.0, 4.0];
        let a_cnt = vec![1_i32, 1, 1, 1];
        let b_val = vec![5.0_f32, 6.0, 7.0, 8.0];
        let b_cnt = vec![1_i32, 1, 1, 1];
        let mut out_val = vec![0.0_f32; 4];
        let mut out_cnt = vec![0_i32; 4];

        let code = tg_matmul_mod_p_f32_max(
            a_val.as_ptr(), a_cnt.as_ptr(), 2, 2,
            b_val.as_ptr(), b_cnt.as_ptr(), 2,
            7,
            out_val.as_mut_ptr(), out_cnt.as_mut_ptr(),
        );
        assert_eq!(code, OK);
        assert_eq!(out_val, vec![9.0_f32, 10.0, 11.0, 12.0]);
        assert_eq!(out_cnt, vec![1_i32, 1, 1, 1]);
    }

    #[test]
    fn matmul_mod_p_invalid_p_returns_invalid() {
        let a_val = vec![1.0_f32; 4]; let a_cnt = vec![1_i32; 4];
        let b_val = vec![1.0_f32; 4]; let b_cnt = vec![1_i32; 4];
        let mut out_val = vec![0.0_f32; 4];
        let mut out_cnt = vec![0_i32; 4];
        let code = tg_matmul_mod_p_f32_max(
            a_val.as_ptr(), a_cnt.as_ptr(), 2, 2,
            b_val.as_ptr(), b_cnt.as_ptr(), 2,
            1,    // p=1 invalid
            out_val.as_mut_ptr(), out_cnt.as_mut_ptr(),
        );
        assert_eq!(code, ERR_INVALID_INPUT);
    }
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cargo test -p tropical-gemm-cuda --features cuda matmul_mod_p_f32_max_smoke -- --nocapture
```
Expected: compile error — `tg_matmul_mod_p_f32_max` not defined.

- [ ] **Step 3: Add slow-path C ABI entries**

In `c_api.rs`, after the imports near the top (around line 41), add:

```rust
use crate::matmul_mod::{matmul_mod_p, matmul_mod_p_pair};
```

After `cabi_count_ground_states_u64!` invocations (around line 199), add:

```rust
// ---------------------------------------------------------------------------
// Spec K: single-prime mod-P matmul. Two C ABI families:
//   - tg_matmul_mod_p_<T>_<D>      — split (val, cnt) input buffers (slow path)
//   - tg_matmul_mod_p_pair_<T>_<D> — pre-packed Pair input buffers   (fast path)
// Both produce SoA (val, cnt) output buffers.
// ---------------------------------------------------------------------------

fn run_matmul_mod_p<T, D>(
    a_val: *const T, a_cnt: *const i32,
    m: usize, k: usize,
    b_val: *const T, b_cnt: *const i32,
    n: usize,
    p: i32,
    out_val: *mut T, out_cnt: *mut i32,
) -> i32
where
    T: tropical_gemm::types::TropicalScalar
        + cudarc::driver::DeviceRepr
        + cudarc::driver::ValidAsZeroBits
        + Default + Clone + Copy
        + crate::pair::PackPair
        + 'static,
    D: tropical_gemm::types::TropicalDirection,
    (T, D): crate::counting_kernel::CountingCudaKernel<T, D>,
{
    if a_val.is_null() || a_cnt.is_null()
        || b_val.is_null() || b_cnt.is_null()
        || out_val.is_null() || out_cnt.is_null()
    {
        store_error("null pointer");
        return ERR_INVALID_INPUT;
    }
    if m == 0 || k == 0 || n == 0 {
        store_error("dimensions must be non-zero");
        return ERR_INVALID_INPUT;
    }
    if p < 2 {
        store_error(format!("modulus must be >= 2, got {}", p));
        return ERR_INVALID_INPUT;
    }

    let a_val_s = unsafe { std::slice::from_raw_parts(a_val, m * k) };
    let a_cnt_s = unsafe { std::slice::from_raw_parts(a_cnt, m * k) };
    let b_val_s = unsafe { std::slice::from_raw_parts(b_val, k * n) };
    let b_cnt_s = unsafe { std::slice::from_raw_parts(b_cnt, k * n) };
    let out_val_s = unsafe { std::slice::from_raw_parts_mut(out_val, m * n) };
    let out_cnt_s = unsafe { std::slice::from_raw_parts_mut(out_cnt, m * n) };

    let ctx = match get_global_context() {
        Ok(c) => c,
        Err(e) => {
            store_error(format!("CUDA context init failed: {}", e));
            return ERR_CUDA;
        }
    };

    match matmul_mod_p::<T, D>(
        ctx, a_val_s, a_cnt_s, m, k, b_val_s, b_cnt_s, n, p,
        out_val_s, out_cnt_s,
    ) {
        Ok(()) => OK,
        Err(e) => { store_error(format!("{}", e)); ERR_CUDA }
    }
}

macro_rules! cabi_matmul_mod_p {
    ($name:ident, $T:ty, $D:ty) => {
        #[no_mangle]
        pub extern "C" fn $name(
            a_val: *const $T, a_cnt: *const i32,
            m: usize, k: usize,
            b_val: *const $T, b_cnt: *const i32,
            n: usize,
            p: i32,
            out_val: *mut $T, out_cnt: *mut i32,
        ) -> c_int {
            let res = catch_unwind(AssertUnwindSafe(|| {
                run_matmul_mod_p::<$T, $D>(
                    a_val, a_cnt, m, k, b_val, b_cnt, n, p, out_val, out_cnt,
                )
            }));
            match res {
                Ok(code) => code,
                Err(_) => { store_error("Rust panic across FFI boundary"); ERR_INTERNAL }
            }
        }
    };
}

cabi_matmul_mod_p!(tg_matmul_mod_p_f32_max, f32, Max);
cabi_matmul_mod_p!(tg_matmul_mod_p_f32_min, f32, Min);
cabi_matmul_mod_p!(tg_matmul_mod_p_f64_max, f64, Max);
cabi_matmul_mod_p!(tg_matmul_mod_p_f64_min, f64, Min);
```

- [ ] **Step 4: Run tests to verify pass**

Run:
```bash
cargo test -p tropical-gemm-cuda --features cuda matmul_mod_p_f32_max_smoke matmul_mod_p_invalid_p_returns_invalid -- --nocapture
```
Expected: 2/2 PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/tropical-gemm-cuda/src/c_api.rs
git commit -m "$(cat <<'EOF'
Spec K: add tg_matmul_mod_p_<T>_<D> C ABI (slow path)

Four extern C entries (f32/f64 × Max/Min) wrapping matmul_mod_p
with split (val, cnt) input buffers, catch_unwind for panic safety,
and standard error-code/last-error-message handling.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: C ABI — fast path entries

**Files:**
- Modify: `crates/tropical-gemm-cuda/src/c_api.rs`

- [ ] **Step 1: Write the failing test**

Append to the `tests` module in `c_api.rs`:

```rust
    #[test]
    fn matmul_mod_p_pair_f32_max_smoke() {
        use crate::pair::PairF32;
        let pair_a = vec![
            PairF32::new(1.0, 1), PairF32::new(2.0, 1),
            PairF32::new(3.0, 1), PairF32::new(4.0, 1),
        ];
        let pair_b = vec![
            PairF32::new(5.0, 1), PairF32::new(6.0, 1),
            PairF32::new(7.0, 1), PairF32::new(8.0, 1),
        ];
        let mut out_val = vec![0.0_f32; 4];
        let mut out_cnt = vec![0_i32; 4];

        let code = tg_matmul_mod_p_pair_f32_max(
            pair_a.as_ptr() as *const std::ffi::c_void, 2, 2,
            pair_b.as_ptr() as *const std::ffi::c_void, 2,
            7,
            out_val.as_mut_ptr(), out_cnt.as_mut_ptr(),
        );
        assert_eq!(code, OK);
        assert_eq!(out_val, vec![9.0_f32, 10.0, 11.0, 12.0]);
        assert_eq!(out_cnt, vec![1_i32, 1, 1, 1]);
    }
```

- [ ] **Step 2: Run to verify failure**

Run:
```bash
cargo test -p tropical-gemm-cuda --features cuda matmul_mod_p_pair_f32_max_smoke -- --nocapture
```
Expected: compile error — `tg_matmul_mod_p_pair_f32_max` not defined.

- [ ] **Step 3: Add fast-path C ABI entries**

In `c_api.rs`, after the slow-path additions from Task 3, append:

```rust
fn run_matmul_mod_p_pair<T, D>(
    pair_a: *const std::ffi::c_void,
    m: usize, k: usize,
    pair_b: *const std::ffi::c_void,
    n: usize,
    p: i32,
    out_val: *mut T, out_cnt: *mut i32,
) -> i32
where
    T: tropical_gemm::types::TropicalScalar
        + cudarc::driver::DeviceRepr
        + cudarc::driver::ValidAsZeroBits
        + Default + Clone + Copy
        + crate::pair::PackPair
        + 'static,
    D: tropical_gemm::types::TropicalDirection,
    (T, D): crate::counting_kernel::CountingCudaKernel<T, D>,
{
    if pair_a.is_null() || pair_b.is_null()
        || out_val.is_null() || out_cnt.is_null()
    {
        store_error("null pointer");
        return ERR_INVALID_INPUT;
    }
    if m == 0 || k == 0 || n == 0 {
        store_error("dimensions must be non-zero");
        return ERR_INVALID_INPUT;
    }
    if p < 2 {
        store_error(format!("modulus must be >= 2, got {}", p));
        return ERR_INVALID_INPUT;
    }

    // SAFETY: caller asserts the pointers reference valid Pair-typed buffers
    // of length m*k and k*n respectively, with alignment matching `PairT`
    // (8 B for PairF32, 16 B for PairF64). Julia's heap allocator returns
    // ≥16 B-aligned arrays, and the Pair stride matches the element size,
    // so this is satisfied for `Matrix{CountingTropical{T, Mod{P, Int32}}}`.
    let pair_a_typed = pair_a as *const <T as crate::pair::PackPair>::Pair;
    let pair_b_typed = pair_b as *const <T as crate::pair::PackPair>::Pair;
    let pair_a_s = unsafe { std::slice::from_raw_parts(pair_a_typed, m * k) };
    let pair_b_s = unsafe { std::slice::from_raw_parts(pair_b_typed, k * n) };
    let out_val_s = unsafe { std::slice::from_raw_parts_mut(out_val, m * n) };
    let out_cnt_s = unsafe { std::slice::from_raw_parts_mut(out_cnt, m * n) };

    let ctx = match get_global_context() {
        Ok(c) => c,
        Err(e) => {
            store_error(format!("CUDA context init failed: {}", e));
            return ERR_CUDA;
        }
    };

    match matmul_mod_p_pair::<T, D>(
        ctx, pair_a_s, m, k, pair_b_s, n, p,
        out_val_s, out_cnt_s,
    ) {
        Ok(()) => OK,
        Err(e) => { store_error(format!("{}", e)); ERR_CUDA }
    }
}

macro_rules! cabi_matmul_mod_p_pair {
    ($name:ident, $T:ty, $D:ty) => {
        #[no_mangle]
        pub extern "C" fn $name(
            pair_a: *const std::ffi::c_void,
            m: usize, k: usize,
            pair_b: *const std::ffi::c_void,
            n: usize,
            p: i32,
            out_val: *mut $T, out_cnt: *mut i32,
        ) -> c_int {
            let res = catch_unwind(AssertUnwindSafe(|| {
                run_matmul_mod_p_pair::<$T, $D>(
                    pair_a, m, k, pair_b, n, p, out_val, out_cnt,
                )
            }));
            match res {
                Ok(code) => code,
                Err(_) => { store_error("Rust panic across FFI boundary"); ERR_INTERNAL }
            }
        }
    };
}

cabi_matmul_mod_p_pair!(tg_matmul_mod_p_pair_f32_max, f32, Max);
cabi_matmul_mod_p_pair!(tg_matmul_mod_p_pair_f32_min, f32, Min);
cabi_matmul_mod_p_pair!(tg_matmul_mod_p_pair_f64_max, f64, Max);
cabi_matmul_mod_p_pair!(tg_matmul_mod_p_pair_f64_min, f64, Min);
```

- [ ] **Step 4: Run tests to verify pass**

Run:
```bash
cargo test -p tropical-gemm-cuda --features cuda matmul_mod -- --nocapture
```
Expected: all matmul_mod tests pass (5 tests now).

- [ ] **Step 5: Verify full crate test suite**

Run:
```bash
cargo test -p tropical-gemm-cuda --features cuda 2>&1 | tail -10
```
Expected: all tests pass, no regressions.

- [ ] **Step 6: Build the cdylib for Julia**

Run:
```bash
cargo build --release -p tropical-gemm-cuda
```
Expected: success; `target/release/libtropical_gemm_cuda.so` exists.

- [ ] **Step 7: Commit**

```bash
git add crates/tropical-gemm-cuda/src/c_api.rs
git commit -m "$(cat <<'EOF'
Spec K: add tg_matmul_mod_p_pair_<T>_<D> C ABI (fast path)

Four extern C entries (f32/f64 × Max/Min) taking pre-packed PairT
input as opaque c_void pointers. Caller asserts alignment matches
PairT (8 B for f32, 16 B for f64); satisfied by Julia's heap-aligned
Matrix{CountingTropical{T, Mod{P, Int32}}} reinterpretation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Julia deps + min-plus type

**Files:**
- Modify: `CountingTropicalGEMM.jl/Project.toml`
- Modify: `CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl`

- [ ] **Step 1: Read current Project.toml**

```bash
cat CountingTropicalGEMM.jl/Project.toml
```

- [ ] **Step 2: Add deps**

Edit `CountingTropicalGEMM.jl/Project.toml` to include `TropicalNumbers`, `Mods`, and `LinearAlgebra` in the `[deps]` block. Add a `[compat]` entry for each. Example block to ensure (replace UUIDs by running `julia -e 'using Pkg; Pkg.add(...)'` if not known):

```toml
[deps]
Libdl = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Mods = "7475f97c-0381-5b22-a0d6-3ad79a8aacd8"
TropicalNumbers = "b3a74e9c-7526-4ef1-bb2d-d35e4c50c1bc"

[compat]
Mods = "2"
TropicalNumbers = "0.6"
julia = "1.9"
```

- [ ] **Step 3: Resolve and instantiate**

```bash
cd CountingTropicalGEMM.jl && julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()' && cd ..
```
Expected: success.

- [ ] **Step 4: Define `CountingTropicalMin` in the Julia module**

Append to `CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl`, just above the closing `end # module`:

```julia
# ---------------------------------------------------------------------------
# Min-plus counterpart of TropicalNumbers.jl's CountingTropical.
# CountingTropical is max-plus; we define a parallel min-plus type so the
# same FFI driver can serve both directions.
# ---------------------------------------------------------------------------
"""
    CountingTropicalMin{T, CT}(n, c)

Min-plus counting tropical number: `n::T` is the value, `c::CT` is the
ground-state multiplicity. Semiring operations: `+` takes the smaller
`n` (sum counts on tie); `*` adds `n` and multiplies counts.
"""
struct CountingTropicalMin{T, CT}
    n::T
    c::CT
end

Base.zero(::Type{CountingTropicalMin{T, CT}}) where {T, CT} =
    CountingTropicalMin{T, CT}(typemax(T), zero(CT))
Base.one(::Type{CountingTropicalMin{T, CT}}) where {T, CT} =
    CountingTropicalMin{T, CT}(zero(T), one(CT))

function Base.:+(a::CountingTropicalMin{T, CT}, b::CountingTropicalMin{T, CT}) where {T, CT}
    if a.n < b.n
        a
    elseif b.n < a.n
        b
    else
        CountingTropicalMin{T, CT}(a.n, a.c + b.c)
    end
end

Base.:*(a::CountingTropicalMin{T, CT}, b::CountingTropicalMin{T, CT}) where {T, CT} =
    CountingTropicalMin{T, CT}(a.n + b.n, a.c * b.c)

Base.:(==)(a::CountingTropicalMin, b::CountingTropicalMin) =
    a.n == b.n && a.c == b.c

export CountingTropicalMin
```

- [ ] **Step 5: Smoke-load to confirm syntax**

```bash
cd CountingTropicalGEMM.jl && julia --project=. -e '
using CountingTropicalGEMM
using Mods
a = CountingTropicalMin{Float32, Mod{7, Int}}(1.0f0, Mod{7}(2))
b = CountingTropicalMin{Float32, Mod{7, Int}}(2.0f0, Mod{7}(3))
println(a + b)
println(a * b)
' && cd ..
```
Expected: prints two `CountingTropicalMin{Float32, Mod{7, Int64}}(...)` values without error.

- [ ] **Step 6: Commit**

```bash
git add CountingTropicalGEMM.jl/Project.toml \
        CountingTropicalGEMM.jl/Manifest.toml \
        CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl
git commit -m "$(cat <<'EOF'
Spec K: add Mods/TropicalNumbers deps and CountingTropicalMin

Min-plus counterpart of TropicalNumbers.jl's CountingTropical
defined locally with semiring + and * for use in the new mod-P
matmul Julia API.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Julia API — slow-path FFI wrapper + `tropical_matmul`

**Files:**
- Modify: `CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl`

- [ ] **Step 1: Add a failing test**

Append to `CountingTropicalGEMM.jl/test/runtests.jl`, before the final `end`:

```julia
    @testset "tropical_matmul slow path (Mod{P, Int}, f32 Max)" begin
        using TropicalNumbers, Mods
        P = 7
        # Random discrete inputs to force ties and exercise mod reduction.
        rng_seed = 1
        Random.seed!(rng_seed)
        A = [CountingTropical{Float32, Mod{P, Int}}(
                Float32(rand(0:3)), Mod{P}(rand(0:P-1))) for i in 1:5, j in 1:8]
        B = [CountingTropical{Float32, Mod{P, Int}}(
                Float32(rand(0:3)), Mod{P}(rand(0:P-1))) for i in 1:8, j in 6]
        # Reference: pure Julia max-plus + count multiply mod P.
        ref = Matrix{CountingTropical{Float32, Mod{P, Int}}}(undef, 5, 6)
        for i in 1:5, j in 1:6
            best_n = -Inf32
            best_c = Mod{P}(0)
            for kk in 1:8
                v = A[i, kk].n + B[kk, j].n
                c = A[i, kk].c * B[kk, j].c
                if v > best_n
                    best_n = v; best_c = c
                elseif v == best_n
                    best_c += c
                end
            end
            ref[i, j] = CountingTropical{Float32, Mod{P, Int}}(best_n, best_c)
        end

        C = tropical_matmul(A, B)
        @test C == ref
    end
```

Also add `using Random` at the top of `runtests.jl` if not already imported.

- [ ] **Step 2: Run to verify failure**

```bash
cd CountingTropicalGEMM.jl && julia --project=. -e 'using Pkg; Pkg.test()' 2>&1 | tail -30; cd ..
```
Expected: error — `tropical_matmul` is not defined / not exported.

- [ ] **Step 3: Add slow-path FFI wrapper and `tropical_matmul`**

In `CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl`, just above the closing `end # module` (and after the `CountingTropicalMin` block from Task 5), add:

```julia
# ---------------------------------------------------------------------------
# Spec K: mod-P single-prime matmul over CountingTropical{T, Mod{P, U}}.
# Two paths:
#   * Slow path (any U): split into Vector{T} + Vector{Int32}, call
#     tg_matmul_mod_p_<T>_<D>.
#   * Fast path (U == Int32): reinterpret the matrix as PairT and call
#     tg_matmul_mod_p_pair_<T>_<D>.
# Both produce SoA outputs, which the wrapper zips back into the matrix.
# ---------------------------------------------------------------------------
using TropicalNumbers
using Mods
using LinearAlgebra

export tropical_matmul, tropical_matmul_min

# Validate prime fits in i32 positive.
@inline function _check_p(p::Integer)
    if !(2 <= p < (Int64(1) << 31))
        throw(ArgumentError("modulus must satisfy 2 <= P < 2^31, got $p"))
    end
end

# Extract (val, cnt) split arrays from a CountingTropical (or CountingTropicalMin)
# matrix in row-major order (column-major Julia matrix → row-major buffer).
function _split_rowmajor(M::AbstractMatrix{<:Union{
        CountingTropical{T, Mod{P, U}},
        CountingTropicalMin{T, Mod{P, U}}
    }}) where {T, P, U}
    rows, cols = size(M)
    val = Vector{T}(undef, rows * cols)
    cnt = Vector{Int32}(undef, rows * cols)
    @inbounds for i in 1:rows, j in 1:cols
        e = M[i, j]
        val[(i - 1) * cols + j] = e.n
        cnt[(i - 1) * cols + j] = Int32(e.c.val)
    end
    return val, cnt
end

# Re-zip flat row-major (val, cnt) into a column-major Matrix{CT{T, Mod{P}}}.
function _zip_rowmajor(::Type{CT}, val::Vector{T}, cnt::Vector{Int32},
                      rows::Int, cols::Int, ::Val{P}, ::Type{U}
                     ) where {CT, T, P, U}
    out = Matrix{CT{T, Mod{P, U}}}(undef, rows, cols)
    @inbounds for i in 1:rows, j in 1:cols
        n = val[(i - 1) * cols + j]
        c = Mod{P, U}(Int(cnt[(i - 1) * cols + j]))
        out[i, j] = CT{T, Mod{P, U}}(n, c)
    end
    return out
end

# Map (T, dir_sym) → (slow_sym, pair_sym) FFI symbol pair.
const _FFI_SYMS = Dict(
    (Float32, :max) => (:tg_matmul_mod_p_f32_max, :tg_matmul_mod_p_pair_f32_max),
    (Float32, :min) => (:tg_matmul_mod_p_f32_min, :tg_matmul_mod_p_pair_f32_min),
    (Float64, :max) => (:tg_matmul_mod_p_f64_max, :tg_matmul_mod_p_pair_f64_max),
    (Float64, :min) => (:tg_matmul_mod_p_f64_min, :tg_matmul_mod_p_pair_f64_min),
)

# Internal slow-path call. ccall with (T, dir_sym) chosen by caller.
function _ccall_slow(slow_sym::Symbol, ::Type{T},
                    a_val::Vector{T}, a_cnt::Vector{Int32}, m::Int, k::Int,
                    b_val::Vector{T}, b_cnt::Vector{Int32}, n::Int,
                    p::Int32,
                    out_val::Vector{T}, out_cnt::Vector{Int32}) where {T}
    _check_version()
    code = ccall((slow_sym, _libpath()), Cint,
        (Ptr{T}, Ptr{Int32}, Csize_t, Csize_t,
         Ptr{T}, Ptr{Int32}, Csize_t,
         Int32,
         Ptr{T}, Ptr{Int32}),
        a_val, a_cnt, m, k,
        b_val, b_cnt, n,
        p,
        out_val, out_cnt)
    if code != Int32(0)
        _throw_for(Int32(code))
    end
    return nothing
end

# Common matmul body. `CT` is CountingTropical (max) or CountingTropicalMin (min).
function _matmul_mod_p_slow(::Type{CT}, dir_sym::Symbol,
                            A::AbstractMatrix{CT{T, Mod{P, U}}},
                            B::AbstractMatrix{CT{T, Mod{P, U}}}
                           ) where {CT, T <: Union{Float32, Float64}, P, U}
    m, k = size(A); k2, n = size(B)
    k == k2 || throw(DimensionMismatch(
        "A is $(size(A)) but B is $(size(B)); inner dims must match"))
    _check_p(P)

    a_val, a_cnt = _split_rowmajor(A)
    b_val, b_cnt = _split_rowmajor(B)
    out_val = Vector{T}(undef, m * n)
    out_cnt = Vector{Int32}(undef, m * n)

    slow_sym, _ = _FFI_SYMS[(T, dir_sym)]
    _ccall_slow(slow_sym, T, a_val, a_cnt, m, k, b_val, b_cnt, n,
                Int32(P), out_val, out_cnt)
    return _zip_rowmajor(CT, out_val, out_cnt, m, n, Val(P), U)
end

# Public entry: max-plus.
function tropical_matmul(A::AbstractMatrix{CountingTropical{T, Mod{P, U}}},
                         B::AbstractMatrix{CountingTropical{T, Mod{P, U}}}
                        ) where {T <: Union{Float32, Float64}, P, U}
    return _matmul_mod_p_slow(CountingTropical, :max, A, B)
end

# Public entry: min-plus.
function tropical_matmul_min(A::AbstractMatrix{CountingTropicalMin{T, Mod{P, U}}},
                             B::AbstractMatrix{CountingTropicalMin{T, Mod{P, U}}}
                            ) where {T <: Union{Float32, Float64}, P, U}
    return _matmul_mod_p_slow(CountingTropicalMin, :min, A, B)
end
```

- [ ] **Step 4: Run the new test**

```bash
cd CountingTropicalGEMM.jl && bash -c 'module load cuda && julia --project=. -e "using Pkg; Pkg.test()"' 2>&1 | tail -25; cd ..
```
Expected: PASS for the new testset; existing tests still pass.

- [ ] **Step 5: Commit**

```bash
git add CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl \
        CountingTropicalGEMM.jl/test/runtests.jl
git commit -m "$(cat <<'EOF'
Spec K: add tropical_matmul / tropical_matmul_min slow path

Julia API operating on Matrix{CountingTropical{T, Mod{P, U}}} (and
CountingTropicalMin counterpart). Splits values and counts at the
boundary, ccalls the new tg_matmul_mod_p_<T>_<D> C ABI entries,
and re-zips the SoA output. Works for any count storage type U.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Julia API — fast-path dispatch on `Mod{P, Int32}`

**Files:**
- Modify: `CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl`
- Modify: `CountingTropicalGEMM.jl/test/runtests.jl`

- [ ] **Step 1: Add a failing test**

Append to `runtests.jl` before the final `end`:

```julia
    @testset "tropical_matmul fast path (Mod{P, Int32}, f64 Max)" begin
        using TropicalNumbers, Mods
        P = 11
        Random.seed!(2)
        A = [CountingTropical{Float64, Mod{P, Int32}}(
                Float64(rand(0:4)), Mod{P, Int32}(rand(0:P-1))) for i in 1:6, j in 1:9]
        B = [CountingTropical{Float64, Mod{P, Int32}}(
                Float64(rand(0:4)), Mod{P, Int32}(rand(0:P-1))) for i in 1:9, j in 7]
        # Reference (same logic as slow-path test).
        ref = Matrix{CountingTropical{Float64, Mod{P, Int32}}}(undef, 6, 7)
        for i in 1:6, j in 1:7
            best_n = -Inf
            best_c = Mod{P, Int32}(0)
            for kk in 1:9
                v = A[i, kk].n + B[kk, j].n
                c = A[i, kk].c * B[kk, j].c
                if v > best_n
                    best_n = v; best_c = c
                elseif v == best_n
                    best_c += c
                end
            end
            ref[i, j] = CountingTropical{Float64, Mod{P, Int32}}(best_n, best_c)
        end

        C = tropical_matmul(A, B)
        @test C == ref
    end
```

- [ ] **Step 2: Run to verify it currently goes through slow path (still passes)**

The existing slow-path code accepts `U == Int32` and routes through the slow C ABI. So the test will pass via slow path. We need to verify the fast path is exercised separately. Add a debug counter first.

```bash
cd CountingTropicalGEMM.jl && bash -c 'module load cuda && julia --project=. -e "using Pkg; Pkg.test()"' 2>&1 | tail -20; cd ..
```
Expected: existing + fast-path test all pass via slow path.

- [ ] **Step 3: Add the fast-path internals**

In `CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl`, after `_matmul_mod_p_slow`, add:

```julia
# Pair structs matching the Rust/CUDA layout (must keep in sync).
struct PairF32
    val::Float32
    cnt::Int32
end
struct PairF64
    val::Float64
    cnt::Int32
    _pad::Int32
end

# Map T → PairT.
_pair_type(::Type{Float32}) = PairF32
_pair_type(::Type{Float64}) = PairF64

# Reinterpret a row-major-ready buffer of CountingTropical{T, Mod{P, Int32}}
# (or its Min variant) as PairT. Caller has already transposed to row-major.
function _row_major_pair_buffer(A::AbstractMatrix{<:Union{
        CountingTropical{T, Mod{P, Int32}},
        CountingTropicalMin{T, Mod{P, Int32}}
    }}) where {T <: Union{Float32, Float64}, P}
    rows, cols = size(A)
    PT = _pair_type(T)
    buf = Vector{PT}(undef, rows * cols)
    @inbounds for i in 1:rows, j in 1:cols
        e = A[i, j]
        # Construct PairT by field; this is the "non-zero-copy" honest path.
        # The struct layout match means GC.@preserve + reinterpret would also
        # work, but constructing fields is robust against future layout drift.
        buf[(i - 1) * cols + j] = PT === PairF32 ?
            PairF32(e.n, e.c.val) :
            PairF64(e.n, e.c.val, Int32(0))
    end
    return buf
end

function _ccall_pair(pair_sym::Symbol, ::Type{T},
                    pair_a::Vector, m::Int, k::Int,
                    pair_b::Vector, n::Int,
                    p::Int32,
                    out_val::Vector{T}, out_cnt::Vector{Int32}) where {T}
    _check_version()
    code = ccall((pair_sym, _libpath()), Cint,
        (Ptr{Cvoid}, Csize_t, Csize_t,
         Ptr{Cvoid}, Csize_t,
         Int32,
         Ptr{T}, Ptr{Int32}),
        pair_a, m, k,
        pair_b, n,
        p,
        out_val, out_cnt)
    if code != Int32(0)
        _throw_for(Int32(code))
    end
    return nothing
end

function _matmul_mod_p_fast(::Type{CT}, dir_sym::Symbol,
                            A::AbstractMatrix{CT{T, Mod{P, Int32}}},
                            B::AbstractMatrix{CT{T, Mod{P, Int32}}}
                           ) where {CT, T <: Union{Float32, Float64}, P}
    m, k = size(A); k2, n = size(B)
    k == k2 || throw(DimensionMismatch(
        "A is $(size(A)) but B is $(size(B)); inner dims must match"))
    _check_p(P)

    pair_a = _row_major_pair_buffer(A)
    pair_b = _row_major_pair_buffer(B)
    out_val = Vector{T}(undef, m * n)
    out_cnt = Vector{Int32}(undef, m * n)

    _, pair_sym = _FFI_SYMS[(T, dir_sym)]
    _ccall_pair(pair_sym, T, pair_a, m, k, pair_b, n,
                Int32(P), out_val, out_cnt)
    return _zip_rowmajor(CT, out_val, out_cnt, m, n, Val(P), Int32)
end

# Override the public entries for U == Int32 to dispatch to the fast path.
function tropical_matmul(A::AbstractMatrix{CountingTropical{T, Mod{P, Int32}}},
                         B::AbstractMatrix{CountingTropical{T, Mod{P, Int32}}}
                        ) where {T <: Union{Float32, Float64}, P}
    return _matmul_mod_p_fast(CountingTropical, :max, A, B)
end

function tropical_matmul_min(A::AbstractMatrix{CountingTropicalMin{T, Mod{P, Int32}}},
                             B::AbstractMatrix{CountingTropicalMin{T, Mod{P, Int32}}}
                            ) where {T <: Union{Float32, Float64}, P}
    return _matmul_mod_p_fast(CountingTropicalMin, :min, A, B)
end
```

- [ ] **Step 4: Run all tests**

```bash
cd CountingTropicalGEMM.jl && bash -c 'module load cuda && julia --project=. -e "using Pkg; Pkg.test()"' 2>&1 | tail -25; cd ..
```
Expected: all tests pass; the `Mod{P, Int32}` test now routes through the fast path.

- [ ] **Step 5: Add a dispatch-coverage assertion**

Append to `runtests.jl`:

```julia
    @testset "tropical_matmul dispatch on U" begin
        using TropicalNumbers, Mods
        # Both should produce the same numeric answer; difference is internal.
        P = 13
        A_int = [CountingTropical{Float32, Mod{P, Int}}(1.0f0, Mod{P}(2))   for _ in 1:3, _ in 1:4]
        B_int = [CountingTropical{Float32, Mod{P, Int}}(1.0f0, Mod{P}(3))   for _ in 1:4, _ in 1:5]
        A_i32 = [CountingTropical{Float32, Mod{P, Int32}}(1.0f0, Mod{P, Int32}(2)) for _ in 1:3, _ in 1:4]
        B_i32 = [CountingTropical{Float32, Mod{P, Int32}}(1.0f0, Mod{P, Int32}(3)) for _ in 1:4, _ in 1:5]
        C_int = tropical_matmul(A_int, B_int)
        C_i32 = tropical_matmul(A_i32, B_i32)
        @test [c.n for c in C_int] == [c.n for c in C_i32]
        @test [Int(c.c.val) for c in C_int] == [Int(c.c.val) for c in C_i32]
    end
```

- [ ] **Step 6: Run all tests again**

```bash
cd CountingTropicalGEMM.jl && bash -c 'module load cuda && julia --project=. -e "using Pkg; Pkg.test()"' 2>&1 | tail -25; cd ..
```
Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl \
        CountingTropicalGEMM.jl/test/runtests.jl
git commit -m "$(cat <<'EOF'
Spec K: add Mod{P, Int32} fast-path dispatch in Julia

When CountingTropical's count storage type U is Int32, dispatch
to the tg_matmul_mod_p_pair_<T>_<D> entry that takes pre-packed
PairT input, skipping the host-side (val, cnt) split. Layout-match
test verifies both paths return identical answers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: `LinearAlgebra.mul!` overloads

**Files:**
- Modify: `CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl`
- Modify: `CountingTropicalGEMM.jl/test/runtests.jl`

- [ ] **Step 1: Add a failing test**

Append to `runtests.jl`:

```julia
    @testset "mul! reuse over CountingTropical" begin
        using TropicalNumbers, Mods, LinearAlgebra
        P = 7
        A = [CountingTropical{Float32, Mod{P, Int32}}(Float32(rand(0:3)), Mod{P, Int32}(rand(0:P-1))) for _ in 1:4, _ in 1:5]
        B = [CountingTropical{Float32, Mod{P, Int32}}(Float32(rand(0:3)), Mod{P, Int32}(rand(0:P-1))) for _ in 1:5, _ in 1:6]
        C = Matrix{CountingTropical{Float32, Mod{P, Int32}}}(undef, 4, 6)
        ref = tropical_matmul(A, B)
        mul!(C, A, B)
        @test C == ref
        # Reuse with different inputs.
        A2 = [CountingTropical{Float32, Mod{P, Int32}}(Float32(rand(0:3)), Mod{P, Int32}(rand(0:P-1))) for _ in 1:4, _ in 1:5]
        ref2 = tropical_matmul(A2, B)
        mul!(C, A2, B)
        @test C == ref2
    end
```

- [ ] **Step 2: Run to verify failure**

```bash
cd CountingTropicalGEMM.jl && bash -c 'module load cuda && julia --project=. -e "using Pkg; Pkg.test()"' 2>&1 | tail -20; cd ..
```
Expected: `MethodError` for `mul!` on these types.

- [ ] **Step 3: Add `mul!` overloads**

In `CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl`, after the `tropical_matmul_min` (Int32) override, add:

```julia
# LinearAlgebra.mul!(C, A, B) — writes the matmul result into preallocated C.
# Implemented by calling tropical_matmul/tropical_matmul_min and copying;
# the underlying FFI does not yet support output-buffer reuse on the device,
# so the savings are at most one host-side allocation.
function LinearAlgebra.mul!(C::AbstractMatrix{CountingTropical{T, Mod{P, U}}},
                            A::AbstractMatrix{CountingTropical{T, Mod{P, U}}},
                            B::AbstractMatrix{CountingTropical{T, Mod{P, U}}}
                           ) where {T <: Union{Float32, Float64}, P, U}
    size(C) == (size(A, 1), size(B, 2)) || throw(DimensionMismatch(
        "C is $(size(C)) but A*B would be $((size(A,1), size(B,2)))"))
    R = tropical_matmul(A, B)
    @inbounds copyto!(C, R)
    return C
end

function LinearAlgebra.mul!(C::AbstractMatrix{CountingTropicalMin{T, Mod{P, U}}},
                            A::AbstractMatrix{CountingTropicalMin{T, Mod{P, U}}},
                            B::AbstractMatrix{CountingTropicalMin{T, Mod{P, U}}}
                           ) where {T <: Union{Float32, Float64}, P, U}
    size(C) == (size(A, 1), size(B, 2)) || throw(DimensionMismatch(
        "C is $(size(C)) but A*B would be $((size(A,1), size(B,2)))"))
    R = tropical_matmul_min(A, B)
    @inbounds copyto!(C, R)
    return C
end
```

- [ ] **Step 4: Run tests**

```bash
cd CountingTropicalGEMM.jl && bash -c 'module load cuda && julia --project=. -e "using Pkg; Pkg.test()"' 2>&1 | tail -20; cd ..
```
Expected: `mul! reuse` test passes; all others still pass.

- [ ] **Step 5: Commit**

```bash
git add CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl \
        CountingTropicalGEMM.jl/test/runtests.jl
git commit -m "$(cat <<'EOF'
Spec K: add LinearAlgebra.mul! overloads for CountingTropical paths

Pre-allocated-output-style API mirroring BLAS mul!. Internally
delegates to tropical_matmul/tropical_matmul_min and copyto!s into
C — true device-side output reuse is a follow-up (CountingTropicalMin
shares the same wrapper).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Min direction + edge cases + error tests

**Files:**
- Modify: `CountingTropicalGEMM.jl/test/runtests.jl`

- [ ] **Step 1: Add Min-direction tests**

Append to `runtests.jl`:

```julia
    @testset "tropical_matmul_min cross-check (f64, fast path)" begin
        using Mods
        P = 13
        Random.seed!(3)
        A = [CountingTropicalMin{Float64, Mod{P, Int32}}(
                Float64(rand(0:4)), Mod{P, Int32}(rand(0:P-1))) for i in 1:6, j in 1:8]
        B = [CountingTropicalMin{Float64, Mod{P, Int32}}(
                Float64(rand(0:4)), Mod{P, Int32}(rand(0:P-1))) for i in 1:8, j in 5]
        ref = Matrix{CountingTropicalMin{Float64, Mod{P, Int32}}}(undef, 6, 5)
        for i in 1:6, j in 1:5
            best_n = Inf
            best_c = Mod{P, Int32}(0)
            for kk in 1:8
                v = A[i, kk].n + B[kk, j].n
                c = A[i, kk].c * B[kk, j].c
                if v < best_n
                    best_n = v; best_c = c
                elseif v == best_n
                    best_c += c
                end
            end
            ref[i, j] = CountingTropicalMin{Float64, Mod{P, Int32}}(best_n, best_c)
        end
        C = tropical_matmul_min(A, B)
        @test C == ref
    end

    @testset "tropical_matmul edge cases" begin
        using Mods
        # 1×1×1
        A = reshape([CountingTropical{Float32, Mod{7, Int32}}(2.0f0, Mod{7, Int32}(3))], 1, 1)
        B = reshape([CountingTropical{Float32, Mod{7, Int32}}(5.0f0, Mod{7, Int32}(4))], 1, 1)
        C = tropical_matmul(A, B)
        @test C[1, 1].n == 7.0f0
        @test C[1, 1].c == Mod{7, Int32}(12 % 7)  # 12 mod 7 = 5

        # P = 2 boundary
        A2 = [CountingTropical{Float32, Mod{2, Int32}}(0.0f0, Mod{2, Int32}(1)) for _ in 1:3, _ in 1:3]
        C2 = tropical_matmul(A2, A2)
        # All values 0+0 = 0, all counts (1*1)+(1*1)+(1*1) = 3 mod 2 = 1.
        @test all(c -> c.n == 0.0f0, C2)
        @test all(c -> c.c.val == 1, C2)

        # All-tie: zeros input, K = 5, count multiplies and sums.
        A3 = [CountingTropical{Float64, Mod{17, Int32}}(0.0, Mod{17, Int32}(2)) for _ in 1:2, _ in 1:5]
        B3 = [CountingTropical{Float64, Mod{17, Int32}}(0.0, Mod{17, Int32}(3)) for _ in 1:5, _ in 1:2]
        C3 = tropical_matmul(A3, B3)
        # Each output cell: 5 ties, each count 2*3=6, total 30 mod 17 = 13.
        @test all(c -> c.n == 0.0, C3)
        @test all(c -> c.c.val == 13, C3)
    end

    @testset "tropical_matmul errors" begin
        using Mods
        # K-mismatch.
        A = [CountingTropical{Float32, Mod{7, Int32}}(0.0f0, Mod{7, Int32}(1)) for _ in 1:2, _ in 1:3]
        Bbad = [CountingTropical{Float32, Mod{7, Int32}}(0.0f0, Mod{7, Int32}(1)) for _ in 1:4, _ in 1:2]
        @test_throws DimensionMismatch tropical_matmul(A, Bbad)

        # Mismatched moduli (different P) — MethodError because no method matches.
        Aother = [CountingTropical{Float32, Mod{11, Int32}}(0.0f0, Mod{11, Int32}(1)) for _ in 1:2, _ in 1:3]
        @test_throws MethodError tropical_matmul(A, Aother)

        # Mismatched U.
        Aint = [CountingTropical{Float32, Mod{7, Int}}(0.0f0, Mod{7}(1)) for _ in 1:2, _ in 1:3]
        @test_throws MethodError tropical_matmul(A, Aint)
    end
```

- [ ] **Step 2: Run all tests**

```bash
cd CountingTropicalGEMM.jl && bash -c 'module load cuda && julia --project=. -e "using Pkg; Pkg.test()"' 2>&1 | tail -25; cd ..
```
Expected: all tests pass — original 17 + ~6 new testsets.

- [ ] **Step 3: Commit**

```bash
git add CountingTropicalGEMM.jl/test/runtests.jl
git commit -m "$(cat <<'EOF'
Spec K: add min direction, edge-case, and error tests

Adds f64 min-plus cross-check with reference, 1x1x1 / P=2 / all-tie
edge cases (verifying mod reduction triggers), and dimension /
modulus / U mismatch error paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Final validation

- [ ] **Step 1: Full Rust crate test sweep**

```bash
cargo test -p tropical-gemm-cuda --features cuda 2>&1 | tail -15
```
Expected: all tests pass, 0 failures.

- [ ] **Step 2: Full Julia test sweep**

```bash
cd CountingTropicalGEMM.jl && bash -c 'module load cuda && julia --project=. -e "using Pkg; Pkg.test()"' 2>&1 | tail -25; cd ..
```
Expected: full testset green; original 17 + new tests.

- [ ] **Step 3: Update Spec J entry in memory.md if it references the old API surface**

If `MEMORY.md` mentions `count_ground_states_gpu_u64` as the only Julia entry, append a note about the new `tropical_matmul` / `tropical_matmul_min` / `mul!` API. (Optional, only if relevant.)

- [ ] **Step 4: Final commit if any docs / memory changes**

```bash
git status
# only if there are doc updates
git add <doc files>
git commit -m "docs: note new Julia mod-P matmul API"
```

---

## Spec coverage check

| Spec section | Implemented in |
|---|---|
| Architecture (4 layers) | Tasks 1–4 (Rust), 5–8 (Julia) |
| `matmul_mod_p` Rust driver | Task 1 |
| `matmul_mod_p_pair` Rust driver | Task 2 |
| Slow-path C ABI (4 entries) | Task 3 |
| Fast-path C ABI (4 entries) | Task 4 |
| Julia deps + `CountingTropicalMin` | Task 5 |
| `tropical_matmul` (slow path) | Task 6 |
| `tropical_matmul_min` (slow path) | Task 6 |
| Mod{P, Int32} fast-path dispatch | Task 7 |
| `LinearAlgebra.mul!` for both directions | Task 8 |
| Reference-equivalence tests (slow + fast × Max + Min × f32 + f64) | Tasks 6, 7, 9 |
| Edge cases (P=2, K=1, 1×1×1, all-tie) | Task 9 |
| Errors (P range, K mismatch, type mismatch) | Tasks 3, 9 |
| Final regression sweep | Task 10 |
