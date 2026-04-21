# Spec C — CUDA kernel + GPU CRT driver — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run `count_ground_states` on GPU. A new CUDA kernel replaces the per-prime `tropical_matmul_t` call; CRT reconstruction remains on the host.

**Architecture:** SoA element layout (two GpuMatrix buffers: values + counts). Four kernel specializations stamped via preprocessor macro from a new `.cu` file compiled with NVRTC at context init. Host driver in `tropical-gemm-cuda` mirrors the CPU spec-B driver surface, reusing `CRT_PRIMES` / `crt_combine` promoted to `pub` in `tropical-gemm::crt`.

**Tech Stack:** Rust 1.87, CUDA via `cudarc` 0.12 + NVRTC, `num-bigint`.

**Spec:** `docs/superpowers/specs/2026-04-21-counting-tropical-cuda-design.md`.

---

## Preconditions (from spec A + B)

- `counting-tropical` branch latest commit `7b248a3` (spec C document).
- Cargo: `. ~/.cargo/env` + `module load cuda python` at shell start.
- CPU CRT driver `count_ground_states` ships from `tropical_gemm::crt`. Helpers `crt_combine`, `CRT_PRIMES`, `choose_primes` are currently `pub(crate)` / private — Task 1 promotes them.
- Baseline tests before touching anything: 304 lib + 6 counting_crt + 5 counting_compose + 24 doctests + 49 cuda lib tests.

## File map

- **Modify:** `crates/tropical-gemm/src/crt.rs` — promote `crt_combine`, `choose_primes` to `pub`. `CRT_PRIMES` is already `pub`.
- **Create:** `crates/tropical-gemm-cuda/kernels/counting_gemm.cu` — 4 kernel specializations via macro.
- **Modify:** `crates/tropical-gemm-cuda/src/context.rs` — compile + load the new .cu as a second module, register 4 new kernel names.
- **Create:** `crates/tropical-gemm-cuda/src/counting_kernel.rs` — `CountingCudaKernel` trait + 4 concrete impls.
- **Modify:** `crates/tropical-gemm-cuda/Cargo.toml` — add `num-bigint`, `num-traits`, `num-integer`.
- **Create:** `crates/tropical-gemm-cuda/src/crt.rs` — `count_ground_states_gpu<T, D>` driver.
- **Modify:** `crates/tropical-gemm-cuda/src/lib.rs` — declare the two new modules, re-export `count_ground_states_gpu` + re-exported `CountedMat`.
- **Create:** `crates/tropical-gemm-cuda/tests/counting_gpu.rs` — GPU vs CPU cross-checks.
- **Modify:** `crates/tropical-gemm-python/src/lib.rs` — `count_ground_states_gpu_py` (gated `#[cfg(feature = "cuda")]`).

---

## Phase 1 — Expose CPU CRT helpers

### Task 1: Make `crt_combine` and `choose_primes` public in `tropical-gemm`

**File:** `crates/tropical-gemm/src/crt.rs`

- [ ] **Step 1: Promote visibility**

In `crates/tropical-gemm/src/crt.rs`:
- Change `pub(crate) fn crt_combine(...)` → `pub fn crt_combine(...)`.
- Change `fn choose_primes(...)` → `pub fn choose_primes(...)`.

These are load-bearing pieces that the GPU driver will reuse verbatim.

- [ ] **Step 2: Add doc examples exercising the public API**

Append a short doctest to each promoted function so users see intended usage. Example for `crt_combine`:

```rust
/// ...existing docs...
///
/// # Example
///
/// ```
/// use num_bigint::BigInt;
/// use tropical_gemm::crt::crt_combine;
/// // x ≡ 2 (mod 3), x ≡ 3 (mod 5) → x = 8 (mod 15)
/// let (x, m) = crt_combine(&BigInt::from(2), &BigInt::from(3), 3, 5);
/// assert_eq!(x, BigInt::from(8));
/// assert_eq!(m, BigInt::from(15));
/// ```
```

For `choose_primes`:

```rust
/// ...existing docs...
///
/// # Example
///
/// ```
/// use num_bigint::BigInt;
/// use tropical_gemm::crt::choose_primes;
/// let (indices, product) = choose_primes(&BigInt::from(5));
/// // Smallest prime in CRT_PRIMES is ~2^30, so one suffices for bound 5.
/// assert_eq!(indices, vec![0]);
/// assert!(product > BigInt::from(5));
/// ```
```

Add `pub use crt::{crt_combine, choose_primes};` to the re-export block in `lib.rs` alongside the existing `count_ground_states` / `CountedMat` re-exports.

- [ ] **Step 3: Verify**

```
. ~/.cargo/env && cargo test -p tropical-gemm --doc 2>&1 | tail -8
. ~/.cargo/env && cargo test -p tropical-gemm --features testing 2>&1 | tail -5
```

Expected: doctests include the two new ones and pass. Full suite still green.

- [ ] **Step 4: Commit**

```bash
git add crates/tropical-gemm/src/crt.rs crates/tropical-gemm/src/lib.rs
git commit -m "$(cat <<'EOF'
Promote crt_combine and choose_primes to pub for cross-crate reuse

tropical-gemm-cuda will reuse these helpers verbatim in its GPU CRT
driver (spec C).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 — CUDA kernel source

### Task 2: Write `counting_gemm.cu`

**File:** `crates/tropical-gemm-cuda/kernels/counting_gemm.cu` (create).

- [ ] **Step 1: Write the kernel source**

Content (exact):

```c
// Counting Tropical GEMM CUDA Kernels (spec C).
//
// SoA element layout: each logical element = (value: T, count: int32).
// Matrices are passed as parallel value and count pointers.
//
// Semiring operations (direction D):
//   tropical_mul: (va, ca) * (vb, cb) = (va + vb, (ca * cb) mod P)
//   tropical_add: strictly-better value wins; on tie, counts add mod P.

// Infinity sentinels for MaxPlus / MinPlus tropical zero.
#define NEG_INF_F32 __int_as_float(0xff800000)
#define POS_INF_F32 __int_as_float(0x7f800000)
#define NEG_INF_F64 __longlong_as_double(0xfff0000000000000LL)
#define POS_INF_F64 __longlong_as_double(0x7ff0000000000000LL)

#define OFFSET_COL(row, col, ld) ((col) * (ld) + (row))

// Row-major addressing: matrices are uploaded row-major from host, so
// A[i, kk] with stride k has offset i * k + kk.
#define OFFSET_ROW(row, col, ncols) ((row) * (ncols) + (col))

// One thread computes one C[i, j] cell. Grid is 2D (m x n threads).
#define COUNTING_KERNEL(name, T, D_ZERO, D_BETTER)                          \
extern "C" __global__ void name(                                            \
    const T*   value_a, const int* count_a,                                 \
    const T*   value_b, const int* count_b,                                 \
    T*         value_c, int*       count_c,                                 \
    int m, int n, int k, int P)                                             \
{                                                                           \
    int i = blockIdx.y * blockDim.y + threadIdx.y;                          \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                          \
    if (i >= m || j >= n) return;                                           \
                                                                            \
    T   acc_val = (D_ZERO);                                                 \
    int acc_cnt = 0;                                                        \
                                                                            \
    for (int kk = 0; kk < k; ++kk) {                                        \
        T   va = value_a[OFFSET_ROW(i, kk, k)];                             \
        int ca = count_a[OFFSET_ROW(i, kk, k)];                             \
        T   vb = value_b[OFFSET_ROW(kk, j, n)];                             \
        int cb = count_b[OFFSET_ROW(kk, j, n)];                             \
        T   pv = va + vb;                                                   \
        int pc = (int)(((long long)ca * (long long)cb) % (long long)P);     \
        if (D_BETTER(pv, acc_val)) {                                        \
            acc_val = pv;                                                   \
            acc_cnt = pc;                                                   \
        } else if (D_BETTER(acc_val, pv)) {                                 \
            /* keep current accumulator */                                  \
        } else {                                                            \
            acc_cnt = (int)(((long long)acc_cnt + (long long)pc)            \
                            % (long long)P);                                \
        }                                                                   \
    }                                                                       \
                                                                            \
    value_c[OFFSET_ROW(i, j, n)] = acc_val;                                 \
    count_c[OFFSET_ROW(i, j, n)] = acc_cnt;                                 \
}

#define MAX_BETTER(a, b) ((a) > (b))
#define MIN_BETTER(a, b) ((a) < (b))

COUNTING_KERNEL(counting_gemm_f32_max, float,  NEG_INF_F32, MAX_BETTER)
COUNTING_KERNEL(counting_gemm_f32_min, float,  POS_INF_F32, MIN_BETTER)
COUNTING_KERNEL(counting_gemm_f64_max, double, NEG_INF_F64, MAX_BETTER)
COUNTING_KERNEL(counting_gemm_f64_min, double, POS_INF_F64, MIN_BETTER)
```

Row-major addressing: the spec-A / spec-B GEMM pipeline uploads matrices row-major (`tropical_matmul_t` takes `&[T]` row-major). This kernel matches that convention — verify at Task 4 via cross-check.

- [ ] **Step 2: Syntax check (no NVRTC yet)**

```
module load cuda && nvcc --ptx --compile-as-tools-patch /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm/crates/tropical-gemm-cuda/kernels/counting_gemm.cu -o /tmp/cgemm.ptx 2>&1 | tail -10 || true
```

If `nvcc` is available, a successful compile confirms the syntax. If not, skip — Task 3 will catch any issues when NVRTC compiles it at context init.

- [ ] **Step 3: Commit (no tests yet — kernel isn't wired up)**

```bash
git add crates/tropical-gemm-cuda/kernels/counting_gemm.cu
git commit -m "$(cat <<'EOF'
Add counting_gemm.cu: CUDA kernels for CountingTropical GEMM

Four specializations via preprocessor macro: {f32, f64} × {Max, Min}.
SoA layout (parallel value + count pointers), runtime modulus P.
One thread per output cell, 2D grid.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3 — Wire kernel into context

### Task 3: Compile and load `counting_gemm.cu` at context init

**Files:** `crates/tropical-gemm-cuda/src/context.rs`, `crates/tropical-gemm-cuda/src/lib.rs`.

- [ ] **Step 1: Extend `context.rs`**

Edit `crates/tropical-gemm-cuda/src/context.rs`:

Add after line 9 (`const KERNEL_SOURCE = …`):

```rust
const COUNTING_KERNEL_SOURCE: &str = include_str!("../kernels/counting_gemm.cu");
```

Add after the existing `KERNEL_NAMES` array (after line 64):

```rust
const COUNTING_KERNEL_NAMES: &[&str] = &[
    "counting_gemm_f32_max",
    "counting_gemm_f32_min",
    "counting_gemm_f64_max",
    "counting_gemm_f64_min",
];
```

Inside `CudaContext::from_device`, after the existing `device.load_ptx(ptx, "tropical_gemm", KERNEL_NAMES)?;` at line 92, add:

```rust
let counting_ptx = cudarc::nvrtc::compile_ptx(COUNTING_KERNEL_SOURCE)?;
device.load_ptx(counting_ptx, "counting_gemm", COUNTING_KERNEL_NAMES)?;
```

In the kernel-cache loop (after line 101), also loop over `COUNTING_KERNEL_NAMES`:

```rust
for name in COUNTING_KERNEL_NAMES {
    let func = device
        .get_func("counting_gemm", name)
        .ok_or_else(|| CudaError::KernelNotFound(name.to_string()))?;
    kernels.insert(*name, func);
}
```

Add default block/grid helpers for the new kernels (below `block_dims_f64`):

```rust
/// Block dimensions for counting kernels (same 16x16 for f32 and f64 for now).
pub fn counting_block_dims() -> (u32, u32, u32) {
    (16, 16, 1)
}

/// Grid dimensions for a counting kernel launch covering m x n output cells.
pub fn counting_grid_dims(m: usize, n: usize) -> (u32, u32, u32) {
    let (bx, by, _) = Self::counting_block_dims();
    let gx = ((n as u32) + bx - 1) / bx;
    let gy = ((m as u32) + by - 1) / by;
    (gx, gy, 1)
}
```

- [ ] **Step 2: Verify context still builds**

```
. ~/.cargo/env && module load cuda && cargo build -p tropical-gemm-cuda 2>&1 | tail -8
```

Expected: clean build. If NVRTC rejects the `.cu`, iterate on Task 2's source until it compiles.

- [ ] **Step 3: Verify existing CUDA tests still pass (context init with extra PTX)**

```
. ~/.cargo/env && module load cuda && cargo test -p tropical-gemm-cuda --lib 2>&1 | tail -5
```

Expected: 49 tests pass. Any regression = the extra PTX load broke something — debug before moving on.

- [ ] **Step 4: Commit**

```bash
git add crates/tropical-gemm-cuda/src/context.rs
git commit -m "$(cat <<'EOF'
Compile and load counting_gemm.cu at CUDA context init

Adds a second NVRTC compile + load_ptx call for the 4 new counting
kernels alongside the existing tropical_gemm module. Adds counting_grid_dims
/ counting_block_dims helpers for launch configuration.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4 — Rust launch wrapper

### Task 4: `CountingCudaKernel` trait and concrete impls

**Files:** `crates/tropical-gemm-cuda/src/counting_kernel.rs` (create), `crates/tropical-gemm-cuda/src/lib.rs` (add `mod counting_kernel;`).

- [ ] **Step 1: Create `counting_kernel.rs`**

```rust
//! Launch wrapper for the CountingTropical CUDA kernels.
//!
//! Reuses `GpuMatrix<T>` for values and `GpuMatrix<i32>` for count residues.
//! The kernel expects row-major data; the rest of the crate already uploads
//! row-major, so no extra transposition is needed.

use crate::context::CudaContext;
use crate::error::Result;
use crate::memory::GpuMatrix;
use cudarc::driver::{DeviceRepr, LaunchAsync, LaunchConfig, ValidAsZeroBits};
use tropical_gemm::types::{Max, Min, TropicalDirection};

pub trait CountingCudaKernel<T, D>
where
    T: DeviceRepr + ValidAsZeroBits + Default + Clone + Copy + 'static,
    D: TropicalDirection,
{
    const KERNEL_NAME: &'static str;

    fn launch_counting_gemm(
        ctx: &CudaContext,
        value_a: &GpuMatrix<T>,
        count_a: &GpuMatrix<i32>,
        value_b: &GpuMatrix<T>,
        count_b: &GpuMatrix<i32>,
        value_c: &mut GpuMatrix<T>,
        count_c: &mut GpuMatrix<i32>,
        modulus: i32,
    ) -> Result<()> {
        let m = value_a.rows();
        let k = value_a.cols();
        let n = value_b.cols();

        assert_eq!(count_a.rows(), m);
        assert_eq!(count_a.cols(), k);
        assert_eq!(value_b.rows(), k);
        assert_eq!(count_b.rows(), k);
        assert_eq!(count_b.cols(), n);
        assert_eq!(value_c.rows(), m);
        assert_eq!(value_c.cols(), n);
        assert_eq!(count_c.rows(), m);
        assert_eq!(count_c.cols(), n);

        let kernel = ctx.get_kernel(Self::KERNEL_NAME)?;
        let cfg = LaunchConfig {
            grid_dim: CudaContext::counting_grid_dims(m, n),
            block_dim: CudaContext::counting_block_dims(),
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(
                cfg,
                (
                    value_a.as_slice(),
                    count_a.as_slice(),
                    value_b.as_slice(),
                    count_b.as_slice(),
                    value_c.as_slice_mut(),
                    count_c.as_slice_mut(),
                    m as i32,
                    n as i32,
                    k as i32,
                    modulus,
                ),
            )?;
        }

        ctx.device().synchronize()?;
        Ok(())
    }
}

// Four concrete impls.
impl CountingCudaKernel<f32, Max> for (f32, Max) {
    const KERNEL_NAME: &'static str = "counting_gemm_f32_max";
}
impl CountingCudaKernel<f32, Min> for (f32, Min) {
    const KERNEL_NAME: &'static str = "counting_gemm_f32_min";
}
impl CountingCudaKernel<f64, Max> for (f64, Max) {
    const KERNEL_NAME: &'static str = "counting_gemm_f64_max";
}
impl CountingCudaKernel<f64, Min> for (f64, Min) {
    const KERNEL_NAME: &'static str = "counting_gemm_f64_min";
}

// Convenience function — picks the right impl at call site.
pub fn launch_counting_gemm<T, D>(
    ctx: &CudaContext,
    value_a: &GpuMatrix<T>,
    count_a: &GpuMatrix<i32>,
    value_b: &GpuMatrix<T>,
    count_b: &GpuMatrix<i32>,
    value_c: &mut GpuMatrix<T>,
    count_c: &mut GpuMatrix<i32>,
    modulus: i32,
) -> Result<()>
where
    T: DeviceRepr + ValidAsZeroBits + Default + Clone + Copy + 'static,
    D: TropicalDirection,
    (T, D): CountingCudaKernel<T, D>,
{
    <(T, D) as CountingCudaKernel<T, D>>::launch_counting_gemm(
        ctx, value_a, count_a, value_b, count_b, value_c, count_c, modulus,
    )
}
```

In `crates/tropical-gemm-cuda/src/lib.rs` add `mod counting_kernel;` near the other `mod` declarations (do not `pub` yet — wrapped by the CRT driver in Task 5).

- [ ] **Step 2: Verify builds cleanly**

```
. ~/.cargo/env && module load cuda && cargo build -p tropical-gemm-cuda 2>&1 | tail -6
```

Expected: clean build.

If `GpuMatrix::as_slice()` / `as_slice_mut()` don't exist, read `crates/tropical-gemm-cuda/src/memory.rs` to find the correct accessor and adjust. The existing kernel wrapper in `src/kernels.rs` will show the pattern — follow it exactly.

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm-cuda/src/counting_kernel.rs crates/tropical-gemm-cuda/src/lib.rs
git commit -m "$(cat <<'EOF'
Add CountingCudaKernel trait + launch wrapper for counting GEMM

Generic wrapper over four (T, D) impls: (f32/f64 × Max/Min). Uses
existing GpuMatrix<Scalar> buffers for values and GpuMatrix<i32> for
count residues. Picks kernel name via associated const.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5 — GPU CRT driver

### Task 5: `count_ground_states_gpu` driver + cross-check test

**Files:**
- `crates/tropical-gemm-cuda/Cargo.toml` — add deps.
- `crates/tropical-gemm-cuda/src/crt.rs` — create.
- `crates/tropical-gemm-cuda/src/lib.rs` — expose.
- `crates/tropical-gemm-cuda/tests/counting_gpu.rs` — create.

- [ ] **Step 1: Add deps**

In `crates/tropical-gemm-cuda/Cargo.toml` under `[dependencies]`:

```toml
num-bigint = "0.4"
num-integer = "0.1"
num-traits = "0.2"
```

- [ ] **Step 2: Create the driver**

Create `crates/tropical-gemm-cuda/src/crt.rs`:

```rust
//! GPU CRT driver for `count_ground_states`.
//!
//! Mirrors the CPU spec-B driver in `tropical_gemm::crt` but dispatches the
//! per-prime matmul to the CUDA kernel. Uploads value matrices once (shared
//! across all prime runs), reuses the count matrices fresh per prime.

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

    // Upload value matrices (row-major). Reused across all prime runs.
    let value_a = GpuMatrix::<T>::from_host(ctx, a_values, m, k)?;
    let value_b = GpuMatrix::<T>::from_host(ctx, b_values, k, n)?;

    let needed = BigInt::from(2) * count_upper_bound + BigInt::one();
    let (prime_indices, _product) = choose_primes(&needed);

    let ncells = m * n;
    let mut values_ref: Option<Vec<T>> = None;
    let mut residue_streams: Vec<Vec<i32>> = Vec::with_capacity(prime_indices.len());

    for &prime_idx in &prime_indices {
        let p = CRT_PRIMES[prime_idx];

        // Count inputs = Mod(1); i.e. i32 buffers filled with 1s.
        let ones_a = vec![1_i32; m * k];
        let ones_b = vec![1_i32; k * n];
        let count_a = GpuMatrix::<i32>::from_host(ctx, &ones_a, m, k)?;
        let count_b = GpuMatrix::<i32>::from_host(ctx, &ones_b, k, n)?;

        let mut value_c = GpuMatrix::<T>::zeros(ctx, m, n)?;
        let mut count_c = GpuMatrix::<i32>::zeros(ctx, m, n)?;

        launch_counting_gemm::<T, D>(
            ctx,
            &value_a, &count_a,
            &value_b, &count_b,
            &mut value_c, &mut count_c,
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

    Ok(CountedMat { nrows: m, ncols: n, values, counts })
}
```

Check that `GpuMatrix::from_host`, `GpuMatrix::zeros`, `GpuMatrix::to_host`, `CudaError::InvalidState` (or similar variant name) exist. Read `src/memory.rs` and `src/error.rs` first. If the error variant is named differently, use the closest match — the goal is a clear string message. If `GpuMatrix::zeros` isn't present, use `GpuMatrix::allocate` or equivalent and rely on the kernel to overwrite (it does — see Task 2's kernel). If no allocator exists that does not require a host buffer, pass a `vec![T::default(); m*n]` / `vec![0_i32; m*n]` and upload.

- [ ] **Step 3: Expose in `lib.rs`**

In `crates/tropical-gemm-cuda/src/lib.rs`:

Add:
```rust
pub mod crt;
pub use crt::count_ground_states_gpu;
pub use tropical_gemm::CountedMat;  // re-export for convenience
```

(Keep the existing `mod counting_kernel;` as non-pub or make `pub(crate)` as matches style.)

- [ ] **Step 4: Create the cross-check test**

Create `crates/tropical-gemm-cuda/tests/counting_gpu.rs`:

```rust
//! GPU vs CPU cross-check for count_ground_states.

use num_bigint::BigInt;
use tropical_gemm::{bound_for_single_matmul, count_ground_states, Max, Min};
use tropical_gemm_cuda::{count_ground_states_gpu, CudaContext};

fn random_ish_matrix(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9e3779b97f4a7c15);
    (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let x = (state >> 33) as u32;
            (x % 7) as f32
        })
        .collect()
}

#[test]
fn gpu_matches_cpu_max_small() {
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (4, 5, 3);
    let a = random_ish_matrix(m, k, 0x1234);
    let b = random_ish_matrix(k, n, 0x5678);
    let bound = bound_for_single_matmul(k);

    let cpu = count_ground_states::<f32, Max>(&a, m, k, &b, n, &bound);
    let gpu = count_ground_states_gpu::<f32, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();

    assert_eq!(gpu.values, cpu.values);
    assert_eq!(gpu.counts, cpu.counts);
}

#[test]
fn gpu_matches_cpu_min_medium() {
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (8, 16, 8);
    let a = random_ish_matrix(m, k, 0xaaaa);
    let b = random_ish_matrix(k, n, 0xbbbb);
    let bound = bound_for_single_matmul(k);

    let cpu = count_ground_states::<f32, Min>(&a, m, k, &b, n, &bound);
    let gpu = count_ground_states_gpu::<f32, Min>(&ctx, &a, m, k, &b, n, &bound).unwrap();

    assert_eq!(gpu.values, cpu.values);
    assert_eq!(gpu.counts, cpu.counts);
}

#[test]
fn gpu_handles_all_ties() {
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (2, 13, 2);
    let a = vec![0.0_f32; m * k];
    let b = vec![0.0_f32; k * n];
    let bound = bound_for_single_matmul(k);

    let gpu = count_ground_states_gpu::<f32, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();

    assert_eq!(gpu.values, vec![0.0; m * n]);
    assert_eq!(gpu.counts, vec![BigInt::from(k); m * n]);
}

#[test]
fn gpu_multi_prime_large_bound() {
    let ctx = CudaContext::new().unwrap();
    let a = vec![0.0_f32; 100];
    let b = vec![0.0_f32; 100];
    let bound = BigInt::from(u128::MAX);

    let gpu = count_ground_states_gpu::<f32, Max>(&ctx, &a, 1, 100, &b, 1, &bound).unwrap();
    assert_eq!(gpu.counts, vec![BigInt::from(100)]);
}

#[test]
fn gpu_f64_matches_cpu() {
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (3, 4, 2);
    let a: Vec<f64> = (0..m * k).map(|x| (x % 5) as f64).collect();
    let b: Vec<f64> = (0..k * n).map(|x| (x % 5) as f64).collect();
    let bound = bound_for_single_matmul(k);

    let cpu = count_ground_states::<f64, Max>(&a, m, k, &b, n, &bound);
    let gpu = count_ground_states_gpu::<f64, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();

    assert_eq!(gpu.values, cpu.values);
    assert_eq!(gpu.counts, cpu.counts);
}
```

- [ ] **Step 5: Run the cross-check**

```
. ~/.cargo/env && module load cuda && cargo test -p tropical-gemm-cuda --test counting_gpu 2>&1 | tail -20
```

Expected: 5 tests pass. If any mismatch: debug by printing a 1x1 example and comparing per-cell. Common bugs:
- Row-major vs column-major mismatch in the kernel.
- `modulus` argument order wrong.
- Uninitialized `count_c` (kernel must overwrite every cell; confirm thread guards `if (i >= m || j >= n) return;` doesn't leave stale values — with `zeros`-initialized buffers it's fine).

- [ ] **Step 6: Also run the full `tropical-gemm-cuda` lib suite to make sure nothing regressed**

```
. ~/.cargo/env && module load cuda && cargo test -p tropical-gemm-cuda --lib 2>&1 | tail -5
```

Expected: 49 tests still pass.

- [ ] **Step 7: Commit**

```bash
git add crates/tropical-gemm-cuda/
git commit -m "$(cat <<'EOF'
Add GPU CRT driver: count_ground_states_gpu

Mirrors the CPU spec-B driver, dispatching per-prime matmul to the
new CUDA kernel. Uploads value matrices once, reuses across prime
runs. CRT reconstruction runs host-side via crt_combine from the CPU
crate.

Cross-check tests validate GPU output equals CPU output on random
matrices (Max and Min, f32 and f64), an all-ties corner case, and
a multi-prime large-bound scenario.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 6 — Python binding

### Task 6: `count_ground_states_gpu_py`

**File:** `crates/tropical-gemm-python/src/lib.rs`.

- [ ] **Step 1: Add the pyfunction**

Read `crates/tropical-gemm-python/src/lib.rs` to see the existing `cuda`-gated pyfunctions (e.g., `tropical_maxplus_matmul_gpu` or similar).

Add at the same gate level:

```rust
#[cfg(feature = "cuda")]
use ::tropical_gemm_cuda::{count_ground_states_gpu, CudaContext};

#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (a, b, direction="min", count_upper_bound=None))]
fn count_ground_states_gpu_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f32>,
    b: PyReadonlyArray2<'py, f32>,
    direction: &str,
    count_upper_bound: Option<&Bound<'py, PyAny>>,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<PyObject>>)> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[0];
    let k = a_shape[1];
    let k2 = b_shape[0];
    let n = b_shape[1];
    if k != k2 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Dimension mismatch: A is {}x{}, B is {}x{}",
            m, k, k2, n
        )));
    }

    let bound: BigInt = match count_upper_bound {
        None => bound_for_single_matmul(k),
        Some(obj) => {
            let s: String = obj.call_method0("__str__")?.extract()?;
            s.parse::<BigInt>().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "count_upper_bound must be a non-negative integer, got {:?} ({})",
                    s, e
                ))
            })?
        }
    };

    let a_data = a.as_slice()?.to_vec();
    let b_data = b.as_slice()?.to_vec();

    let result = py
        .allow_threads(|| -> Result<_, String> {
            let ctx = CudaContext::new().map_err(|e| format!("CUDA init: {}", e))?;
            match direction {
                "max" => count_ground_states_gpu::<f32, Max>(&ctx, &a_data, m, k, &b_data, n, &bound)
                    .map_err(|e| format!("GPU compute: {}", e)),
                "min" => count_ground_states_gpu::<f32, Min>(&ctx, &a_data, m, k, &b_data, n, &bound)
                    .map_err(|e| format!("GPU compute: {}", e)),
                other => Err(format!("direction must be 'max' or 'min', got {:?}", other)),
            }
        })
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;

    // Reshape outputs (identical to the CPU count_ground_states_py post-processing).
    let values_array = numpy::ndarray::Array2::from_shape_vec((m, n), result.values)
        .expect("values length matches m*n")
        .into_pyarray(py);

    // Reuse the same BigInt → Python int conversion approach as the CPU binding.
    let counts_py: Vec<PyObject> = result
        .counts
        .into_iter()
        .map(|bn| {
            let s = bn.to_string();
            py.eval(
                std::ffi::CString::new(format!("int({})", s)).unwrap().as_c_str(),
                None,
                None,
            )
            .map(|b| b.unbind())
        })
        .collect::<PyResult<Vec<_>>>()?;
    let counts_array = numpy::ndarray::Array2::from_shape_vec((m, n), counts_py)
        .expect("counts length matches m*n")
        .into_pyarray(py);

    Ok((values_array, counts_array))
}
```

Register it in the `#[pymodule]` body (also `#[cfg(feature = "cuda")]`-gated):

```rust
#[cfg(feature = "cuda")]
m.add_function(wrap_pyfunction!(count_ground_states_gpu_py, m)?)?;
```

- [ ] **Step 2: Build**

```
. ~/.cargo/env && module load cuda python && cargo build -p tropical-gemm-python --features cuda 2>&1 | tail -8
```

Expected: clean build. If the python crate's `cuda` feature isn't named exactly `cuda`, grep its `Cargo.toml` and match.

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm-python/src/lib.rs
git commit -m "$(cat <<'EOF'
Expose count_ground_states_gpu to Python (behind cuda feature)

Mirrors the CPU count_ground_states_py surface. Creates a fresh
CudaContext per call; future work can cache a singleton if launch
overhead matters. GIL released during the GPU compute.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

### Task 7: Python round-trip test (env-skipped)

**File:** `crates/tropical-gemm-python/tests/test_count_ground_states_gpu.py` (create).

- [ ] **Step 1: Create**

```python
"""Round trip for count_ground_states_gpu."""

import numpy as np
import pytest

try:
    import tropical_gemm
    _fn = getattr(
        tropical_gemm, "count_ground_states_gpu",
        getattr(tropical_gemm, "count_ground_states_gpu_py", None),
    )
    HAVE_EXT = _fn is not None
except ImportError:
    HAVE_EXT = False
    _fn = None

pytestmark = pytest.mark.skipif(not HAVE_EXT, reason="tropical_gemm[cuda] extension not built")


def test_trivial_gpu_1x1():
    a = np.array([[3.0]], dtype=np.float32)
    b = np.array([[4.0]], dtype=np.float32)
    values, counts = _fn(a, b, "max")
    assert values[0, 0] == 7.0
    assert int(counts[0, 0]) == 1


def test_ties_merge_gpu_max():
    a = np.array([[2.0, 3.0]], dtype=np.float32)
    b = np.array([[3.0], [2.0]], dtype=np.float32)
    values, counts = _fn(a, b, "max")
    assert values[0, 0] == 5.0
    assert int(counts[0, 0]) == 2


def test_gpu_matches_cpu():
    a = np.random.RandomState(42).randint(0, 5, size=(8, 12)).astype(np.float32)
    b = np.random.RandomState(43).randint(0, 5, size=(12, 6)).astype(np.float32)
    gpu_v, gpu_c = _fn(a, b, "max")
    cpu_v, cpu_c = tropical_gemm.count_ground_states(a, b, "max")
    np.testing.assert_array_equal(gpu_v, cpu_v)
    # Object arrays — compare elementwise via int cast.
    assert gpu_c.shape == cpu_c.shape
    for (i, j), x in np.ndenumerate(gpu_c):
        assert int(x) == int(cpu_c[i, j])
```

- [ ] **Step 2: Commit**

```bash
git add crates/tropical-gemm-python/tests/test_count_ground_states_gpu.py
git commit -m "$(cat <<'EOF'
Add Python round-trip test for count_ground_states_gpu

Skips when the extension isn't built. Cross-checks GPU output against
CPU count_ground_states on a random input.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 7 — Final gate

### Task 8: Workspace regression

- [ ] **Step 1: Run all**

```
. ~/.cargo/env && module load cuda python
cargo test -p tropical-gemm --features testing 2>&1 | tail -5
cargo test -p tropical-gemm-cuda --lib 2>&1 | tail -5
cargo test -p tropical-gemm-cuda --test counting_gpu 2>&1 | tail -5
cargo build -p tropical-gemm-python --features cuda 2>&1 | tail -5
```

Expected tallies:
- tropical-gemm lib: 304 / 0 fail (+ 2 new doctests from Task 1: 26 total doctests).
- tropical-gemm-cuda lib: 49 / 0.
- tropical-gemm-cuda counting_gpu integration: 5 / 0.
- tropical-gemm-python cuda build: clean.

Fix anything that regressed.

- [ ] **Step 2: No commit unless fixes.**

---

## Out of scope

- MaxMul counting semiring on GPU.
- Integer T (`i32`, `i64`) GPU counting. f32/f64 only for v1.
- Streams or async concurrency across prime launches. Synchronous baseline.
- Argmax-tracking counting on GPU.
- Kernel tiling / shared memory / warp-level reductions. Naive 1-thread-per-cell kernel. Optimization is a follow-up once correctness is established.

## Self-review notes

- **Spec coverage.** Kernel (§1) → Task 2-3. Launch wrapper (§2) → Task 4. GPU CRT driver (§3) → Task 5. Python (§4) → Task 6-7. Testing (§testing) → Task 5 cross-checks + Task 7 Python round-trip. Out-of-scope items match spec.
- **Placeholder scan.** None — every step has concrete code. Task 5 Step 2 acknowledges possible `GpuMatrix` API mismatch and directs the implementer to `src/memory.rs` with specific fallback rules.
- **Type consistency.** `CountedMat<T>`, `count_ground_states_gpu<T, D>`, `CountingCudaKernel<T, D>`, `CRT_PRIMES`, `crt_combine`, `choose_primes` — names consistent across tasks. The trait method name is `launch_counting_gemm` everywhere.
- **Risks.** Row-major vs col-major is the #1 bug source; Task 5's cross-check tests against the CPU reference catch it. If they fail, swap the row/col offsets in the kernel.
