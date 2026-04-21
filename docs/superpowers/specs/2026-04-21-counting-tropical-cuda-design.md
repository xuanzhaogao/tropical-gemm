# Spec C — CUDA kernel for `CountingTropical` + GPU CRT driver

**Date:** 2026-04-21
**Status:** design approved
**Depends on:** spec A (`2026-04-21-counting-tropical-compose-design.md`), spec B (`2026-04-21-counting-tropical-crt-design.md`).
**Scope:** GPU port of `count_ground_states`. Same semantics, same `CountedMat<T>` return. Rust host driver + CUDA kernel + Python entry point.

## Goal

Run the spec-B CRT-based counting matmul on GPU. The kernel per-prime launch
replaces the CPU `tropical_matmul_t::<CountingTropical<T, Mod<P>, D>>` call;
everything else (CRT reconstruction, value-field invariant, `count_upper_bound`
contract) is unchanged.

## Architecture

### 1. CUDA kernel — SoA layout

Element layout on device is Struct-of-Arrays: one `GpuMatrix<T>` for tropical
values, one `GpuMatrix<i32>` for count residues. The kernel takes six pointer
arguments (value/count, A/B/C). No new `DeviceRepr` type; reuses existing
`GpuMatrix<Scalar>` infrastructure.

Kernel signature (per `T`, `D`):

```c
__global__ void counting_gemm_<T>_<D>(
    const T*   value_a, const int* count_a,   // m x k, column-major
    const T*   value_b, const int* count_b,   // k x n
    T*         value_c, int*       count_c,   // m x n (output)
    int m, int n, int k,
    int P                                      // runtime modulus
);
```

Inner loop per `(i, j)` thread:

```c
T   acc_val = D_ZERO;         // -INF for MaxPlus, +INF for MinPlus
int acc_cnt = 0;
for (int kk = 0; kk < k; ++kk) {
    T   va = value_a[OFFSET(i, kk, lda)];
    int ca = count_a[OFFSET(i, kk, lda)];
    T   vb = value_b[OFFSET(kk, j, ldb)];
    int cb = count_b[OFFSET(kk, j, ldb)];
    T   pv = va + vb;
    int pc = (int)(((long long)ca * cb) % P);
    if      (D_BETTER(pv, acc_val))  { acc_val = pv; acc_cnt = pc; }
    else if (D_BETTER(acc_val, pv))  { /* keep */ }
    else                              { acc_cnt = (acc_cnt + pc) % P; }
}
value_c[OFFSET(i, j, ldc)] = acc_val;
count_c[OFFSET(i, j, ldc)] = acc_cnt;
```

Four kernel instantiations via preprocessor macro: `{float, double} × {Max, Min}`.
Do **not** template on `P`: keeping `P` as a runtime argument avoids 16× kernel
cache pressure. The per-thread `%` cost on a 30-bit `int` is cheap compared to
memory traffic for any realistic matrix size.

Follow the existing `kernels/tropical_gemm.cu` pattern: preprocessor macros to
stamp out concrete kernels, NVRTC compile at context init, kernel lookup by
name.

### 2. Rust launch wrapper

New file `crates/tropical-gemm-cuda/src/counting_kernel.rs`:

```rust
pub(crate) trait CountingCudaKernel<T, D>
where T: TropicalScalar + DeviceRepr + ValidAsZeroBits + Default + Clone,
      D: TropicalDirection,
{
    const KERNEL_NAME: &'static str;
    fn launch_counting_gemm(
        ctx: &CudaContext,
        value_a: &GpuMatrix<T>, count_a: &GpuMatrix<i32>,
        value_b: &GpuMatrix<T>, count_b: &GpuMatrix<i32>,
        value_c: &mut GpuMatrix<T>, count_c: &mut GpuMatrix<i32>,
        modulus: i32,
    ) -> Result<()>;
}
```

Concrete impls: `(f32, Max)`, `(f32, Min)`, `(f64, Max)`, `(f64, Min)`. Each
looks up its kernel name (`counting_gemm_f32_max`, etc.) and launches with
tile-based grid dims (mirror existing kernel).

### 3. GPU CRT driver

New file `crates/tropical-gemm-cuda/src/crt.rs`:

```rust
pub fn count_ground_states_gpu<T, D>(
    ctx: &CudaContext,
    a_values: &[T], m: usize, k: usize,
    b_values: &[T], n: usize,
    count_upper_bound: &BigInt,
) -> Result<CountedMat<T>>
where T: TropicalScalar + DeviceRepr + ValidAsZeroBits + Default + Clone,
      (T, D): CountingCudaKernel<T, D>,
      D: TropicalDirection;
```

Algorithm:

1. Upload `a_values`, `b_values` to `GpuMatrix<T>` once (reused across primes).
2. Pick primes via `choose_primes(2 * count_upper_bound + 1)` — import from
   `tropical_gemm::crt`.
3. For each chosen prime `p`:
   - Allocate `GpuMatrix<i32>` `count_a` and `count_b`, device-fill with `1`.
   - Allocate output `GpuMatrix<T>` `value_c` and `GpuMatrix<i32>` `count_c`.
   - Call `launch_counting_gemm(ctx, value_a, count_a, value_b, count_b, value_c, count_c, p)`.
   - Download `count_c` to a host `Vec<i32>`; push into `residue_streams`.
   - For first prime: download `value_c` to host `Vec<T>` as the reference.
   - For subsequent primes: download `value_c` and assert equal to reference
     (cross-prime invariant). If different, `return Err(Error::from("CRT
     invariant violated: value field differs across primes"))`.
4. Run CRT reconstruction host-side via `crt_combine` (re-exported from
   `tropical_gemm::crt`).
5. Return `CountedMat { nrows: m, ncols: n, values, counts }`.

### 4. Python entry point

In `crates/tropical-gemm-python/src/lib.rs`, add (gated behind the existing
`cuda` feature):

```rust
#[cfg(feature = "cuda")]
#[pyfunction]
#[pyo3(signature = (a, b, direction="min", count_upper_bound=None))]
fn count_ground_states_gpu_py<'py>(...) -> PyResult<(Array2<f32>, Array2<PyObject>)>;
```

Mirrors the CPU `count_ground_states_py` surface. Uses the process-wide
`CudaContext` singleton (existing pattern in this crate).

## File map

- **Create:** `crates/tropical-gemm-cuda/kernels/counting_gemm.cu` (~200 LOC).
- **Modify:** `crates/tropical-gemm-cuda/src/context.rs` or
  `crates/tropical-gemm-cuda/src/lib.rs` — wire up the new `.cu` source via
  `include_str!` and register the four new kernel names.
- **Create:** `crates/tropical-gemm-cuda/src/counting_kernel.rs` (~100 LOC).
- **Create:** `crates/tropical-gemm-cuda/src/crt.rs` (~150 LOC).
- **Modify:** `crates/tropical-gemm/src/crt.rs` — promote `crt_combine`,
  `CRT_PRIMES`, `choose_primes` to `pub` so the cuda crate can reuse them.
- **Modify:** `crates/tropical-gemm-cuda/Cargo.toml` — add `num-bigint`,
  `num-traits`, `num-integer`.
- **Modify:** `crates/tropical-gemm-cuda/src/lib.rs` — re-export the GPU driver
  and `CountedMat`.
- **Create:** `crates/tropical-gemm-cuda/tests/counting_gpu.rs` — cross-check
  GPU vs CPU `count_ground_states` on deterministic inputs.
- **Modify:** `crates/tropical-gemm-python/src/lib.rs` — add the Python
  binding, gated on `cuda` feature.

## Testing

1. **GPU ↔ CPU equivalence.** Three deterministic random inputs (small, medium,
   k ≥ 64) compared cell-by-cell; both values and counts must match exactly.
2. **All-zeros all-ties.** 1×100 * 100×1 all-zero input — true count is 100,
   GPU must reproduce it.
3. **Both directions.** Max and Min tested separately.
4. **Cross-prime invariant.** A test where the GPU value output is deliberately
   corrupted on the second prime (via a small custom mock, if feasible) triggers
   the panic. If mocking is awkward, skip — the invariant is trivially
   preserved for deterministic inputs.
5. **Large-count regression.** `u128::MAX` bound forces multiple primes; result
   still correct.
6. **Python round-trip** (`count_ground_states_gpu_py`) — trivial 1×1, tie
   merge, object-dtype ints.

Use the existing `test_gpu_mat_*` harness pattern for setup (create context,
small matrices, compare against CPU reference).

## Out of scope

- `MaxMul` counting semiring (not on CPU either).
- Integer `T` (`i32`, `i64`). Accept `f32` / `f64` only for v1.
- Streams / kernel concurrency across primes. One synchronous launch per prime.
- Argmax-tracking counting on GPU.

## Risks

- **NVRTC compile time** for four new kernel symbols adds ~400 ms to first
  `CudaContext::new()`. Acceptable; existing code already compiles ~20 kernels
  at init.
- **Uninitialized `count_c` output matrix.** The kernel overwrites every cell
  unconditionally (no accumulation into existing `C`), so no zero-init is
  required. Confirm during implementation.
- **Value-field float determinism across primes.** Same caveat as spec B: the
  CPU reduction order is deterministic; on GPU, warp-level ordering is also
  deterministic for a fixed grid config, so `value_c` equality across primes
  holds for all practical inputs. Pin this in the test harness — if it ever
  fails, the fix is to use a `T`-typed integer, not to relax the invariant.
- **Small problems: GPU loses to CPU.** PCIe upload overhead dominates for
  matrices under ~256×256. Not a correctness issue; document as expected.
