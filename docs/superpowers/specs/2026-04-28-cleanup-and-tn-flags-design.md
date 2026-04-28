# Cleanup + N/T flags: minimal mod-P counting tropical GEMM

**Date:** 2026-04-28
**Status:** Design — pending user approval
**Branch:** `main`

## Goal

Strip the `tropical-gemm-cuda` crate and `CountingTropicalGEMM.jl` package down to a single, BLAS-style mod-P counting tropical matmul on device-resident inputs, with `'N'`/`'T'` per-operand transpose flags. Everything related to CRT, BigInt, multi-prime, ground-state counting, host wrappers, and other accumulated surface goes away.

## Non-goals

- Touching the parent `tropical-gemm` crate. Its CPU counting tropical, BigInt CRT, and `CountingTropical{T, CT}` infrastructure stay as-is.
- Host-Matrix convenience entries. Users with host data are responsible for `CuArray(...)` upload.
- `mul!`, `LinearAlgebra` integration, BLAS-via-`Base.*`. Just one named function and an in-place variant.
- Conjugate-transpose (`'C'`). Tropical max-plus has no conjugation; we don't model it.
- Min direction as a separate function. Direction is encoded in the element type — `ModCountingTropical{T, P}` (max-plus) vs `ModCountingTropicalMin{T, P}` (min-plus).

## Public API (Julia)

```julia
"""
    tropical_matmul(tA::Char, tB::Char, A, B) -> CuMatrix

Mod-P counting tropical matrix multiplication on the GPU.

`A` and `B` must be `CuMatrix{ModCountingTropical{T, P}}` (max-plus) or
`CuMatrix{ModCountingTropicalMin{T, P}}` (min-plus) with the same `T` and `P`.
`T ∈ {Float32, Float64}`. `2 ≤ P < 2^31`. Modulus and direction are inferred
from the element type; mismatched element types raise `MethodError`.

`tA`, `tB ∈ {'N', 'T'}` follow the BLAS gemm convention:
- `'N'`: operand stored as is — A is M×K (row-major), B is K×N (row-major).
- `'T'`: operand stored transposed — A is K×M (row-major), B is N×K (row-major).
  Effectively, treats column-major Julia bytes of an M×K matrix as a row-major
  K×M layout, so `'T','T'` produces algebraic A·B for column-major inputs.

Returns a fresh `CuMatrix` of the same element type and shape M×N (where
M = size(op(A), 1), K = size(op(A), 2) = size(op(B), 1), N = size(op(B), 2)).

Throws `DimensionMismatch` for shape mismatch, `ArgumentError` for invalid
flags or `P` out of range, `CountingTropicalGEMMError` for FFI/CUDA failures.
"""
function tropical_matmul end

"""
    tropical_matmul!(tA::Char, tB::Char, A, B, C) -> C

In-place variant. `C` must be a preallocated `CuMatrix` of the same element
type and shape `(size(op(A), 1), size(op(B), 2))`.
"""
function tropical_matmul! end
```

That is the entire public surface — two functions plus the four types (`ModCountingTropical{T, P}`, `ModCountingTropicalMin{T, P}`, error type, version helper).

## Kernel design

### Compile-time specialization

Four CUDA kernel template instantiations per `(T, D)` combo, one per `(transA, transB) ∈ {NN, NT, TN, TT}`. Total kernel matrix:

| | T = Float32 | T = Float64 |
|---|---|---|
| Max | 4 | 4 |
| Min | 4 | 4 |

= **16 kernel symbols** baked into the NVRTC source. Each combo has its memory access pattern hard-coded — no runtime branching on the flag. Coalescing is per-variant best-effort.

### NVRTC + lazy compile

cudarc compiles per kernel-name lazily on first `get_kernel(name)`. So a user who only does `'T','T'` matmuls only pays the compile cost for that variant (~7 s on first call). Other variants compile only when first invoked. Memory cost: 16 cubin entries in the cache.

### Stride / indexing per variant

Let A be the M×K (post-transpose-flag) operand. The kernel iterates `k = 0..K-1`. Read `A[i, k]` and `B[k, j]`:

| Variant | A indexing | B indexing |
|---|---|---|
| NN | `A[i*K + k]` | `B[k*N + j]` |
| NT | `A[i*K + k]` | `B[j*K + k]`  ← B^T as N×K row-major |
| TN | `A[k*M + i]`  ← A^T as K×M row-major | `B[k*N + j]` |
| TT | `A[k*M + i]` | `B[j*K + k]` |

The existing kernel only does NN. The other three variants are mechanical re-indexings of the same algorithmic core (max-plus inner product with mod-P count multiply + Barrett).

### Existing warpk dispatch

Drop. The warpk path was specialized for parallelism-starved small-MN-large-K shapes via `launch_counting_gemm_ones` (the "input counts are all 1" path). With this redesign there's no input-counts-are-1 case to specialize for (general AoS only), and the user's primary path is the device API where the warpk codepath was already a marginal win at edges. Simpler to drop and revisit if measurement justifies.

### Modulus

Same Barrett reduction as today: kernel takes `i32 modulus` and `u64 mu = floor(2^64 / P)`. Host precomputes `mu`. Modulus encoded at runtime (no per-P specialization).

### Direction tag

`Max` vs `Min` is a compile-time template parameter (existing pattern). The kernel's reduction op switches accordingly.

## Rust surface (after cleanup)

```
crates/tropical-gemm-cuda/src/
  context.rs         (kept; primary CUDA ctx, lazy global)
  counting_kernel.rs (single launch fn, takes (transA, transB) Char + (T, D) generics)
  error.rs           (kept; drop ERR_BOUND_TOO_LARGE)
  gpu_mat.rs         (kept; used internally if needed)
  kernels.rs         (CUDA source: 16 specialized kernels)
  lib.rs             (slimmed; drop crt module, drop tropical_matmul_gpu)
  matmul_mod.rs      (one driver: tropical_matmul_kernel)
  memory.rs          (kept)
  pair.rs            (kept: PairF32, PairF64 structs and DeviceRepr impls;
                      drop PackPair trait + pack_* helpers)
  c_api.rs           (4 entries: tg_tropical_matmul_<T>_<D>; drop everything else)
```

### C ABI (one family, 4 entries)

```c
int tg_tropical_matmul_<T>_<D>(
    char     tA,
    char     tB,
    uint64_t a_dev,         // device pointer; bytes are PairT
    size_t   a_rows,
    size_t   a_cols,
    uint64_t b_dev,
    size_t   b_rows,
    size_t   b_cols,
    int32_t  p,
    uint64_t out_dev        // device pointer; bytes are PairT, length M*N
);
```

Returns `0` ok, `1` invalid input (null/dim/p<2/bad flag), `3` CUDA error, `4` panic. Wrapped in `catch_unwind`, sets thread-local last-error message.

`(M, K, N)` are derived from `(a_rows, a_cols, b_rows, b_cols, tA, tB)` inside the wrapper:

| tA | tB | M | K | N |
|---|---|---|---|---|
| 'N' | 'N' | a_rows | a_cols (= b_rows) | b_cols |
| 'N' | 'T' | a_rows | a_cols (= b_cols) | b_rows |
| 'T' | 'N' | a_cols | a_rows (= b_rows) | b_cols |
| 'T' | 'T' | a_cols | a_rows (= b_cols) | b_rows |

Wrapper validates the consistency check (the inner-K dimension match between A and B) and rejects with `1` if mismatched.

### Rust driver

```rust
pub fn tropical_matmul_kernel<T, D>(
    ctx: &CudaContext,
    tA: char, tB: char,
    a_dev_ptr: u64, a_rows: usize, a_cols: usize,
    b_dev_ptr: u64, b_rows: usize, b_cols: usize,
    p: i32,
    out_val_ptr: u64, out_cnt_ptr: u64,
) -> Result<()>
```

Validates flags ∈ {'N','T'}, derives (M, K, N) per the table, validates K-match, validates `p ≥ 2`, syncs the device (CUDA.jl coordination), launches the kernel for the matching (T, D, tA, tB) variant, syncs again. Returns nothing — output bytes already in caller's device buffers.

## Julia surface (after cleanup)

```
CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl
  module CountingTropicalGEMM
    using CUDA
    using Libdl

    # Public types
    export ModCountingTropical, ModCountingTropicalMin
    export tropical_matmul, tropical_matmul!
    export CountingTropicalGEMMError

    # ... ModCountingTropical[Min] structs + semiring ops (kept verbatim)
    # ... PairF32/PairF64 (internal, used for reinterpret)
    # ... _check_version, _libpath, _throw_for (kept)
    # ... __init__ (CUDA env workaround note kept)

    # Two public functions
    function tropical_matmul(tA::Char, tB::Char, A::CuMatrix, B::CuMatrix) ... end
    function tropical_matmul!(tA::Char, tB::Char, A::CuMatrix, B::CuMatrix, C::CuMatrix) ... end
  end
```

### Element-type dispatch

Each public function is a single method using `@eval` to fan out to the four `(T, D) ∈ {f32,f64} × {Max,Min}` ccall thunks based on the element type's parametric structure. Direction comes from whether the element is `ModCountingTropical` (Max) or `ModCountingTropicalMin` (Min).

### Validation

- `tA`, `tB ∈ {'N', 'T'}` else `ArgumentError`.
- `2 ≤ P < 2^31` else `ArgumentError`.
- A and B element types must be the same parametric kind with same `T`, `P` else `MethodError`.
- Inner dim match (K from A vs K from B per flag combo) else `DimensionMismatch`.
- `tropical_matmul!`: C size must match `(M, N)` else `DimensionMismatch`.

### Layout reinterpret + AoS output

`A::CuMatrix{ModCountingTropical{Float32, P}}` has bytes byte-compatible with `CuMatrix{PairF32}` (Task 6 verified). Wrapper does `reinterpret(PairF32, A)` → `CuMatrix{PairF32}` → grab device pointer → ccall. Identically for `Float64`/`PairF64` and the Min variant.

**The kernel writes AoS output** (one `PairT` per cell, not separate `T` + `i32` SoA buffers). This change vs. the current kernel keeps inputs and output in symmetric layout, eliminates the post-kernel zip pass, and lets `tropical_matmul` allocate a single fresh `CuMatrix{ModCountingTropical{T, P}}` and pass a single output pointer. One packed 8 B (f32) or 16 B (f64) store per cell — should coalesce as well as or better than separate scalar stores on Nvidia's recent SMs.

### `tropical_matmul!`

Same as `tropical_matmul` but the output buffer is the caller-provided `C::CuMatrix{ModCT}`. We `reinterpret` C similarly and pass its device pointer as the output `PairT*`. Returns `C`.

## What gets deleted

### Rust

- `crates/tropical-gemm-cuda/src/crt.rs` — entire file.
- From `c_api.rs`:
  - `tg_count_ground_states_gpu_u64_<T>_<D>` (4 fns)
  - `tg_bench_kernel_only_u64_<T>_<D>` (4 fns)
  - `tg_matmul_mod_p_<T>_<D>` slow-path (4 fns)
  - `tg_matmul_mod_p_pair_<T>_<D>` host-pair (4 fns)
  - `tg_matmul_mod_p_pair_dev_<T>_<D>` (4 fns) — replaced by `tg_tropical_matmul_<T>_<D>`
  - `bench_kernel_only_impl`, `cabi_bench_kernel_only`, `run_u64`, `run_matmul_mod_p`, `run_matmul_mod_p_pair`, `run_matmul_mod_p_dev`, all related macros
  - `ERR_BOUND_TOO_LARGE` constant + `classify_cuda_error` u64-bound branch
- From `matmul_mod.rs`:
  - `matmul_mod_p`, `matmul_mod_p_pair`, `matmul_mod_p_kernel_only`, `run_packed`, related tests
- From `counting_kernel.rs`:
  - `launch_counting_gemm_ones`, `launch_counting_gemm_dev_ptr` (replaced by single new fn that takes flags)
  - `KERNEL_NAME_WARPK`, warpk dispatch logic, transposed-B helper
- From `pair.rs`:
  - `PackPair` trait, `pack_f32`, `pack_f64`, `pack_f32_ones`, `pack_f64_ones`, `pack_pair`, `pack_ones`. Keep `PairF32`, `PairF64` structs and `DeviceRepr`/`ValidAsZeroBits` impls.
- From `lib.rs`:
  - `pub mod crt;` declaration
  - `tropical_matmul_gpu` and related convenience fns (the older non-counting tropical path) — verify whether these are used by anything else first; if not, delete.
- CUDA kernels source (`kernels/counting_gemm.cu` or wherever):
  - Drop ones-specialized kernels, warpk variants, transposed-B-warpk variants.
  - Add 16 specialized kernels (4 transpose combos × 2 dtypes × 2 dirs).

### Julia

- Module body deletions:
  - `count_ground_states_gpu_u64`, `bench_kernel_only_u64`, `CountedMatU64`, `BoundTooLargeError`, `ERR_BOUND_TOO_LARGE`
  - `Max`, `Min` tag types (no longer needed; element type encodes direction)
  - `TropicalMatrix` struct + methods + tests
  - `CountingTropicalMin` (the local Mods-interop type — distinct from `ModCountingTropicalMin`!)
  - `tropical_matmul_min`, `tropical_matmul_dev`, `tropical_matmul_dev_min`, `cuda_pair_buffer`
  - Existing host-Matrix `tropical_matmul`, `mul!` overloads
  - `using LinearAlgebra`, `using Mods`, `using TropicalNumbers`
  - `Base.convert(::ModCT, ::CountingTropical)` interop methods
  - `_FFI_SYMS`, `_MOD_FAST_SYMS` dicts (replaced by single new dispatch table)
  - `_pair_type`, `_modulus`, `_check_mod_p`, `_row_major_pair`, `_zip_to_modct`, `_tropical_matmul_core` (replaced by simpler internals)
  - `_tg_mod_pair_ccall`, `_ensure_cuda_jl_context` (or kept as needed)
- `Project.toml` deps: drop `LinearAlgebra`, `Mods`, `TropicalNumbers`. Keep `CUDA`, `Libdl`. Drop `Random` from `[extras]` if no remaining tests use it.
- Tests: drop everything not exercising `ModCountingTropical[Min]` or the new `tropical_matmul`/`tropical_matmul!`. Keep ~30 of the 103.
- `bench/bench.jl`, `bench/bench_huge.jl` — delete (use the old CRT/u64 path). Rewrite `bench/bench_mul.jl` for the new API.

## Tests

After cleanup the suite has ~30 tests:

1. **`ModCountingTropical[Min]` types and semiring** — kept verbatim from current Task 6 testset.
2. **`tropical_matmul('N','N',...)`** — small Max f32 and Min f64 with cell-by-cell reference cross-check.
3. **`tropical_matmul('T','T',...)`** — verify column-major Julia inputs round-trip correctly.
4. **`tropical_matmul('N','T',...)` and `('T','N',...)`** — verify each indexing variant against the same reference.
5. **`tropical_matmul!`** — preallocated-C reuse with two different inputs.
6. **Edge cases**: 1×1×1, K=1, P=2, P=2^31-1, all-tie inputs.
7. **Errors**: bad flag (`'C'`, `'X'`), bad P (1, 2^31), DimensionMismatch on K-mismatch, MethodError on direction or T mismatch.

## Migration

This is breaking. No back-compat shim. Memory entry will note "Spec K, L, J entries deleted; only Spec M (this) remains." Existing user code using the old `tropical_matmul(A::Matrix, B::Matrix)` host signature, `mul!`, `count_ground_states_gpu_u64`, `TropicalMatrix`, etc., will break.

## Open questions / risks

1. **NVRTC compile time on first call**: 16 specialized kernels × ~0.5 s each = ~8 s if all compile at once. Cudarc compiles lazily per name, so users only pay for the variants they invoke (typically 1–2). Acceptable.
2. **Kernel write-AoS-output**: existing kernels write SoA. Switching to AoS output is a small modification but does need verification that f64 16-byte AoS stores coalesce as well as separate f64 + i32 stores (they should — modern Nvidia GPUs prefer 16 B aligned writes).
3. **Modulus compile-time specialization** is **not** in scope. P stays a runtime arg threaded through Barrett reduction. Spec K had this in roadmap; defer.

## Roadmap items deferred (not in scope)

- NVRTC per-prime templating (compile-time `P`).
- On-device transpose pre-kernel for cases where data isn't already in row-major.
- Streaming / async API for overlapping uploads with compute.
- Output buffer reuse across calls beyond what `tropical_matmul!` provides.

## File-level diff plan

```
crates/tropical-gemm-cuda/src/c_api.rs                   -550 / +90 LOC
crates/tropical-gemm-cuda/src/crt.rs                     -350         (delete file)
crates/tropical-gemm-cuda/src/counting_kernel.rs         -200 / +100
crates/tropical-gemm-cuda/src/matmul_mod.rs              -250 / +90
crates/tropical-gemm-cuda/src/pair.rs                    -50  / 0
crates/tropical-gemm-cuda/src/lib.rs                     -20  / 0
crates/tropical-gemm-cuda/kernels/counting_gemm.cu       -150 / +200  (drop warpk/ones, add 16 NN/NT/TN/TT)
CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl      -500 / +120
CountingTropicalGEMM.jl/Project.toml                     ~5   / ~0
CountingTropicalGEMM.jl/test/runtests.jl                 -350 / +150
CountingTropicalGEMM.jl/bench/bench.jl                   -40           (delete)
CountingTropicalGEMM.jl/bench/bench_huge.jl              -50           (delete)
CountingTropicalGEMM.jl/bench/bench_mul.jl               -100 / +80
```

Net: ~−2000 / +830 LOC, mostly deletion.
