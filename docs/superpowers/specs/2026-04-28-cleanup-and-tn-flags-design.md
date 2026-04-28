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

`tA`, `tB ∈ {'N', 'T'}` follow the **column-major BLAS gemm convention**
(matches CUBLAS / Julia's `LinearAlgebra.BLAS.gemm`). Operands and output are
column-major `CuMatrix` (Julia's natural layout):
- `tA == 'N'`: `A` is `M×K` column-major; `op(A) = A`.
- `tA == 'T'`: `A` is `K×M` column-major; `op(A) = A^T` (logical `M×K`).
- Same for `tB` with `(K, N)`.

So `tropical_matmul('N', 'N', A, B)` on Julia inputs computes the
algebraic `A * B` directly — no swap, no transpose. `tropical_matmul('T', 'T', A, B)` computes `A^T * B^T`. Etc.

Returns a fresh `CuMatrix` of the same element type and column-major shape
`M × N` where `M = size(op(A), 1)`, `K = size(op(A), 2) = size(op(B), 1)`,
`N = size(op(B), 2)`.

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

### NVRTC + eager compile (revised — codex-flagged)

cudarc 0.12.1 compiles the entire `.cu` source via `compile_ptx()` and `load_ptx()` resolves every named function at module load time. The existing `CudaContext::from_device` (`context.rs:135-158`) eagerly compiles + caches all kernel symbols at first context init. So **all 16 specialized kernels compile up front on first `tropical_matmul` call**, not lazily per-name.

First-call cost estimate: current 4-kernel set takes ~7 s NVRTC. Going to 16 (4 transposes × 2 dtypes × 2 dirs) is ~4× more code. Expected: **~25-30 s on first call, paid once per process**, then warm thereafter. Document in the public docstring; consider an optional precompile-ahead hook in a future iteration.

### Stride / indexing per variant — column-major (revised)

Operands are column-major `CuMatrix`. Logical `op(A)` is `M×K`, `op(B)` is `K×N`. The kernel iterates `k ∈ [0, K)` for the inner product. Storage layout (post-flag) and read indices:

| Variant | A storage | A read `A[i,k]` | B storage | B read `B[k,j]` |
|---|---|---|---|---|
| NN | `M×K` col-major | `A[i + k*M]` | `K×N` col-major | `B[k + j*K]` |
| NT | `M×K` col-major | `A[i + k*M]` | `N×K` col-major | `B[j + k*N]` |
| TN | `K×M` col-major | `A[k + i*K]` | `K×N` col-major | `B[k + j*K]` |
| TT | `K×M` col-major | `A[k + i*K]` | `N×K` col-major | `B[j + k*N]` |

Output is always `M×N` col-major: `C[i + j*M]`.

The existing kernel is row-major NN. **All 16 specializations are rewritten to be column-major-natural** so the Julia caller passes flags through directly without any swap-operands trick. Memory access patterns per variant — coalescing at warp level prefers the dimension stored contiguously in memory (in column-major: row index for matrix `M×K` is contiguous). Each kernel template hand-codes the access pattern that coalesces best for its `(transA, transB)` combo.

### Warpk dispatch dropped (revised — codex-flagged)

Codex correctly flagged that the current general AoS path has a warpk variant that wins ~6× for small-MN-large-K shapes (`counting_kernel.rs:42-54, 86-103, 180-193`; `counting_gemm.cu:99-157`). **For Spec M v1 we drop warpk** and accept the regression in that regime, in exchange for the simpler 16-kernel surface. If subsequent measurement shows the small-MN regime matters, add warpk back as a follow-up Spec N (would multiply the kernel count by some factor — e.g., NN+warpk-NN, etc.).

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

### C ABI (one family, 4 entries) — revised

The Julia wrapper is responsible for computing logical `(M, K, N)` from Julia matrix shapes and flags before the ccall. The C ABI takes the logical dims directly — no derivation table on the Rust side, fewer chances for the wrapper and Rust to disagree about shape conventions.

```c
int tg_tropical_matmul_<T>_<D>(
    char     tA,            // 'N' or 'T'
    char     tB,
    size_t   M,             // logical rows of op(A) = rows of C
    size_t   K,             // logical cols of op(A) = rows of op(B)
    size_t   N,             // logical cols of op(B) = cols of C
    uint64_t a_dev,         // CUdeviceptr; bytes are PairT, column-major
    uint64_t b_dev,
    int32_t  p,
    uint64_t out_dev        // CUdeviceptr; bytes are PairT, column-major M×N
);
```

Returns `0` ok, `1` invalid input (null/p<2/bad flag), `3` CUDA error, `4` panic. Wrapped in `catch_unwind`, sets thread-local last-error message.

### Rust driver

```rust
pub fn tropical_matmul_kernel<T, D>(
    ctx: &CudaContext,
    tA: char, tB: char,
    m: usize, k: usize, n: usize,
    a_dev_ptr: u64,
    b_dev_ptr: u64,
    p: i32,
    out_dev_ptr: u64,
) -> Result<()>
```

Validates flags ∈ {'N','T'} (else `InvalidState`), validates `p ≥ 2`, validates `m, k, n > 0`, syncs the device (CUDA.jl coordination), looks up the kernel for the matching `(T, D, tA, tB)` quadruple, launches it, syncs again. Output bytes are already in the caller's device buffer; returns `Ok(())`.

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

### Element-type dispatch (revised — codex-flagged)

Define **separate methods per element type** so element-type mismatches naturally raise `MethodError` at dispatch time. Two methods per public function:

```julia
function tropical_matmul(tA::Char, tB::Char,
                        A::CuMatrix{ModCountingTropical{T, P}},
                        B::CuMatrix{ModCountingTropical{T, P}}
                       ) where {T <: Union{Float32, Float64}, P}
    _tropical_matmul_impl(tA, tB, :max, A, B)
end

function tropical_matmul(tA::Char, tB::Char,
                        A::CuMatrix{ModCountingTropicalMin{T, P}},
                        B::CuMatrix{ModCountingTropicalMin{T, P}}
                       ) where {T <: Union{Float32, Float64}, P}
    _tropical_matmul_impl(tA, tB, :min, A, B)
end
```

Mismatched element types (`A::ModCountingTropical{Float32, 7}` × `B::ModCountingTropicalMin{Float32, 7}`, or different `T`/`P` between the two args) → no method matches → `MethodError`. Inside `_tropical_matmul_impl`, four ccall thunks (one per `(T, dir)` combo) are reached via `@eval`-generated lookup keyed on the runtime tuple `(T, dir)`.

### Validation

- `tA`, `tB ∈ {'N', 'T'}` else `ArgumentError`.
- `2 ≤ P < 2^31` else `ArgumentError`.
- Inner-K match between operands (after applying flags) else `DimensionMismatch`.
- `tropical_matmul!`: `C` size must equal `(M, N)` else `DimensionMismatch`.
- Element-type mismatch → `MethodError` at dispatch (no manual check needed).

### Pointer extraction (no Julia-level reinterpret) — revised

Codex flagged that Julia 1.11's `reinterpret` rejects converting `CuMatrix{ModCountingTropical{Float64, P}}` → `CuMatrix{PairF64}` due to padding-incompatibility with `PairF64`'s explicit `_pad::Int32` (we hit this in Spec K Task 7).

**Resolution**: skip Julia-level reinterpret entirely. The `ccall` takes the device pointer as an opaque `UInt64`:

```julia
a_dev_u64 = UInt64(UInt(pointer(A)))   # pointer(::CuArray{T}) returns CuPtr{T}
b_dev_u64 = UInt64(UInt(pointer(B)))
out = CuMatrix{eltype(A)}(undef, M, N)
out_dev_u64 = UInt64(UInt(pointer(out)))
ccall((sym, _libpath()), Cint, (Cchar, Cchar, Csize_t, Csize_t, Csize_t,
                                UInt64, UInt64, Int32, UInt64),
      tA, tB, M, K, N, a_dev_u64, b_dev_u64, Int32(P), out_dev_u64)
```

Bytes of `Matrix{ModCountingTropical{Float32, P}}` are `(Float32, Int32)` per element, 8 B, naturally aligned to 4 (Julia) but the `CuArray` heap is 16-aligned, so kernel-required 8 B element alignment is satisfied.

Bytes of `Matrix{ModCountingTropical{Float64, P}}` are `(Float64, Int32, padding)` per element, 16 B aligned to 8 (Julia) — matches `PairF64`'s 16 B element layout (`val::Float64` at offset 0, `cnt::Int32` at offset 8, `_pad` at offset 12). Reinterpret would fail (struct-incompat in Julia's view), but **byte-level** the layouts match — and the ccall just passes a raw byte pointer, so Julia's reinterpret rejection is sidestepped.

The Rust side is responsible for casting `*const c_void` → `*const PairT` and using the bytes. This is identical to how Spec L's `matmul_mod_p_kernel_only` already works.

### AoS output

The kernel writes a single `PairT` per output cell (packed `(value, count)` 8 B for f32, 16 B for f64). The Julia wrapper allocates `out = CuMatrix{eltype(A)}(undef, M, N)` and passes its device pointer. No post-kernel zip pass. (Existing kernel writes SoA — needs modification for Spec M.)

### `tropical_matmul!`

Same flow, but `C` is caller-provided. Wrapper validates `size(C) == (M, N)` and `eltype(C) == eltype(A)`. Returns `C`.

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
  - `launch_counting_gemm_ones` (CRT path's count=1 specialization).
  - `launch_counting_gemm_dev_ptr` (replaced by a single new `launch_tropical_matmul<T, D>` that takes `(tA, tB)` flags).
  - `launch_counting_gemm` (replaced; existing one is row-major NN only).
  - `KERNEL_NAME_WARPK`, warpk dispatch logic, transposed-B helper. (Re-add as Spec N if measured to matter.)
  - `DevPtr` wrapper — keep (still needed for raw-pointer kernel-arg passing).
- From `pair.rs`:
  - `PackPair` trait, `pack_f32`, `pack_f64`, `pack_f32_ones`, `pack_f64_ones`, `pack_pair`, `pack_ones`. Keep `PairF32`, `PairF64` structs and `DeviceRepr`/`ValidAsZeroBits` impls.
- From `lib.rs`:
  - `pub mod crt;` declaration only.
  - **Do NOT delete `tropical_matmul_gpu`** — codex verified it is consumed by `crates/tropical-gemm-python/src/lib.rs:1667, 1906` and documented in `docs/src/gpu.md:113-121`. That's the older non-counting tropical path. Out of scope for Spec M (parent crate / Python bindings stay untouched).
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

1. **NVRTC eager-compile cost**: codex verified cudarc 0.12.1 compiles all kernels in `compile_ptx()` and `load_ptx()` resolves all functions at module load (`cudarc-0.12.1/src/nvrtc/safe.rs:71-92`, `src/driver/safe/ptx.rs:17-55`). The repo's `CudaContext::from_device` (`context.rs:135-158`) already eagerly compiles + caches everything. Going from 4 → 16 kernels means **all 16 compile up front** on first context init. Expected first-call cost ~25–30 s, paid once per process. Documented in the public docstring.
2. **Kernel write-AoS-output**: existing kernels write SoA. Switching to AoS output requires verifying that 16 B aligned writes for `PairF64` coalesce as well as separate `f64 + i32` stores (should, on Ampere+; verify on Turing).
3. **Warpk regression risk**: dropping warpk in v1 surrenders the small-MN-large-K win (~6× on small shapes). If post-cleanup measurement shows that regime matters, add warpk back as Spec N.
4. **Pointer extraction stability**: the `UInt64(UInt(pointer(::CuArray)))` form depends on CUDA.jl returning a `CuPtr{T}` that can be converted to `UInt`. This works on CUDA.jl v5.x; pin in compat.
5. **Modulus compile-time specialization** is **not** in scope. `P` stays a runtime arg threaded through Barrett reduction. Defer.

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
