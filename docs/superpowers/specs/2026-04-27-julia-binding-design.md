# Spec J — `CountingTropicalGEMM.jl` Julia binding

**Date:** 2026-04-27
**Status:** design (auto-approved)
**Branch:** `counting-tropical`
**Depends on:** spec I (`count_ground_states_gpu_u64`).
**Scope:** add a Julia interface to `count_ground_states_gpu_u64` for the four `(T ∈ {f32, f64}, D ∈ {Max, Min})` combos. Distributed as a Julia package `CountingTropicalGEMM.jl/` at the repo root, backed by a C ABI added to the existing `tropical-gemm-cuda` crate (now built as both `rlib` and `cdylib`).

## Architecture

Three layers:

1. **Rust C ABI** in `crates/tropical-gemm-cuda/src/c_api.rs`. Eight `extern "C" fn` wrapping `count_ground_states_gpu_u64::<T, D>` for the four combos plus context init / version / last-error helpers. Lazy global `CudaContext` via existing `get_global_context`. Each function body wrapped in `std::panic::catch_unwind` to prevent UB on Rust panic.

2. **Cargo cdylib output**. Add `crate-type = ["rlib", "cdylib"]` to `tropical-gemm-cuda/Cargo.toml`. `cargo build --release -p tropical-gemm-cuda` produces both `libtropical_gemm_cuda.rlib` (used by `tropical-gemm-python`) and `libtropical_gemm_cuda.so`.

3. **Julia package `CountingTropicalGEMM.jl/`** at repo root:
   - `Project.toml` — name, uuid, version. No external Julia deps.
   - `src/CountingTropicalGEMM.jl` — main module: `Max`/`Min` direction tags, `CountedMatU64{T}` struct, `count_ground_states_gpu_u64` methods (4, dispatched on `T` × direction tag).
   - `test/runtests.jl` — small parity sanity (3 tests).
   - `README.md` — install instructions and usage example.

## C ABI surface

```c
// Returns 0 on success, non-zero error code on failure.
// All inputs row-major. Caller allocates out_values[m*n], out_counts[m*n].
int32_t tg_count_ground_states_gpu_u64_f32_max(
    const float* a, size_t m, size_t k,
    const float* b, size_t n,
    uint64_t count_upper_bound,
    float*    out_values,
    uint64_t* out_counts
);
// Same shape for: f32_min, f64_max, f64_min (with `double` for f64).

// Returns the error message from the last failed call on this thread, or
// NULL if there has not been one. Pointer valid until the next call.
const char* tg_last_error_message(void);

// API version (compile-time constant for now). Lets the Julia side
// version-check against the loaded library.
int32_t tg_api_version(void);
```

Error codes: 0 = ok, 1 = invalid input (dim mismatch, null pointer), 2 = bound exceeds u64 envelope, 3 = CUDA error (kernel launch, memory, etc.), 4 = panic / other internal error.

## Julia surface

```julia
module CountingTropicalGEMM

export count_ground_states_gpu_u64, CountedMatU64, Max, Min

struct Max end
struct Min end

struct CountedMatU64{T}
    values::Matrix{T}
    counts::Matrix{UInt64}
end

struct CountingTropicalGEMMError <: Exception
    code::Int32
    msg::String
end

function count_ground_states_gpu_u64(
    ::Type{Max}, A::Matrix{Float32}, B::Matrix{Float32}, bound::UInt64
)::CountedMatU64{Float32} ... end
# + same for (Min, Float32), (Max, Float64), (Min, Float64).

end
```

Dispatch on direction-type and scalar element-type. Single user-facing entry, four method bodies that `ccall` into the matching C symbol.

## Library resolution

`__init__()` looks up `libtropical_gemm_cuda` in this order:

1. `ENV["TROPICAL_GEMM_LIB"]` if set (absolute path).
2. `joinpath(@__DIR__, "..", "..", "target", "release", "libtropical_gemm_cuda.so")` — finds the workspace dev-build output.
3. `Libdl.find_library("tropical_gemm_cuda")` — standard search paths.

If none found, defer the error to first `ccall` with a clear message. Resolved path stored in a `const Ref{String}`.

## Memory layout

Julia's `Matrix{T}` is **column-major**; the kernel expects **row-major**. Wrapper transposes inputs at the boundary (`A_rm = collect(transpose(A))`) and reshape the row-major outputs back to Julia matrices. The transpose cost is O(M·K + K·N) host bytes — negligible compared to the kernel time at any meaningful problem size.

## Error handling

C functions return `Int32`; Julia wrapper checks the code and on non-zero calls `tg_last_error_message()` and throws `CountingTropicalGEMMError(code, msg)`. Code 2 (bound too large) maps to a distinct subtype `BoundTooLargeError <: CountingTropicalGEMMError` so callers can catch it specifically and fall back.

## Tests

`test/runtests.jl`:

1. **f32 Max small case** — A, B with hand-computable result; assert values + counts.
2. **f64 Min larger case** — small randomized check against a Julia-side O(M·N·K) loop reference.
3. **Bound too large** — pass `bound = UInt64(1) << 62`, assert `BoundTooLargeError` thrown.

No Julia↔Rust cross-check inside the package — the Rust side has 21/21 integration tests covering correctness. Julia tests verify FFI plumbing only.

## Risks

- **Cross-language panic safety.** Each `extern "C" fn` wrapped in `std::panic::catch_unwind`; panic returns code 4 with message stored in TLS.
- **Static init time on first call.** NVRTC compilation of all kernels is ~7 sec; happens lazily on first `count_ground_states_gpu_u64` call. Document. Future: explicit `init()` call to pay the cost upfront.
- **Library not found on first call.** Resolved-path check happens at module load; missing library throws clearly.
- **Thread-local error message lifetime.** Pointer valid until the next call on the same thread. Documented in C header comments and Julia docstring.

## Non-goals

- BigInt entry point.
- Direct `CuArray` interop (avoids the H↔D round-trip per call). Possible follow-up if call-rate matters.
- Cross-compiled binary artifact / Julia registry release.
- Python parity (`tropical-gemm-python` PyO3 bindings unchanged).

## Roll-out

1. Add `c_api` module + cdylib crate-type. Sanity-check symbol export with `nm` on the produced `.so`.
2. Scaffold `CountingTropicalGEMM.jl/` with Project.toml + main module.
3. Add Julia tests; run on A100 node (need GPU + NVRTC).
4. Commit.

If anything in the above shape regresses the Rust crate's existing tests, stop and investigate before adding more.

## Outcome (measured 2026-04-27 on A100-SXM4-80GB)

Landed cleanly. `cargo build --release -p tropical-gemm-cuda` produces both `libtropical_gemm_cuda.rlib` (used by Python crate) and `libtropical_gemm_cuda.so`. All 6 expected symbols exported (`tg_api_version`, `tg_last_error_message`, four `tg_count_ground_states_gpu_u64_*`).

Julia package tests (8/8 green) on A100:
1. `f32 Max small` — hand-verifiable 2×2 case.
2. `f64 Min vs reference` — randomized 12×17 × 17×11 against pure-Julia loop.
3. `all-ties large K` — M=5, K=200, N=7, asserts counts == K.
4. `BoundTooLargeError` — bound = 2^62, asserts typed exception.
5. `DimensionMismatch` — asserts native Julia `DimensionMismatch` thrown.

**Files:**
- `crates/tropical-gemm-cuda/Cargo.toml` — added `crate-type = ["rlib", "cdylib"]`.
- `crates/tropical-gemm-cuda/src/c_api.rs` — 4 entry points + last-error TLS + version + panic-catch wrappers.
- `crates/tropical-gemm-cuda/src/lib.rs` — `pub mod c_api;`.
- `CountingTropicalGEMM.jl/Project.toml` — package metadata.
- `CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl` — main module (~150 LOC).
- `CountingTropicalGEMM.jl/test/runtests.jl` — 5 testsets.
- `CountingTropicalGEMM.jl/README.md` — usage + library resolution docs.
