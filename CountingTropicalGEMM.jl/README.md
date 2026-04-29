# CountingTropicalGEMM.jl

Julia bindings for a fast CUDA implementation of the **mod-P counting
tropical GEMM**: a BLAS-style matrix multiplication on the
counting tropical semiring with counts reduced modulo a fixed prime/integer
`P`. The kernels live in the Rust crate `tropical-gemm-cuda` (one workspace
level up); this package wraps them as a Julia-friendly API operating
directly on `CuMatrix` of dedicated element types.

Backed by a tiled shared-memory CUDA kernel (Spec N) with layout-aware
coalesced loads, peaking around **640 G tropical-ops/s** on an RTX 6000 at
4096³ (Float32, max-plus, modulus 7).

## Semiring

For element `(v, c)` (value `v`, count `c`):

- **Multiplication:** `(va, ca) ⊗ (vb, cb) = (va + vb, (ca · cb) mod P)`.
- **Addition (max-plus):** keep the larger `v`. On a tie, sum counts mod `P`.
- **Addition (min-plus):** keep the smaller `v`. Same tie rule.

The package provides two element types — one per direction:

| Julia type | Direction | `+` keeps |
|---|---|---|
| `ModCountingTropical{T, P}`    | max-plus | larger `v` |
| `ModCountingTropicalMin{T, P}` | min-plus | smaller `v` |

`T ∈ {Float32, Float64}`; `P` is a compile-time `Int` with `2 ≤ P < 2^31`.
Counts are stored as `Int32` in `[0, P)`.

`Base.show` renders elements as `(v₊, c_P)` (max-plus) or `(v₋, c_P)`
(min-plus). For example, `ModCountingTropical{Float32, 7}(3.0, 2)` prints
as `(3.0₊, 2₇)`.

## Requirements

- A CUDA-capable GPU and a working CUDA toolkit (NVRTC, libcuda).
- Rust toolchain (stable, 1.70+) to build the backing `cdylib`.
- Julia ≥ 1.10.

On clusters where the system `libcuda.so` is older than what CUDA.jl ships,
launch Julia with `JULIA_CUDA_USE_COMPAT=false` so the Julia and Rust
sides share the same driver.

## Build the backing library

The package `dlopen`s a Rust shared library
(`libtropical_gemm_cuda.{so,dylib,dll}`). Build it once from the workspace
root, before `using` the package:

```bash
# From the workspace root (one level above this directory):
cargo build --release -p tropical-gemm-cuda
# Produces: target/release/libtropical_gemm_cuda.{so,dylib,dll}
```

Rebuild whenever the Rust crate or CUDA kernel source changes.

The package's `__init__` looks for the library in this order:

1. `ENV["TROPICAL_GEMM_LIB"]` — absolute path, if set.
2. `<package>/../target/release/libtropical_gemm_cuda.<ext>`.
3. The system loader path (whatever `Libdl.dlopen` resolves).

Set `ENV["TROPICAL_GEMM_LIB"]` if the package lives outside the workspace.

## Install

In dev mode (recommended while iterating on the kernel):

```julia
using Pkg
Pkg.develop(path="path/to/tropical-gemm/CountingTropicalGEMM.jl")
```

## Quick start

```julia
using CountingTropicalGEMM, CUDA

const P = 7                     # modulus (compile-time)
M, K, N = 64, 128, 64

# Build random max-plus inputs on the GPU.
A = CuArray([ModCountingTropical{Float32, P}(Float32(rand(0:9)),
                                              Int32(rand(1:P-1)))
             for _ in 1:M, _ in 1:K])
B = CuArray([ModCountingTropical{Float32, P}(Float32(rand(0:9)),
                                              Int32(rand(1:P-1)))
             for _ in 1:K, _ in 1:N])

# C = A * B in the counting tropical semiring (max-plus).
C = tropical_matmul('N', 'N', A, B)            # M×N CuMatrix

# Pull a result back to host:
host = Array(C)
host[1, 1]   # ModCountingTropical{Float32, 7} — prints like (5.0₊, 3₇)
```

For min-plus, just swap the element type:

```julia
A = CuArray([ModCountingTropicalMin{Float32, P}(...) for _ in 1:M, _ in 1:K])
B = CuArray([ModCountingTropicalMin{Float32, P}(...) for _ in 1:K, _ in 1:N])
C = tropical_matmul('N', 'N', A, B)
```

## API

```julia
tropical_matmul(tA::Char, tB::Char, A, B) -> C
tropical_matmul!(tA::Char, tB::Char, A, B, C) -> C
```

- `tA, tB ∈ {'N', 'T'}` — BLAS-style transpose flags. The result is
  `op(A, tA) * op(B, tB)` where `op(X, 'N') = X` and `op(X, 'T') = Xᵀ`.
- `A` and `B` are `CuMatrix{ModCountingTropical{T, P}}` (max-plus) or
  `CuMatrix{ModCountingTropicalMin{T, P}}` (min-plus); both arguments
  must share element type, `T`, and `P`.
- `C` (in the in-place form) is preallocated and must have the right
  output shape; both element types must match `A` and `B`.

Logical shapes follow column-major BLAS convention:

| `tA` | `tB` | `A` storage | `B` storage | `C` |
|:-:|:-:|---|---|---|
| `'N'` | `'N'` | `M×K` | `K×N` | `M×N` |
| `'N'` | `'T'` | `M×K` | `N×K` | `M×N` |
| `'T'` | `'N'` | `K×M` | `K×N` | `M×N` |
| `'T'` | `'T'` | `K×M` | `N×K` | `M×N` |

## Errors

- `CountingTropicalGEMMError(code, msg)` — propagated from the Rust side.
  - `code = 1` → invalid input (bad flag, dimension mismatch, P < 2,
    null device pointer).
  - `code = 3` → CUDA error (kernel launch / context).
  - `code = 4` → internal panic across the FFI boundary.
- `DimensionMismatch` — Julia-side check that `C` matches the result
  shape implied by `tA, tB, size(A), size(B)`.

The thrown `msg` carries the error string returned by the Rust ABI's
last-error TLS slot — useful when triaging CUDA failures.

## Performance notes

The kernel is a shared-memory tiled GEMM with **layout-aware coalesced
global loads**, so all four `(tA, tB)` combinations reach essentially the
same throughput. RTX 6000 (Turing sm_75), `ModCountingTropical{Float32, 7}`:

| Shape | NN | NT | TN | TT |
|---:|---:|---:|---:|---:|
| 1024³ | 544 | 524 | 524 | 548 G/s |
| 4096³ | 642 | 627 | 613 | 647 G/s |

The first call into a kernel triggers an NVRTC compile (~30 s wallclock
on Turing); subsequent calls reuse the cached module.

## Tests

```julia
using Pkg; Pkg.test("CountingTropicalGEMM")
```

Or directly:

```bash
JULIA_CUDA_USE_COMPAT=false julia --project=. test/runtests.jl
```

Requires a CUDA GPU. Verifies type semantics, all four `(tA, tB)` flag
combos for both directions and dtypes, in-place reuse, and error paths.

## Benchmark

```bash
JULIA_CUDA_USE_COMPAT=false julia --project=. bench/bench_mul.jl
```

Sweeps `M=N=K ∈ {128, 256, 512, 1024, 2048, 4096}` and reports
ms/call and G tropical-ops/s for each `(tA, tB)`. Recorded results live
in `bench/RESULTS.md`.

## Troubleshooting

- **`Unable to dynamically load the "nvrtc" shared library`**
  → `LD_LIBRARY_PATH` lacks libnvrtc. Load your CUDA module
  (e.g. `module load cuda`) in the same shell before starting Julia.
- **`CUDA_ERROR_OPERATING_SYSTEM` on first matmul call**
  → Driver mismatch between CUDA.jl's bundled `libcuda` stub and the
  system driver. Restart Julia with
  `JULIA_CUDA_USE_COMPAT=false julia --project=.` (env var must be set
  *before* `using CUDA`).
- **`libtropical_gemm_cuda not found`** at `using` time
  → Run `cargo build --release -p tropical-gemm-cuda` from the workspace
  root, or set `ENV["TROPICAL_GEMM_LIB"]` to its absolute path.
- **Stale results after kernel changes**
  → Rebuild the Rust library
  (`cargo build --release -p tropical-gemm-cuda`) and restart Julia so
  the new `.so` is loaded.

## Layout note (for advanced users)

Internally each element is stored as a packed 8 B (`Float32`) or 16 B
(`Float64`, with 4 B padding) `(value, count)` struct with C layout
matching the Rust crate's `PairF32` / `PairF64`. The Julia structs
`ModCountingTropical{T, P}` and `ModCountingTropicalMin{T, P}` are
designed to be `reinterpret`-compatible with these. The kernel always
treats the buffers as column-major; the `'T'` flag flips logical
indexing rather than copying or transposing data.
