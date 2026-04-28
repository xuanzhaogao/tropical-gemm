# CountingTropicalGEMM.jl

Julia bindings for the counting-tropical CUDA matmul in `tropical-gemm-cuda`.
Computes per-cell `(value, count)` for max-plus / min-plus tropical matrix
multiplication on a CUDA GPU. Counts are returned as `UInt64`.

## Build

The package wraps a Rust `cdylib`. Build it once before `using` the package:

```bash
cd ../  # workspace root (one level up from this package)
cargo build --release -p tropical-gemm-cuda
# Produces target/release/libtropical_gemm_cuda.{so,dylib,dll}
```

The package's `__init__` looks for the library in this order:

1. `ENV["TROPICAL_GEMM_LIB"]` if set (absolute path to the shared library)
2. `<this-package>/../target/release/libtropical_gemm_cuda.<ext>`
3. Standard system library search paths

Set `ENV["TROPICAL_GEMM_LIB"]` if the package lives outside the workspace.

## Install (dev mode)

```julia
using Pkg
Pkg.develop(path="path/to/tropical-gemm/CountingTropicalGEMM.jl")
```

## Usage

```julia
using CountingTropicalGEMM

# A is M×K, B is K×N, both Matrix{Float32} or Matrix{Float64}.
# bound is the per-cell count upper bound (UInt64; must fit in u63).
A = rand(Float32, 64, 128)
B = rand(Float32, 128, 64)
res = count_ground_states_gpu_u64(Max, A, B, UInt64(128))

res.values  # 64×64 Matrix{Float32}: max-plus value at each cell
res.counts  # 64×64 Matrix{UInt64} : number of ground states per cell
```

Direction tags: `Max` or `Min`. Element types: `Float32` or `Float64`.

## Errors

- `BoundTooLargeError` — `count_upper_bound` exceeds the u64 envelope
  (would need ≥ 3 of the 30-bit CRT primes). For larger bounds, use the
  Rust `count_ground_states_gpu` (BigInt) entry point.
- `CountingTropicalGEMMError(code, msg)` — invalid input, CUDA failure,
  or internal error.
- `DimensionMismatch` — `A`'s `K` does not match `B`'s row count.

## Tests

```julia
using Pkg; Pkg.test("CountingTropicalGEMM")
```

Requires a CUDA GPU; runs the kernel and verifies parity against a
Julia reference implementation.
