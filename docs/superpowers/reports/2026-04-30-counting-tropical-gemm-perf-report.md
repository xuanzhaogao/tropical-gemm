# Counting Tropical GEMM Performance Report

**Date:** 2026-04-30
**Hardware:** A100-SXM4-80GB (workergpu049), CUDA 12.5
**Branch:** `counting-tropical`
**Element type:** `ModCountingTropical{Float32, P}` (8-byte AoS pair: f32 value + i32 count mod P)

## Headline numbers

| Metric | Result |
|---|---|
| Peak kernel throughput @ 4096³ (P=7) | **2 852 G tropical-ops/s** |
| Sustained throughput @ 8192³ → 32768³ | **~1 690–2 870 G/s** (flat) |
| vs Julia auto-compiled `mul!` @ 32768³ | **8.35×** |
| L=25 spin-glass count (2³ primes, A100) | **1.14 s** |
| L=25 vs GenericTensorNetworks GPU path | **38.85×** |

## Speedup ladder on the kernel itself (4096³, TT, A100)

| Spec | Path | G/s | × Spec N |
|---|---|---:|---:|
| N (sync, Barrett-in-loop) | sync | 746 | 1.00× |
| P (pipelined-structure, no cp.async) | `_pl` Barrett | 771 | 1.03× |
| Q (defer-mod, u64 cnt acc) | `_pl` u64 | 1845 | 2.47× |
| **R (defer-mod, u32 cnt acc, P≤7)** | **`_plu32`** | **2852** | **3.82×** |

Spec Q removed the per-step Barrett reduction by deferring `mod P` to
the epilogue (algebraically equivalent because
`(a + b) mod P = ((a mod P) + (b mod P)) mod P`). Spec R noticed via
SASS + ncu that the remaining bottleneck was the u64 carry chain
(`IMAD.WIDE.U32 + IMAD.X + IADD3` per FMA) and switched the cnt
accumulator to u32 for small-P workloads.

## Direct kernel head-to-head vs Julia auto-compiled `mul!`

`CuMatrix{ModCountingTropical{Float32, 2965819}}` vs
`CuMatrix{CountingTropical{Float64, Int64}}` (the GTN-style element
type), `mul!` on each path, min-of-N reps:

| Shape | ours ms | ours G/s | julia ms | julia G/s | speedup |
|---:|---:|---:|---:|---:|---:|
| 512³ | 0.32 | 836 | 0.43 | 618 | 1.35× |
| 1024³ | 1.69 | 1268 | 4.02 | 535 | 2.37× |
| 2048³ | 11.11 | 1546 | 45.77 | 375 | 4.12× |
| 4096³ | 80.69 | 1703 | 622.88 | 221 | 7.72× |
| 8192³ | 653.27 | 1683 | 5336.32 | 206 | 8.17× |
| 16384³ | 5212.14 | 1688 | 43239.72 | 203 | 8.30× |
| 32768³ | 41742.36 | **1686** | 348404.82 | 202 | **8.35×** |

Note: the head-to-head uses the 22-bit prime path (`_pl` u64). At P≤7
the `_plu32` path runs at ~2 850 G/s instead of ~1 700, pushing the
ratio at large shapes well past 10×.

Julia's auto-compiled path plateaus at ~200 G/s — that's the
GPUArrays generic GEMM bound on a non-standard semiring (no fusion,
two scalar fields, generic reduction). Our kernel scales flat from
4096³ to 32768³ at ~1.7 TG/s (u64 path) or ~2.85 TG/s (u32 path).

## Roofline analysis (A100, 4096³)

- HBM ceiling: 4 MiB/block × 4096 blocks = 16 GiB / 1.5 TB/s ≈ 10.6 ms
  → memory-only bound ≈ **13 TG/s**.
- We measure 80–48 ms wall depending on path → **memory utilization
  6–14%, not memory-bound**.
- Compute side: A100 has 4 separate pipes (FP32 / INT32 / SFU / Tensor)
  per SM. With separate-pipe ILP the realistic CUDA-core ceiling for
  this AoS pair-multiply pattern is ~3 TG/s. Spec R hits 2.85 TG/s ≈
  **95% of the realistic compute ceiling**.

`ncu` confirms Spec Q at 82.8% SM throughput, occupancy 25%, limited
by both registers (96–128/thread) and shared memory (33 KiB/block).
Per the matmul-kernel-optimization skill: occupancy is not the lever
on a compute-bound kernel — instruction count per FMA is.

## End-to-end: spin glass with 2-prime / 3-prime CRT

Tensor network contraction via OMEinsum + custom `mul!` overload on
`CuMatrix{ModCountingTropical{Float32, P}}`. Each pairwise contraction
becomes a `tropical_matmul` call.

Auto prime picker:
1. `max_contraction_k(optcode, sizes)` walks the OMEinsum
   `DynamicNestedEinsum` tree, returns the largest reduction-K seen
   across all binary contraction steps.
2. `pick_fast_crt_primes(k_max; target_envelope_log2)` picks the
   largest distinct primes such that `K · (P-1)² < 2^32` (Spec R u32
   gate, faster) or `< 2^63` (Spec Q u64 gate). Greedily takes primes
   until `prod(primes) ≥ 2^target_envelope_log2`.

| L | grid spins | k_max | primes (gate) | kernel total | GTN-GPU | speedup |
|---:|---:|---:|---|---:|---:|---:|
| 20 | 400 | 65 536 (2¹⁶) | 3 × 23-bit (`:u64`) | 1.05 s | 0.96 s | 0.94× |
| **25** | **625** | **32 768 (2¹⁵)** | **3 × 24-bit (`:u64`)** | **1.14 s** | **44.23 s** | **38.85×** |

Both runs reconstruct the exact ground-state count and match GTN-GPU
bit-for-bit.

For L=20 we slightly lose because the OMEinsum tree contains many
small matmuls where our per-matmul edge is only 1.4–2.4×; with 3 CRT
passes vs GTN's 1, we don't fully amortize. For L=25 the matmuls are
much larger (M up to 2³² elements per binary step), and the per-matmul
edge hits 8×, swamping the 3-pass overhead.

L=25 ground-state count (recovered exactly):
**621 237 554 380 800 000** ≈ 2⁵⁹ ≈ 6.2 × 10¹⁷.

## Bug fixes that enabled L=25

OMEinsum's binary contraction reshapes a rank-25 tensor into matmul
shapes like `M = 2²³, N = 2, K = 2`. Two CUDA grid-shape limits
tripped here:

1. Original launcher put M on `gridDim.y` (capped at 65 535).
   M-blocks = 2²³ / 64 = 131 072 → `INVALID_VALUE` at launch.
   **Fix:** swap to put M on `gridDim.x` (cap 2³¹).

2. Even after the swap, lopsided steps with `M = 2², N = 2²³` put
   N-blocks = 65 536 on `gridDim.y` — exactly one over the limit.
   **Fix:** linearize N-blocks across `gridDim.y × gridDim.z`
   (each cap 65 535 → product 4.3 × 10⁹). Kernel does
   `n_blk = blockIdx.y + blockIdx.z·gridDim.y` and early-returns
   for over-launched blocks when `n_blocks` isn't a multiple of
   `gridDim.y`.

Both fixes are general — useful for any caller passing rank-N tensor
reshapes through this kernel.

## Reproducing

```bash
# Build
cd /path/to/tropical-gemm
cargo build --release -p tropical-gemm-cuda

# Library tests (79 should pass)
cargo test --release -p tropical-gemm-cuda --lib

# Head-to-head kernel bench (ours vs Julia auto)
JULIA_CUDA_USE_COMPAT=false julia --project=examples \
  examples/bench_kernel_vs_julia.jl

# Per-shape u32 vs u64 path comparison
JULIA_CUDA_USE_COMPAT=false julia --project=examples \
  examples/bench_u32_probe.jl

# L=25 spin-glass (auto picks primes from contraction tree)
JULIA_CUDA_USE_COMPAT=false julia --project=examples -e '
include("examples/spin_glass_demo.jl")
demo_grid_crt(25; ntrials=3, niters=15, target_envelope_log2=56)'
```

The Julia path requires CUDA.jl with `JULIA_CUDA_USE_COMPAT=false` on
this cluster (older driver vs CUDA.jl's bundled forward-compat
libcuda.so).

## Open follow-ups

- **Single-pass dual-prime kernel.** Each thread tracks two cnt
  accumulators (mod P₁ and mod P₂) per output element. One contraction
  pass yields both residues. Estimated ~2× over current 2-prime CRT
  for spin-glass-style workloads. Cost: ~doubles cnt-related register
  footprint, may require new kernel variants.
- **u128 acc for large-P fast path.** Extends the Spec Q u64 envelope
  to arbitrary P (including M31 primes) at the cost of one extra ADC
  per FMA. Useful only if a caller specifically needs M31-class CRT.
- **cp.async via `cuda::pipeline`.** Spec P.1 is still deferred. Now
  has less leverage since Spec Q+R already hit ~95% of the realistic
  CUDA-core compute ceiling — expected gain probably <10%.
- **RTX 6000 numbers.** Current session only had A100 access; the
  Spec Q+R deltas should re-measure on Turing for completeness.

## File-level summary

| Layer | File | Notable changes today |
|---|---|---|
| CUDA kernel | `crates/tropical-gemm-cuda/kernels/counting_gemm.cu` | Macro now parameterized by `ACC_T`; emits 16 `_pl` (u64) + 16 `_plu32` (u32) variants. Grid-axis swap and N-z-linearization for huge tensors. |
| Launcher | `crates/tropical-gemm-cuda/src/counting_kernel.rs` | Three-way variant pick: `_plu32` if `K·(P-1)² < 2^32`, `_pl` if `< 2^63`, sync otherwise. Grid `.x` is M-axis; N folds into `.y × .z`. |
| Context | `crates/tropical-gemm-cuda/src/context.rs` | Registered 16 `_plu32` kernel names. |
| Demo | `examples/spin_glass_demo.jl` | `max_contraction_k`, `pick_fast_crt_primes`, generic N-prime CRT loop, GTN-GPU reference comparison. |
| Bench | `examples/bench_kernel_vs_julia.jl` | Head-to-head against Julia auto-compiled `mul!`. |
| Bench | `examples/bench_u32_probe.jl` | u32 vs u64 dispatch comparison. |
| Bench results | `CountingTropicalGEMM.jl/bench/RESULTS.md` | Spec Q + Spec R sections + head-to-head table. |

## Commits landed in this session

```
2d6d3f5  Spin-glass demo: auto-pick CRT primes from contraction tree, plus grid fixes for huge tensors
839e726  Spec R: u32 cnt-acc small-P fast path — 2.85 TG/s @ 4096³ (3.8× baseline)
37d037b  bench: head-to-head ours vs Julia auto-mul!, plus fast-path CRT in demo
0e31aff  Spec Q: defer Barrett to write-out — 2.4× on A100 (771 → 1845 G/s)
```
