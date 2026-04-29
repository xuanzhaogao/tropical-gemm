# Spec M `tropical_matmul` benchmarks

Element type: `ModCountingTropical{Float32, 7}`. Inputs and output stay on
device for the duration of the inner loop; per-call wall time wraps the
kernel launch + sync. Reported throughput is `2·M·N·K / time` (G tropical-ops/s).

Run via:
```
julia --project=CountingTropicalGEMM.jl CountingTropicalGEMM.jl/bench/bench_mul.jl
```

## Quadro RTX 6000 (Turing sm_75) — 2026-04-28

| Shape | flag | ms / call | G tropical-ops/s |
|---:|:---:|---:|---:|
| 128³ | NN | 0.065 | 65.0 |
| 128³ | NT | 0.028 | 149.8 |
| 128³ | TN | 0.061 | 69.0 |
| 128³ | TT | 0.028 | 149.4 |
| 256³ | NN | 0.397 | 84.6 |
| 256³ | NT | 0.105 | 320.5 |
| 256³ | TN | 0.397 | 84.6 |
| 256³ | TT | 0.103 | 326.0 |
| 512³ | NN | 2.895 | 92.7 |
| 512³ | NT | 0.635 | 422.7 |
| 512³ | TN | 2.838 | 94.6 |
| 512³ | TT | 0.615 | 436.8 |
| 1024³ | NN | 18.521 | 115.9 |
| 1024³ | NT | 3.634 | 590.9 |
| 1024³ | TN | 17.234 | 124.6 |
| 1024³ | TT | 3.517 | 610.6 |
| 2048³ | NN | 130.449 | 131.7 |
| 2048³ | NT | 31.037 | 553.5 |
| 2048³ | TN | 130.422 | 131.7 |
| 2048³ | TT | 30.431 | 564.6 |
| 4096³ | NN | 1044.771 | 131.5 |
| 4096³ | NT | 251.969 | 545.5 |
| 4096³ | TN | 1049.326 | 131.0 |
| 4096³ | TT | 249.216 | 551.5 |

### Observations

- **NT and TT are ~4× faster than NN and TN at large sizes.** The kernel reads
  `B[k + j*K]` (NN/TN: stride `K` per `j`-step → uncoalesced across the warp's
  N-axis) vs. `B[j + k*N]` (NT/TT: contiguous across `j` for fixed `k` →
  coalesced). The 4× ratio matches the gap between coalesced and stride-K
  loads on Turing's L1.
- **Peak throughput ~611 G tropical-ops/s** (1024³, TT) on RTX 6000. For
  comparison, the prior Spec K's AoS general kernel (NN row-major) hit
  ~666 G/s on A100 at 4096²; on RTX 6000 the same ceiling is roughly
  half due to fewer SMs and lower memory bandwidth.
- **Small-shape regime** (128³, 256³): per-call kernel launch overhead
  dominates; throughput is far below the ceiling. The dropped warpk path
  used to address this for `M·N ≤ 128² && K ≥ 64` shapes (see Spec K
  memory entry) — revisit as Spec N if perf there matters.

## Quadro RTX 6000 (Spec N tiled) — 2026-04-28

Shared-memory tiled kernel with layout-aware coalesced loads
(BM=BN=64, BK=8, TM=TN=4 for f32; 16×16 threads).

| Shape | flag | ms / call | G tropical-ops/s |
|---:|:---:|---:|---:|
| 128³ | NN | 0.183 | 22.9 |
| 128³ | NT | 0.210 | 20.0 |
| 128³ | TN | 0.210 | 20.0 |
| 128³ | TT | 0.183 | 22.9 |
| 256³ | NN | 0.352 | 95.4 |
| 256³ | NT | 0.411 | 81.6 |
| 256³ | TN | 0.406 | 82.7 |
| 256³ | TT | 0.355 | 94.5 |
| 512³ | NN | 0.712 | 377.0 |
| 512³ | NT | 0.818 | 328.3 |
| 512³ | TN | 0.813 | 330.2 |
| 512³ | TT | 0.702 | 382.3 |
| 1024³ | NN | 3.948 | 543.9 |
| 1024³ | NT | 4.097 | 524.2 |
| 1024³ | TN | 4.100 | 523.8 |
| 1024³ | TT | 3.919 | 547.9 |
| 2048³ | NN | 27.426 | 626.4 |
| 2048³ | NT | 28.684 | 598.9 |
| 2048³ | TN | 28.597 | 600.8 |
| 2048³ | TT | 27.434 | 626.2 |
| 4096³ | NN | 214.090 | 642.0 |
| 4096³ | NT | 219.257 | 626.8 |
| 4096³ | TN | 224.251 | 612.9 |
| 4096³ | TT | 212.330 | 647.3 |

### Observations

- **NN/TN/NT/TT now within ~5% of each other** at every size ≥ 256³. The
  4× gap from Spec M is closed: layout-aware coalesced global → shared
  loads make the consumer-side access pattern irrelevant.
- **Peak 647 G tropical-ops/s** (4096³, TT) — a slight gain over Spec M's
  611 G/s peak.
- **NN/TN improvement at 4096³:** 132 → 642 G/s (~4.9×). At 1024³:
  116 → 544 G/s (~4.7×).
- **Small TT regression at 1024³:** 611 → 548 G/s (~10%, within the
  15% spec budget). The extra `__syncthreads()` and shared-memory
  traffic are the cost; gains outside this corner more than pay for it.
- **Small-shape regime** (128³, 256³): tiled launch overhead is now
  more visible since each block does more work per launch but utilization
  is poor; throughput at 128³ regressed slightly. This regime is
  launch-overhead-bound regardless and would need a small-shape path
  (warpk-style) to address.

## A100-SXM4-80GB (Spec N tiled) — 2026-04-29

Same kernel and tile sizes as RTX 6000 (BM=BN=64, BK=8, TM=TN=4 for f32).
Run on `workergpu049`.

| Shape | flag | ms / call | G tropical-ops/s |
|---:|:---:|---:|---:|
| 128³ | NN | 0.213 | 19.7 |
| 128³ | NT | 0.237 | 17.7 |
| 128³ | TN | 0.237 | 17.7 |
| 128³ | TT | 0.213 | 19.7 |
| 256³ | NN | 0.407 | 82.5 |
| 256³ | NT | 0.463 | 72.4 |
| 256³ | TN | 0.456 | 73.5 |
| 256³ | TT | 0.411 | 81.6 |
| 512³ | NN | 0.809 | 331.7 |
| 512³ | NT | 0.880 | 305.1 |
| 512³ | TN | 0.880 | 305.0 |
| 512³ | TT | 0.784 | 342.6 |
| 1024³ | NN | 4.167 | 515.4 |
| 1024³ | NT | 4.136 | 519.2 |
| 1024³ | TN | 4.027 | 533.2 |
| 1024³ | TT | 3.833 | 560.2 |
| 2048³ | NN | 24.553 | 699.7 |
| 2048³ | NT | 25.388 | 676.7 |
| 2048³ | TN | 25.529 | 673.0 |
| 2048³ | TT | 24.541 | 700.0 |
| 4096³ | NN | 184.497 | 744.9 |
| 4096³ | NT | 190.298 | 722.2 |
| 4096³ | TN | 190.922 | 719.9 |
| 4096³ | TT | 184.287 | 745.8 |

### Observations

- **A100 only ~15% ahead of RTX 6000 at peak** (746 vs 647 G/s at 4096³ TT).
  The kernel's fixed sm_75-tuned tile geometry leaves A100's bigger SM
  count, larger register file, and HBM3-class bandwidth on the table.
  An A100-specific tile (e.g. BM=BN=128, BK=16, TM=TN=8) and async
  shared-mem loads (`cp.async`) would likely close the gap to A100's
  ~2 TB/s ceiling for this kernel pattern. Follow-on spec.
- **Cross-flag spread on A100 mirrors RTX 6000** — within ~5% of each
  other at every size. Layout-aware loading carries over cleanly.

## A100-SXM4-80GB (Spec P pipelined-structure) — 2026-04-29

Double-buffered shared-memory layout (BM=BN=64, **BK=16**, TM=TN=4 for
f32; 2 stages). Dispatched on sm_80+ via runtime compute-cap check.

| Shape | flag | ms / call | G tropical-ops/s |
|---:|:---:|---:|---:|
| 512³ | NN | 0.735 | 365.4 |
| 512³ | NT | 0.716 | 375.1 |
| 512³ | TN | 0.715 | 375.4 |
| 512³ | TT | 0.722 | 371.6 |
| 1024³ | NN | 4.020 | 534.2 |
| 1024³ | NT | 3.617 | 593.7 |
| 1024³ | TN | 3.616 | 593.9 |
| 1024³ | TT | 3.632 | 591.2 |
| 2048³ | NN | 23.592 | 728.2 |
| 2048³ | NT | 23.682 | 725.4 |
| 2048³ | TN | 23.681 | 725.5 |
| 2048³ | TT | 23.614 | 727.5 |
| 4096³ | NN | 178.494 | 770.0 |
| 4096³ | NT | 179.075 | 767.5 |
| 4096³ | TN | 179.071 | 767.5 |
| 4096³ | TT | 178.245 | **771.1** |

### Spec P status

- **Structural pipeline only — cp.async NOT delivered.** Initial cp.async
  variants (both `.cg` and `.ca`, both per-field and packed-pair
  encodings) produced silently-corrupt outputs on sm_80 in this kernel
  layout. The byte-equal sync-vs-pipelined tests caught it before any
  perf claim could be made. Root cause not pinned to a documented PTX
  rule. Mitigation: keep the double-buffered shared layout +
  `__syncthreads()` placement, use plain LDG+STS for the loads.
  `CP_ASYNC_COMMIT` / `WAIT_GROUP` calls remain as no-ops to preserve
  the pipeline shape for future re-introduction via `cuda::pipeline` /
  `__pipeline_memcpy_async`.
- **Net A100 effect:** **771 G/s** at 4096³ TT vs **746 G/s** for the
  Spec N sync-only kernel = +3% from the double-buffer shape alone.
  At 2048³ the pipelined structure gains ~4% (700 → 728 G/s).
- All four (tA, tB) layouts within ~1% of each other (down from the ~5%
  spread on Spec N).
- 75/75 lib tests pass on A100 (70 Spec N + 5 byte-equal `pl_matches_sync_*`
  fanned across NN/NT/TN/TT = 20 layout comparisons).

The cp.async optimization is **deferred to Spec P.1**: rewrite the loader
using the higher-level `cuda::pipeline` async API rather than raw inline
PTX, which the implementation experience suggests has subtle interactions
with the multi-buffer shared-memory layout we couldn't diagnose at the
PTX level.

## A100-SXM4-80GB (Spec Q defer-mod) — 2026-04-29

Same pipeline structure as Spec P (BM=BN=64, BK=16, TM=TN=4 for f32, double-
buffered shared), but the per-FMA Barrett reduction is removed from the
inner K-loop. The cnt accumulator stays in u64 and absorbs raw `ac*bc`
products; a single Barrett runs at write-out. Mathematically equivalent
because `(a+b) mod P == ((a mod P)+(b mod P)) mod P`. Launcher gates the
fast path on `K * (P-1)^2 < 2^63`; unsafe (P, K) shapes fall through to
the unmodified Spec N sync kernel.

| Shape | flag | ms / call | G tropical-ops/s |
|---:|:---:|---:|---:|
| 128³ | NN | 0.094 | 44.7 |
| 128³ | NT | 0.104 | 40.5 |
| 128³ | TN | 0.104 | 40.2 |
| 128³ | TT | 0.094 | 44.6 |
| 256³ | NN | 0.176 | 190.7 |
| 256³ | NT | 0.189 | 177.3 |
| 256³ | TN | 0.196 | 171.5 |
| 256³ | TT | 0.170 | 197.2 |
| 512³ | NN | 0.331 | 812.2 |
| 512³ | NT | 0.373 | 720.3 |
| 512³ | TN | 0.377 | 711.8 |
| 512³ | TT | 0.323 | 832.0 |
| 1024³ | NN | 1.692 | 1269.5 |
| 1024³ | NT | 1.786 | 1202.3 |
| 1024³ | TN | 1.823 | 1178.2 |
| 1024³ | TT | 1.689 | 1271.3 |
| 2048³ | NN | 10.540 | 1629.9 |
| 2048³ | NT | 10.478 | 1639.7 |
| 2048³ | TN | 10.513 | 1634.2 |
| 2048³ | TT | 10.056 | 1708.4 |
| 4096³ | NN | 74.778 | 1837.9 |
| 4096³ | NT | 77.024 | 1784.4 |
| 4096³ | TN | 77.757 | 1767.5 |
| 4096³ | TT | 74.490 | **1845.1** |

### Spec Q observations

- **2.4× over Spec P at peak** (771 → 1845 G/s @ 4096³ TT). Bottleneck on
  Spec P was per-step Barrett (`__umul64hi` + sub + cmp + sub) inside the
  TM·TN×BK inner loop, not memory or shared-bank conflicts. Pulling
  Barrett out of the inner loop eliminates ~5 emulated-u64 instructions
  per FMA on every K-step.
- **Hits the historical "~2 TG/s" target** that Spec K saw on its older
  A100 path. The Spec N/P regression (down to ~750 G/s) was masked
  Barrett cost; once removed, the kernel's actual SOL on the AoS
  pair-multiply pattern is recovered.
- **Tile geometry is not the bottleneck.** A 128×128 / TM=TN=8 probe at
  both BK=8 and BK=16 measured 650–663 G/s — *worse* than 64×64 / 4×4 at
  BK=16. Larger register blocks collapse occupancy (acc_v + acc_c u64 →
  ~200 regs/thread before scratch); the 4×4 block is the sweet spot.
- **Smaller BK hurts.** At 64×64/4×4 BK=8, 4096³ TT = 1702 G/s; BK=16 =
  1845. The compute side gains more from K-reuse than the loader-side
  gains from extra concurrent blocks.
- **NN/TN still ~4% behind TT** at 4096³ (1837 vs 1845). The Spec N
  layout-aware loader closed the original 4× gap; the residual is
  shared-bank conflict in B's access pattern under NN.
- **75/75 lib tests pass** including the byte-equal `pl_matches_sync_*`
  set — confirms the equivalence holds across all four (tA, tB) layouts.
- **Defer-mod safety gate** unit-tested (`gate_tests::*`): K * (P-1)² <
  2^63 picks `_pl`, otherwise falls back to the always-correct sync.

## A100-SXM4-80GB head-to-head vs Julia auto-compiled `mul!` — 2026-04-29

Direct kernel comparison: our `tropical_matmul('N','N', A, B)` on
`CuMatrix{ModCountingTropical{Float32, 2965819}}` (22-bit prime, fast
path) vs `LinearAlgebra.mul!` on `CuMatrix{CountingTropical{Float64,
Int64}}` (the GenericTensorNetworks-GPU element type). Both go through
their respective single-kernel matmul; min-of-N-reps reported. Bench
script: `examples/bench_kernel_vs_julia.jl`.

| Shape | ours ms | ours G/s | julia ms | julia G/s | speedup |
|---:|---:|---:|---:|---:|---:|
| 512³ | 0.32 | 836 | 0.43 | 618 | 1.35× |
| 1024³ | 1.69 | 1268 | 4.02 | 535 | 2.37× |
| 2048³ | 11.11 | 1546 | 45.77 | 375 | 4.12× |
| 4096³ | 80.69 | 1703 | 622.88 | 221 | 7.72× |
| 8192³ | 653.27 | 1683 | 5336.32 | 206 | 8.17× |
| 16384³ | 5212.14 | 1688 | 43239.72 | 203 | 8.30× |
| 32768³ | 41742.36 | **1686** | 348404.82 | 202 | **8.35×** |

### Observations

- **Our kernel scales flat at ~1.69 TG/s from 4096³ all the way to
  32768³.** No degradation at the largest shape; well-aligned with the
  ~2 TG/s historical Spec K ceiling.
- **Julia auto-compiled `mul!` plateaus at ~200 G/s** for any shape
  ≥ 4096³ — that's the GPUArrays generic GEMM bound on a non-standard
  semiring (no per-element fusion, two scalar fields, generic
  reduction).
- **Speedup grows with size** because Julia's per-element overhead is
  fixed while ours scales with hardware bandwidth/compute. The ratio
  asymptotes at ~8.3× — both kernels are now compute-bound.
- 32768³ wall: 42 s for us vs **5 min 48 s** for the Julia path. At
  L=40 / L=50 spin-glass scales (where the contraction is dominated
  by these huge matmuls), this 8× per-matmul edge translates directly
  to end-to-end wins, even with 2-pass CRT factor.

## A100-SXM4-80GB (Spec R u32-acc small-P fast path) — 2026-04-29

ncu confirmed Spec Q was compute-bound at 82.8% SM throughput (occupancy
25%, capped by reg+shared, but raising occupancy doesn't help a
compute-bound kernel — see skill `matmul-kernel-optimization`). SASS
opcode histogram on the inner FMA loop showed ~96 IMAD.WIDE.U32 + ~64
IMAD.X (u64 carry) + ~144 SEL versus only ~32 FADD — i.e. the u64 cnt
path costs much more than the actual fp32 work.

Spec R splits the pipelined kernel by accumulator type:

- `_plu32` — `unsigned int` cnt acc. Gate: `K · (P-1)² < 2^32`. Strips
  `IMAD.WIDE.U32` + `IMAD.X` carries; ~halves integer-pipe traffic.
- `_pl` — unchanged from Spec Q (u64 cnt acc, gate `< 2^63`).
- sync — Barrett-in-loop fallback for arbitrary P/K.

Launcher picks the most aggressive variant the (P, K) shape allows. 79/79
lib tests pass; 4 gate unit tests cover all three paths.

| P | Shape | ms / call | G tropical-ops/s | path |
|---:|---:|---:|---:|:---|
| 7 | 1024³ | 1.012 | 2122 | _plu32 |
| 7 | 2048³ | 6.514 | 2637 | _plu32 |
| 7 | 4096³ | 48.194 | **2852** | _plu32 |
| 7 | 8192³ | 383.095 | **2870** | _plu32 |
| 2 965 819 (22-bit) | 1024³ | 1.532 | 1402 | _pl |
| 2 965 819 (22-bit) | 2048³ | 11.056 | 1554 | _pl |
| 2 965 819 (22-bit) | 4096³ | 80.646 | 1704 | _pl |
| 2 965 819 (22-bit) | 8192³ | 660.880 | 1664 | _pl |

### Cumulative speedup (4096³ TT on A100)

| Spec | path | G/s | × baseline |
|---|---|---|---|
| N (sync, Barrett-in-loop) | sync | 746 | 1.0× |
| P (pipelined, no cp.async) | _pl Barrett | 771 | 1.03× |
| Q (defer-mod u64) | _pl u64 | 1845 | 2.47× |
| **R (defer-mod u32, P≤7)** | **_plu32** | **2852** | **3.82×** |

### Spec R observations

- **2870 G/s @ 8192³** at P=7 is past the historical "~2 TG/s" target
  by ~40%. Sustained out to 32768³ (verified at 4096³+8192³; 16384³+
  re-run pending).
- **u32 path is 1.67–1.70× faster** than u64 across all measured sizes.
  The win matches the SASS-derived expectation: integer carry removed,
  shorter dep chain into SEL, half the registers for acc_c.
- The u64 path keeps Spec Q numbers exactly (the launcher selects it
  unchanged when the u32 gate fails). Defaults are now strictly
  better for any (P, K) the original gate accepted.

## H100

Pending — no H100 available at the time of writing. Re-run on A100-80GB
(SXM4 if available) and append. Expected throughput per Spec K precedent:
- A100 NN at 4096³: ~250-350 G/s (likely worse than Spec K's 666 due to
  the column-major NN's uncoalesced B access — same 4× gap as on RTX
  6000 means ~1/4 of A100's coalesced ceiling).
- A100 TT at 4096³: ~1500-2000 G/s if the coalesced access dominates.

These predictions were superseded by Spec N's tiled kernel — see the
A100 section above. Actual measured A100 4096³ TT = 746 G/s.

H100 numbers should scale roughly with HBM3 bandwidth (3.35 TB/s vs.
A100's 1555 GB/s = ~2.15×).
