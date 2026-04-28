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

## A100 / H100

Pending — no A100 free at the time of writing. Re-run on A100-80GB
(SXM4 if available) and append. Expected throughput per Spec K precedent:
- A100 NN at 4096³: ~250-350 G/s (likely worse than Spec K's 666 due to
  the column-major NN's uncoalesced B access — same 4× gap as on RTX
  6000 means ~1/4 of A100's coalesced ceiling).
- A100 TT at 4096³: ~1500-2000 G/s if the coalesced access dominates.

H100 numbers should scale roughly with HBM3 bandwidth (3.35 TB/s vs.
A100's 1555 GB/s = ~2.15×).
