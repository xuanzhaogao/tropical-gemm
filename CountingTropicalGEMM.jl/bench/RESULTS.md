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

## A100 / H100

Pending — no A100 free at the time of writing. Re-run on A100-80GB
(SXM4 if available) and append. Expected throughput per Spec K precedent:
- A100 NN at 4096³: ~250-350 G/s (likely worse than Spec K's 666 due to
  the column-major NN's uncoalesced B access — same 4× gap as on RTX
  6000 means ~1/4 of A100's coalesced ceiling).
- A100 TT at 4096³: ~1500-2000 G/s if the coalesced access dominates.

H100 numbers should scale roughly with HBM3 bandwidth (3.35 TB/s vs.
A100's 1555 GB/s = ~2.15×).
