# Spec E — Warp-K-reduction counting kernel + AoS pair loads

**Date:** 2026-04-27
**Status:** design approved
**Branch:** `counting-tropical`
**Depends on:** spec C (`2026-04-21-counting-tropical-cuda-design.md`), spec D (`2026-04-21-counting-kernel-tiled-design.md`).
**Scope:** add a second counting-kernel variant that splits the K reduction across a warp; then layer AoS `(value, count)` packing on top to halve global loads. Existing naive kernel kept as the fallback / small-K path.

## Goal

Lift counting-kernel throughput on A100+ above the current ~600 G tropical-ops/s ceiling toward the MaxPlus reference (~1500 G-ops/s), using two stacked changes:

- **Stage A — warp-K-reduction:** 32 threads cooperate on each output cell, splitting K. Final warp shuffle reduces partial `(val, cnt)` pairs.
- **Stage B — AoS `int2` element layout:** pack `(value_bits, count)` per element, single 8-byte load per element instead of two 4-byte loads.

Both stages stack multiplicatively in the inner loop. Targeted regime: medium-to-large M·N where L2 is already absorbing reuse on the naive kernel and inner-loop instructions/loads dominate.

## Non-goals

- Shared-memory tiling. Spec D measured this as a loss; not revisited here.
- Autotuning / per-shape dispatch. First cut uses a simple K threshold heuristic.
- Host-level chunking for problems exceeding device memory. Separate work.
- NVRTC per-prime templating. Independent optimization, not in this spec.

## Stage A — Warp-K-reduction kernel

### Parallelization shape

- **One warp (32 threads) per output cell (i, j).** Each thread owns K-stride positions `k = lane, lane+32, lane+64, …`.
- **Block:** 4 warps = 128 threads = 4 output cells per block. Block dim `(32, 4)`: `threadIdx.x` is the lane (0..31), `threadIdx.y` selects which of the 4 cells the warp computes.
- **Grid:** `dim3(ceil(N / 1), ceil(M / 4), 1)` — y-direction packs 4 rows of cells per block, x-direction one column per block. (Concretely: `gridDim = (N, ceil_div(M, 4), 1)`, `blockIdx.x = j`, `(blockIdx.y * 4 + threadIdx.y) = i`.)

### Per-thread inner loop

```c
int lane = threadIdx.x;        // 0..31
int i    = blockIdx.y * 4 + threadIdx.y;
int j    = blockIdx.x;
if (i >= M) return;            // tail rows

T   acc_val = INIT_VAL;
u64 acc_cnt = 0;

for (int k = lane; k < K; k += 32) {
    T   va = value_a[i*K + k];
    u32 ca = (u32)count_a[i*K + k];
    T   vb = value_b[k*N + j];
    u32 cb = (u32)count_b[k*N + j];
    T   pv = va + vb;
    u64 pc = barrett_mod((u64)ca * cb, P, MU);
    bool win = BETTER(pv, acc_val);
    bool tie = (pv == acc_val);
    acc_val  = win ? pv : acc_val;
    acc_cnt  = win ? pc : (tie ? barrett_mod(acc_cnt + pc, P, MU) : acc_cnt);
}
```

Note: count is reduced under modulus *during* the K-loop (not lazy) because partial `acc_cnt + pc` could otherwise overflow u64 after enough ties (each addend ≤ P < 2³¹, but with 2K K-stride steps and consistent ties, the sum could reach K·P ≈ 2⁶⁰ — still within u64, but only safely. Reducing each step keeps headroom and matches the existing naive kernel).

### Warp reduction

Five-step `__shfl_xor_sync` tree using the tropical-add operator:

```c
for (int off = 16; off > 0; off >>= 1) {
    T   ov_lo = __shfl_xor_sync(0xffffffff, acc_val, off);
    u32 oc_lo = __shfl_xor_sync(0xffffffff, (u32)(acc_cnt & 0xffffffff), off);
    u32 oc_hi = __shfl_xor_sync(0xffffffff, (u32)(acc_cnt >> 32),         off);
    u64 oc    = ((u64)oc_hi << 32) | oc_lo;

    bool win  = BETTER(ov_lo, acc_val);
    bool tie  = (ov_lo == acc_val);
    T   nv    = win ? ov_lo : acc_val;
    u64 nc    = win ? oc
              : tie ? barrett_mod(acc_cnt + oc, P, MU)
              : acc_cnt;
    acc_val = nv;
    acc_cnt = nc;
}
```

(`__shfl_xor_sync` is 32-bit; u64 is shuffled as two halves.)

Lane 0 writes:

```c
if (lane == 0) {
    value_c[i*N + j] = acc_val;
    count_c[i*N + j] = (int)barrett_mod(acc_cnt, P, MU);
}
```

### Boundary handling

- **M boundary:** `if (i >= M) return` at block entry (whole warp exits together — safe for `__shfl_xor_sync` because all participating lanes are guarded uniformly).
- **K boundary:** the K-stride loop naturally terminates with `k < K` predication; no tail loop needed.
- **N boundary:** grid is `gridDim.x = N` exactly, no over-launch.

### Memory access pattern

- **A:** at iteration `s`, lanes read `A[i, 32s + lane]` — 32 contiguous elements within a warp → **coalesced**.
- **B:** at iteration `s`, lanes read `B[32s + lane, j]` — same column `j`, 32 different rows, strided by N → **non-coalesced**. L2 absorbs reuse across warps that share `j`. Expect this to limit the speedup ceiling at very large N. Quantified by benchmark, not predicted.
- **Output:** lane 0 of each warp writes one `(value_c, count_c)` cell. Within a block, 4 warps write 4 cells contiguous in M for fixed j — non-ideal but only one write per cell.

### Dispatch

Add new kernel symbols:

| Symbol | Variant |
|---|---|
| `counting_gemm_f32_max_warpk` | new |
| `counting_gemm_f32_min_warpk` | new |
| `counting_gemm_f64_max_warpk` | new |
| `counting_gemm_f64_min_warpk` | new |

Existing naive kernels (`counting_gemm_f32_max`, …) remain unchanged.

Trait `CountingCudaKernel<T, D>` gets a second method `launch_counting_gemm_warpk` (or a runtime branch inside `launch_counting_gemm` keyed on K). First cut: simple inline branch in `launch_counting_gemm`:

```rust
if k >= 64 {
    launch warpk variant
} else {
    launch naive variant
}
```

K threshold of 64 is a conservative starting point: each thread does ≥2 K-stride iterations, enough to amortize the reduction. Tune after benchmarking.

### Testing

The existing `counting_gpu` integration tests already exercise correctness across sizes, primes, and directions. The dispatch change routes large-K cases to the new kernel; small-K cases stay on naive. Add three explicit tests forcing the warpk path:

1. `warpk_small_k_boundary` — K=32, K=33, K=64 (boundary of the heuristic and of the warp stride).
2. `warpk_non_aligned` — M=37, K=131, N=29 to hit all three boundary paths simultaneously.
3. `warpk_matches_naive` — for one fixed (M, K, N), run both variants directly and assert byte-equal output. Forces the comparison even if dispatch logic changes later.

## Stage B — AoS `(value, count)` element layout

### Element type

Define a device-side compound type for each `T`:

```c
struct __align__(8)  PairF32 { float  val; int cnt; };  // 8 bytes
struct __align__(16) PairF64 { double val; long long cnt; };  // 16 bytes
```

The cnt field stays as a signed integer mod P (i32 for f32-paired pair; i64 for f64-paired pair to hold u64-equivalent counts when paired with f64 values, kept consistent with current i32 count buffers — TBD: keep i32 for f64 too, since CRT primes fit in i31).

**Decision:** keep cnt as `i32` for both f32 and f64. Reasoning: counts come from CRT_PRIMES (all < 2³¹), match existing buffer types, and the f64 pair becomes an awkward 12-byte aligned-16 layout — pad with explicit `int _pad` to 16 bytes for f64. Cost: 4 wasted bytes per element on f64. Acceptable for now.

```c
struct __align__(8)  PairF32 { float  val; int cnt; };          //  8 B
struct __align__(16) PairF64 { double val; int cnt; int _pad; };// 16 B
```

### Buffer layout change

`count_ground_states_gpu` currently uploads four buffers: `value_a, count_a, value_b, count_b`. With AoS:

- Host-side: pack `(value_a[i], count_a[i])` into `Vec<PairT>` once before upload. Counts are all-ones for the entry point; the packing is `vec.iter().map(|&v| Pair { val: v, cnt: 1 }).collect()`. (One-shot; not in the per-prime hot loop.)
- Device-side: kernel receives 2 buffers (`pair_a`, `pair_b`) instead of 4, plus 2 output buffers (`pair_c` or kept SoA — see below).

**Output layout decision:** keep output SoA (`value_c: T*`, `count_c: int*`). Reason: callers consume them separately (CRT keeps residue streams as `Vec<i32>`, value field is downloaded once). Packing the output would force an unpack on host, no benefit.

So Stage B kernel signature:

```c
counting_gemm_f32_max_warpk_aos(
    const PairF32* __restrict__ pair_a,
    const PairF32* __restrict__ pair_b,
    float* __restrict__ value_c,
    int*   __restrict__ count_c,
    int M, int N, int K, int P, u64 MU
);
```

### Inner-loop change

```c
PairF32 a = pair_a[i*K + k];
PairF32 b = pair_b[k*N + j];
T   pv = a.val + b.val;
u64 pc = barrett_mod((u64)(u32)a.cnt * (u32)b.cnt, P, MU);
// rest identical
```

Two LDG.E.64 (one per pair) replace four LDG.E.32. ~50% fewer memory instructions in the inner loop.

### Rust-side wiring

- Add `cudarc::driver::DeviceRepr` impl for `PairF32`, `PairF64` (manually, since they're plain repr-C structs).
- Extend `GpuMatrix<T>` callers to either (a) hold a `GpuMatrix<PairF32>` directly, or (b) keep separate buffers but kernel-side reinterpret the pointers — easier interim is (a).
- New trait method (or trait variant) `launch_counting_gemm_aos` taking the packed buffers.
- `count_ground_states_gpu` driver: pack once before the prime loop (counts are all-ones, so the pack is a `vec![Pair{val, cnt: 1}; mn]`-style operation per matrix, done once, reused across primes).

### Dispatch (after Stage B)

Update the heuristic to: AoS-warpk for K ≥ 64, naive otherwise. The non-AoS warpk variant from Stage A becomes redundant once B is measured to win — keep it temporarily for A/B comparison, then remove.

## Roll-out plan

1. Land Stage A: warpk kernel + dispatch + tests + bench. Measure on A100.
2. If A is a clear win (e.g., ≥1.3× kernel-only at 4096²), proceed to B.
3. Land Stage B: AoS layout + new kernel variant + dispatch update. Measure.
4. Update `project_counting_status.md` memory with new numbers; close out the spec.

If A is *not* a win on A100 (e.g., L2 traffic from non-coalesced B reads outweighs the K-parallelism gain), pause and reassess before B.

## Risks

- **Non-coalesced B reads.** Acknowledged in the design. Mitigation: rely on A100's L2 (~40 MB) to absorb. If it dominates, a fallback is to flip the warp's role (K-stride along A's row instead, using transposed B) — out of scope for this spec.
- **Reduction operator under shuffle.** Tropical-add with ties is associative and commutative; the tree reduce is correct as long as the operator is implemented identically at each step. The test `warpk_matches_naive` enforces this against the existing kernel.
- **u64 split shuffle.** `__shfl_xor_sync` is 32-bit only. Splitting `acc_cnt` into hi/lo halves and recombining is straightforward but adds two extra shuffles per step (10 shuffles total instead of 5 for u32). Cost is small; flagged here so a reader doesn't see it as a bug.
- **i32 cnt for f64 pair (Stage B padding).** 4 wasted bytes per f64 element. Acceptable; flagged for future reconsideration if f64 throughput becomes a priority.
