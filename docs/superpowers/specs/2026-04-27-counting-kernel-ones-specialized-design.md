# Spec G — `ones`-specialized counting kernel

**Date:** 2026-04-27
**Status:** design (ready to implement)
**Branch:** `counting-tropical`
**Depends on:** specs C, E, F.
**Scope:** add a value-only counting kernel specialized for the all-ones input-counts case (the only entry point today via `count_ground_states_gpu`). Both naive and warpk variants. The general AoS kernels stay in place as the fallback for any future non-ones entry point.

## Goal

Cut the inner-loop work for the dominant production path. Today the AoS general kernel does on each k-step:

- 2× LDG.E.64 to read `(value, count)` pairs
- 1 fp add, 1 u32×u32→u64 mul, 1 Barrett mod, 3-way compare/update on a u64 accumulator

For the all-ones case (counts are uniformly `1`), the multiply, Barrett, and u64 accumulator are all wasted work. Output counts at cell (i, j) equal the number of k positions tied at the max — bounded by K. For K ≤ 2³², the count fits in `u32` and a single Barrett mod at the end of the loop suffices.

**Expected:** **1.6–2.1× kernel speedup** on the dominant 4096² square path (target ~1100–1400 G-ops/s, approaching the MaxPlus 1500 G-ops/s reference). Bigger relative gains in the warpk small-shape regime where the inner loop is more memory-bound.

## Non-goals

- Replacing the AoS general kernels. They remain for future callers passing non-trivial input counts (e.g., chained matmul, Spec E follow-on).
- Public API changes. `count_ground_states_gpu(a, m, k, b, n, …)` already takes value-only inputs; callers see the same signature.

## Design

### Kernel signature

Value-only inputs (SoA), output unchanged from existing kernels:

```c
extern "C" __global__ void counting_gemm_f32_max_ones(
    const float* __restrict__ value_a,
    const float* __restrict__ value_b,
    float*       __restrict__ value_c,
    int*         __restrict__ count_c,
    int M, int N, int K, int P, unsigned long long MU
);
```

Same for `_f32_min_ones`, `_f64_max_ones`, `_f64_min_ones` and the four `_warpk_ones` variants.

### Inner loop (naive)

```c
unsigned int acc_cnt = 0;        // u32 — max value K, fits trivially.
T            acc_val = (INIT_VAL);

for (int k = 0; k < K; ++k) {
    T va = value_a[OFFSET_ROW(i, k, K)];
    T vb = value_b[OFFSET_ROW(k, j, N)];
    T pv = va + vb;
    bool win = BETTER(pv, acc_val);
    bool tie = (pv == acc_val);
    acc_val  = win ? pv : acc_val;
    acc_cnt  = win ? 1u : (tie ? (acc_cnt + 1u) : acc_cnt);
}
value_c[OFFSET_ROW(i, j, N)] = acc_val;
count_c[OFFSET_ROW(i, j, N)] = (int)barrett_mod((unsigned long long)acc_cnt,
                                                (unsigned long long)P, MU);
```

What's gone vs. AoS: count loads, the u32×u32→u64 multiply, per-step Barrett, u64 accumulator. What's left: 2 fp loads, 1 fp add, 1 fp compare, 1 increment. That's the lower bound of useful work for tropical+counting.

### Inner loop (warpk)

Same K-stride structure as `counting_gemm_<T>_<dir>_warpk`, but with the simpler accumulator:

```c
for (int k = lane; k < K; k += 32) {
    T va = value_a[OFFSET_ROW(i, k, K)];
    T vb = value_b[OFFSET_ROW(k, j, N)];
    T pv = va + vb;
    bool win = BETTER(pv, acc_val);
    bool tie = (pv == acc_val);
    acc_val  = win ? pv : acc_val;
    acc_cnt  = win ? 1u : (tie ? (acc_cnt + 1u) : acc_cnt);
}
```

### Warp reduction (warpk)

`acc_cnt` is u32 — single shuffle per step instead of the hi/lo split currently needed for u64. **Five shuffles per acc_cnt instead of ten.**

```c
for (int off = 16; off > 0; off >>= 1) {
    T            ov = __shfl_xor_sync(0xffffffff, acc_val, off);
    unsigned int oc = __shfl_xor_sync(0xffffffff, acc_cnt, off);
    bool win = BETTER(ov, acc_val);
    bool tie = (ov == acc_val);
    acc_val = win ? ov : acc_val;
    acc_cnt = win ? oc : (tie ? acc_cnt + oc : acc_cnt);
}
if (lane == 0) {
    value_c[OFFSET_ROW(i, j, N)] = acc_val;
    count_c[OFFSET_ROW(i, j, N)] = (int)barrett_mod((unsigned long long)acc_cnt,
                                                    (unsigned long long)P, MU);
}
```

### Overflow analysis

`acc_cnt` is u32; maximum value is K (every k-step ties). For the stated use cases up to K=2¹⁶, headroom is 2³² / 2¹⁶ = 2¹⁶× — vastly safe. Even K=2³¹ fits. Single Barrett at the end (after warp reduction) lands the residue mod P.

### Driver change

`count_ground_states_gpu` (in `src/crt.rs`) currently packs all-ones counts into AoS pair buffers and routes through the AoS kernels. Change:

- Upload value buffers as `GpuMatrix<T>` (no pair pack, no count buffer).
- Add a `launch_counting_gemm_ones<T, D>` entry point that dispatches naive vs warpk based on the same `M·N ≤ 64²` shape heuristic and routes to the new kernel symbols.
- The AoS `launch_counting_gemm` stays callable for future general-counts callers.

For now, the driver's only entry point is all-ones, so the AoS code path becomes orphaned in production but remains live for tests/bench.

### Buffer layout

Value buffers stay row-major and are uploaded verbatim — no packing pass on the host (replaces the AoS pack, saving ~M·K + K·N element copies on the host). Net e2e benefit on top of the kernel speedup.

### Trait

```rust
pub trait CountingCudaKernel<T, D> {
    const KERNEL_NAME: &'static str;          // existing AoS naive
    const KERNEL_NAME_WARPK: &'static str;    // existing AoS warpk
    const KERNEL_NAME_ONES: &'static str;     // new
    const KERNEL_NAME_WARPK_ONES: &'static str; // new
    // ...
}
```

Add `launch_counting_gemm_ones<T, D>` free fn and trait method that take value-only buffers (`GpuMatrix<T>`, no `Pair` involvement). Same shape-aware dispatch.

### Testing

Existing 13 integration tests in `tests/counting_gpu.rs` already use `count_ground_states_gpu` (which will route to ones path post-migration) — they cover correctness for free across shapes / primes / directions / scalars.

Add three direct ones-path tests per spec E precedent:

1. `ones_k_threshold_boundary` — K ∈ {32, 63, 64, 65, 95, 128} matches CPU oracle (covers naive↔warpk boundary).
2. `ones_non_aligned_dims` — M=37, K=131, N=29, exercises tail-row + tail-K-stride predication on warpk-ones.
3. `ones_all_ties_large_k` — K=200, all inputs zero, asserts count = K mod P (most reduction-sensitive case under the new u32 accumulator).

### Bench

Extend `bench_kernel_only.rs` to also time the new `_ones` kernels at the same shapes used today. Compare against the AoS general kernels on the same hardware.

### Roll-out

1. Add 8 ones kernels (4 naive + 4 warpk) to `kernels/counting_gemm.cu`.
2. Add `KERNEL_NAME_ONES` / `KERNEL_NAME_WARPK_ONES` consts and trait method.
3. Add `launch_counting_gemm_ones` free fn.
4. Update `count_ground_states_gpu` driver to upload value-only and call ones path.
5. Run tests + bench. Verify ≥1.5× kernel speedup at 4096² square; verify warpk small-shape ≥1.5×.
6. Update memory + commit.

If wins are at-or-above expectations, no further action. If wins are below 1.3×, profile to see whether the gain is masked by output-write traffic — possible follow-up: lazier writes / write coalescing.

### Risks

- **Overflow if K very large.** Mitigated: u32 holds K ≤ 2³², covers the entire stated regime with 2¹⁶× headroom.
- **Future non-ones callers.** Mitigated: AoS general kernels stay; routing is a one-line switch in the driver.
- **Cache-pressure equivalence with AoS.** Inner loop now does 2× fp32 loads instead of 2× pair loads. Both same total bytes (8 B/k for f32). The win is in instruction count and the disappeared int multiply / Barrett, not memory traffic. Don't expect this to magically fix bandwidth-bound regimes.
- **Tie correctness under warp reduce.** The `acc_cnt + oc` is a plain u32 add, no Barrett. Tested by `ones_all_ties_large_k`.

## Outcome (measured 2026-04-27 on A100-SXM4-80GB)

**Massively above expectations.** Spec predicted 1.6–2.1× kernel speedup; measured 2.9–3.2× on the dominant naive path. The estimate underweighted how expensive the count multiply + Barrett were relative to fp work.

**Naive path (square shapes, f32 Max, 1 prime, kernel-only):**

| Size | AoS general G/s | ones G/s | speedup |
|---|---|---|---|
| 128² | 163 | 348 | 2.13× |
| 256² | 409 | 1203 | 2.94× |
| 512² | 570 | 1729 | 3.03× |
| 1024² | 625 | 1944 | 3.11× |
| **2048²** | 665 | **2136** | **3.21×** |
| **4096²** | 665 | **1946** | **2.93×** |

The ones kernel **exceeds the prior MaxPlus reference (~1500 G/s, no counting)** at all sizes ≥ 512². Likely explanation: removing both the count loads (halving bandwidth) and the count arithmetic frees up enough cycles that the kernel becomes compute-bound on a streamlined critical path, beating older MaxPlus implementations.

**Warpk path:** modest 1.0–1.2× — that regime is bottlenecked by strided-B reads, so dropping count machinery doesn't help as much.

**Tests:** 16/16 green (3 new ones-path tests + 13 prior). Existing integration tests now route through the ones path via `count_ground_states_gpu`, so coverage of the production path is identical to before.

**Files:**
- `kernels/counting_gemm.cu`: 8 new kernels (4 naive ones + 4 warpk ones).
- `src/context.rs`, `src/counting_kernel.rs`: kernel name registry, trait consts, `launch_counting_gemm_ones` method + free fn.
- `src/crt.rs`: driver routes through ones path; AoS pack code removed.
- `tests/counting_gpu.rs`: 3 ones-path correctness tests.
- `examples/bench_kernel_only.rs`: AoS-vs-ones head-to-head bench.

**AoS general kernels remain in tree** for future callers passing non-trivial input counts (chained matmul, etc.). They are now orphaned in production but are reachable via `launch_counting_gemm` for benchmarks and future work.
