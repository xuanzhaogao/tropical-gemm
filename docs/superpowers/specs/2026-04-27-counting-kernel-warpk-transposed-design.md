# Spec H — Warpk with transposed B for coalesced lane loads

**Date:** 2026-04-27
**Status:** design (ready to implement)
**Branch:** `counting-tropical`
**Depends on:** spec E (warpk dispatch), spec G (ones-specialized kernel).
**Scope:** fix the strided-B access pattern in the warpk-ones kernel by uploading B in transposed (N×K row-major) layout when the warpk regime is selected. Re-measure the warpk-vs-naive crossover after the fix; the warpk envelope likely expands.

## Goal

Lift the warpk-ones throughput from the current 184–231 G/s ceiling toward the naive-ones level (~2000 G/s). The current warpk kernel's only memory-bound bottleneck is non-coalesced B reads — every other piece of the inner loop is the same as naive-ones, which hits 1946 G/s at 4096².

**Expected:** 1.5–2.5× kernel speedup in the warpk regime (target ~400–600 G/s @ M=N=64, K=4096). Larger relative gain at larger N (where the strided pattern hurts most), so this also potentially expands the warpk dispatch envelope past `M·N ≤ 64²`.

## The problem

Current warpk kernel inner loop (per lane, per K-stride step `s`):

```c
T va = value_a[i*K + 32s + lane];   // 32 contiguous → coalesced ✓
T vb = value_b[(32s + lane)*N + j]; // 32 elements at fixed j, strided by N → non-coalesced ✗
```

The B access fans out to 32 separate cache lines per warp-step. At large N, L2 can't absorb the working set across the warp's lifetime, so the kernel is bandwidth-starved — not by HBM throughput but by L2 transactions.

## The fix

Upload B as B^T (shape N × K row-major) when the driver dispatches to warpk. Then:

```c
T vb = value_b_t[j*K + 32s + lane];  // same j, 32 contiguous k → coalesced ✓
```

Identical access pattern to A — 32 contiguous loads per warp-step, perfectly coalesced. The transpose runs once host-side (or on-device) before the prime loop and is amortized across all primes.

## Design

### Kernel signature change

Replace the existing 4 `counting_gemm_<T>_<dir>_warpk_ones` kernels with new ones that read B^T:

```c
extern "C" __global__ void counting_gemm_f32_max_warpk_ones(
    const float* __restrict__ value_a,    // M × K row-major (unchanged)
    const float* __restrict__ value_b_t,  // N × K row-major (was K × N)
    float*       __restrict__ value_c,
    int*         __restrict__ count_c,
    int M, int N, int K, int P, unsigned long long MU
);
```

Inner loop:

```c
for (int k = lane; k < K; k += 32) {
    T va = value_a  [OFFSET_ROW(i, k, K)];   // M-row * K + k
    T vb = value_b_t[OFFSET_ROW(j, k, K)];   // N-row * K + k  ← transposed
    // rest identical
}
```

Same shape constraints as before (`blockDim = (32, 4)`, `gridDim = (N, ceil(M/4), 1)`). No change to warp reduction, output layout, or boundary handling.

### Naive path stays the same

The naive kernel's access pattern is *already* coalesced (32 lanes share `i`, varying `j` → 32 contiguous B columns at fixed `k`). Transposing would *break* that. So:

- Naive uses non-transposed B (current layout).
- Warpk uses transposed B (new layout).
- Driver branches on `use_warpk` and uploads the appropriate buffer.

### Driver change

In `count_ground_states_gpu` (`src/crt.rs`):

```rust
let use_warpk = k >= COUNTING_WARPK_K_THRESHOLD
    && m.saturating_mul(n) <= COUNTING_WARPK_MN_CEILING;

let value_a_dev = GpuMatrix::<T>::from_host(ctx, a_values, m, k)?;
let value_b_dev = if use_warpk {
    let b_t = transpose_row_major::<T>(b_values, k, n); // (k×n) → (n×k)
    GpuMatrix::<T>::from_host(ctx, &b_t, n, k)?
} else {
    GpuMatrix::<T>::from_host(ctx, b_values, k, n)?
};

// kernel launch unchanged in shape; the kernel name selects which layout it expects
```

The transpose helper is a 10-line host-side function; no need for a CUDA transpose kernel at the warpk regime's small N (≤ 64). Cost at M=N=64, K=4096 is ~64×4096 = 256K elements = 1 ms host-side, amortized over 1+ prime kernel launches that take ~150 µs each. With 1 prime the transpose adds ~7× the kernel time — non-trivial. With 2+ primes the relative cost drops fast.

**Mitigation if 1-prime warpk transpose cost is unacceptable:** add an on-device transpose mini-kernel (parallelized, runs at HBM bandwidth → ~10 µs). Defer until measurement says it matters.

### Trait + dispatch

No new kernel symbols (the existing `KERNEL_NAME_WARPK_ONES` constant points at the new transposed-B kernel). Internal-only contract change: when dispatching warpk-ones, the driver must upload B transposed.

The dispatch heuristic (`use_warpk = K >= 64 && M*N <= 64*64`) **probably needs to be re-tuned upward** after this lands. Once warpk's B reads are coalesced, the kernel becomes broadly competitive — possibly up to `M*N ≤ 256²` or higher. Re-measure after landing and adjust `COUNTING_WARPK_MN_CEILING` accordingly.

### Tests

Existing tests cover correctness via CPU oracle parity in the warpk regime (M·N ≤ 64²). The kernel rewrite changes the **input layout** but not the output, so the existing tests stay valid as long as the driver uploads correctly transposed.

Add one direct test for layout sanity:

- `warpk_transposed_b_layout` — set up A and B with deliberately asymmetric values (e.g., A[i,k] = i+k, B[k,j] = 100*k + j), force warpk dispatch (K=128, M=N=8), assert CPU parity. Catches the case where the host transpose helper is wrong but the kernel happens to mask it.

### Bench

Extend `bench_kernel_only.rs` to bench the warpk path at expanded M·N to see where the new crossover sits:

- M=N=32 K=4096 (existing)
- M=N=64 K=4096 (existing)
- M=N=128 K=4096 (new — was a warpk loss before; expected to win after)
- M=N=256 K=4096 (new)
- M=N=128 K=512 (new — small-K corner)

### Roll-out

1. Add `transpose_row_major::<T>` host helper (10 LOC).
2. Replace the 4 `_warpk_ones` kernels in `counting_gemm.cu` with transposed-B versions.
3. Update driver to upload transposed B when warpk-ones dispatched.
4. Add `warpk_transposed_b_layout` test.
5. Build + run all tests on A100. Verify 16/16 still green.
6. Run extended bench. Identify new crossover point.
7. Update `COUNTING_WARPK_MN_CEILING` based on measurement.
8. Update memory + commit.

### Risks

- **Transpose cost dominates at 1-prime, small problems.** Quantified above. Mitigation: on-device transpose if it becomes an issue. Likely irrelevant once CRT runs ≥2 primes (the common case).
- **Crossover shift not measured cleanly.** Bench at multiple M·N points to find the real boundary, not just confirm wins at the old points. Don't blindly raise `COUNTING_WARPK_MN_CEILING` without data.
- **Transpose helper correctness.** A row/column swap is easy to get wrong. The `warpk_transposed_b_layout` test with asymmetric values catches this.
- **AoS warpk path becomes inconsistent.** The AoS general warpk kernel still uses non-transposed B. We're not touching it (currently orphaned in production). When a non-ones caller appears, it'll need its own transposed variant or stay slow on warpk dispatch.

### Non-goals

- On-device transpose kernel (defer until needed).
- Fixing AoS warpk (orphaned in production).
- Changing the naive path (already optimal coalescing).
- Auto-tuning `COUNTING_WARPK_MN_CEILING` per shape — manual re-tune after measurement.

## Outcome (measured 2026-04-27 on A100-SXM4-80GB)

**Massively above expectations across the entire tested range.** Spec predicted 1.5–2.5×; warpk-ones (transposed B) now wins at every M·N tested up to 1024².

| Shape (K=4096) | naive-ones G/s | warpk-ones (transposed B) G/s | speedup |
|---|---|---|---|
| 16² | 8 | 178 | 21.7× |
| 32² | 33 | 605 | 18.5× |
| **64² (old ceiling)** | 122 | **1324** | **10.9×** |
| 96² | 271 | 1610 | 5.95× |
| **128² (new ceiling)** | 480 | **1903** | **3.96×** |
| 192² | 1034 | 2097 | 2.03× |
| 256² | 1384 | 2210 | 1.60× |
| 512² | 1786 | 2316 | 1.30× |
| **1024²** | 2031 | **2525** | 1.24× |

Peak warpk-ones throughput hits **2.5 TG/s** at 1024² — beyond the prior MaxPlus reference.

**Dispatch ceiling raised from 64² → 128²** (conservative, accounts for host-side transpose cost on single-prime calls). The kernel itself wins much further out, but the transpose overhead is non-amortized in single-prime calls. Multi-prime CRT calls would profitably use warpk up to ~512²; future on-device transpose would push the threshold higher still.

**Tests:** 17/17 integration green (added `warpk_transposed_b_layout` with deliberately asymmetric A/B values to catch transpose-helper bugs). 56/56 lib green. Total 73/73.

**Files:**
- `kernels/counting_gemm.cu` — `_warpk_ones` kernels read `value_b_t[j, k]` (N×K row-major) instead of `value_b[k, j]`.
- `src/crt.rs` — added `transpose_row_major::<T>` host helper; driver branches B upload on `use_warpk`.
- `src/counting_kernel.rs` — `launch_counting_gemm_ones` now takes M, K, N explicitly (B's `GpuMatrix` shape is layout-dependent).
- `src/context.rs` — `COUNTING_WARPK_MN_CEILING` raised 4096 → 16384.
- `tests/counting_gpu.rs` — `warpk_transposed_b_layout` test.
- `examples/bench_warpk_crossover.rs` — sweep across shape grid to find the new crossover.

**Follow-up suggested by data:** on-device transpose kernel would let `COUNTING_WARPK_MN_CEILING` rise to ~256² or higher and capture the 1.24–1.60× wins at 256–1024² that single-prime calls currently leave on the table.