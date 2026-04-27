# Spec F — AoS `(value, count)` element layout for counting kernels

**Date:** 2026-04-27
**Status:** design (pending user review)
**Branch:** `counting-tropical`
**Depends on:** spec C (CUDA counting kernel), spec E (warpk variant + dispatch).
**Scope:** change the on-device element layout from SoA (`value_a`, `count_a` as separate buffers) to AoS (`pair_a` as a single buffer of `(value, count)` structs). Apply to both the naive and warpk counting kernels. Output buffers stay SoA.

## Goal

Halve global memory transactions per inner-loop step. The current kernels issue 4× LDG.E.32 per step (2 values + 2 counts). With AoS, each (value, count) element is loaded as a single 8-byte transaction (f32 pair) — 2× LDG.E.64 per step. **Expected: 1.3–1.5× kernel speedup on the dominant large-shape regime** (naive kernel, where memory pressure on the inner loop is the next bottleneck after Barrett).

This is the optimization roadmap's Tier 1 #2 item, deferred from spec E because it was originally proposed in conjunction with warpk and warpk turned out to be useful only for small shapes. AoS-on-naive attacks the regime that actually matters for large problems.

## Non-goals

- Output AoS. Callers (CRT host driver, Python bindings) consume values and counts separately. Output stays SoA — no caller-facing churn.
- Removing the SoA kernels in this spec. They stay as fallbacks until AoS is measured to win at all shapes; final cleanup is a follow-up commit.
- Changing the public Rust API (`count_ground_states_gpu`). The pack/unpack happens inside the host driver; callers keep passing `&[T]` and `&[i32]` (or all-ones implicitly).

## Design

### Pair element types

```rust
#[repr(C, align(8))]
#[derive(Clone, Copy, Default)]
pub struct PairF32 { pub val: f32, pub cnt: i32 }   // 8 B exactly

#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
pub struct PairF64 { pub val: f64, pub cnt: i32, _pad: i32 }  // 16 B (4 B pad)
```

C-side mirrors:

```c
struct __align__(8)  PairF32 { float  val; int cnt; };
struct __align__(16) PairF64 { double val; int cnt; int _pad; };
```

`DeviceRepr` and `ValidAsZeroBits` impls are manual (both traits are unsafe but trivial for plain `repr(C)` POD with no padding-of-pointers concerns; the f64 padding byte is uninit but never read by the kernel).

**Why keep `cnt` as `i32` for f64?** All CRT primes are < 2³¹, so i32 mod p is sufficient. Promoting to i64 would balloon f64 input to 32 B/element with no correctness gain. The 4-byte pad on f64 is a one-time cost (acceptable: 16 MB wasted at 2048² f64 input, negligible compared to the kernel time saved).

### Buffer layout

`count_ground_states_gpu` driver replaces the four uploads `value_a, count_a, value_b, count_b` with two:

```rust
// Pack once, before the prime loop. Counts are all-ones in the entry point.
let pair_a: Vec<PairT> = a_values.iter().map(|&v| PairT { val: v, cnt: 1 }).collect();
let pair_b: Vec<PairT> = b_values.iter().map(|&v| PairT { val: v, cnt: 1 }).collect();
let dev_pair_a = GpuMatrix::<PairT>::from_host(ctx, &pair_a, m, k)?;
let dev_pair_b = GpuMatrix::<PairT>::from_host(ctx, &pair_b, k, n)?;
```

Output buffers `value_c: GpuMatrix<T>`, `count_c: GpuMatrix<i32>` stay unchanged. The pack runs once per call (not per prime), so cost is amortized across the prime loop.

### Kernel signature

```c
extern "C" __global__ void counting_gemm_f32_max_aos(
    const PairF32* __restrict__ pair_a,
    const PairF32* __restrict__ pair_b,
    float*         __restrict__ value_c,
    int*           __restrict__ count_c,
    int M, int N, int K, int P, unsigned long long MU
);
```

Same for `_min`, `_f64_max`, `_f64_min`. And the warpk siblings: `counting_gemm_<T>_<dir>_warpk_aos`.

### Inner loop change (naive)

Before (SoA):

```c
T   va = value_a[OFFSET_ROW(i, k, K)];
u32 ca = (u32)count_a[OFFSET_ROW(i, k, K)];
T   vb = value_b[OFFSET_ROW(k, j, N)];
u32 cb = (u32)count_b[OFFSET_ROW(k, j, N)];
```

After (AoS):

```c
PairF32 a = pair_a[OFFSET_ROW(i, k, K)];   // single LDG.E.64
PairF32 b = pair_b[OFFSET_ROW(k, j, N)];   // single LDG.E.64
T   va = a.val;  u32 ca = (u32)a.cnt;
T   vb = b.val;  u32 cb = (u32)b.cnt;
```

The rest of the loop body and reduction are byte-identical.

### Kernel macros

Refactor `counting_gemm.cu` so the loop body is a shared macro and the load/store wrapper differs per layout. Alternative: duplicate the body as a separate macro. **Decision:** duplicate. Two macros (`COUNTING_GEMM` and `COUNTING_GEMM_AOS`, plus warpk variants) keeps each readable; the body is short (~20 lines). Cross-macro abstraction would obscure the fast path for marginal DRY benefit.

### Rust trait

`CountingCudaKernel<T, D>` gains a third name and launcher:

```rust
const KERNEL_NAME_AOS: &'static str;
const KERNEL_NAME_WARPK_AOS: &'static str;
```

`launch_counting_gemm` updated to dispatch in two dimensions:
- AoS vs SoA (preferred AoS once measured).
- Naive vs warpk (existing K + M·N heuristic).

First cut: **always use AoS**. Keep SoA kernels accessible via separate trait method `launch_counting_gemm_soa` for benchmarking only. After the AoS win is verified, retire the SoA path in a follow-up commit.

### Dispatch (after Spec F lands)

```
use_warpk = (K >= 64) && (M*N <= 64*64)
kernel_name = match (use_warpk, /* always AoS */) {
    (false, _) => KERNEL_NAME_AOS,
    (true,  _) => KERNEL_NAME_WARPK_AOS,
}
```

The pair pack happens in `count_ground_states_gpu` once, both naive and warpk kernels read the same packed buffers — no per-dispatch packing.

### Tests

Existing 13 `counting_gpu` tests (CPU oracle parity at varied shapes / primes / directions) cover correctness for free once dispatch routes through AoS. Add three layout-specific tests:

1. `aos_pack_roundtrip` — host-side: pack a `(values, counts)` pair, confirm field accessors return the originals byte-for-byte. Pure unit test, no GPU. Catches `repr(C)` / alignment regressions.
2. `aos_matches_soa_naive` — for one (M, K, N), call both kernels directly and assert byte-equal output. Forces a regression alarm if the layout change ever silently corrupts.
3. `aos_matches_soa_warpk` — same for the warpk pair.

### Bench

Extend `bench_kernel_warpk.rs` (or fork to `bench_kernel_aos.rs`) to run AoS-naive vs SoA-naive and AoS-warpk vs SoA-warpk at the same six shapes used in spec E. Single-launch + amortized variants.

### Roll-out plan

1. Land pair types + `DeviceRepr` impls + host pack helper.
2. Land AoS kernels (naive + warpk) alongside the SoA originals.
3. Update dispatch to use AoS by default; keep SoA accessible.
4. Run full test suite + AoS-vs-SoA bench on A100. Verify byte-equal output and ≥1.3× kernel speedup at 4096².
5. If wins are uniform: retire SoA kernels in a follow-up commit, remove now-dead `KERNEL_NAME` (rename `KERNEL_NAME_AOS` → `KERNEL_NAME`).
6. Update `project_counting_status.md` memory.

### Risks

- **Alignment.** `PairF32` is 8-byte aligned, naturally; `PairF64` is 16-byte. cudarc's `htod_copy` of a `Vec<PairT>` should give 256-byte device alignment. Confirm post-merge; otherwise add an `alloc_zeros::<PairT>` + `htod_copy_into` path.
- **Pack overhead.** ~M·K + K·N elements at ~5 ns per `Vec::push`-equivalent. At 2048² square: 2 × 4M elements × 5 ns ≈ 40 ms. The driver currently spends ~13 ms on the SoA `vec![1; …]` ones-alloc — replacing it with the pack adds ~25 ms net. Acceptable: this is host-side work that runs in parallel with the first prime's H→D transfer; net e2e impact is small. Verify in bench.
- **f64 padding byte.** Uninit memory written to device. Kernel never reads it, but Valgrind / cuda-memcheck might flag the H→D copy if the source `Vec<PairF64>` was not fully initialized. Mitigation: explicit `_pad: 0` in the constructor.
- **DeviceRepr impl correctness.** Low risk — both types are POD `repr(C)` with no pointer fields. Standard manual impl pattern.
- **Cross-platform behavior.** PTX uses LD.GLOBAL.E.U64 for 8-byte loads regardless of arch (sm_75 onward). No A100-specific assumption.

## Decision points (need user input before implementing)

1. **Replace SoA kernels in this spec, or keep both as a transition?** Recommendation: keep both for one commit (so AoS-vs-SoA can be A/B benched), then retire SoA in a follow-up.
2. **f64 cnt as i32-with-pad (16 B element) or i64 (also 16 B but no pad)?** Recommendation: i32-with-pad, matches existing buffer semantics and keeps host-side pack code uniform across f32/f64. The pad is internal kernel detail; callers don't see it.
3. **Pack inside `count_ground_states_gpu`, or expose a pre-packed input API?** Recommendation: pack inside the driver. The all-ones count case is the only entry point today; future callers with non-trivial counts can be supported by a second entry point later.
