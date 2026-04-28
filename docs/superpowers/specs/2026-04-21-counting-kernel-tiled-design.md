# Spec D — Tiled CUDA kernel for CountingTropical

**Date:** 2026-04-21
**Status:** design approved
**Depends on:** spec C (`2026-04-21-counting-tropical-cuda-design.md`).
**Scope:** replace the naive one-thread-per-cell `counting_gemm.cu` with a
BLIS-style tiled kernel mirroring the existing `tropical_gemm.cu` pattern.
Same API (`count_ground_states_gpu`), same kernel names, same trait dispatch.

## Goal

Bring the `CountingTropical` GPU kernel up to the same optimization tier as
the existing `tropical_gemm.cu` kernels: two-level cache blocking (shared
memory block tile + register thread tile), cooperative tile loads, and
`__restrict__` pointers.

## Design

### Tile parameters

| | f32 | f64 |
|---|---|---|
| `BLOCK_SIZE_M × BLOCK_SIZE_K × BLOCK_SIZE_N` | 64 × 32 × 64 | 32 × 16 × 32 |
| `THREAD_SIZE_M × THREAD_SIZE_N` | 4 × 4 | 4 × 4 |
| Threads per block | 256 (16×16) | 64 (8×8) |
| Shared memory per block | 32 KB (2×8 KB value + 2×8 KB count) | ~12 KB |

f32 tile sizes match the existing `tropical_gemm.cu` f32 kernel. f64 halves
block dims (same as existing f64 tropical kernel) to keep shared memory
within the 48 KB baseline for `double` + `int` tile pairs.

### Shared memory

Each block reserves four tiles: value + count for both A and B.

```c
__shared__ T   As_val[BLOCK_SIZE_M * BLOCK_SIZE_K];
__shared__ int As_cnt[BLOCK_SIZE_M * BLOCK_SIZE_K];
__shared__ T   Bs_val[BLOCK_SIZE_K * BLOCK_SIZE_N];
__shared__ int Bs_cnt[BLOCK_SIZE_K * BLOCK_SIZE_N];
```

### Per-thread accumulators

```c
T   val_accum[THREAD_SIZE_M * THREAD_SIZE_N];   // init = INIT_VAL
int cnt_accum[THREAD_SIZE_M * THREAD_SIZE_N];   // init = 0
T   regs_a_val[THREAD_SIZE_M];
int regs_a_cnt[THREAD_SIZE_M];
T   regs_b_val[THREAD_SIZE_N];
int regs_b_cnt[THREAD_SIZE_N];
```

### Inner loop structure

```
for (tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {
    // Cooperative tile load (256 threads load the 64×32 A tile and 32×64 B tile).
    // Predicated: if (row < M && col < K) { load } else { pad (INIT_VAL, 0) }
    __syncthreads();
    for (int k = 0; k < BLOCK_SIZE_K; ++k) {
        // Load 4 A and 4 B (value, count) pairs into registers.
        // 4×4 outer product with three-way compare-update.
        for (tm = 0..THREAD_SIZE_M) for (tn = 0..THREAD_SIZE_N) {
            T   pv = regs_a_val[tm] + regs_b_val[tn];
            int pc = (int)((long long)regs_a_cnt[tm] * regs_b_cnt[tn] % P);
            if (BETTER(pv, val_accum[idx])) {
                val_accum[idx] = pv;
                cnt_accum[idx] = pc;
            } else if (BETTER(val_accum[idx], pv)) {
                // keep
            } else {
                cnt_accum[idx] = (int)(((long long)cnt_accum[idx] + pc) % P);
            }
        }
    }
    __syncthreads();
}
// Write the 4×4 tile per thread to value_c and count_c (predicated).
```

All `for` loops over `THREAD_SIZE_*` and `BLOCK_SIZE_K` get `_Pragma("unroll")`.

### Padding correctness

Tile-load padding uses `(INIT_VAL, 0)` for out-of-bounds cells.

- MaxPlus: `INIT_VAL = -INF`. Padding product: `-INF + any = -INF, 0 × any mod P = 0`. `BETTER(-INF, acc)` false for finite `acc`, `BETTER(acc, -INF)` true → kept. Initial accumulator `-INF` hits tie branch: `(0 + 0) mod P = 0`. Safe.
- MinPlus: `INIT_VAL = +INF`. Symmetric.

### Row-major vs column-major

The current `counting_gemm.cu` uses row-major addressing (matches CPU
`tropical_matmul_t` convention). The existing `tropical_gemm.cu` uses
column-major — a different historical choice. The tiled counting kernel
keeps **row-major** to stay consistent with the rest of the counting pipeline
(driver, Rust wrapper, Python binding). This means the tile-load helper
indices mirror the existing kernel's shape but with row/col roles swapped.

### Grid / block dims

`counting_block_dims_f32() = (16, 16, 1)` and `counting_block_dims_f64() = (8, 8, 1)`.
`counting_grid_dims_f32(m, n)` and `counting_grid_dims_f64(m, n)` compute block
counts using the respective `BLOCK_SIZE_M / BLOCK_SIZE_N`. The `CountingCudaKernel`
trait gains a method `launch_dims(m, n) -> (grid, block)` that each impl
fills in with the right helper, so the launch wrapper doesn't branch on `T`
internally.

## File impact

- **Rewrite:** `crates/tropical-gemm-cuda/kernels/counting_gemm.cu` — tiled macro replacing the naive one.
- **Modify:** `crates/tropical-gemm-cuda/src/context.rs` — split `counting_block_dims` / `counting_grid_dims` into `_f32` and `_f64` variants.
- **Modify:** `crates/tropical-gemm-cuda/src/counting_kernel.rs` — trait method `launch_dims()` + per-impl grid/block selection.
- **Modify:** `crates/tropical-gemm-cuda/tests/counting_gpu.rs` — add 3 new tests (large, off-boundary, f64 medium).

## Testing

### Preserved (must still pass)

- All 6 existing `counting_gpu` tests.
- `counting_gpu gpu_layout_contract_asymmetric` (2×3 × 3×2 hand-checked).
- CPU-side `counting_crt` tests (unchanged).

### New

- **Large shape.** 512×512 × 512×512 `f32` Max. GPU values + counts match CPU `count_ground_states`.
- **Off-block-boundary shape.** 17×19 × 19×23 `f32` Max. Exercises the predicated bounds guards for every tile edge.
- **f64 medium shape.** 128×128 × 128×128 `f64` Max.

All new tests compare GPU output against CPU `count_ground_states`.

### Optional (not in CI)

Simple manual benchmark with `std::time::Instant` around the GPU call at
512² and 1024² on `f32`. Not a regression gate; log numbers in the final
commit for future reference.

## Out of scope

- Kernel variants with argmax tracking for counting — separate spec.
- Integer `T` (`i32`, `i64`) counting kernels — separate spec.
- Further perf work (warp-level reductions, Tensor Cores, async copies) — separate follow-up if benchmarks motivate it.
- MinPlus specific tile sizes — MinPlus reuses the Max direction's tile sizes; the only diff is `INIT_VAL` and `BETTER` macro.

## Risks

- **Shared memory pressure on f64.** 48 KB is the per-block baseline on most NVIDIA cards (50-series and later have more; older devices have less). If compilation or runtime exceeds the limit on the target device, halve `BLOCK_SIZE_K` (16 → 8) for f64. Keep the same macro shape — just adjust the compile-time constants in the f64 specialization.
- **Index-bug surface.** The tiled macro with two parallel tiles (value + count) doubles the chance of a typo. Mitigation: the 3 new tests (large, off-boundary, f64) plus the existing 6 tests cover asymmetric shapes, boundary cases, and both scalar types. Any index bug surfaces as a GPU ≠ CPU mismatch.
- **Modular-arithmetic precision.** Same as spec C: 30-bit primes × 30-bit counts fit in `int64_t` intermediate. No change.
- **Register pressure.** Four 4-element arrays + two 16-element accumulators = 40 registers per thread for f32. With 256 threads/block that's 10240 registers/block, well within the 64K register file on modern SMs. For f64 the pattern doubles per-value but 64 threads/block halves the total → still under budget.
