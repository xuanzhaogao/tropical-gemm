# Spec P: cp.async double-buffered tiled counting GEMM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pipeline the tiled counting tropical GEMM kernel with `cp.async`-based double buffering on Ampere+ (sm_80) so the next K-tile's global→shared load overlaps the current K-tile's compute, lifting A100 throughput beyond the current 763 G tropical-ops/s baseline.

**Architecture:** Drop in a second variant of `TROPICAL_MATMUL_TILED_BODY` that uses two shared-memory buffers per operand and a 1-stage prefetch with `cp.async.cg.shared.global` + `cp.async.commit_group` + `cp.async.wait_group`. The current synchronous variant is preserved so sm_75 (RTX 6000) keeps working — runtime dispatch picks the pipelined variant only on `compute_capability >= (8, 0)`. Tile geometry stays at the A100-tuned 64×64 / BK=32 / 4×4 reg.

**Tech Stack:** CUDA C (NVRTC at runtime), PTX inline `cp.async`, cudarc 0.12.

**Spec:** none separate — this plan is the spec. Builds on `docs/superpowers/specs/2026-04-28-tiled-counting-gemm-design.md` (Spec N).

---

## File Map

- **Modify** `crates/tropical-gemm-cuda/kernels/counting_gemm.cu` — add `TROPICAL_MATMUL_TILED_PIPELINED_BODY` macro (cp.async + 2-buffer ping-pong), add 16 `*_pipelined_*` kernels (NN/NT/TN/TT × f32/f64 × max/min). Existing 16 sync kernels stay intact.
- **Modify** `crates/tropical-gemm-cuda/src/counting_kernel.rs` — add a `prefer_pipelined` flag computed once from the device's compute capability; route to `*_pipelined_*` kernel names when set. The 4-character suffix dispatch (NN/NT/TN/TT) and tile-dim math stay the same.
- **Modify** `crates/tropical-gemm-cuda/src/context.rs` — register the 16 new kernel names alongside the existing 16.
- **Modify** `crates/tropical-gemm-cuda/src/matmul_mod.rs` — no signature change; add a named-kernel launch helper plus five byte-equal sync-vs-pipelined tests (covers all four `(tA, tB)` layouts and the ragged-K / ragged-M-N edge cases). Tests skip cleanly on sm_<80.
- **Modify** `CountingTropicalGEMM.jl/bench/RESULTS.md` — append a new "## A100-SXM4-80GB (Spec P pipelined)" section with the bench numbers.

The compute body, write-out, and `LOAD_*_DECOMP_*` macros are reused unchanged. Only the load phase is rewritten.

---

### Task 1: Add the `cp.async` low-level helper macros

**Files:**
- Modify: `crates/tropical-gemm-cuda/kernels/counting_gemm.cu`

- [ ] **Step 1: Insert helper macros after the existing barrett_mod definition (around line 31)**

Insert before the `struct __align__(8) PairF32 { ... }` line:

```c
// ---- Spec P: cp.async helpers (sm_80+) ------------------------------------
//
// `cp.async.cg.shared.global` performs an asynchronous 4/8/16-byte copy
// from global memory into shared memory without occupying a register and
// without blocking the issuing warp. The copy is committed to a "group"
// via `cp.async.commit_group` and waited on with `cp.async.wait_group N`,
// which blocks until at most N groups are still in flight.
//
// `.cg` (cache global) is the right hint for streaming loads we read once
// per K-tile. Use `.ca` if subsequent re-reads from L1 would help (not
// our case).
//
// CP_ASYNC_CG_4 copies 4 bytes (one f32 or one i32). CP_ASYNC_CG_8 copies
// 8 bytes (a double or a PairF32). PairF64's 16 B is moved as one 8-byte
// cp.async for the `val` (double) plus one 4-byte cp.async for `cnt`
// (int). The 4-byte `_pad` slot is left untouched in shared memory; the
// compute phase never reads it.
#if __CUDA_ARCH__ >= 800
#define CP_ASYNC_CG_4(smem_ptr_u32, gmem_ptr) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 4;\n" :: "r"(smem_ptr_u32), "l"(gmem_ptr))
#define CP_ASYNC_CG_8(smem_ptr_u32, gmem_ptr) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 8;\n" :: "r"(smem_ptr_u32), "l"(gmem_ptr))
#define CP_ASYNC_COMMIT()  asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" :: "n"(N))
// __cvta_generic_to_shared converts a generic shared pointer to the
// 32-bit shared-state-space address that cp.async expects.
#define SMEM_PTR(ptr) static_cast<unsigned>(__cvta_generic_to_shared(ptr))
#else
// Stubs for sm_<80; the pipelined kernel will not be launched there, but
// it must still compile so NVRTC can build all 16 specializations on a
// shared NVRTC pass.
#define CP_ASYNC_CG_4(s, g) do { (void)(s); (void)(g); } while (0)
#define CP_ASYNC_CG_8(s, g) do { (void)(s); (void)(g); } while (0)
#define CP_ASYNC_COMMIT()
#define CP_ASYNC_WAIT_GROUP(N) ((void)0)
#define SMEM_PTR(ptr) 0u
#endif
```

- [ ] **Step 2: Verify the kernel still compiles via NVRTC at runtime**

`cargo build` only checks Rust — the `.cu` file is compiled by NVRTC the
first time `CudaContext::from_device()` runs. To actually validate the
new helpers don't break the existing kernels, run:

```
srun --jobid=6308637 --overlap bash -lc 'export PATH=$HOME/.cargo/bin:$PATH; module load cuda; cargo build --release -p tropical-gemm-cuda && cargo test --release -p tropical-gemm-cuda --lib -- --skip pipelined 2>&1 | tail -5'
```

Expected: 70/70 sync tests pass (Spec N), confirming NVRTC eagerly
compiled all 16 sync kernels with the new helpers in scope. (We `--skip
pipelined` because none of those tests exist yet.)

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm-cuda/kernels/counting_gemm.cu
git commit -m "Spec P Task 1: add cp.async helper macros for sm_80+"
```

---

### Task 2: Add the pipelined kernel body macro

**Files:**
- Modify: `crates/tropical-gemm-cuda/kernels/counting_gemm.cu`

- [ ] **Step 1: Insert the pipelined body macro after `TROPICAL_MATMUL_TILED_BODY`**

Append after the closing `}` of `TROPICAL_MATMUL_TILED_BODY` (around line 186):

```c
// Spec P: cp.async pipelined variant. Two shared-memory buffers per
// operand (As_v[2][...], etc.) ping-pong: while the compute phase reads
// buffer cur, the loader prefetches the next K-tile into buffer 1-cur.
//
// Pipeline layout (NUM_STAGES = 2):
//   prefetch tile 0 → buffer 0 → commit
//   for kk = BK; kk < K; kk += BK:
//       prefetch tile (kk/BK) → buffer (kk/BK & 1) → commit
//       wait_group 1                       // ensure tile (kk/BK)-1 ready
//       __syncthreads()                    // smem visibility for compute
//       compute on buffer ((kk/BK)-1) & 1
//       __syncthreads()                    // smem reuse safety
//   wait_group 0; __syncthreads(); compute final tile
//
// Shared memory usage:
//   f32 BM=BN=64, BK=16:
//     4 slabs × 2 stages × BK × (BM+1) × 4 B
//       = 4 × 2 × 16 × 65 × 4 = 33,280 B ≈ 32.5 KiB / block. Fits under
//     A100's 48-KiB-without-opt-in static-shared limit.
//   f64 BM=BN=32, BK=8:
//     (val) 2 stages × BK × (BM+1) × 8 B × 2 ops
//   + (cnt) 2 stages × BK × (BM+1) × 4 B × 2 ops
//       = (2 × 8 × 33 × 8 × 2) + (2 × 8 × 33 × 4 × 2)
//       = 8,448 + 4,224 = 12,672 B ≈ 12.4 KiB / block. Fits comfortably.
//
// IMPORTANT: BK for the pipelined f32 kernel is 16, not 32 (the sync
// baseline). Doubling BK to match would push shared past the 48 KiB
// static limit and require cudaFuncSetAttribute opt-in. We start with
// BK=16 to keep the implementation simple; if the pipelined win at
// BK=16 doesn't beat sync@BK=32, the appendix at the bottom of this
// plan covers the dynamic-shared opt-in path to try BK=32 pipelined.
#define TROPICAL_MATMUL_TILED_PIPELINED_BODY(T, PAIR, INIT_VAL, BETTER,        \
    A_OFF, B_OFF, LOAD_A_DECOMP, LOAD_B_DECOMP,                                \
    BM_, BN_, BK_, TM_, TN_)                                                   \
{                                                                              \
    __shared__ T   As_v[2][BK_][BM_ + 1];                                      \
    __shared__ int As_c[2][BK_][BM_ + 1];                                      \
    __shared__ T   Bs_v[2][BK_][BN_ + 1];                                      \
    __shared__ int Bs_c[2][BK_][BN_ + 1];                                      \
                                                                               \
    int tx = threadIdx.x, ty = threadIdx.y;                                    \
    int tid = ty * blockDim.x + tx;                                            \
    int threads_per_block = blockDim.x * blockDim.y;                           \
    int block_i0 = blockIdx.y * (BM_);                                         \
    int block_j0 = blockIdx.x * (BN_);                                         \
                                                                               \
    T                  acc_v[TM_][TN_];                                        \
    unsigned long long acc_c[TM_][TN_];                                        \
    _Pragma("unroll")                                                          \
    for (int ti = 0; ti < (TM_); ++ti)                                         \
        _Pragma("unroll")                                                      \
        for (int tj = 0; tj < (TN_); ++tj) {                                   \
            acc_v[ti][tj] = (INIT_VAL);                                        \
            acc_c[ti][tj] = 0ULL;                                              \
        }                                                                      \
                                                                               \
    const unsigned long long Pull = (unsigned long long)P;                     \
    const int A_TILE = (BM_) * (BK_);                                          \
    const int B_TILE = (BN_) * (BK_);                                          \
                                                                               \
    /* Per-element copy size for the `val` slot is sizeof(T):                 \
     *   f32 → 4 bytes (CP_ASYNC_CG_4), f64 → 8 bytes (CP_ASYNC_CG_8).         \
     * The `cnt` slot is always 4 bytes (int).                                 \
     * For ragged-K (gk >= K), write the (INIT_VAL, 0) sentinel directly via  \
     * shared-memory store (cp.async cannot synthesize arbitrary fill bytes).  \
     * The mixed-store-and-cp.async-into-the-same-buffer hazard is contained:  \
     * each thread writes exactly one of the two paths per (sk, si) slot, and \
     * different threads always own different slots, so no slot is written   \
     * twice in one tile. The subsequent __syncthreads() before compute       \
     * (after CP_ASYNC_WAIT_GROUP) makes both kinds of writes visible.        \
     *                                                                        \
     * Macro `CP_ASYNC_VAL_T(smem, gmem)` dispatches on sizeof(T):            \
     *   f32 (sizeof=4) → CP_ASYNC_CG_4, f64 (sizeof=8) → CP_ASYNC_CG_8.      \
     * Implemented as a `if constexpr` since NVRTC running in C++17 mode      \
     * supports it; if NVRTC rejects it, fall back to a tagged dispatch.      \
     */                                                                       \
    auto issue_load_A = [&](int kk_base, int buf) {                            \
        for (int idx = tid; idx < A_TILE; idx += threads_per_block) {          \
            int sk_a, si_a;                                                    \
            LOAD_A_DECOMP(idx, (BM_), (BK_), sk_a, si_a);                      \
            int gi = block_i0 + si_a;                                          \
            int gk = kk_base + sk_a;                                           \
            if (gi < M && gk < K) {                                            \
                const PAIR* gptr = &pair_a[A_OFF(gi, gk, M, K)];               \
                if constexpr (sizeof(T) == 8) {                                \
                    CP_ASYNC_CG_8(SMEM_PTR(&As_v[buf][sk_a][si_a]),            \
                                  reinterpret_cast<const T*>(&gptr->val));     \
                } else {                                                       \
                    CP_ASYNC_CG_4(SMEM_PTR(&As_v[buf][sk_a][si_a]),            \
                                  reinterpret_cast<const T*>(&gptr->val));     \
                }                                                              \
                CP_ASYNC_CG_4(SMEM_PTR(&As_c[buf][sk_a][si_a]),                \
                              reinterpret_cast<const int*>(&gptr->cnt));       \
            } else {                                                           \
                As_v[buf][sk_a][si_a] = (INIT_VAL);                            \
                As_c[buf][sk_a][si_a] = 0;                                     \
            }                                                                  \
        }                                                                      \
    };                                                                         \
    auto issue_load_B = [&](int kk_base, int buf) {                            \
        for (int idx = tid; idx < B_TILE; idx += threads_per_block) {          \
            int sk_b, sj_b;                                                    \
            LOAD_B_DECOMP(idx, (BN_), (BK_), sk_b, sj_b);                      \
            int gk = kk_base + sk_b;                                           \
            int gj = block_j0 + sj_b;                                          \
            if (gj < N && gk < K) {                                            \
                const PAIR* gptr = &pair_b[B_OFF(gk, gj, K, N)];               \
                if constexpr (sizeof(T) == 8) {                                \
                    CP_ASYNC_CG_8(SMEM_PTR(&Bs_v[buf][sk_b][sj_b]),            \
                                  reinterpret_cast<const T*>(&gptr->val));     \
                } else {                                                       \
                    CP_ASYNC_CG_4(SMEM_PTR(&Bs_v[buf][sk_b][sj_b]),            \
                                  reinterpret_cast<const T*>(&gptr->val));     \
                }                                                              \
                CP_ASYNC_CG_4(SMEM_PTR(&Bs_c[buf][sk_b][sj_b]),                \
                              reinterpret_cast<const int*>(&gptr->cnt));       \
            } else {                                                           \
                Bs_v[buf][sk_b][sj_b] = (INIT_VAL);                            \
                Bs_c[buf][sk_b][sj_b] = 0;                                     \
            }                                                                  \
        }                                                                      \
    };                                                                         \
    auto compute_on = [&](int buf, int kk_end) {                               \
        for (int kk2 = 0; kk2 < kk_end; ++kk2) {                               \
            T   av[TM_]; int ac[TM_];                                          \
            T   bv[TN_]; int bc[TN_];                                          \
            _Pragma("unroll")                                                  \
            for (int ti = 0; ti < (TM_); ++ti) {                               \
                av[ti] = As_v[buf][kk2][ty * (TM_) + ti];                      \
                ac[ti] = As_c[buf][kk2][ty * (TM_) + ti];                      \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tj = 0; tj < (TN_); ++tj) {                               \
                bv[tj] = Bs_v[buf][kk2][tx * (TN_) + tj];                      \
                bc[tj] = Bs_c[buf][kk2][tx * (TN_) + tj];                      \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int ti = 0; ti < (TM_); ++ti)                                 \
                _Pragma("unroll")                                              \
                for (int tj = 0; tj < (TN_); ++tj) {                           \
                    T pv = av[ti] + bv[tj];                                    \
                    unsigned long long prod =                                  \
                        (unsigned long long)(unsigned)ac[ti] *                 \
                        (unsigned long long)(unsigned)bc[tj];                  \
                    unsigned long long pc = barrett_mod(prod, Pull, MU);       \
                    bool win = BETTER(pv, acc_v[ti][tj]);                      \
                    bool tie = (pv == acc_v[ti][tj]);                          \
                    acc_v[ti][tj] = win ? pv : acc_v[ti][tj];                  \
                    acc_c[ti][tj] = win ? pc :                                 \
                        (tie ? (acc_c[ti][tj] + pc) : acc_c[ti][tj]);          \
                }                                                              \
        }                                                                      \
    };                                                                         \
                                                                               \
    /* Prefetch the first tile (kk=0) into buffer 0. */                       \
    issue_load_A(0, 0);                                                        \
    issue_load_B(0, 0);                                                        \
    CP_ASYNC_COMMIT();                                                         \
                                                                               \
    int cur_buf = 0;                                                           \
    for (int kk = (BK_); kk < K; kk += (BK_)) {                                \
        int next_buf = 1 - cur_buf;                                            \
        issue_load_A(kk, next_buf);                                            \
        issue_load_B(kk, next_buf);                                            \
        CP_ASYNC_COMMIT();                                                     \
        /* Wait for the *previous* tile (the one we're about to compute on). */\
        CP_ASYNC_WAIT_GROUP(1);                                                \
        __syncthreads();                                                       \
                                                                               \
        int kk_prev_end = (BK_);                                               \
        compute_on(cur_buf, kk_prev_end);                                      \
                                                                               \
        __syncthreads();                                                       \
        cur_buf = next_buf;                                                    \
    }                                                                          \
                                                                               \
    /* Drain the final tile. */                                                \
    CP_ASYNC_WAIT_GROUP(0);                                                    \
    __syncthreads();                                                           \
    int last_kk_base = ((K - 1) / (BK_)) * (BK_);                              \
    int last_kk_end = K - last_kk_base; /* in [1, BK_] */                      \
    compute_on(cur_buf, last_kk_end);                                          \
                                                                               \
    /* Write-out (identical to sync variant). */                               \
    _Pragma("unroll")                                                          \
    for (int ti = 0; ti < (TM_); ++ti) {                                       \
        int gi = block_i0 + ty * (TM_) + ti;                                   \
        if (gi >= M) continue;                                                 \
        _Pragma("unroll")                                                      \
        for (int tj = 0; tj < (TN_); ++tj) {                                   \
            int gj = block_j0 + tx * (TN_) + tj;                               \
            if (gj >= N) continue;                                             \
            PAIR out;                                                          \
            out.val = acc_v[ti][tj];                                           \
            out.cnt = (int)barrett_mod(acc_c[ti][tj], Pull, MU);               \
            out_c[gi + gj * M] = out;                                          \
        }                                                                      \
    }                                                                          \
}
```

- [ ] **Step 2: NVRTC validation — the macro is unused so far, so this only
checks the file parses. Defer real validation to Task 3 Step 4 where the
expansions force NVRTC to compile a pipelined kernel.**

Run: `bash -lc 'export PATH=$HOME/.cargo/bin:$PATH; cargo build --release -p tropical-gemm-cuda 2>&1 | tail -3'`
Expected: cargo build succeeds (Rust-side check only).

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm-cuda/kernels/counting_gemm.cu
git commit -m "Spec P Task 2: define cp.async pipelined kernel body macro"
```

---

### Task 3: Emit the 16 pipelined kernel specializations

**Files:**
- Modify: `crates/tropical-gemm-cuda/kernels/counting_gemm.cu`

- [ ] **Step 1: Add `DEFINE_TILED_*_PIPELINED` macros and 16 expansions**

Append after the existing 16 `DEFINE_TILED_F64(...)` lines (around line 226 in the current file):

```c
#define DEFINE_TILED_F32_PIPELINED(NAME, INIT_VAL, BETTER, A_OFF, B_OFF, LOAD_A, LOAD_B) \
extern "C" __global__ void NAME(                                               \
    const PairF32* __restrict__ pair_a,                                        \
    const PairF32* __restrict__ pair_b,                                        \
    PairF32* __restrict__ out_c,                                               \
    int M, int N, int K, int P, unsigned long long MU                          \
)                                                                              \
TROPICAL_MATMUL_TILED_PIPELINED_BODY(float, PairF32, INIT_VAL, BETTER, A_OFF, B_OFF, \
                                     LOAD_A, LOAD_B, 64, 64, 16, 4, 4)

#define DEFINE_TILED_F64_PIPELINED(NAME, INIT_VAL, BETTER, A_OFF, B_OFF, LOAD_A, LOAD_B) \
extern "C" __global__ void NAME(                                               \
    const PairF64* __restrict__ pair_a,                                        \
    const PairF64* __restrict__ pair_b,                                        \
    PairF64* __restrict__ out_c,                                               \
    int M, int N, int K, int P, unsigned long long MU                          \
)                                                                              \
TROPICAL_MATMUL_TILED_PIPELINED_BODY(double, PairF64, INIT_VAL, BETTER, A_OFF, B_OFF, \
                                     LOAD_A, LOAD_B, 32, 32, 8, 2, 4)

DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_max_NN_pl, NEG_INF_F32, MAX_BETTER, A_OFF_N, B_OFF_N, LOAD_A_DECOMP_N, LOAD_B_DECOMP_N)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_max_NT_pl, NEG_INF_F32, MAX_BETTER, A_OFF_N, B_OFF_T, LOAD_A_DECOMP_N, LOAD_B_DECOMP_T)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_max_TN_pl, NEG_INF_F32, MAX_BETTER, A_OFF_T, B_OFF_N, LOAD_A_DECOMP_T, LOAD_B_DECOMP_N)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_max_TT_pl, NEG_INF_F32, MAX_BETTER, A_OFF_T, B_OFF_T, LOAD_A_DECOMP_T, LOAD_B_DECOMP_T)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_min_NN_pl, POS_INF_F32, MIN_BETTER, A_OFF_N, B_OFF_N, LOAD_A_DECOMP_N, LOAD_B_DECOMP_N)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_min_NT_pl, POS_INF_F32, MIN_BETTER, A_OFF_N, B_OFF_T, LOAD_A_DECOMP_N, LOAD_B_DECOMP_T)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_min_TN_pl, POS_INF_F32, MIN_BETTER, A_OFF_T, B_OFF_N, LOAD_A_DECOMP_T, LOAD_B_DECOMP_N)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_min_TT_pl, POS_INF_F32, MIN_BETTER, A_OFF_T, B_OFF_T, LOAD_A_DECOMP_T, LOAD_B_DECOMP_T)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_max_NN_pl, NEG_INF_F64, MAX_BETTER, A_OFF_N, B_OFF_N, LOAD_A_DECOMP_N, LOAD_B_DECOMP_N)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_max_NT_pl, NEG_INF_F64, MAX_BETTER, A_OFF_N, B_OFF_T, LOAD_A_DECOMP_N, LOAD_B_DECOMP_T)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_max_TN_pl, NEG_INF_F64, MAX_BETTER, A_OFF_T, B_OFF_N, LOAD_A_DECOMP_T, LOAD_B_DECOMP_N)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_max_TT_pl, NEG_INF_F64, MAX_BETTER, A_OFF_T, B_OFF_T, LOAD_A_DECOMP_T, LOAD_B_DECOMP_T)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_min_NN_pl, POS_INF_F64, MIN_BETTER, A_OFF_N, B_OFF_N, LOAD_A_DECOMP_N, LOAD_B_DECOMP_N)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_min_NT_pl, POS_INF_F64, MIN_BETTER, A_OFF_N, B_OFF_T, LOAD_A_DECOMP_N, LOAD_B_DECOMP_T)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_min_TN_pl, POS_INF_F64, MIN_BETTER, A_OFF_T, B_OFF_N, LOAD_A_DECOMP_T, LOAD_B_DECOMP_N)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_min_TT_pl, POS_INF_F64, MIN_BETTER, A_OFF_T, B_OFF_T, LOAD_A_DECOMP_T, LOAD_B_DECOMP_T)
```

The `_pl` suffix is short for "pipelined".

- [ ] **Step 2: Add the 16 names to the kernel-name registry**

Open `crates/tropical-gemm-cuda/src/context.rs` and find `COUNTING_KERNEL_NAMES` (a `&[&str]`). Append the 16 new names. The existing constant currently lists the 16 sync names — add 16 `_pl` siblings:

```rust
pub const COUNTING_KERNEL_NAMES: &[&str] = &[
    // existing 16 sync entries (unchanged) ...
    "tropical_matmul_f32_max_NN_pl",
    "tropical_matmul_f32_max_NT_pl",
    "tropical_matmul_f32_max_TN_pl",
    "tropical_matmul_f32_max_TT_pl",
    "tropical_matmul_f32_min_NN_pl",
    "tropical_matmul_f32_min_NT_pl",
    "tropical_matmul_f32_min_TN_pl",
    "tropical_matmul_f32_min_TT_pl",
    "tropical_matmul_f64_max_NN_pl",
    "tropical_matmul_f64_max_NT_pl",
    "tropical_matmul_f64_max_TN_pl",
    "tropical_matmul_f64_max_TT_pl",
    "tropical_matmul_f64_min_NN_pl",
    "tropical_matmul_f64_min_NT_pl",
    "tropical_matmul_f64_min_TN_pl",
    "tropical_matmul_f64_min_TT_pl",
];
```

(Read the file before editing to confirm the exact constant name and surrounding syntax.)

- [ ] **Step 3: Cargo build (Rust-side)**

Run: `bash -lc 'export PATH=$HOME/.cargo/bin:$PATH; cargo build --release -p tropical-gemm-cuda 2>&1 | tail -5'`
Expected: success.

- [ ] **Step 4: NVRTC smoke test — force compilation of one pipelined kernel**

This is the first real validation that:
- The pipelined macro body parses through NVRTC.
- The lambdas inside `extern "C" __global__` are accepted (NVRTC's C++17
  mode allows them; if not, this step fails and we fall back to repeating
  the load loop bodies inline via a preprocessor expansion).
- `__cvta_generic_to_shared` and the inline PTX `cp.async` syntax are
  accepted at this NVRTC version.

Add a temporary smoke test to `crates/tropical-gemm-cuda/src/matmul_mod.rs`'s
`mod tests` block:

```rust
#[test]
fn nvrtc_compiles_pipelined_kernel() {
    use cudarc::driver::DevicePtr;
    let ctx = crate::get_global_context().expect("CUDA ctx");
    // Look up the kernel by literal `_pl` name. If NVRTC failed on this
    // name, get_kernel returns Err.
    let _k = ctx.get_kernel("tropical_matmul_f32_max_NN_pl")
        .expect("pipelined kernel must compile under NVRTC");
}
```

Run on the A100 allocation:
```
srun --jobid=6308637 --overlap bash -lc 'export PATH=$HOME/.cargo/bin:$PATH; module load cuda; cargo test --release -p tropical-gemm-cuda --lib nvrtc_compiles_pipelined_kernel 2>&1 | tail -8'
```

Expected: 1 passed.

If NVRTC errors with a specific message, fix according to the error
class:
- `error: identifier "cp" is undefined` (or similar) — the `__CUDA_ARCH__`
  guard in CP_ASYNC_CG_* didn't pick up sm_80; adjust the conditional.
- `error: lambda definition is not allowed in an extern "C" __global__` —
  rewrite the three lambdas as `#define INNER_LOAD_A(...)` macros that
  re-expand at each call site.
- `error: __cvta_generic_to_shared not found` — replace with PTX:
  `unsigned smem_addr; asm("cvta.to.shared.u32 %0, %1;" : "=r"(smem_addr) : "l"(__cvta_generic_to_shared(ptr)));`.

If NVRTC reports a tile-related shared-memory limit (>48 KiB without
opt-in), see "Shared-memory opt-in" appendix at the end of this plan.

- [ ] **Step 5: Commit (smoke test included; Task 5 will replace it with a stronger sync-vs-pl comparison)**

```bash
git add crates/tropical-gemm-cuda/kernels/counting_gemm.cu \
        crates/tropical-gemm-cuda/src/context.rs \
        crates/tropical-gemm-cuda/src/matmul_mod.rs
git commit -m "Spec P Task 3: emit 16 cp.async pipelined kernel specializations + NVRTC smoke test"
```

---

### Task 4: Runtime dispatch — pick `_pl` variant on sm_80+

**Files:**
- Modify: `crates/tropical-gemm-cuda/src/counting_kernel.rs`

- [ ] **Step 1: Read the current `launch_tropical_matmul` to confirm where the kernel name is built**

Run: `sed -n '60,80p' crates/tropical-gemm-cuda/src/counting_kernel.rs`
Expected: a block that builds a name like `tropical_matmul_f32_max_NN` from a base name + `(tA, tB)` suffix.

- [ ] **Step 2: Add a per-context compute-cap check**

A process-global `OnceLock` would be wrong: this crate exposes
`CudaContext::new_on_device(device_id)` and `from_device(device)`
(see `src/context.rs:97`), so a single process can hold multiple
contexts on different physical GPUs. The capability check must be
per-context.

Cheapest correct implementation: just call the cudarc attribute query
each launch — it's a cached driver-side read, ~hundreds of nanoseconds.
Insert at the top of `counting_kernel.rs` (after the existing `use`
block):

```rust
/// True iff the device backing `ctx` has compute capability ≥ 8.0
/// (Ampere+). The cp.async pipelined `_pl` kernels target sm_80+; on
/// older devices we route to the sync kernels.
fn prefer_pipelined(ctx: &crate::context::CudaContext) -> bool {
    use cudarc::driver::sys::CUdevice_attribute::*;
    ctx.device()
        .attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map(|major| major >= 8)
        .unwrap_or(false)
}
```

If `device().attribute(...)` is not the right API in this cudarc 0.12,
the alternate form is:

```rust
ctx.device().compute_capability().map(|(maj, _min)| maj >= 8).unwrap_or(false)
```

Pick whichever compiles; either is fine.

- [ ] **Step 3: Append `_pl` to the kernel name when pipelined is preferred**

In `launch_tropical_matmul`, locate the line that constructs `kernel_name_owned`:

```rust
let kernel_name_owned: String = format!(
    "{}_{}",
    <(T, D) as TropicalMatmulKernelName<T, D>>::BASE_NAME,
    suffix
);
```

Replace with:

```rust
let suffix_with_variant = if prefer_pipelined(ctx) {
    format!("{}_pl", suffix)
} else {
    suffix.to_string()
};
let kernel_name_owned: String = format!(
    "{}_{}",
    <(T, D) as TropicalMatmulKernelName<T, D>>::BASE_NAME,
    suffix_with_variant
);
```

The Box::leak path that follows (turning the String into `&'static str`) is unchanged — there are now 16 leaked names per process instead of 8 per (T,D) pair, still bounded.

- [ ] **Step 4: Build**

Run: `bash -lc 'export PATH=$HOME/.cargo/bin:$PATH; cargo build --release -p tropical-gemm-cuda 2>&1 | tail -3'`
Expected: success.

- [ ] **Step 5: Commit**

```bash
git add crates/tropical-gemm-cuda/src/counting_kernel.rs
git commit -m "Spec P Task 4: dispatch to _pl pipelined kernels on sm_80+"
```

---

### Task 5: Correctness gate — pipelined matches sync byte-for-byte

**Files:**
- Modify: `crates/tropical-gemm-cuda/src/matmul_mod.rs`

The Spec N tests compare a single dispatched kernel run against the CPU
reference. That doesn't prove the `_pl` kernel matches the sync kernel:
on pre-sm_80 the dispatcher silently falls back, and even on sm_80 a
broken dispatch could keep using the sync path.

This task adds **direct kernel-name comparisons** that bypass the
dispatcher: launch `tropical_matmul_f32_max_NN` (sync) and
`tropical_matmul_f32_max_NN_pl` (pipelined) on the same input and
require byte-equal output. On sm_<80 this test is skipped (the `_pl`
kernel is sm_80-specific PTX).

- [ ] **Step 1: Add a layout-agnostic named-kernel launch helper**

In the `mod tests` block of `matmul_mod.rs`, add:

```rust
fn launch_named_kernel_f32(
    kernel_name: &'static str,
    m: usize, k: usize, n: usize,
    a_dev: u64, b_dev: u64, p: i32, out_dev: u64,
) -> Result<(), String> {
    use cudarc::driver::LaunchAsync;
    use crate::counting_kernel::DevPtr;
    let ctx = crate::get_global_context().map_err(|e| format!("{e}"))?;
    let kernel = ctx.get_kernel(kernel_name).map_err(|e| format!("{e}"))?;
    // Same grid/block math the launcher uses for f32 (BM=BN=64, TM=TN=4)
    // — output tile shape and addressing are independent of (tA, tB).
    let block: (u32, u32, u32) = (16, 16, 1);
    let grid: (u32, u32, u32) = (
        ((n + 63) / 64) as u32,
        ((m + 63) / 64) as u32,
        1,
    );
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: grid, block_dim: block, shared_mem_bytes: 0,
    };
    let mu: u64 = if p > 1 { ((1u128 << 64) / p as u128) as u64 } else { 0 };
    ctx.device().synchronize().map_err(|e| format!("{e}"))?;
    unsafe {
        kernel.launch(cfg, (
            DevPtr(a_dev), DevPtr(b_dev), DevPtr(out_dev),
            m as i32, n as i32, k as i32, p, mu,
        )).map_err(|e| format!("{e}"))?;
    }
    ctx.device().synchronize().map_err(|e| format!("{e}"))?;
    Ok(())
}

fn cuda_arch_at_least_80() -> bool {
    use cudarc::driver::sys::CUdevice_attribute::*;
    crate::get_global_context().ok().and_then(|ctx| {
        ctx.device().attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR).ok()
    }).map(|m| m >= 8).unwrap_or(false)
}

/// Compare sync vs pipelined for a given f32_max layout and shape. The
/// caller supplies the (tA, tB)-encoded layout suffix once; we run both
/// `*_<sfx>` and `*_<sfx>_pl` on the SAME (a_dev, b_dev) inputs and
/// require byte-equal outputs.
///
/// Note: a_dev / b_dev hold the same packed bytes regardless of layout
/// (the kernel reinterprets them via the layout-aware A_OFF / B_OFF
/// macros), so a single random buffer of `m * k` (resp `k * n`) Pairs
/// works for every layout. M and N get swapped on the storage side
/// in the kernel under the 'T' flag, but the *test* doesn't care
/// because we're only checking sync == pipelined for that exact layout.
fn compare_sync_vs_pl_f32_max(layout_sfx: &'static str,
                               sync_name: &'static str,
                               pl_name:   &'static str,
                               m: usize, k: usize, n: usize, p: i32) {
    if !cuda_arch_at_least_80() {
        eprintln!("compare_sync_vs_pl[{layout_sfx}]: skipped on sm_<80");
        return;
    }
    use cudarc::driver::DevicePtr;
    let ctx = crate::get_global_context().expect("CUDA ctx");
    let a_host = rand_pairs_f32(m * k, p);
    let b_host = rand_pairs_f32(k * n, p);
    let a_dev = ctx.device().htod_copy(a_host).unwrap();
    let b_dev = ctx.device().htod_copy(b_host).unwrap();
    let out_sync = ctx.device().alloc_zeros::<crate::pair::PairF32>(m * n).unwrap();
    let out_pl   = ctx.device().alloc_zeros::<crate::pair::PairF32>(m * n).unwrap();

    launch_named_kernel_f32(
        sync_name, m, k, n,
        *a_dev.device_ptr(), *b_dev.device_ptr(), p, *out_sync.device_ptr(),
    ).expect("sync kernel launch");
    launch_named_kernel_f32(
        pl_name, m, k, n,
        *a_dev.device_ptr(), *b_dev.device_ptr(), p, *out_pl.device_ptr(),
    ).expect("pipelined kernel launch");

    let s = ctx.device().dtoh_sync_copy(&out_sync).unwrap();
    let p_out = ctx.device().dtoh_sync_copy(&out_pl).unwrap();
    for idx in 0..(m * n) {
        assert_eq!(s[idx].val, p_out[idx].val,
            "[{layout_sfx}] val mismatch at {idx} (m={m}, k={k}, n={n})");
        assert_eq!(s[idx].cnt, p_out[idx].cnt,
            "[{layout_sfx}] cnt mismatch at {idx} (m={m}, k={k}, n={n})");
    }
}
```

- [ ] **Step 2: Add the byte-equal test fixtures across all 4 layouts**

```rust
// Helper that fans out (m, k, n, p) across NN / NT / TN / TT.
fn check_all_layouts(m: usize, k: usize, n: usize, p: i32) {
    for (sfx, sync_n, pl_n) in [
        ("NN", "tropical_matmul_f32_max_NN", "tropical_matmul_f32_max_NN_pl"),
        ("NT", "tropical_matmul_f32_max_NT", "tropical_matmul_f32_max_NT_pl"),
        ("TN", "tropical_matmul_f32_max_TN", "tropical_matmul_f32_max_TN_pl"),
        ("TT", "tropical_matmul_f32_max_TT", "tropical_matmul_f32_max_TT_pl"),
    ] {
        compare_sync_vs_pl_f32_max(sfx, sync_n, pl_n, m, k, n, p);
    }
}

// Exact tile multiples — easiest case.
#[test] fn pl_matches_sync_64_64_64()  { check_all_layouts(64, 64, 64, 7); }
// 10 K-tiles (BK_pl = 16) → 9 pipeline handoffs.
#[test] fn pl_matches_sync_64_160_64() { check_all_layouts(64, 160, 64, 7); }
// Ragged K (last tile has < BK_pl=16 elements).
#[test] fn pl_matches_sync_64_33_64()  { check_all_layouts(64, 33, 64, 7); }
// Ragged M and N (sentinel-write path).
#[test] fn pl_matches_sync_65_65_65()  { check_all_layouts(65, 65, 65, 7); }
// Ragged across all three.
#[test] fn pl_matches_sync_100_33_77() { check_all_layouts(100, 33, 77, 11); }
```

- [ ] **Step 3: Drop the temporary smoke test from Task 3 if it's still there**

The Task 3 `nvrtc_compiles_pipelined_kernel` test is now subsumed by
the byte-equal tests (which also force NVRTC to compile the `_pl`
kernel). Delete it.

- [ ] **Step 4: Run all kernel tests on the A100 allocation**

Run:
```
srun --jobid=6308637 --overlap bash -lc 'export PATH=$HOME/.cargo/bin:$PATH; module load cuda; cargo test --release -p tropical-gemm-cuda 2>&1 | tail -15'
```

Expected: 70 sync tests (Spec N) + 5 new byte-equal tests (each fans
out across 4 layouts internally) = 75 passing test functions, 90 layout
comparisons. On sm_<80 the 5 new tests print "skipped" per layout and
pass trivially.

If any byte-equal test fails, the assert message tells you which layout
(NN / NT / TN / TT) and (M, K, N). Useful triage:
1. **Single layout fails, others pass** → bug is in the LOAD_*_DECOMP_*
   wiring for that layout in Task 3 (the macro chose the wrong
   fastest-axis decomposition).
2. **All 4 layouts fail at the ragged-K case** → bug is in the final-tile
   drain (`last_kk_end < BK_`).
3. **All 4 layouts fail at ragged-M/N** → bug is in the sentinel-write
   race; the `__syncthreads()` after `CP_ASYNC_WAIT_GROUP` should make
   both the cp.async writes and the direct sentinel stores visible.
4. **All 4 layouts fail at exact-multiples** → bug is in the prefetch /
   wait_group ordering (Task 2 macro structure).

- [ ] **Step 5: Commit**

```bash
git add crates/tropical-gemm-cuda/src/matmul_mod.rs
git commit -m "Spec P Task 5: byte-equal sync-vs-pipelined correctness tests"
```

---

### Task 6: A100 bench, record results, decide if win is real

**Files:**
- Modify: `CountingTropicalGEMM.jl/bench/RESULTS.md`

- [ ] **Step 1: Run the bench on the A100 allocation**

Run: `srun --jobid=6308637 --overlap bash -lc 'module load cuda; JULIA_CUDA_USE_COMPAT=false julia --project=CountingTropicalGEMM.jl CountingTropicalGEMM.jl/bench/bench_mul.jl' 2>&1 | tee /tmp/specp_bench.txt | tail -32`

Expected: 24-line table with shapes 128…4096 and flags NN/NT/TN/TT.

Record the 4096³ TT throughput. If it's ≥ 900 G/s, the pipeline is working as designed. If it's ≤ 770 G/s (within noise of the 763 baseline), the prefetch is not overlapping with compute — likely Task 2 has a logic bug, see "If the win doesn't materialize" below.

- [ ] **Step 2: Append a new section to `bench/RESULTS.md`**

Add this block after the existing "## A100-SXM4-80GB (Spec N tiled) — 2026-04-29" section:

```markdown
## A100-SXM4-80GB (Spec P pipelined) — 2026-04-29

cp.async double-buffered version of the Spec N kernel. f32 tile is
**BM=BN=64, BK=16, TM=TN=4** — BK is half the sync baseline (BK=32) to
fit the doubled (2-stage) shared memory under the A100 48 KiB
static-shared limit without requiring `cudaFuncSetAttribute` opt-in.
f64 tile unchanged from sync at BM=BN=32, BK=8, TM=2, TN=4.

Therefore the head-to-head comparison records pipelined@BK=16 vs
sync@BK=32 — the throughput question being answered is "does cp.async
pipelining at BK=16 beat synchronous loads at BK=32?". If the answer is
no, the appendix's dynamic-shared opt-in path lets us try
pipelined@BK=32 in a follow-on.

| Shape | flag | ms / call | G tropical-ops/s |
|---:|:---:|---:|---:|
<24 rows from /tmp/specp_bench.txt>

### Observations

- 4096³ TT pipelined@BK=16: <X> G/s, vs 763 G/s sync@BK=32 = <ratio>×.
- All four flags within ~5% (cp.async preserves the layout-aware
  coalescing from Spec N).
- RTX 6000 falls back to the sync kernel via runtime dispatch — its
  numbers are unchanged from the existing Spec N table.
- If the pipelined throughput < sync, the appendix outlines the dynamic
  shared opt-in path (Spec P.1) to lift the BK cap and re-bench.
```

Replace `<X>` and `<ratio>` with the measured values from /tmp/specp_bench.txt.

- [ ] **Step 3: Commit**

```bash
git add CountingTropicalGEMM.jl/bench/RESULTS.md
git commit -m "Spec P: record A100 bench for cp.async pipelined kernel"
```

---

## If the win doesn't materialize

If Task 6 shows ≤ 5% improvement, common causes ranked by likelihood:

1. **`cp.async` issued but not overlapping with compute.** The pipeline
   has only 1 stage in flight, so the wait_group blocks immediately.
   Check by inspecting nsys: the `cp.async.wait_group` should show as
   *not* blocking (i.e., the prior commit already completed).
2. **Smem padding `[BM_+1]` makes loader strides non-aligned for cp.async.4.**
   Try removing the +1 padding (revert to `[BM_]`) for the pipelined
   variant only; bank conflicts are second-order vs the cp.async win.
3. **NVRTC declined to use cp.async** — falls through to scalar copy.
   `cuobjdump --dump-sass <ptx>` should show `LDGSTS` instructions; if
   it shows `LDG` + `STS` instead, the asm volatile didn't take.
4. **3-stage pipeline needed.** A100 prefers 3 stages for non-tensor-core
   GEMM. Bump `NUM_STAGES = 3`, change `wait_group(1)` to `wait_group(2)`,
   re-bench.

Each is a small follow-up task — don't bundle into this plan unless 1+ is
hit. The plan is complete when Task 6 lands a measurable improvement
*or* the hit is documented and the design committed for follow-up.

---

## Appendix — Shared-memory opt-in (escalation if BK=16 pipelined < sync@BK=32)

A100 allows up to ~163 KiB of shared per block but only with **dynamic**
shared (`extern __shared__ char smem[];`) and a runtime opt-in via
`cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`.
Static `__shared__` declarations are capped at 48 KiB on Ampere by
default.

If Task 6 shows BK=16 pipelined ≤ BK=32 sync, escalate by:

1. Replacing the four static arrays with one `extern __shared__` slab
   partitioned into As_v / As_c / Bs_v / Bs_c at known offsets.
2. Bumping pipelined f32 BK from 16 to 32 (66 KiB total).
3. In `counting_kernel.rs`, after `ctx.get_kernel(kernel_name)`,
   call cudarc's equivalent of `cudaFuncSetAttribute` once per kernel.
4. Passing the size at launch via `LaunchConfig::shared_mem_bytes`.

That's a separate spec (Spec P.1) — only pursue if BK=16 doesn't pull on
its own. Most A100 GEMM-pattern wins come from cp.async pipelining with
modest tile depth, not from extreme BK.

## Self-review

- **Spec coverage.** The opening "Goal" and "Architecture" require: cp.async helpers (Task 1), pipelined body (Task 2), 16 expansions + name registry (Task 3), runtime dispatch (Task 4), correctness gate (Task 5), bench + RESULTS (Task 6). Every item maps to a task.
- **Placeholder scan.** The phrase "see Task N" appears once, in the "If the win doesn't materialize" section, with concrete diagnostics for each bullet. No "TBD"/"implement later"/"add validation" left in the steps.
- **Type/symbol consistency.** Kernel names use `_pl` suffix everywhere (the macro, the registry, the dispatch). The pipelined f32 tile is `64, 64, 16, 4, 4` (BK=16 to fit static shared on A100); the sync f32 tile remains `64, 64, 32, 4, 4`. The pipelined f64 tile is `32, 32, 8, 2, 4` matching the sync. `prefer_pipelined` and `BASE_NAME` are referenced consistently across Tasks 4 and 5.

## Codex-review revisions (vs. v1 of this plan)

1. **PairF64 16-byte loads** (codex finding 1): the load macros now
   dispatch on `sizeof(T)` via `if constexpr` so f64's `val` field
   moves via CP_ASYNC_CG_8 (8 B), and only the i32 `cnt` uses CG_4. The
   4-byte `_pad` slot is never touched.
2. **Cargo build is not a kernel-validation gate** (codex finding 2):
   Task 3 now adds an explicit NVRTC smoke step (look up kernel by
   name, fail loudly on compile error). Task 5's byte-equal tests are
   the canonical end-to-end validation.
3. **Per-context, not OnceLock-global, capability cache** (codex
   finding 3): `prefer_pipelined(ctx)` now queries the device every
   launch via cudarc's attribute call (cheap, cached driver-side).
4. **Force-`_pl`-vs-force-sync byte-equal tests** (codex finding 4):
   Task 5 was rewritten to bypass the dispatcher and look up both
   `tropical_matmul_f32_max_NN` and `tropical_matmul_f32_max_NN_pl` by
   literal name, run on identical inputs, and require exact equality.
   On sm_<80 these tests skip cleanly.
5. **A100 static-shared 48 KiB cap** (uncovered while addressing #4):
   pipelined f32 starts at BK=16 (32 KiB shared) instead of BK=32
   (~65 KiB, would need dynamic-shared opt-in). Appendix documents the
   escalation path.

## Codex-review revisions round 2

6. **Task 6 bench section had stale BK=32 geometry** — corrected to
   BK=16 with explicit note that the head-to-head is pipelined@BK=16
   vs sync@BK=32.
7. **Task 5 only covered the NN layout** — refactored into a
   `check_all_layouts(m,k,n,p)` fan-out so each of the 5 ragged/edge
   shapes runs against NN, NT, TN, and TT (20 layout comparisons
   total). Failure messages tag the layout for fast triage.
8. **File map said "one extra inline test"** — updated to reflect the
   helper + 5 multi-layout tests.
