# Spec D — Tiled counting CUDA kernel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the naive one-thread-per-cell `counting_gemm.cu` with a BLIS-style tiled kernel, matching the existing `tropical_gemm.cu` optimization tier.

**Architecture:** Two-level blocking (shared-memory block tile + 4×4 register thread tile), `__restrict__` pointers, cooperative tile loads with predicated bounds checks only at load time. Two parallel tile pairs per block: one for `T` values and one for `int` count residues. One macro instantiated 4× for `{f32, f64} × {Max, Min}`. Same kernel names, same API, same 6-pointer signature.

**Tech Stack:** CUDA via NVRTC, Rust `cudarc` 0.12.

**Spec:** `docs/superpowers/specs/2026-04-21-counting-kernel-tiled-design.md`

---

## Preconditions

- Branch `counting-tropical`, latest commit `77bce95` (spec D document).
- `. ~/.cargo/env && module load cuda` at shell start.
- Baseline: all 6 `counting_gpu` integration tests pass; 49 CUDA lib tests pass; 304 CPU lib tests pass.

## File map

- **Rewrite:** `crates/tropical-gemm-cuda/kernels/counting_gemm.cu` — tiled macro replacing the naive one.
- **Modify:** `crates/tropical-gemm-cuda/src/context.rs` — split `counting_block_dims` / `counting_grid_dims` into `_f32` / `_f64` variants.
- **Modify:** `crates/tropical-gemm-cuda/src/counting_kernel.rs` — trait method `launch_dims(m, n) -> (grid, block)` with per-impl selection.
- **Modify:** `crates/tropical-gemm-cuda/tests/counting_gpu.rs` — add 3 new tests (large, off-boundary, f64 medium).

---

## Phase 1 — Tiled kernel source

### Task 1: Rewrite `counting_gemm.cu` with tiled macro

**File:** `crates/tropical-gemm-cuda/kernels/counting_gemm.cu` (rewrite).

- [ ] **Step 1: Replace the file with the tiled kernel**

Replace the entire contents of `crates/tropical-gemm-cuda/kernels/counting_gemm.cu` with:

```c
// Counting Tropical GEMM CUDA Kernels (spec D tiled version).
//
// SoA element layout: each logical element = (value: T, count: int32).
// Matrices are passed as parallel value and count pointers, row-major.
//
// Two-level blocking (BLIS style) mirroring tropical_gemm.cu: a block tile
// loaded into shared memory, a 4x4 thread tile kept in registers. Two
// parallel tiles per block (value + count).
//
// Semiring operations (direction D):
//   tropical_mul: (va, ca) * (vb, cb) = (va + vb, (ca * cb) mod P)
//   tropical_add: strictly-better value wins; on tie, counts add mod P.

// Infinity sentinels for MaxPlus / MinPlus tropical zero.
#define NEG_INF_F32 __int_as_float(0xff800000)
#define POS_INF_F32 __int_as_float(0x7f800000)
#define NEG_INF_F64 __longlong_as_double(0xfff0000000000000LL)
#define POS_INF_F64 __longlong_as_double(0x7ff0000000000000LL)

// Row-major offset.
#define OFFSET_ROW(row, col, ncols) ((row) * (ncols) + (col))

#define MAX_BETTER(a, b) ((a) > (b))
#define MIN_BETTER(a, b) ((a) < (b))

// ============================================================================
// F32 COUNTING KERNEL MACRO
// ============================================================================
// Block sizes: 64x32x64, Thread sizes: 4x4. Threads per block: 256 (16x16).
// Shared memory per block: 4 * 64 * 32 bytes value + 4 * 64 * 32 bytes count
//                        = 8 KB + 8 KB (As pair) + 8 KB + 8 KB (Bs pair) = 32 KB.

#define COUNTING_GEMM_F32(NAME, INIT_VAL, BETTER)                              \
extern "C" __global__ void NAME(                                               \
    const float* __restrict__ value_a, const int* __restrict__ count_a,        \
    const float* __restrict__ value_b, const int* __restrict__ count_b,        \
    float* __restrict__ value_c, int* __restrict__ count_c,                    \
    int M, int N, int K, int P                                                 \
) {                                                                            \
    const int BLOCK_SIZE_M = 64;                                               \
    const int BLOCK_SIZE_K = 32;                                               \
    const int BLOCK_SIZE_N = 64;                                               \
    const int THREAD_SIZE_M = 4;                                               \
    const int THREAD_SIZE_N = 4;                                               \
                                                                               \
    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;                             \
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;                             \
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;                              \
                                                                               \
    /* Block covers rows [BLOCK_IDY*BLOCK_SIZE_M, +BLOCK_SIZE_M)               \
       and cols [BLOCK_IDX*BLOCK_SIZE_N, +BLOCK_SIZE_N) of C. */               \
    int BLOCK_IDY = blockIdx.y;  /* row-block index */                         \
    int BLOCK_IDX = blockIdx.x;  /* col-block index */                         \
                                                                               \
    const int tid = threadIdx.y * bszm + threadIdx.x;                          \
                                                                               \
    __shared__ float As_val[BLOCK_SIZE_M * BLOCK_SIZE_K];                      \
    __shared__ int   As_cnt[BLOCK_SIZE_M * BLOCK_SIZE_K];                      \
    __shared__ float Bs_val[BLOCK_SIZE_K * BLOCK_SIZE_N];                      \
    __shared__ int   Bs_cnt[BLOCK_SIZE_K * BLOCK_SIZE_N];                      \
                                                                               \
    float val_accum[THREAD_SIZE_M * THREAD_SIZE_N];                            \
    int   cnt_accum[THREAD_SIZE_M * THREAD_SIZE_N];                            \
    float regs_a_val[THREAD_SIZE_M];                                           \
    int   regs_a_cnt[THREAD_SIZE_M];                                           \
    float regs_b_val[THREAD_SIZE_N];                                           \
    int   regs_b_cnt[THREAD_SIZE_N];                                           \
                                                                               \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {                  \
        val_accum[i] = INIT_VAL;                                               \
        cnt_accum[i] = 0;                                                      \
    }                                                                          \
                                                                               \
    /* Tile-load mapping: each of 256 threads loads multiple elements.         \
       A tile is BLOCK_SIZE_M x BLOCK_SIZE_K = 64x32 (row-major).              \
       B tile is BLOCK_SIZE_K x BLOCK_SIZE_N = 32x64. */                       \
    const int A_TILE_ROW = tid / BLOCK_SIZE_K;                                 \
    const int A_TILE_COL = tid % BLOCK_SIZE_K;                                 \
    const int B_TILE_ROW = tid / BLOCK_SIZE_N;                                 \
    const int B_TILE_COL = tid % BLOCK_SIZE_N;                                 \
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;         \
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_N;         \
                                                                               \
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {           \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {            \
            int row = BLOCK_SIZE_M * BLOCK_IDY + A_TILE_ROW + i;               \
            int col = tile_idx + A_TILE_COL;                                   \
            float v = INIT_VAL;                                                \
            int   c = 0;                                                       \
            if (row < M && col < K) {                                          \
                v = value_a[OFFSET_ROW(row, col, K)];                          \
                c = count_a[OFFSET_ROW(row, col, K)];                          \
            }                                                                  \
            int s_idx = OFFSET_ROW(A_TILE_ROW + i, A_TILE_COL, BLOCK_SIZE_K);  \
            As_val[s_idx] = v;                                                 \
            As_cnt[s_idx] = c;                                                 \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {            \
            int row = tile_idx + B_TILE_ROW + i;                               \
            int col = BLOCK_SIZE_N * BLOCK_IDX + B_TILE_COL;                   \
            float v = INIT_VAL;                                                \
            int   c = 0;                                                       \
            if (row < K && col < N) {                                          \
                v = value_b[OFFSET_ROW(row, col, N)];                          \
                c = count_b[OFFSET_ROW(row, col, N)];                          \
            }                                                                  \
            int s_idx = OFFSET_ROW(B_TILE_ROW + i, B_TILE_COL, BLOCK_SIZE_N);  \
            Bs_val[s_idx] = v;                                                 \
            Bs_cnt[s_idx] = c;                                                 \
        }                                                                      \
                                                                               \
        __syncthreads();                                                       \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {                               \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                int r = threadIdx.y * THREAD_SIZE_M + tm;                      \
                regs_a_val[tm] = As_val[OFFSET_ROW(r, k, BLOCK_SIZE_K)];       \
                regs_a_cnt[tm] = As_cnt[OFFSET_ROW(r, k, BLOCK_SIZE_K)];       \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                       \
                int c = threadIdx.x * THREAD_SIZE_N + tn;                      \
                regs_b_val[tn] = Bs_val[OFFSET_ROW(k, c, BLOCK_SIZE_N)];       \
                regs_b_cnt[tn] = Bs_cnt[OFFSET_ROW(k, c, BLOCK_SIZE_N)];       \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                _Pragma("unroll")                                              \
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                   \
                    int idx = tm * THREAD_SIZE_N + tn;                         \
                    float pv = regs_a_val[tm] + regs_b_val[tn];                \
                    int   pc = (int)(((long long)regs_a_cnt[tm]                \
                                     * (long long)regs_b_cnt[tn])              \
                                     % (long long)P);                          \
                    if (BETTER(pv, val_accum[idx])) {                          \
                        val_accum[idx] = pv;                                   \
                        cnt_accum[idx] = pc;                                   \
                    } else if (BETTER(val_accum[idx], pv)) {                   \
                        /* keep */                                             \
                    } else {                                                   \
                        cnt_accum[idx] = (int)(((long long)cnt_accum[idx]      \
                                                + (long long)pc)               \
                                                % (long long)P);               \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
    /* Write the 4x4 tile per thread. */                                       \
    _Pragma("unroll")                                                          \
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                               \
        _Pragma("unroll")                                                      \
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                           \
            int row = BLOCK_SIZE_M * BLOCK_IDY + threadIdx.y * THREAD_SIZE_M + tm; \
            int col = BLOCK_SIZE_N * BLOCK_IDX + threadIdx.x * THREAD_SIZE_N + tn; \
            if (row < M && col < N) {                                          \
                int idx = tm * THREAD_SIZE_N + tn;                             \
                value_c[OFFSET_ROW(row, col, N)] = val_accum[idx];             \
                count_c[OFFSET_ROW(row, col, N)] = cnt_accum[idx];             \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// ============================================================================
// F64 COUNTING KERNEL MACRO
// ============================================================================
// Block sizes: 32x16x32, Thread sizes: 4x4. Threads per block: 64 (8x8).
// Shared memory: 2 * (32 * 16 * 8 + 32 * 16 * 4) = 2 * (4KB + 2KB) * 2 = 24 KB.

#define COUNTING_GEMM_F64(NAME, INIT_VAL, BETTER)                              \
extern "C" __global__ void NAME(                                               \
    const double* __restrict__ value_a, const int* __restrict__ count_a,       \
    const double* __restrict__ value_b, const int* __restrict__ count_b,       \
    double* __restrict__ value_c, int* __restrict__ count_c,                   \
    int M, int N, int K, int P                                                 \
) {                                                                            \
    const int BLOCK_SIZE_M = 32;                                               \
    const int BLOCK_SIZE_K = 16;                                               \
    const int BLOCK_SIZE_N = 32;                                               \
    const int THREAD_SIZE_M = 4;                                               \
    const int THREAD_SIZE_N = 4;                                               \
                                                                               \
    const int bszm = BLOCK_SIZE_M / THREAD_SIZE_M;                             \
    const int bszn = BLOCK_SIZE_N / THREAD_SIZE_N;                             \
    const int THREAD_NUM_PER_BLOCK = bszm * bszn;                              \
                                                                               \
    int BLOCK_IDY = blockIdx.y;                                                \
    int BLOCK_IDX = blockIdx.x;                                                \
                                                                               \
    const int tid = threadIdx.y * bszm + threadIdx.x;                          \
                                                                               \
    __shared__ double As_val[BLOCK_SIZE_M * BLOCK_SIZE_K];                     \
    __shared__ int    As_cnt[BLOCK_SIZE_M * BLOCK_SIZE_K];                     \
    __shared__ double Bs_val[BLOCK_SIZE_K * BLOCK_SIZE_N];                     \
    __shared__ int    Bs_cnt[BLOCK_SIZE_K * BLOCK_SIZE_N];                     \
                                                                               \
    double val_accum[THREAD_SIZE_M * THREAD_SIZE_N];                           \
    int    cnt_accum[THREAD_SIZE_M * THREAD_SIZE_N];                           \
    double regs_a_val[THREAD_SIZE_M];                                          \
    int    regs_a_cnt[THREAD_SIZE_M];                                          \
    double regs_b_val[THREAD_SIZE_N];                                          \
    int    regs_b_cnt[THREAD_SIZE_N];                                          \
                                                                               \
    _Pragma("unroll")                                                          \
    for (int i = 0; i < THREAD_SIZE_M * THREAD_SIZE_N; ++i) {                  \
        val_accum[i] = INIT_VAL;                                               \
        cnt_accum[i] = 0;                                                      \
    }                                                                          \
                                                                               \
    const int A_TILE_ROW = tid / BLOCK_SIZE_K;                                 \
    const int A_TILE_COL = tid % BLOCK_SIZE_K;                                 \
    const int B_TILE_ROW = tid / BLOCK_SIZE_N;                                 \
    const int B_TILE_COL = tid % BLOCK_SIZE_N;                                 \
    const int A_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_K;         \
    const int B_TILE_ROW_STRIDE = THREAD_NUM_PER_BLOCK / BLOCK_SIZE_N;         \
                                                                               \
    for (int tile_idx = 0; tile_idx < K; tile_idx += BLOCK_SIZE_K) {           \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_M; i += A_TILE_ROW_STRIDE) {            \
            int row = BLOCK_SIZE_M * BLOCK_IDY + A_TILE_ROW + i;               \
            int col = tile_idx + A_TILE_COL;                                   \
            double v = INIT_VAL;                                               \
            int    c = 0;                                                      \
            if (row < M && col < K) {                                          \
                v = value_a[OFFSET_ROW(row, col, K)];                          \
                c = count_a[OFFSET_ROW(row, col, K)];                          \
            }                                                                  \
            int s_idx = OFFSET_ROW(A_TILE_ROW + i, A_TILE_COL, BLOCK_SIZE_K);  \
            As_val[s_idx] = v;                                                 \
            As_cnt[s_idx] = c;                                                 \
        }                                                                      \
        _Pragma("unroll")                                                      \
        for (int i = 0; i < BLOCK_SIZE_K; i += B_TILE_ROW_STRIDE) {            \
            int row = tile_idx + B_TILE_ROW + i;                               \
            int col = BLOCK_SIZE_N * BLOCK_IDX + B_TILE_COL;                   \
            double v = INIT_VAL;                                               \
            int    c = 0;                                                      \
            if (row < K && col < N) {                                          \
                v = value_b[OFFSET_ROW(row, col, N)];                          \
                c = count_b[OFFSET_ROW(row, col, N)];                          \
            }                                                                  \
            int s_idx = OFFSET_ROW(B_TILE_ROW + i, B_TILE_COL, BLOCK_SIZE_N);  \
            Bs_val[s_idx] = v;                                                 \
            Bs_cnt[s_idx] = c;                                                 \
        }                                                                      \
                                                                               \
        __syncthreads();                                                       \
                                                                               \
        _Pragma("unroll")                                                      \
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {                               \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                int r = threadIdx.y * THREAD_SIZE_M + tm;                      \
                regs_a_val[tm] = As_val[OFFSET_ROW(r, k, BLOCK_SIZE_K)];       \
                regs_a_cnt[tm] = As_cnt[OFFSET_ROW(r, k, BLOCK_SIZE_K)];       \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                       \
                int c = threadIdx.x * THREAD_SIZE_N + tn;                      \
                regs_b_val[tn] = Bs_val[OFFSET_ROW(k, c, BLOCK_SIZE_N)];       \
                regs_b_cnt[tn] = Bs_cnt[OFFSET_ROW(k, c, BLOCK_SIZE_N)];       \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                _Pragma("unroll")                                              \
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                   \
                    int idx = tm * THREAD_SIZE_N + tn;                         \
                    double pv = regs_a_val[tm] + regs_b_val[tn];               \
                    int    pc = (int)(((long long)regs_a_cnt[tm]               \
                                      * (long long)regs_b_cnt[tn])             \
                                      % (long long)P);                         \
                    if (BETTER(pv, val_accum[idx])) {                          \
                        val_accum[idx] = pv;                                   \
                        cnt_accum[idx] = pc;                                   \
                    } else if (BETTER(val_accum[idx], pv)) {                   \
                        /* keep */                                             \
                    } else {                                                   \
                        cnt_accum[idx] = (int)(((long long)cnt_accum[idx]      \
                                                + (long long)pc)               \
                                                % (long long)P);               \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
    _Pragma("unroll")                                                          \
    for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                               \
        _Pragma("unroll")                                                      \
        for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                           \
            int row = BLOCK_SIZE_M * BLOCK_IDY + threadIdx.y * THREAD_SIZE_M + tm; \
            int col = BLOCK_SIZE_N * BLOCK_IDX + threadIdx.x * THREAD_SIZE_N + tn; \
            if (row < M && col < N) {                                          \
                int idx = tm * THREAD_SIZE_N + tn;                             \
                value_c[OFFSET_ROW(row, col, N)] = val_accum[idx];             \
                count_c[OFFSET_ROW(row, col, N)] = cnt_accum[idx];             \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// ============================================================================
// INSTANTIATIONS
// ============================================================================

COUNTING_GEMM_F32(counting_gemm_f32_max, NEG_INF_F32, MAX_BETTER)
COUNTING_GEMM_F32(counting_gemm_f32_min, POS_INF_F32, MIN_BETTER)
COUNTING_GEMM_F64(counting_gemm_f64_max, NEG_INF_F64, MAX_BETTER)
COUNTING_GEMM_F64(counting_gemm_f64_min, POS_INF_F64, MIN_BETTER)
```

- [ ] **Step 2: Optional `nvcc` syntax check**

```
module load cuda && nvcc --ptx /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm/crates/tropical-gemm-cuda/kernels/counting_gemm.cu -o /tmp/cgemm.ptx 2>&1 | tail -15
```

Expected: clean compile. If `nvcc` isn't available, skip — Task 2 will catch syntax issues via NVRTC.

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm-cuda/kernels/counting_gemm.cu
git commit -m "$(cat <<'EOF'
Rewrite counting_gemm.cu as BLIS-style tiled kernel (spec D)

Two-level blocking mirroring tropical_gemm.cu: shared-memory block
tile (64x32x64 for f32, 32x16x32 for f64) with a 4x4 register tile
per thread. Four parallel tiles per block (A value, A count, B value,
B count). Four kernel specializations via macro: f32/f64 × Max/Min.

Replaces the naive one-thread-per-cell kernel. Same kernel names, same
API, same signature — internal change only.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2 — Update host-side grid/block helpers and launch dispatch

### Task 2: Split context helpers into `_f32` / `_f64` variants

**File:** `crates/tropical-gemm-cuda/src/context.rs`.

- [ ] **Step 1: Replace the two `counting_*` helpers**

Find the existing helpers in `crates/tropical-gemm-cuda/src/context.rs` (added in spec C). They look like:

```rust
pub fn counting_block_dims() -> (u32, u32, u32) {
    (16, 16, 1)
}
pub fn counting_grid_dims(m: usize, n: usize) -> (u32, u32, u32) {
    let (bx, by, _) = Self::counting_block_dims();
    let gx = ((n as u32) + bx - 1) / bx;
    let gy = ((m as u32) + by - 1) / by;
    (gx, gy, 1)
}
```

Replace them with:

```rust
/// f32 counting kernel block dims (16 × 16 = 256 threads).
pub fn counting_block_dims_f32() -> (u32, u32, u32) {
    (16, 16, 1)
}

/// f32 counting kernel grid dims.
/// Matches BLOCK_SIZE_M = 64 (rows) and BLOCK_SIZE_N = 64 (cols).
pub fn counting_grid_dims_f32(m: usize, n: usize) -> (u32, u32, u32) {
    const BLOCK_M: u32 = 64;
    const BLOCK_N: u32 = 64;
    let gx = ((n as u32) + BLOCK_N - 1) / BLOCK_N;
    let gy = ((m as u32) + BLOCK_M - 1) / BLOCK_M;
    (gx, gy, 1)
}

/// f64 counting kernel block dims (8 × 8 = 64 threads).
pub fn counting_block_dims_f64() -> (u32, u32, u32) {
    (8, 8, 1)
}

/// f64 counting kernel grid dims.
/// Matches BLOCK_SIZE_M = 32 (rows) and BLOCK_SIZE_N = 32 (cols).
pub fn counting_grid_dims_f64(m: usize, n: usize) -> (u32, u32, u32) {
    const BLOCK_M: u32 = 32;
    const BLOCK_N: u32 = 32;
    let gx = ((n as u32) + BLOCK_N - 1) / BLOCK_N;
    let gy = ((m as u32) + BLOCK_M - 1) / BLOCK_M;
    (gx, gy, 1)
}
```

Rationale for the thread layout: the tiled kernel uses `threadIdx.y * THREAD_SIZE_M + tm` for rows and `threadIdx.x * THREAD_SIZE_N + tn` for cols. So `blockDim.x = BLOCK_SIZE_N / THREAD_SIZE_N = 16` and `blockDim.y = BLOCK_SIZE_M / THREAD_SIZE_M = 16` for f32; 8 × 8 for f64.

- [ ] **Step 2: Build to confirm no callers still reference the old names**

```
. ~/.cargo/env && module load cuda && cargo build -p tropical-gemm-cuda 2>&1 | tail -10
```

Expected: compilation error pointing to `src/counting_kernel.rs` still calling `CudaContext::counting_grid_dims` / `counting_block_dims` without the `_f32` suffix. That's fixed in Task 3.

- [ ] **Step 3: Do not commit yet — Task 3 must land together.**

---

### Task 3: Update `counting_kernel.rs` to dispatch grid/block by `T`

**File:** `crates/tropical-gemm-cuda/src/counting_kernel.rs`.

- [ ] **Step 1: Extend the trait with a `launch_dims` method**

Replace the existing `trait CountingCudaKernel<T, D>` and its 4 impls with:

```rust
pub trait CountingCudaKernel<T, D>
where
    T: DeviceRepr + ValidAsZeroBits + Default + Clone + Copy + 'static,
    D: TropicalDirection,
{
    const KERNEL_NAME: &'static str;

    /// Returns (grid_dim, block_dim) for a launch covering `m × n` output cells.
    fn launch_dims(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32));

    fn launch_counting_gemm(
        ctx: &CudaContext,
        value_a: &GpuMatrix<T>,
        count_a: &GpuMatrix<i32>,
        value_b: &GpuMatrix<T>,
        count_b: &GpuMatrix<i32>,
        value_c: &mut GpuMatrix<T>,
        count_c: &mut GpuMatrix<i32>,
        modulus: i32,
    ) -> Result<()> {
        let m = value_a.rows();
        let k = value_a.cols();
        let n = value_b.cols();

        assert_eq!(count_a.rows(), m);
        assert_eq!(count_a.cols(), k);
        assert_eq!(value_b.rows(), k);
        assert_eq!(count_b.rows(), k);
        assert_eq!(count_b.cols(), n);
        assert_eq!(value_c.rows(), m);
        assert_eq!(value_c.cols(), n);
        assert_eq!(count_c.rows(), m);
        assert_eq!(count_c.cols(), n);

        let kernel = ctx.get_kernel(Self::KERNEL_NAME)?;
        let (grid_dim, block_dim) = Self::launch_dims(m, n);
        let cfg = LaunchConfig {
            grid_dim,
            block_dim,
            shared_mem_bytes: 0,
        };

        unsafe {
            kernel.launch(
                cfg,
                (
                    value_a.as_slice(),
                    count_a.as_slice(),
                    value_b.as_slice(),
                    count_b.as_slice(),
                    value_c.as_slice_mut(),
                    count_c.as_slice_mut(),
                    m as i32,
                    n as i32,
                    k as i32,
                    modulus,
                ),
            )?;
        }

        ctx.device().synchronize()?;
        Ok(())
    }
}

impl CountingCudaKernel<f32, Max> for (f32, Max) {
    const KERNEL_NAME: &'static str = "counting_gemm_f32_max";
    fn launch_dims(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
        (CudaContext::counting_grid_dims_f32(m, n), CudaContext::counting_block_dims_f32())
    }
}
impl CountingCudaKernel<f32, Min> for (f32, Min) {
    const KERNEL_NAME: &'static str = "counting_gemm_f32_min";
    fn launch_dims(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
        (CudaContext::counting_grid_dims_f32(m, n), CudaContext::counting_block_dims_f32())
    }
}
impl CountingCudaKernel<f64, Max> for (f64, Max) {
    const KERNEL_NAME: &'static str = "counting_gemm_f64_max";
    fn launch_dims(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
        (CudaContext::counting_grid_dims_f64(m, n), CudaContext::counting_block_dims_f64())
    }
}
impl CountingCudaKernel<f64, Min> for (f64, Min) {
    const KERNEL_NAME: &'static str = "counting_gemm_f64_min";
    fn launch_dims(m: usize, n: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
        (CudaContext::counting_grid_dims_f64(m, n), CudaContext::counting_block_dims_f64())
    }
}
```

(Keep the existing `launch_counting_gemm` free function that delegates to the trait — the body doesn't change.)

- [ ] **Step 2: Verify build**

```
. ~/.cargo/env && module load cuda && cargo build -p tropical-gemm-cuda 2>&1 | tail -6
```

Expected: clean build.

- [ ] **Step 3: Run the existing counting_gpu tests — these are the correctness gate**

```
. ~/.cargo/env && module load cuda && cargo test -p tropical-gemm-cuda --test counting_gpu 2>&1 | tail -12
```

Expected: all 6 existing tests pass. If any fail, debug — most likely causes:
- **Row/col swap in kernel**: the asymmetric layout test will flag this.
- **Off-by-one in tile bounds**: the all-ties test at size 2×13×2 exercises small tiles.
- **Grid dim miscount**: the 8×16×8 medium test would produce wrong answers for cells in blocks 2+.

Common debug path: print a single 1×1 or 2×2 result, compare to CPU; if 1×1 works but 2×2 doesn't, the tile coordinate math is wrong. The plan's kernel code is reviewed but any transcription typo is possible — read the macro carefully.

- [ ] **Step 4: Run the full CUDA lib test suite to catch regressions**

```
. ~/.cargo/env && module load cuda && cargo test -p tropical-gemm-cuda --lib 2>&1 | tail -5
```

Expected: 49 tests pass (unchanged — the tiled kernel doesn't touch anything the lib tests use directly).

- [ ] **Step 5: Commit Tasks 2 + 3 together**

```bash
git add crates/tropical-gemm-cuda/src/context.rs crates/tropical-gemm-cuda/src/counting_kernel.rs
git commit -m "$(cat <<'EOF'
Wire tiled counting kernel grid/block dims per T

Split counting_block_dims / counting_grid_dims into _f32 and _f64
variants matching the tile sizes of the new tiled kernel (64x64 for
f32, 32x32 for f64). Add a launch_dims associated method on
CountingCudaKernel so the launch wrapper doesn't branch on T
internally; each of the 4 impls returns the right helper pair.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3 — New tests

### Task 4: Add large, off-boundary, and f64 medium tests

**File:** `crates/tropical-gemm-cuda/tests/counting_gpu.rs`.

- [ ] **Step 1: Append the three tests to the end of the file**

```rust
/// Large shape. Exercises the tile loop across many block iterations.
#[test]
fn gpu_large_shape_f32() {
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (512, 512, 512);
    let a = random_ish_matrix(m, k, 0xdead);
    let b = random_ish_matrix(k, n, 0xbeef);
    let bound = bound_for_single_matmul(k);

    let cpu = count_ground_states::<f32, Max>(&a, m, k, &b, n, &bound);
    let gpu = count_ground_states_gpu::<f32, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();

    assert_eq!(gpu.values, cpu.values);
    assert_eq!(gpu.counts, cpu.counts);
}

/// Off-block-boundary shape. Every dim is prime / not a multiple of block
/// size, stressing the predicated tile-load bounds checks for all edges.
#[test]
fn gpu_off_boundary_shape() {
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (17, 19, 23);
    let a = random_ish_matrix(m, k, 0x1111);
    let b = random_ish_matrix(k, n, 0x2222);
    let bound = bound_for_single_matmul(k);

    let cpu = count_ground_states::<f32, Max>(&a, m, k, &b, n, &bound);
    let gpu = count_ground_states_gpu::<f32, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();

    assert_eq!(gpu.values, cpu.values);
    assert_eq!(gpu.counts, cpu.counts);
}

/// f64 medium shape. Exercises the f64 tiled macro specifically.
#[test]
fn gpu_f64_medium_shape() {
    let ctx = CudaContext::new().unwrap();
    let (m, k, n) = (128, 128, 128);
    let mut state = 0xcafef00du64;
    let mut gen = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((state >> 33) as u32 % 5) as f64
    };
    let a: Vec<f64> = (0..m * k).map(|_| gen()).collect();
    let b: Vec<f64> = (0..k * n).map(|_| gen()).collect();
    let bound = bound_for_single_matmul(k);

    let cpu = count_ground_states::<f64, Max>(&a, m, k, &b, n, &bound);
    let gpu = count_ground_states_gpu::<f64, Max>(&ctx, &a, m, k, &b, n, &bound).unwrap();

    assert_eq!(gpu.values, cpu.values);
    assert_eq!(gpu.counts, cpu.counts);
}
```

- [ ] **Step 2: Run all counting_gpu tests**

```
. ~/.cargo/env && module load cuda && cargo test -p tropical-gemm-cuda --test counting_gpu 2>&1 | tail -15
```

Expected: 9 tests pass (6 original + 3 new). The CPU comparison takes real time for 512×512 and 128×128 f64 (the CPU driver runs 1 matmul per prime × multiple primes × O(mnk)); expect a few seconds per test.

If the large test fails but small/off-boundary pass, the tile loop probably has an issue around multiple tile iterations — print a 128×128 subset vs CPU to isolate.

- [ ] **Step 3: Commit**

```bash
git add crates/tropical-gemm-cuda/tests/counting_gpu.rs
git commit -m "$(cat <<'EOF'
Add large, off-boundary, and f64 medium tests for tiled kernel

- 512x512x512 f32 Max: exercises the tile loop over many iterations.
- 17x19x23 f32 Max: every dim prime, stresses predicated tile-load
  bounds checks at every edge simultaneously.
- 128x128x128 f64 Max: validates the f64 macro on a non-trivial shape.

All three cross-check against the CPU count_ground_states oracle.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4 — Final regression gate

### Task 5: Run the full workspace suite and report numbers

- [ ] **Step 1: Run the full suite**

```
. ~/.cargo/env && module load cuda && module load python
cargo test -p tropical-gemm --features testing 2>&1 | grep "test result" | head -5
echo CUDA:
cargo test -p tropical-gemm-cuda 2>&1 | grep "test result" | head -5
echo PYTHON:
cargo build -p tropical-gemm-python --features cuda 2>&1 | tail -3
```

Expected tallies:
- tropical-gemm lib (`--features testing`): 304 / 0.
- tropical-gemm counting_compose: 5 / 0.
- tropical-gemm counting_crt: 6 / 0.
- tropical-gemm doctests: 26 / 0.
- tropical-gemm-cuda lib: 49 / 0.
- tropical-gemm-cuda counting_gpu: 9 / 0.
- tropical-gemm-python cuda build: clean.

- [ ] **Step 2: No commit unless fixes.**

If anything regresses, investigate. The most common cause is a kernel index typo caught by the large or off-boundary test.

---

## Out of scope for this plan

- Further GPU perf work (warp-level reductions, Tensor Cores, async copies, `cuBLAS`-style `ldc` stride passing). The goal is parity with the existing tropical GEMM kernels, not beating them.
- Benchmark harness (`cargo bench`). Mentioned in spec as optional; skipping unless the implementer has time at the end.
- MinPlus specialization beyond what the macro gives. f32/f64 × Max/Min is the full grid.
- Integer `T`, argmax, chained matmul — separate specs.

## Self-review notes

- **Spec coverage.** Kernel tile layout (spec §Tile parameters / Shared memory / Per-thread accumulators / Inner loop / Padding / Row-major) → Task 1. Grid/block helpers (spec §Grid / block dims) → Task 2. Launch dispatch per T (spec §Grid / block dims trailing paragraph) → Task 3. Testing (spec §Testing) → Task 4. File impact (spec §File impact) covered by all four tasks.
- **Placeholder scan.** No TBDs. Kernel source is the full code, not fragmentary. All commands have expected output. Test code is complete.
- **Type consistency.** `CountingCudaKernel<T, D>` trait gains `launch_dims`; the free `launch_counting_gemm` function still delegates. `counting_block_dims_f32` / `_f64` and `counting_grid_dims_f32` / `_f64` named consistently. Kernel names unchanged from spec C (`counting_gemm_{f32,f64}_{max,min}`).
- **Risks.** Kernel code transcription is the primary risk; mitigated by running all 6 pre-existing cross-check tests (including the asymmetric layout one) before adding the 3 new tests. If the macro has a bug, Task 3 Step 3 catches it before Task 4 extends scope.
