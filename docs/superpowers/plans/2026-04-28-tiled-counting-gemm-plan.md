# Spec N: Tiled Counting Tropical GEMM — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the naive Spec M counting tropical GEMM kernels with shared-memory tiled kernels that close the NN/TN ↔ NT/TT performance gap on RTX 6000.

**Architecture:** Single C macro template `TROPICAL_MATMUL_TILED_BODY` parameterized by dtype (f32/f64), direction (max/min), `BETTER` op, and four layout-aware loader-decomposition macros (`LOAD_A_DECOMP_N/T`, `LOAD_B_DECOMP_N/T`). Tile sizes are dtype-specific compile-time constants (f32: BM=BN=64, BK=8, TM=TN=4 → 16×16 threads; f64: BM=BN=32, BK=8, TM=2, TN=4 → 16×8 threads).

**Tech Stack:** CUDA C (NVRTC-compiled at runtime), Rust (cudarc 0.12.1), Julia 1.x.

**Spec:** `docs/superpowers/specs/2026-04-28-tiled-counting-gemm-design.md`

---

## File Map

- **Modify** `crates/tropical-gemm-cuda/kernels/counting_gemm.cu` — replace `TROPICAL_MATMUL_BODY` with tiled body; add four `LOAD_*_DECOMP` macros; thread `BM_T/BN_T/BK_T/TM_T/TN_T` into the 16 `DEFINE_TROPICAL_MATMUL` calls (split into f32 and f64 groups so each can have its own tile sizes).
- **Modify** `crates/tropical-gemm-cuda/src/counting_kernel.rs` — block dim and grid dim now depend on dtype; introduce `tile_dims<T>()` helper.
- **Modify** `crates/tropical-gemm-cuda/src/matmul_mod.rs` — add tile-edge and ragged-K tests.
- **Modify** `CountingTropicalGEMM.jl/bench/RESULTS.md` — append Spec N table after benchmark.

---

### Task 1: Layout-aware loader macros + tiled kernel body (f32)

**Files:**
- Modify: `crates/tropical-gemm-cuda/kernels/counting_gemm.cu`

- [ ] **Step 1: Add tile-size constants and loader-decomposition macros**

Insert after the existing `#define B_OFF_T(...)` block (around line 52), before `TROPICAL_MATMUL_BODY`:

```c
// ---- Spec N: tile-size constants and layout-aware loader decompositions ----
//
// Tile sizes per dtype:
//   f32: BM=BN=64, BK=8, TM=TN=4 → block = (BN/TN, BM/TM) = (16, 16) = 256 threads.
//   f64: BM=BN=32, BK=8, TM=2,  TN=4 → block = (BN/TN, BM/TM) = (8, 16) = 128 threads.
//
// LOAD_*_DECOMP_{N,T}: maps a linear loader index `idx` (0..BM*BK or 0..BN*BK)
// to the (sk, si) or (sk, sj) shared-tile slot, with the contiguous global
// axis varying fastest across `idx` so consecutive lane indices read
// consecutive global addresses.
//
// 'N' op A: pair_a[i + k*M] is M-contiguous → fastest-axis = si.
// 'T' op A: pair_a[k + i*K] is K-contiguous → fastest-axis = sk.
// 'N' op B: pair_b[k + j*K] is K-contiguous → fastest-axis = sk.
// 'T' op B: pair_b[j + k*N] is N-contiguous → fastest-axis = sj.
#define LOAD_A_DECOMP_N(idx, BM_, BK_, sk_, si_) \
    do { (sk_) = (idx) / (BM_); (si_) = (idx) % (BM_); } while (0)
#define LOAD_A_DECOMP_T(idx, BM_, BK_, sk_, si_) \
    do { (si_) = (idx) / (BK_); (sk_) = (idx) % (BK_); } while (0)
#define LOAD_B_DECOMP_N(idx, BN_, BK_, sk_, sj_) \
    do { (sj_) = (idx) / (BK_); (sk_) = (idx) % (BK_); } while (0)
#define LOAD_B_DECOMP_T(idx, BN_, BK_, sk_, sj_) \
    do { (sk_) = (idx) / (BN_); (sj_) = (idx) % (BN_); } while (0)
```

- [ ] **Step 2: Replace `TROPICAL_MATMUL_BODY` with the tiled body**

Replace the existing `#define TROPICAL_MATMUL_BODY(...)` block (lines 54–82) with:

```c
// Tiled tropical matmul body. Block computes a BM×BN output tile with
// each thread accumulating a TM×TN sub-tile in registers. Loader walks
// 256 threads × ceil(BM*BK/256) iterations covering the full A tile,
// then likewise for B; the LOAD_*_DECOMP macros pick the per-layout
// fastest-varying axis for warp-coalesced global reads.
#define TROPICAL_MATMUL_TILED_BODY(T, PAIR, INIT_VAL, BETTER,                  \
    A_OFF, B_OFF, LOAD_A_DECOMP, LOAD_B_DECOMP,                                \
    BM_, BN_, BK_, TM_, TN_)                                                   \
{                                                                              \
    __shared__ T   As_v[BK_][BM_ + 1];                                         \
    __shared__ int As_c[BK_][BM_ + 1];                                         \
    __shared__ T   Bs_v[BK_][BN_ + 1];                                         \
    __shared__ int Bs_c[BK_][BN_ + 1];                                         \
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
    for (int kk = 0; kk < K; kk += (BK_)) {                                    \
        for (int idx = tid; idx < A_TILE; idx += threads_per_block) {          \
            int sk_a, si_a;                                                    \
            LOAD_A_DECOMP(idx, (BM_), (BK_), sk_a, si_a);                      \
            int gi = block_i0 + si_a;                                          \
            int gk = kk + sk_a;                                                \
            if (gi < M && gk < K) {                                            \
                PAIR a = pair_a[A_OFF(gi, gk, M, K)];                          \
                As_v[sk_a][si_a] = a.val; As_c[sk_a][si_a] = a.cnt;            \
            } else {                                                           \
                As_v[sk_a][si_a] = (INIT_VAL); As_c[sk_a][si_a] = 0;           \
            }                                                                  \
        }                                                                      \
        for (int idx = tid; idx < B_TILE; idx += threads_per_block) {          \
            int sk_b, sj_b;                                                    \
            LOAD_B_DECOMP(idx, (BN_), (BK_), sk_b, sj_b);                      \
            int gk = kk + sk_b;                                                \
            int gj = block_j0 + sj_b;                                          \
            if (gj < N && gk < K) {                                            \
                PAIR b = pair_b[B_OFF(gk, gj, K, N)];                          \
                Bs_v[sk_b][sj_b] = b.val; Bs_c[sk_b][sj_b] = b.cnt;            \
            } else {                                                           \
                Bs_v[sk_b][sj_b] = (INIT_VAL); Bs_c[sk_b][sj_b] = 0;           \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
                                                                               \
        int kk_end = (K - kk < (BK_)) ? (K - kk) : (BK_);                      \
        for (int kk2 = 0; kk2 < kk_end; ++kk2) {                               \
            T   av[TM_]; int ac[TM_];                                          \
            T   bv[TN_]; int bc[TN_];                                          \
            _Pragma("unroll")                                                  \
            for (int ti = 0; ti < (TM_); ++ti) {                               \
                av[ti] = As_v[kk2][ty * (TM_) + ti];                           \
                ac[ti] = As_c[kk2][ty * (TM_) + ti];                           \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tj = 0; tj < (TN_); ++tj) {                               \
                bv[tj] = Bs_v[kk2][tx * (TN_) + tj];                           \
                bc[tj] = Bs_c[kk2][tx * (TN_) + tj];                           \
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
        __syncthreads();                                                       \
    }                                                                          \
                                                                               \
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

- [ ] **Step 3: Replace the 16 `DEFINE_TROPICAL_MATMUL` lines with tiled versions split by dtype**

Replace lines 84–113 (`DEFINE_TROPICAL_MATMUL` macro and the 16 expansions) with:

```c
#define DEFINE_TILED_F32(NAME, INIT_VAL, BETTER, A_OFF, B_OFF, LOAD_A, LOAD_B) \
extern "C" __global__ void NAME(                                               \
    const PairF32* __restrict__ pair_a,                                        \
    const PairF32* __restrict__ pair_b,                                        \
    PairF32* __restrict__ out_c,                                               \
    int M, int N, int K, int P, unsigned long long MU                          \
)                                                                              \
TROPICAL_MATMUL_TILED_BODY(float, PairF32, INIT_VAL, BETTER, A_OFF, B_OFF,     \
                           LOAD_A, LOAD_B, 64, 64, 8, 4, 4)

#define DEFINE_TILED_F64(NAME, INIT_VAL, BETTER, A_OFF, B_OFF, LOAD_A, LOAD_B) \
extern "C" __global__ void NAME(                                               \
    const PairF64* __restrict__ pair_a,                                        \
    const PairF64* __restrict__ pair_b,                                        \
    PairF64* __restrict__ out_c,                                               \
    int M, int N, int K, int P, unsigned long long MU                          \
)                                                                              \
TROPICAL_MATMUL_TILED_BODY(double, PairF64, INIT_VAL, BETTER, A_OFF, B_OFF,    \
                           LOAD_A, LOAD_B, 32, 32, 8, 2, 4)

DEFINE_TILED_F32(tropical_matmul_f32_max_NN, NEG_INF_F32, MAX_BETTER, A_OFF_N, B_OFF_N, LOAD_A_DECOMP_N, LOAD_B_DECOMP_N)
DEFINE_TILED_F32(tropical_matmul_f32_max_NT, NEG_INF_F32, MAX_BETTER, A_OFF_N, B_OFF_T, LOAD_A_DECOMP_N, LOAD_B_DECOMP_T)
DEFINE_TILED_F32(tropical_matmul_f32_max_TN, NEG_INF_F32, MAX_BETTER, A_OFF_T, B_OFF_N, LOAD_A_DECOMP_T, LOAD_B_DECOMP_N)
DEFINE_TILED_F32(tropical_matmul_f32_max_TT, NEG_INF_F32, MAX_BETTER, A_OFF_T, B_OFF_T, LOAD_A_DECOMP_T, LOAD_B_DECOMP_T)

DEFINE_TILED_F32(tropical_matmul_f32_min_NN, POS_INF_F32, MIN_BETTER, A_OFF_N, B_OFF_N, LOAD_A_DECOMP_N, LOAD_B_DECOMP_N)
DEFINE_TILED_F32(tropical_matmul_f32_min_NT, POS_INF_F32, MIN_BETTER, A_OFF_N, B_OFF_T, LOAD_A_DECOMP_N, LOAD_B_DECOMP_T)
DEFINE_TILED_F32(tropical_matmul_f32_min_TN, POS_INF_F32, MIN_BETTER, A_OFF_T, B_OFF_N, LOAD_A_DECOMP_T, LOAD_B_DECOMP_N)
DEFINE_TILED_F32(tropical_matmul_f32_min_TT, POS_INF_F32, MIN_BETTER, A_OFF_T, B_OFF_T, LOAD_A_DECOMP_T, LOAD_B_DECOMP_T)

DEFINE_TILED_F64(tropical_matmul_f64_max_NN, NEG_INF_F64, MAX_BETTER, A_OFF_N, B_OFF_N, LOAD_A_DECOMP_N, LOAD_B_DECOMP_N)
DEFINE_TILED_F64(tropical_matmul_f64_max_NT, NEG_INF_F64, MAX_BETTER, A_OFF_N, B_OFF_T, LOAD_A_DECOMP_N, LOAD_B_DECOMP_T)
DEFINE_TILED_F64(tropical_matmul_f64_max_TN, NEG_INF_F64, MAX_BETTER, A_OFF_T, B_OFF_N, LOAD_A_DECOMP_T, LOAD_B_DECOMP_N)
DEFINE_TILED_F64(tropical_matmul_f64_max_TT, NEG_INF_F64, MAX_BETTER, A_OFF_T, B_OFF_T, LOAD_A_DECOMP_T, LOAD_B_DECOMP_T)

DEFINE_TILED_F64(tropical_matmul_f64_min_NN, POS_INF_F64, MIN_BETTER, A_OFF_N, B_OFF_N, LOAD_A_DECOMP_N, LOAD_B_DECOMP_N)
DEFINE_TILED_F64(tropical_matmul_f64_min_NT, POS_INF_F64, MIN_BETTER, A_OFF_N, B_OFF_T, LOAD_A_DECOMP_N, LOAD_B_DECOMP_T)
DEFINE_TILED_F64(tropical_matmul_f64_min_TN, POS_INF_F64, MIN_BETTER, A_OFF_T, B_OFF_N, LOAD_A_DECOMP_T, LOAD_B_DECOMP_N)
DEFINE_TILED_F64(tropical_matmul_f64_min_TT, POS_INF_F64, MIN_BETTER, A_OFF_T, B_OFF_T, LOAD_A_DECOMP_T, LOAD_B_DECOMP_T)
```

- [ ] **Step 4: Build to verify the kernel compiles via NVRTC at first call** — proceed to Task 2 first; build is invoked there.

---

### Task 2: Update launcher to use dtype-specific tile dims

**Files:**
- Modify: `crates/tropical-gemm-cuda/src/counting_kernel.rs`

- [ ] **Step 1: Add a `tile_dims<T>` helper and rewire grid/block dims**

Replace the body of `launch_tropical_matmul` (the launch-config section, currently lines 76–87 of `counting_kernel.rs`) so block/grid dims come from a per-dtype helper. Insert this trait+impls block above `pub fn launch_tropical_matmul`:

```rust
/// Spec N: per-dtype tile dimensions. Block dim = (BN/TN, BM/TM, 1).
/// Grid dim = (ceil(N/BN), ceil(M/BM), 1).
trait TileDims {
    const BM: usize;
    const BN: usize;
    const TM: usize;
    const TN: usize;
}
impl TileDims for f32 {
    const BM: usize = 64; const BN: usize = 64; const TM: usize = 4; const TN: usize = 4;
}
impl TileDims for f64 {
    const BM: usize = 32; const BN: usize = 32; const TM: usize = 2; const TN: usize = 4;
}
```

Then add `T: TileDims` to the function's where clause and replace the block/grid lines with:

```rust
    let block: (u32, u32, u32) = ((T::BN / T::TN) as u32, (T::BM / T::TM) as u32, 1);
    let grid: (u32, u32, u32) = (
        ((n + T::BN - 1) / T::BN) as u32,
        ((m + T::BM - 1) / T::BM) as u32,
        1,
    );
```

- [ ] **Step 2: Build**

Run: `cargo build -p tropical-gemm-cuda 2>&1 | tail -40`
Expected: success. If a missing `TileDims` bound is reported in `matmul_mod.rs`, add `+ TileDims` there too (the bound flows through via the same `T` generic).

- [ ] **Step 3: Commit Tasks 1+2**

```bash
git add crates/tropical-gemm-cuda/kernels/counting_gemm.cu \
        crates/tropical-gemm-cuda/src/counting_kernel.rs \
        crates/tropical-gemm-cuda/src/matmul_mod.rs
git commit -m "Spec N: tiled counting tropical GEMM kernel + dtype-aware launcher"
```

---

### Task 3: Tile-edge and ragged-K correctness tests

**Files:**
- Modify: `crates/tropical-gemm-cuda/src/matmul_mod.rs`

- [ ] **Step 1: Read existing test scaffolding**

Read `crates/tropical-gemm-cuda/src/matmul_mod.rs` to find the existing `mod tests` block and the helpers used by current 2×2/4×4 tests (random-input pattern, host reference if any). Reuse helpers; add a CPU reference inline only if none exists.

- [ ] **Step 2: Add a host-reference helper inside `mod tests`**

If no `cpu_reference` helper exists, add this near the top of `mod tests`:

```rust
fn cpu_ref_max_f32(
    tA: char, tB: char, m: usize, k: usize, n: usize,
    a: &[crate::pair::PairF32], b: &[crate::pair::PairF32], p: i32,
) -> Vec<crate::pair::PairF32> {
    let mut out = vec![crate::pair::PairF32::new(f32::NEG_INFINITY, 0); m * n];
    let p_u = p as u64;
    for j in 0..n {
        for i in 0..m {
            let mut acc_v = f32::NEG_INFINITY;
            let mut acc_c: u64 = 0;
            for kk in 0..k {
                let av = if tA == 'N' { a[i + kk * m] } else { a[kk + i * k] };
                let bv = if tB == 'N' { b[kk + j * k] } else { b[j + kk * n] };
                let pv = av.val + bv.val;
                let pc = ((av.cnt as u64) * (bv.cnt as u64)) % p_u;
                if pv > acc_v { acc_v = pv; acc_c = pc; }
                else if pv == acc_v { acc_c = (acc_c + pc) % p_u; }
            }
            out[i + j * m] = crate::pair::PairF32::new(acc_v, (acc_c % p_u) as i32);
        }
    }
    out
}
```

- [ ] **Step 3: Add tile-edge and ragged-K tests**

Append to `mod tests` (replace dimensions, layouts as appropriate). Each test allocates random Pair input, runs the kernel via the public `tropical_matmul_kernel<f32, Max>(...)` (existing entry point), and asserts each output cell matches `cpu_ref_max_f32`. The 5 tests below should be added verbatim with their parameters:

```rust
fn rand_pairs_f32(n: usize, p: i32) -> Vec<crate::pair::PairF32> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(42);
    (0..n).map(|_| crate::pair::PairF32::new(
        rng.gen_range(0.0..4.0),
        rng.gen_range(1..p),
    )).collect()
}

fn run_and_check_f32_max(tA: char, tB: char, m: usize, k: usize, n: usize, p: i32) {
    use cudarc::driver::DevicePtr;
    let ctx = crate::get_global_context().expect("CUDA ctx");
    let a_len = m * k; let b_len = k * n;
    let a_host = rand_pairs_f32(a_len, p);
    let b_host = rand_pairs_f32(b_len, p);
    let expect = cpu_ref_max_f32(tA, tB, m, k, n, &a_host, &b_host, p);
    let a_dev = ctx.device().htod_copy(a_host).unwrap();
    let b_dev = ctx.device().htod_copy(b_host).unwrap();
    let out_dev = ctx.device().alloc_zeros::<crate::pair::PairF32>(m * n).unwrap();
    crate::matmul_mod::tropical_matmul_kernel::<f32, tropical_gemm::types::Max>(
        ctx, tA, tB, m, k, n,
        *a_dev.device_ptr(), *b_dev.device_ptr(), p, *out_dev.device_ptr(),
    ).unwrap();
    let got = ctx.device().dtoh_sync_copy(&out_dev).unwrap();
    for idx in 0..(m*n) {
        assert_eq!(got[idx].val, expect[idx].val,
            "val mismatch at {} (tA={},tB={},M={},K={},N={})", idx, tA, tB, m, k, n);
        assert_eq!(got[idx].cnt, expect[idx].cnt,
            "cnt mismatch at {} (tA={},tB={},M={},K={},N={})", idx, tA, tB, m, k, n);
    }
}

#[test] fn tile_edge_nn_65_65_65_f32() { run_and_check_f32_max('N','N', 65, 65, 65, 7); }
#[test] fn tile_edge_tt_100_33_77_f32() { run_and_check_f32_max('T','T', 100, 33, 77, 11); }
#[test] fn tile_exact_nt_128_128_128_f32() { run_and_check_f32_max('N','T', 128, 128, 128, 13); }

#[test] fn ragged_k_bk_plus_1_nn_f32() { run_and_check_f32_max('N','N', 8, 9, 8, 7); }
#[test] fn ragged_k_bk_plus_1_nt_f32() { run_and_check_f32_max('N','T', 8, 9, 8, 7); }
#[test] fn ragged_k_bk_plus_1_tn_f32() { run_and_check_f32_max('T','N', 8, 9, 8, 7); }
#[test] fn ragged_k_bk_plus_1_tt_f32() { run_and_check_f32_max('T','T', 8, 9, 8, 7); }

#[test] fn ragged_k_2bk_minus_1_nn_f32() { run_and_check_f32_max('N','N', 8, 15, 8, 7); }
#[test] fn ragged_k_1_tt_f32() { run_and_check_f32_max('T','T', 8, 1, 8, 7); }
```

- [ ] **Step 4: Verify rand crate is available**

Run: `grep -n '^rand' crates/tropical-gemm-cuda/Cargo.toml`
If absent, add to `[dev-dependencies]`: `rand = "0.8"`.

- [ ] **Step 5: Build and run tests**

Run: `cargo test -p tropical-gemm-cuda 2>&1 | tail -60`
Expected: all tests pass (existing 5 + new 9). NVRTC compile takes ~30s on
first run. If any test fails, the kernel has a correctness bug — diagnose
before proceeding.

- [ ] **Step 6: Commit**

```bash
git add crates/tropical-gemm-cuda/src/matmul_mod.rs crates/tropical-gemm-cuda/Cargo.toml
git commit -m "Spec N: tile-edge and ragged-K correctness tests"
```

---

### Task 4: Run Julia regression suite

**Files:**
- (Tests only; no source changes.)

- [ ] **Step 1: Run Julia tests**

Run from repo root:
```
JULIA_CUDA_USE_COMPAT=false julia --project=CountingTropicalGEMM.jl CountingTropicalGEMM.jl/test/runtests.jl 2>&1 | tail -40
```

Expected: all 26 tests pass. Failure here indicates either a regression in
the C ABI (unchanged) or a kernel correctness bug not caught by the Rust
tests.

- [ ] **Step 2: If on the cluster, run via the standard SLURM wrapper instead**

Run:
```
srun --partition=gpu --gpus=1 --time=00:15:00 \
    bash -lc "cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && \
    JULIA_CUDA_USE_COMPAT=false julia --project=CountingTropicalGEMM.jl \
    CountingTropicalGEMM.jl/test/runtests.jl"
```

Expected: same.

(No commit; tests-only step.)

---

### Task 5: Benchmark and append to RESULTS.md

**Files:**
- Modify: `CountingTropicalGEMM.jl/bench/RESULTS.md`

- [ ] **Step 1: Run bench on RTX 6000**

Run on the cluster RTX 6000 node:
```
srun --partition=gpu --gpus=1 --time=00:30:00 \
    bash -lc "cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && \
    JULIA_CUDA_USE_COMPAT=false julia --project=CountingTropicalGEMM.jl \
    CountingTropicalGEMM.jl/bench/bench_mul.jl"
```

Capture stdout into a temp file (e.g. via `tee /tmp/specn_rtx6000.txt`).

- [ ] **Step 2: Append a new section to `RESULTS.md`**

Add the following block after the existing "## Quadro RTX 6000 (Turing
sm_75) — 2026-04-28" section, before the "## A100 / H100" heading:

```markdown
## Quadro RTX 6000 (Spec N tiled) — 2026-04-28

| Shape | flag | ms / call | G tropical-ops/s |
|---:|:---:|---:|---:|
<one row per bench output, 24 rows total>

### Observations

- All four flags converge to within <X>× of each other (was 4×).
- Peak <Y> G/s at <shape, flag>. Spec M peak was 611 G/s at 1024³ TT.
- NN/TN at 1024³: <Z> G/s (was 116/125 G/s).
```

Replace `<X>`, `<Y>`, `<Z>` and the table rows with the actual numbers from
the bench output. If observed perf misses the spec criteria (NN/TN ≥ 500
G/s at 1024³, TT ≥ 519 G/s), record the result honestly and stop here for
human triage rather than landing.

- [ ] **Step 3: Commit**

```bash
git add CountingTropicalGEMM.jl/bench/RESULTS.md
git commit -m "Spec N: record tiled-kernel bench on RTX 6000"
```

---

## Self-review checklist (controller)

- [x] Spec coverage: Task 1 covers loader macros + tiled body + 16 specializations; Task 2 covers launcher; Task 3 covers correctness tests (tile-edge + ragged-K for all 4 flags); Task 4 covers Julia regression; Task 5 covers benchmark.
- [x] No placeholders in code blocks.
- [x] Type/symbol consistency: `tropical_matmul_kernel`, `PairF32`, `LOAD_*_DECOMP_*`, tile constants are reused across tasks.
