# Spec N: Tiled Counting Tropical GEMM (close the NN/TN ↔ NT/TT gap)

## Problem

The Spec M kernels in `crates/tropical-gemm-cuda/kernels/counting_gemm.cu` are
naive: each thread computes one output element with a direct `for k` loop over
global memory. Bench results on RTX 6000 (`CountingTropicalGEMM.jl/bench/RESULTS.md`):

| size | NN | TN | NT | TT |
|---:|---:|---:|---:|---:|
| 1024³ | 116 G/s | 125 G/s | 591 G/s | 611 G/s |
| 4096³ | 132 G/s | 131 G/s | 546 G/s | 552 G/s |

NN/TN are ~4× slower than NT/TT. Within a warp, `threadIdx.x` varies along
`j`. For `B_OFF_N(k,j) = k + j*K`, B accesses across the warp are stride-K
(uncoalesced). For `B_OFF_T(k,j) = j + k*N`, accesses are stride-1 (coalesced).
A's pattern is broadcast in both layouts (good for the consumer; *load* still
needs care).

## Goal

Make all four (tA, tB) variants land within ~1.3× of each other, and lift
absolute throughput. Specifically on RTX 6000 at 1024³ ModCountingTropical{Float32,7}:

- All four flags ≥ 500 G tropical-ops/s.
- TT performance regresses no more than 15% vs current 611 G/s (acceptable
  cost of the new sync/shared-memory traffic; the goal is closing the gap).
- NN performance increases ≥ 3× over current 116 G/s.

## Approach

Shared-memory tiled GEMM with register tiling. Single template covers all
16 specializations. The template takes **layout-aware load macros** so the
*global → shared* phase is always coalesced regardless of source transpose.

**Block tile:** BM × BN output elements per block.
**K-tile:** BK reduction-axis elements per inner iteration.
**Thread tile:** TM × TN output elements per thread (registers).
**Block dims:** (BN/TN) × (BM/TM) threads = (16, 16) = 256 threads.

Tile constants per dtype (chosen for sm_75 register budget; verified post-hoc
by `cuobjdump --dump-sass` for register count, see Test Plan):
- f32: BM=BN=64, BK=8, TM=TN=4.
- f64: BM=BN=32, BK=8, TM=2, TN=4.

The f32 path targets 64–80 regs/thread; the f64 path stays under that ceiling
because each `acc_c` register is u64 and `acc_v` is f64. If post-build sass
reports >96 regs/thread on either, fall back one notch (TM/TN halved) before
landing.

### Layout-aware coalesced load

The previous draft used a single `(sk, si)` mapping for all layouts. Codex
correctly flagged that this only coalesces when the source axis matches `si`'s
fastest-varying direction. Fix: parameterize the macro by **A's contiguous
axis** and **B's contiguous axis** so loads always walk lane-stride-1 in
global memory.

For the four cases:
- `A` op-`N`: A is M×K col-major, `pair_a[i + k*M]`. Contiguous axis = M (i).
- `A` op-`T`: A is K×M col-major, `pair_a[k + i*K]`. Contiguous axis = K (k).
- `B` op-`N`: B is K×N col-major, `pair_b[k + j*K]`. Contiguous axis = K (k).
- `B` op-`T`: B is N×K col-major, `pair_b[j + k*N]`. Contiguous axis = N (j).

We expose two load macros per operand:
```c
// LOAD_A_*: produce (val, cnt) for shared element (sk, si) where
//   global_i = block_i0 + si, global_k = kk + sk.
// 'N': lane fastest-axis = si (M-contig). 'T': lane fastest-axis = sk (K-contig).
#define LOAD_A_N(/* tid, sk_var, si_var */) /* tid -> (sk = tid/BM, si = tid%BM) */
#define LOAD_A_T(/* tid, sk_var, si_var */) /* tid -> (si = tid/BK, sk = tid%BK) */

// LOAD_B_*: produce (val, cnt) for shared element (sk, sj).
// 'N': lane fastest-axis = sk (K-contig). 'T': lane fastest-axis = sj (N-contig).
#define LOAD_B_N(/* tid, sk_var, sj_var */) /* tid -> (sj = tid/BK, sk = tid%BK) */
#define LOAD_B_T(/* tid, sk_var, sj_var */) /* tid -> (sk = tid/BN, sj = tid%BN) */
```

Each macro just defines the (`sk`, `si`/`sj`) decomposition of the linear
loader index `idx`; the actual `pair_a[A_OFF(...)]` access is then
unit-stride across consecutive lanes of the warp.

For BM·BK = 64·8 = 512 elements with 256 threads, each thread loads 2
elements (loop `s = 0,1`, `idx = tid + s*256`). Same for B.

### Shared layout (bank-conflict mitigation)

Stored as separate value/count arrays to avoid AoS interleave grief. **Add
+1 padding to the leading dim consumed columnwise** to break stride-32 bank
aliasing on the f32 read path:

```c
__shared__ T  As_v[BK][BM + 1]; __shared__ int As_c[BK][BM + 1];
__shared__ T  Bs_v[BK][BN + 1]; __shared__ int Bs_c[BK][BN + 1];
```

The compute phase reads `Bs_v[kk2][tx*TN + tj]` for tj = 0..TN-1 across
`tx = 0..15`. With 32-bit banks and BN+1 = 65, the lane→bank map breaks
the 4-stride aliasing that would otherwise hit 8-way conflicts on f32. For
f64, banks are 32-bit pairs; pad still helps.

### Compute body (f32 example, TM=TN=4)

```c
__shared__ float As_v[BK][BM+1]; __shared__ int As_c[BK][BM+1];
__shared__ float Bs_v[BK][BN+1]; __shared__ int Bs_c[BK][BN+1];

int tx = threadIdx.x, ty = threadIdx.y;       // 0..15
int tid = ty * blockDim.x + tx;               // 0..255
int block_i0 = blockIdx.y * BM;
int block_j0 = blockIdx.x * BN;

float              acc_v[TM][TN];
unsigned long long acc_c[TM][TN];
#pragma unroll
for (int ti=0; ti<TM; ++ti)
  #pragma unroll
  for (int tj=0; tj<TN; ++tj) { acc_v[ti][tj] = INIT_VAL; acc_c[ti][tj] = 0; }

const unsigned long long Pull = (unsigned long long)P;

for (int kk = 0; kk < K; kk += BK) {
    // Two loader iterations per thread (512 elements / 256 threads).
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        int idx = tid + s * 256;

        // A: layout-aware decomposition (LOAD_A_N or LOAD_A_T).
        int sk_a, si_a;  LOAD_A_DECOMP(idx, sk_a, si_a);
        int gi = block_i0 + si_a, gk = kk + sk_a;
        if (gi < M && gk < K) {
            PAIR a = pair_a[A_OFF(gi, gk, M, K)];
            As_v[sk_a][si_a] = a.val; As_c[sk_a][si_a] = a.cnt;
        } else {
            As_v[sk_a][si_a] = INIT_VAL; As_c[sk_a][si_a] = 0;
        }

        int sk_b, sj_b;  LOAD_B_DECOMP(idx, sk_b, sj_b);
        int gj = block_j0 + sj_b, gk2 = kk + sk_b;
        if (gj < N && gk2 < K) {
            PAIR b = pair_b[B_OFF(gk2, gj, K, N)];
            Bs_v[sk_b][sj_b] = b.val; Bs_c[sk_b][sj_b] = b.cnt;
        } else {
            Bs_v[sk_b][sj_b] = INIT_VAL; Bs_c[sk_b][sj_b] = 0;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int kk2 = 0; kk2 < BK; ++kk2) {
        float av[TM]; int ac[TM];
        float bv[TN]; int bc[TN];
        #pragma unroll
        for (int ti=0; ti<TM; ++ti) {
            av[ti] = As_v[kk2][ty*TM + ti];
            ac[ti] = As_c[kk2][ty*TM + ti];
        }
        #pragma unroll
        for (int tj=0; tj<TN; ++tj) {
            bv[tj] = Bs_v[kk2][tx*TN + tj];
            bc[tj] = Bs_c[kk2][tx*TN + tj];
        }
        #pragma unroll
        for (int ti=0; ti<TM; ++ti)
          #pragma unroll
          for (int tj=0; tj<TN; ++tj) {
            float pv = av[ti] + bv[tj];
            unsigned long long prod =
                (unsigned long long)(unsigned)ac[ti] *
                (unsigned long long)(unsigned)bc[tj];
            unsigned long long pc = barrett_mod(prod, Pull, MU);
            bool win = BETTER(pv, acc_v[ti][tj]);
            bool tie = (pv == acc_v[ti][tj]);
            acc_v[ti][tj] = win ? pv : acc_v[ti][tj];
            acc_c[ti][tj] = win ? pc :
                            (tie ? (acc_c[ti][tj] + pc) : acc_c[ti][tj]);
          }
    }
    __syncthreads();
}

#pragma unroll
for (int ti=0; ti<TM; ++ti) {
    int gi = block_i0 + ty*TM + ti;
    if (gi >= M) continue;
    #pragma unroll
    for (int tj=0; tj<TN; ++tj) {
        int gj = block_j0 + tx*TN + tj;
        if (gj >= N) continue;
        PAIR out;
        out.val = acc_v[ti][tj];
        out.cnt = (int)barrett_mod(acc_c[ti][tj], Pull, MU);
        out_c[gi + gj * M] = out;
    }
}
```

The 16 `DEFINE_TROPICAL_MATMUL` expansions stay; the body is selected by
two `LOAD_*_DECOMP` macros (one for A, one for B). Each (transA, transB)
combo binds the correct decomposition.

### Launch config update (`counting_kernel.rs`)

Block dim stays (16, 16). Grid dim becomes
`(ceil(N / BN), ceil(M / BM))`. BM/BN depend on dtype, so the launcher
selects via the `T` generic. Constants live as `#define`s in the .cu file
and as `const usize` mirrors in the launcher.

## Components

1. **`counting_gemm.cu`** — replaced body. Two new tile-size groups
   (f32: 64×64×8, f64: 32×32×8). Layout-aware load macros wired into the
   16 `DEFINE_TROPICAL_MATMUL` expansions.
2. **`counting_kernel.rs`** — grid-dim arithmetic uses dtype-correct BM/BN.
3. **Tests** in `matmul_mod.rs` — see below.
4. **Bench refresh** — re-run on RTX 6000, append a new "## Quadro RTX 6000
   (Spec N tiled) — 2026-04-28" section to `bench/RESULTS.md`.

## Tests (correctness)

Existing 5 tests stay green. Add tile-edge and ragged-K cases — every
boundary class for every (transA, transB):

1. NN 65×65×65 f64 Min — M and N tile-boundary +1.
2. TT 100×33×77 f32 Max — fully ragged.
3. NT 128×128×128 f32 Max — exact tile multiples.
4. **Ragged K** (one for each flag): K = BK + 1, K = 2·BK − 1, K = 1
   covering NN, NT, TN, TT with M = N = 8 (cheap, fast). Compares against
   a host reference.

The host reference is a 30-line CPU loop that reproduces the exact
tropical-add-with-tie-counts semantics; small dims so it runs in <10 ms.

## Out of Scope

- Tile autotuning across GPUs. Two fixed configs (f32 / f64); A100/H100
  tune is a follow-up spec.
- Double-buffered shared loads (`cp.async`, k-tile pipelining). Single-
  buffered with `__syncthreads` is enough for the gap fix.
- Tensor-core / DPX instructions. Counting tropical's `tie ? add : keep`
  is not a tensor-core shape.
- AoS PairT input layout — unchanged.

## Risks

- **Register pressure (sm_75)**. f32 tile estimated 64–80 regs; f64 tile
  ~70 regs. Build inspects `cuobjdump --dump-sass` after compile; if any
  kernel exceeds 96 regs, halve the failing dimension and rebuild.
- **Bank conflicts**. Mitigated by `[BK][BM+1]` / `[BK][BN+1]` padding.
  Verified after bench: if NN/TN throughput is still <300 G/s on RTX 6000,
  inspect with Nsight Compute and switch to swizzled layout.
- **TT regression**. Allowance is 15%; the extra `__syncthreads` and
  shared traffic cost is real. If TT regresses >15%, the design lands as a
  net loss and we either tune tile sizes per (tA, tB) or revert.
- **Modulus accumulation**. `acc_c` is u64; per step `pc < P ≤ 2³¹`,
  `acc_c` resets on `win` and only sums on `tie` — at K ≤ 2²⁰ no overflow.
- **Cross-GPU portability**. Two configs (f32, f64) chosen for sm_75. On
  sm_80 / sm_90 they should still run correctly and faster than Spec M;
  best-tuned numbers come in a follow-on spec.

## Test Plan

1. `cargo build -p tropical-gemm-cuda` (compiles all 16 kernels).
2. `cuobjdump --dump-sass <built-ptx>` — record register count per kernel.
   If any > 96, adjust tile and rebuild.
3. `cargo test -p tropical-gemm-cuda` — all inline tests including the new
   tile-edge and ragged-K cases.
4. `julia --project=CountingTropicalGEMM.jl
   CountingTropicalGEMM.jl/test/runtests.jl` — Julia binding regression.
5. `julia --project=CountingTropicalGEMM.jl
   CountingTropicalGEMM.jl/bench/bench_mul.jl` on the RTX 6000 node —
   append fresh table to `RESULTS.md` under a new heading "Spec N tiled".
6. Compare side-by-side with Spec M numbers. Land if NN/TN ≥ 500 G/s and
   TT ≥ 519 G/s (= 0.85 × 611) at 1024³.
