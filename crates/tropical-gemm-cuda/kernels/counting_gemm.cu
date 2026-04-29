// Counting Tropical GEMM CUDA kernels (Spec M).
//
// Element layout: AoS — each input element is a packed (value, count) struct
// (PairF32: 8 B; PairF64: 16 B with 4 B padding). Inputs and output are
// column-major. Sixteen specializations cover (transA, transB) ∈ {N,T}² ×
// dtype ∈ {f32, f64} × direction ∈ {max, min}.
//
// Semiring (direction D):
//   tropical_mul: (va, ca) * (vb, cb) = (va + vb, (ca * cb) mod P)
//   tropical_add: strictly-better value wins; on tie, counts add mod P.
// Counts are reduced under modulus P via Barrett reduction with a host-
// precomputed mu = floor(2^64 / P).

#define NEG_INF_F32 __int_as_float(0xff800000)
#define POS_INF_F32 __int_as_float(0x7f800000)
#define NEG_INF_F64 __longlong_as_double(0xfff0000000000000LL)
#define POS_INF_F64 __longlong_as_double(0x7ff0000000000000LL)

#define OFFSET_ROW(row, col, ncols) ((row) * (ncols) + (col))

#define MAX_BETTER(a, b) ((a) > (b))
#define MIN_BETTER(a, b) ((a) < (b))

__device__ __forceinline__ unsigned long long barrett_mod(
    unsigned long long x, unsigned long long P, unsigned long long mu)
{
    unsigned long long q = __umul64hi(x, mu);
    unsigned long long r = x - q * P;
    if (r >= P) r -= P;
    return r;
}

// ---- Spec P: cp.async helpers (sm_80+) ------------------------------------
//
// `cp.async.cg.shared.global` performs an asynchronous 4/8/16-byte copy
// from global memory into shared memory without occupying a register and
// without blocking the issuing warp. The copy is committed to a "group"
// via `cp.async.commit_group` and waited on with `cp.async.wait_group N`,
// which blocks until at most N groups are still in flight.
//
// PTX restriction: `cp.async.cg` is only valid for cp_size = 16 bytes.
// For 4- or 8-byte transfers we MUST use `.ca` (cache all). Issuing
// `cp.async.cg ... 8` produced silently-corrupt loads on sm_80 (the
// pipelined kernel returned wrong cnt values even at M=K=N=64), so this
// header now picks the encoding by size: CA for 4/8 B, CG for 16 B.
//
// PairF32's 8 B is moved as one CA_8 cp.async (val and cnt together).
// PairF64's 16 B is moved as one CG_16 cp.async (val, cnt, and pad).
#if __CUDA_ARCH__ >= 800
#define CP_ASYNC_CA_4(smem_ptr_u32, gmem_ptr) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(smem_ptr_u32), "l"(gmem_ptr) : "memory")
#define CP_ASYNC_CA_8(smem_ptr_u32, gmem_ptr) \
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(smem_ptr_u32), "l"(gmem_ptr) : "memory")
#define CP_ASYNC_CG_16(smem_ptr_u32, gmem_ptr) \
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(smem_ptr_u32), "l"(gmem_ptr) : "memory")
#define CP_ASYNC_COMMIT()  asm volatile("cp.async.commit_group;\n" ::: "memory")
#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory")
// __cvta_generic_to_shared converts a generic shared pointer to the
// 32-bit shared-state-space address that cp.async expects.
#define SMEM_PTR(ptr) static_cast<unsigned>(__cvta_generic_to_shared(ptr))
#else
// Stubs for sm_<80; the pipelined kernel will not be launched there, but
// it must still compile so NVRTC can build all 16 specializations on a
// shared NVRTC pass.
#define CP_ASYNC_CA_4(s, g) do { (void)(s); (void)(g); } while (0)
#define CP_ASYNC_CA_8(s, g) do { (void)(s); (void)(g); } while (0)
#define CP_ASYNC_CG_16(s, g) do { (void)(s); (void)(g); } while (0)
#define CP_ASYNC_COMMIT()
#define CP_ASYNC_WAIT_GROUP(N) ((void)0)
#define SMEM_PTR(ptr) 0u
#endif

struct __align__(8)  PairF32 { float  val; int cnt; };
struct __align__(16) PairF64 { double val; int cnt; int _pad; };

// ============================================================================
// Spec M: Column-major counting tropical GEMM with per-operand N/T flags.
// Inputs and output are column-major AoS Pair buffers (PairF32 8 B, PairF64
// 16 B). Sixteen specializations: (transA, transB) × dtype × direction.
// ============================================================================

// Element addressing per operand layout. (i, k) is the logical position in
// op(A); (k, j) is the logical position in op(B).
// 'N' op: A is M×K col-major, A[i,k] = pair_a[i + k*M].
// 'T' op: A is K×M col-major, A[k,i] = pair_a[k + i*K].
// 'N' op: B is K×N col-major, B[k,j] = pair_b[k + j*K].
// 'T' op: B is N×K col-major, B[j,k] = pair_b[j + k*N].
// Output C is M×N col-major AoS: out[i + j*M] = (acc_val, acc_cnt).
#define A_OFF_N(i, k, M_, K_) ((i) + (k) * (M_))
#define A_OFF_T(i, k, M_, K_) ((k) + (i) * (K_))
#define B_OFF_N(k, j, K_, N_) ((k) + (j) * (K_))
#define B_OFF_T(k, j, K_, N_) ((j) + (k) * (N_))

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

// Spec P (pipelined-structure variant) + Spec Q (defer-mod fast path).
// Two shared-memory buffers per operand (As_v[2][...], etc.) ping-pong:
// while the compute phase reads buffer cur, the loader stores the next
// K-tile into buffer 1-cur.
//
// SPEC Q (defer-mod): the inner FMA does NOT call barrett_mod. The cnt
// accumulator stores the raw u64 sum of `ac*bc` products; the epilogue
// runs a single Barrett at write-out. This is mathematically equivalent
// to per-step modding because (a + b) mod P == ((a mod P) + (b mod P))
// mod P. The launcher must ensure `K * (P-1)^2 < 2^63` before picking
// this kernel — otherwise the u64 acc_c overflows. Empirical A100 gain:
// ~2.4× over per-step Barrett (771 → 1845 G/s @ 4096³ TT, P=7).
//
// The pipeline layout is unchanged from Spec P's original cp.async design:
//
//   load tile 0 -> buffer 0
//   for kk = BK; kk < K; kk += BK:
//       load tile (kk/BK) -> buffer (kk/BK & 1)
//       __syncthreads()
//       compute on buffer ((kk/BK)-1) & 1
//       __syncthreads()
//   compute final tile from cur_buf
//
// HISTORICAL NOTE on Spec P (BUG FIX): the original Spec P kernel issued
// cp.async.{ca,cg}.shared.global for both `val` and `cnt` slabs. On sm_80
// this produced silently-wrong `cnt` values (val was always correct):
// e.g. M=K=N=64 gave cnt = 2*expected. We narrowed it down by replacing
// only the `cnt` cp.async with a plain STS while keeping val's cp.async,
// and the cnt result was *still* wrong — i.e. a 4-byte cp.async issued
// to a separate __shared__ int slab corrupts that slab's contents on
// sm_80 in this kernel. We could not pin the failure on a documented
// PTX rule (alignment, .ca vs .cg, memory clobber, fence ordering all
// checked). The mitigation here is to drop cp.async entirely and use
// plain LDG+STS for both slabs while preserving the double-buffered
// pipeline structure. CP_ASYNC_COMMIT/WAIT_GROUP calls remain in place
// as harmless no-ops over an empty cp.async group; they keep the
// pipeline shape intact for future re-introduction of cp.async with
// (perhaps) the cuda::pipeline / __pipeline_memcpy_async API.
// ACC_T is the cnt accumulator type: `unsigned long long` for the general
// defer-mod fast path (gate K·(P-1)² < 2^63), `unsigned int` for the small-P
// fast path (gate K·(P-1)² < 2^32). The second saves IMAD.WIDE.U32 + IMAD.X
// carries per FMA — empirically ~1.7× over the u64 form on A100 at P≤7.
#define TROPICAL_MATMUL_TILED_PIPELINED_BODY(T, ACC_T, PAIR, INIT_VAL, BETTER, \
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
    T     acc_v[TM_][TN_];                                                     \
    ACC_T acc_c[TM_][TN_];                                                     \
    _Pragma("unroll")                                                          \
    for (int ti = 0; ti < (TM_); ++ti)                                         \
        _Pragma("unroll")                                                      \
        for (int tj = 0; tj < (TN_); ++tj) {                                   \
            acc_v[ti][tj] = (INIT_VAL);                                        \
            acc_c[ti][tj] = (ACC_T)0;                                          \
        }                                                                      \
                                                                               \
    const unsigned long long Pull = (unsigned long long)P;                     \
    const int A_TILE = (BM_) * (BK_);                                          \
    const int B_TILE = (BN_) * (BK_);                                          \
                                                                               \
    auto issue_load_A = [&](int kk_base, int buf) {                            \
        for (int idx = tid; idx < A_TILE; idx += threads_per_block) {          \
            int sk_a, si_a;                                                    \
            LOAD_A_DECOMP(idx, (BM_), (BK_), sk_a, si_a);                      \
            int gi = block_i0 + si_a;                                          \
            int gk = kk_base + sk_a;                                           \
            if (gi < M && gk < K) {                                            \
                const PAIR* gptr = &pair_a[A_OFF(gi, gk, M, K)];               \
                As_v[buf][sk_a][si_a] = gptr->val;                             \
                As_c[buf][sk_a][si_a] = gptr->cnt;                             \
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
                Bs_v[buf][sk_b][sj_b] = gptr->val;                             \
                Bs_c[buf][sk_b][sj_b] = gptr->cnt;                             \
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
                    /* defer-mod: acc_c absorbs raw products in ACC_T; one    \
                       Barrett at epilogue. Gate enforced by launcher. */     \
                    ACC_T prod =                                               \
                        (ACC_T)(unsigned)ac[ti] * (ACC_T)(unsigned)bc[tj];     \
                    bool win = BETTER(pv, acc_v[ti][tj]);                      \
                    bool tie = (pv == acc_v[ti][tj]);                          \
                    acc_v[ti][tj] = win ? pv : acc_v[ti][tj];                  \
                    acc_c[ti][tj] = win ? prod :                               \
                        (tie ? (acc_c[ti][tj] + prod) : acc_c[ti][tj]);        \
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
        CP_ASYNC_WAIT_GROUP(1);                                                \
        __syncthreads();                                                       \
                                                                               \
        compute_on(cur_buf, (BK_));                                            \
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

#define DEFINE_TILED_F32(NAME, INIT_VAL, BETTER, A_OFF, B_OFF, LOAD_A, LOAD_B) \
extern "C" __global__ void NAME(                                               \
    const PairF32* __restrict__ pair_a,                                        \
    const PairF32* __restrict__ pair_b,                                        \
    PairF32* __restrict__ out_c,                                               \
    int M, int N, int K, int P, unsigned long long MU                          \
)                                                                              \
TROPICAL_MATMUL_TILED_BODY(float, PairF32, INIT_VAL, BETTER, A_OFF, B_OFF,     \
                           LOAD_A, LOAD_B, 64, 64, 32, 4, 4)

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

#define DEFINE_TILED_F32_PIPELINED(NAME, ACC_T, INIT_VAL, BETTER, A_OFF, B_OFF, LOAD_A, LOAD_B) \
extern "C" __global__ void NAME(                                               \
    const PairF32* __restrict__ pair_a,                                        \
    const PairF32* __restrict__ pair_b,                                        \
    PairF32* __restrict__ out_c,                                               \
    int M, int N, int K, int P, unsigned long long MU                          \
)                                                                              \
TROPICAL_MATMUL_TILED_PIPELINED_BODY(float, ACC_T, PairF32, INIT_VAL, BETTER, A_OFF, B_OFF, \
                                     LOAD_A, LOAD_B, 64, 64, 16, 4, 4)

#define DEFINE_TILED_F64_PIPELINED(NAME, ACC_T, INIT_VAL, BETTER, A_OFF, B_OFF, LOAD_A, LOAD_B) \
extern "C" __global__ void NAME(                                               \
    const PairF64* __restrict__ pair_a,                                        \
    const PairF64* __restrict__ pair_b,                                        \
    PairF64* __restrict__ out_c,                                               \
    int M, int N, int K, int P, unsigned long long MU                          \
)                                                                              \
TROPICAL_MATMUL_TILED_PIPELINED_BODY(double, ACC_T, PairF64, INIT_VAL, BETTER, A_OFF, B_OFF, \
                                     LOAD_A, LOAD_B, 32, 32, 8, 2, 4)

// _pl = u64 cnt accumulator (gate K·(P-1)² < 2^63).
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_max_NN_pl, unsigned long long, NEG_INF_F32, MAX_BETTER, A_OFF_N, B_OFF_N, LOAD_A_DECOMP_N, LOAD_B_DECOMP_N)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_max_NT_pl, unsigned long long, NEG_INF_F32, MAX_BETTER, A_OFF_N, B_OFF_T, LOAD_A_DECOMP_N, LOAD_B_DECOMP_T)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_max_TN_pl, unsigned long long, NEG_INF_F32, MAX_BETTER, A_OFF_T, B_OFF_N, LOAD_A_DECOMP_T, LOAD_B_DECOMP_N)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_max_TT_pl, unsigned long long, NEG_INF_F32, MAX_BETTER, A_OFF_T, B_OFF_T, LOAD_A_DECOMP_T, LOAD_B_DECOMP_T)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_min_NN_pl, unsigned long long, POS_INF_F32, MIN_BETTER, A_OFF_N, B_OFF_N, LOAD_A_DECOMP_N, LOAD_B_DECOMP_N)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_min_NT_pl, unsigned long long, POS_INF_F32, MIN_BETTER, A_OFF_N, B_OFF_T, LOAD_A_DECOMP_N, LOAD_B_DECOMP_T)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_min_TN_pl, unsigned long long, POS_INF_F32, MIN_BETTER, A_OFF_T, B_OFF_N, LOAD_A_DECOMP_T, LOAD_B_DECOMP_N)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_min_TT_pl, unsigned long long, POS_INF_F32, MIN_BETTER, A_OFF_T, B_OFF_T, LOAD_A_DECOMP_T, LOAD_B_DECOMP_T)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_max_NN_pl, unsigned long long, NEG_INF_F64, MAX_BETTER, A_OFF_N, B_OFF_N, LOAD_A_DECOMP_N, LOAD_B_DECOMP_N)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_max_NT_pl, unsigned long long, NEG_INF_F64, MAX_BETTER, A_OFF_N, B_OFF_T, LOAD_A_DECOMP_N, LOAD_B_DECOMP_T)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_max_TN_pl, unsigned long long, NEG_INF_F64, MAX_BETTER, A_OFF_T, B_OFF_N, LOAD_A_DECOMP_T, LOAD_B_DECOMP_N)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_max_TT_pl, unsigned long long, NEG_INF_F64, MAX_BETTER, A_OFF_T, B_OFF_T, LOAD_A_DECOMP_T, LOAD_B_DECOMP_T)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_min_NN_pl, unsigned long long, POS_INF_F64, MIN_BETTER, A_OFF_N, B_OFF_N, LOAD_A_DECOMP_N, LOAD_B_DECOMP_N)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_min_NT_pl, unsigned long long, POS_INF_F64, MIN_BETTER, A_OFF_N, B_OFF_T, LOAD_A_DECOMP_N, LOAD_B_DECOMP_T)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_min_TN_pl, unsigned long long, POS_INF_F64, MIN_BETTER, A_OFF_T, B_OFF_N, LOAD_A_DECOMP_T, LOAD_B_DECOMP_N)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_min_TT_pl, unsigned long long, POS_INF_F64, MIN_BETTER, A_OFF_T, B_OFF_T, LOAD_A_DECOMP_T, LOAD_B_DECOMP_T)

// _plu32 = u32 cnt accumulator (gate K·(P-1)² < 2^32). ~1.7× faster than _pl
// on A100; eliminates IMAD.WIDE.U32 + IMAD.X carry chain in the hot inner loop.
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_max_NN_plu32, unsigned int, NEG_INF_F32, MAX_BETTER, A_OFF_N, B_OFF_N, LOAD_A_DECOMP_N, LOAD_B_DECOMP_N)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_max_NT_plu32, unsigned int, NEG_INF_F32, MAX_BETTER, A_OFF_N, B_OFF_T, LOAD_A_DECOMP_N, LOAD_B_DECOMP_T)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_max_TN_plu32, unsigned int, NEG_INF_F32, MAX_BETTER, A_OFF_T, B_OFF_N, LOAD_A_DECOMP_T, LOAD_B_DECOMP_N)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_max_TT_plu32, unsigned int, NEG_INF_F32, MAX_BETTER, A_OFF_T, B_OFF_T, LOAD_A_DECOMP_T, LOAD_B_DECOMP_T)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_min_NN_plu32, unsigned int, POS_INF_F32, MIN_BETTER, A_OFF_N, B_OFF_N, LOAD_A_DECOMP_N, LOAD_B_DECOMP_N)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_min_NT_plu32, unsigned int, POS_INF_F32, MIN_BETTER, A_OFF_N, B_OFF_T, LOAD_A_DECOMP_N, LOAD_B_DECOMP_T)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_min_TN_plu32, unsigned int, POS_INF_F32, MIN_BETTER, A_OFF_T, B_OFF_N, LOAD_A_DECOMP_T, LOAD_B_DECOMP_N)
DEFINE_TILED_F32_PIPELINED(tropical_matmul_f32_min_TT_plu32, unsigned int, POS_INF_F32, MIN_BETTER, A_OFF_T, B_OFF_T, LOAD_A_DECOMP_T, LOAD_B_DECOMP_T)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_max_NN_plu32, unsigned int, NEG_INF_F64, MAX_BETTER, A_OFF_N, B_OFF_N, LOAD_A_DECOMP_N, LOAD_B_DECOMP_N)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_max_NT_plu32, unsigned int, NEG_INF_F64, MAX_BETTER, A_OFF_N, B_OFF_T, LOAD_A_DECOMP_N, LOAD_B_DECOMP_T)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_max_TN_plu32, unsigned int, NEG_INF_F64, MAX_BETTER, A_OFF_T, B_OFF_N, LOAD_A_DECOMP_T, LOAD_B_DECOMP_N)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_max_TT_plu32, unsigned int, NEG_INF_F64, MAX_BETTER, A_OFF_T, B_OFF_T, LOAD_A_DECOMP_T, LOAD_B_DECOMP_T)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_min_NN_plu32, unsigned int, POS_INF_F64, MIN_BETTER, A_OFF_N, B_OFF_N, LOAD_A_DECOMP_N, LOAD_B_DECOMP_N)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_min_NT_plu32, unsigned int, POS_INF_F64, MIN_BETTER, A_OFF_N, B_OFF_T, LOAD_A_DECOMP_N, LOAD_B_DECOMP_T)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_min_TN_plu32, unsigned int, POS_INF_F64, MIN_BETTER, A_OFF_T, B_OFF_N, LOAD_A_DECOMP_T, LOAD_B_DECOMP_N)
DEFINE_TILED_F64_PIPELINED(tropical_matmul_f64_min_TT_plu32, unsigned int, POS_INF_F64, MIN_BETTER, A_OFF_T, B_OFF_T, LOAD_A_DECOMP_T, LOAD_B_DECOMP_T)
