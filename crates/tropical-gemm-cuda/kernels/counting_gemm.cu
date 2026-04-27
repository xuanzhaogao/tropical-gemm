// Counting Tropical GEMM CUDA Kernels — naive (one thread per output cell).
//
// SoA element layout: each logical element = (value: T, count: int32).
// Matrices are row-major.
//
// Discovered empirically that GPUArrays.jl's naive matmul outperforms
// our hand-tiled BLIS-style kernel for this workload by ~4x. Reason:
//   - 2x memory traffic (val + cnt per element) halves shared-mem occupancy.
//   - Per-thread u64 accumulator + Barrett state costs many registers,
//     capping occupancy further.
//   - L1/L2 caches handle data reuse adequately for a naive pattern.
// So we drop the shared-mem tiling and keep only the Barrett optimization.
//
// Semiring (direction D):
//   tropical_mul: (va, ca) * (vb, cb) = (va + vb, (ca * cb) mod P)
//   tropical_add: strictly-better value wins; on tie, counts add mod P.

#define NEG_INF_F32 __int_as_float(0xff800000)
#define POS_INF_F32 __int_as_float(0x7f800000)
#define NEG_INF_F64 __longlong_as_double(0xfff0000000000000LL)
#define POS_INF_F64 __longlong_as_double(0x7ff0000000000000LL)

#define OFFSET_ROW(row, col, ncols) ((row) * (ncols) + (col))

#define MAX_BETTER(a, b) ((a) > (b))
#define MIN_BETTER(a, b) ((a) < (b))

// Barrett reduction: x mod P via host-precomputed mu = floor(2^64 / P).
__device__ __forceinline__ unsigned long long barrett_mod(
    unsigned long long x, unsigned long long P, unsigned long long mu)
{
    unsigned long long q = __umul64hi(x, mu);
    unsigned long long r = x - q * P;
    if (r >= P) r -= P;
    return r;
}

// One thread per output cell.
#define COUNTING_GEMM(NAME, T, INIT_VAL, BETTER)                               \
extern "C" __global__ void NAME(                                               \
    const T*   __restrict__ value_a, const int* __restrict__ count_a,          \
    const T*   __restrict__ value_b, const int* __restrict__ count_b,          \
    T*         __restrict__ value_c, int*       __restrict__ count_c,          \
    int M, int N, int K, int P, unsigned long long MU                          \
) {                                                                            \
    int i = blockIdx.y * blockDim.y + threadIdx.y;                             \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                             \
    if (i >= M || j >= N) return;                                              \
                                                                               \
    T                  acc_val = (INIT_VAL);                                   \
    unsigned long long acc_cnt = 0;                                            \
    const unsigned long long Pull = (unsigned long long)P;                     \
                                                                               \
    for (int k = 0; k < K; ++k) {                                              \
        T            va = value_a[OFFSET_ROW(i, k, K)];                        \
        unsigned int ca = (unsigned int)count_a[OFFSET_ROW(i, k, K)];          \
        T            vb = value_b[OFFSET_ROW(k, j, N)];                        \
        unsigned int cb = (unsigned int)count_b[OFFSET_ROW(k, j, N)];          \
        T            pv = va + vb;                                             \
        unsigned long long prod = (unsigned long long)ca * (unsigned long long)cb; \
        unsigned long long pc = barrett_mod(prod, Pull, MU);                   \
        bool win = BETTER(pv, acc_val);                                        \
        bool tie = (pv == acc_val);                                            \
        acc_val = win ? pv : acc_val;                                          \
        acc_cnt = win ? pc : (tie ? (acc_cnt + pc) : acc_cnt);                 \
    }                                                                          \
    value_c[OFFSET_ROW(i, j, N)] = acc_val;                                    \
    count_c[OFFSET_ROW(i, j, N)] = (int)barrett_mod(acc_cnt, Pull, MU);        \
}

COUNTING_GEMM(counting_gemm_f32_max, float,  NEG_INF_F32, MAX_BETTER)
COUNTING_GEMM(counting_gemm_f32_min, float,  POS_INF_F32, MIN_BETTER)
COUNTING_GEMM(counting_gemm_f64_max, double, NEG_INF_F64, MAX_BETTER)
COUNTING_GEMM(counting_gemm_f64_min, double, POS_INF_F64, MIN_BETTER)

// ============================================================================
// Spec E — Stage A: warp-K-reduction variant.
//
// 32 threads of a warp cooperate on one output cell (i, j):
//   - thread `lane` (0..31) walks K-stride: k = lane, lane+32, lane+64, ...
//   - after the K-loop, a 5-step __shfl_xor_sync tree reduces 32 partial
//     (acc_val, acc_cnt) pairs using the tropical-add operator.
//   - lane 0 writes the result.
//
// Block layout: blockDim = (32, 4, 1). threadIdx.x = lane, threadIdx.y picks
// which of the 4 output rows the warp computes. Grid: (N, ceil(M/4), 1).
//
// Memory access:
//   A: lanes at one step read 32 contiguous A elements -> coalesced.
//   B: lanes at one step read 32 elements at fixed col j, strided by N
//      -> non-coalesced. L2 absorbs reuse across warps sharing j.
// ============================================================================

#define COUNTING_GEMM_WARPK(NAME, T, INIT_VAL, BETTER)                         \
extern "C" __global__ void NAME(                                               \
    const T*   __restrict__ value_a, const int* __restrict__ count_a,          \
    const T*   __restrict__ value_b, const int* __restrict__ count_b,          \
    T*         __restrict__ value_c, int*       __restrict__ count_c,          \
    int M, int N, int K, int P, unsigned long long MU                          \
) {                                                                            \
    int lane = threadIdx.x;                                                    \
    int i    = blockIdx.y * blockDim.y + threadIdx.y;                          \
    int j    = blockIdx.x;                                                     \
    if (i >= M) return;                                                        \
                                                                               \
    T                  acc_val = (INIT_VAL);                                   \
    unsigned long long acc_cnt = 0;                                            \
    const unsigned long long Pull = (unsigned long long)P;                     \
                                                                               \
    for (int k = lane; k < K; k += 32) {                                       \
        T            va = value_a[OFFSET_ROW(i, k, K)];                        \
        unsigned int ca = (unsigned int)count_a[OFFSET_ROW(i, k, K)];          \
        T            vb = value_b[OFFSET_ROW(k, j, N)];                        \
        unsigned int cb = (unsigned int)count_b[OFFSET_ROW(k, j, N)];          \
        T            pv = va + vb;                                             \
        unsigned long long prod = (unsigned long long)ca * (unsigned long long)cb; \
        unsigned long long pc = barrett_mod(prod, Pull, MU);                   \
        bool win = BETTER(pv, acc_val);                                        \
        bool tie = (pv == acc_val);                                            \
        acc_val  = win ? pv : acc_val;                                         \
        acc_cnt  = win ? pc                                                    \
                       : (tie ? barrett_mod(acc_cnt + pc, Pull, MU) : acc_cnt);\
    }                                                                          \
                                                                               \
    /* 5-step warp-shuffle tree reduction. u64 acc_cnt shuffled hi/lo. */      \
    for (int off = 16; off > 0; off >>= 1) {                                   \
        T   ov    = __shfl_xor_sync(0xffffffff, acc_val, off);                 \
        unsigned int oc_lo = __shfl_xor_sync(                                  \
            0xffffffff, (unsigned int)(acc_cnt & 0xffffffffULL), off);         \
        unsigned int oc_hi = __shfl_xor_sync(                                  \
            0xffffffff, (unsigned int)(acc_cnt >> 32), off);                   \
        unsigned long long oc =                                                \
            ((unsigned long long)oc_hi << 32) | (unsigned long long)oc_lo;     \
        bool win = BETTER(ov, acc_val);                                        \
        bool tie = (ov == acc_val);                                            \
        unsigned long long nc = win ? oc                                       \
            : (tie ? barrett_mod(acc_cnt + oc, Pull, MU) : acc_cnt);           \
        T nv = win ? ov : acc_val;                                             \
        acc_val = nv;                                                          \
        acc_cnt = nc;                                                          \
    }                                                                          \
                                                                               \
    if (lane == 0) {                                                           \
        value_c[OFFSET_ROW(i, j, N)] = acc_val;                                \
        count_c[OFFSET_ROW(i, j, N)] = (int)barrett_mod(acc_cnt, Pull, MU);    \
    }                                                                          \
}

COUNTING_GEMM_WARPK(counting_gemm_f32_max_warpk, float,  NEG_INF_F32, MAX_BETTER)
COUNTING_GEMM_WARPK(counting_gemm_f32_min_warpk, float,  POS_INF_F32, MIN_BETTER)
COUNTING_GEMM_WARPK(counting_gemm_f64_max_warpk, double, NEG_INF_F64, MAX_BETTER)
COUNTING_GEMM_WARPK(counting_gemm_f64_min_warpk, double, POS_INF_F64, MIN_BETTER)
