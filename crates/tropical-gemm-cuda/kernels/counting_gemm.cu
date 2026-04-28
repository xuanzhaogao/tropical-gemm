// Counting Tropical GEMM CUDA kernels.
//
// Element layout: AoS — each input element is a packed (value, count) struct
// (PairF32: 8 B; PairF64: 16 B with 4 B padding). One LDG per element in the
// inner loop. Output buffers stay SoA (callers consume value and count
// separately). Matrices are row-major.
//
// Two parallelization strategies, dispatched by the host based on M*N:
//   - naive  (`counting_gemm_<T>_<dir>`):       one thread per output cell.
//                                               Wins for square / large M*N
//                                               (coalesced access, full SMs).
//   - warpk  (`counting_gemm_<T>_<dir>_warpk`): 32 threads cooperate on each
//                                               output cell, K-stride loop,
//                                               5-step shfl_xor reduction.
//                                               Wins for small M*N + large K.
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

#define TROPICAL_MATMUL_BODY(T, PAIR, INIT_VAL, BETTER, A_OFF, B_OFF)          \
{                                                                              \
    int i = blockIdx.y * blockDim.y + threadIdx.y;                             \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                             \
    if (i >= M || j >= N) return;                                              \
                                                                               \
    T                  acc_val = (INIT_VAL);                                   \
    unsigned long long acc_cnt = 0;                                            \
    const unsigned long long Pull = (unsigned long long)P;                     \
                                                                               \
    for (int k = 0; k < K; ++k) {                                              \
        PAIR a = pair_a[A_OFF(i, k, M, K)];                                    \
        PAIR b = pair_b[B_OFF(k, j, K, N)];                                    \
        T            pv = a.val + b.val;                                       \
        unsigned int ca = (unsigned int)a.cnt;                                 \
        unsigned int cb = (unsigned int)b.cnt;                                 \
        unsigned long long prod = (unsigned long long)ca * (unsigned long long)cb; \
        unsigned long long pc = barrett_mod(prod, Pull, MU);                   \
        bool win = BETTER(pv, acc_val);                                        \
        bool tie = (pv == acc_val);                                            \
        acc_val = win ? pv : acc_val;                                          \
        acc_cnt = win ? pc : (tie ? (acc_cnt + pc) : acc_cnt);                 \
    }                                                                          \
                                                                               \
    PAIR out;                                                                  \
    out.val = acc_val;                                                         \
    out.cnt = (int)barrett_mod(acc_cnt, Pull, MU);                             \
    out_c[(i) + (j) * M] = out;                                                \
}

#define DEFINE_TROPICAL_MATMUL(NAME, T, PAIR, INIT_VAL, BETTER, A_OFF, B_OFF)  \
extern "C" __global__ void NAME(                                               \
    const PAIR* __restrict__ pair_a,                                           \
    const PAIR* __restrict__ pair_b,                                           \
    PAIR* __restrict__ out_c,                                                  \
    int M, int N, int K, int P, unsigned long long MU                          \
)                                                                              \
TROPICAL_MATMUL_BODY(T, PAIR, INIT_VAL, BETTER, A_OFF, B_OFF)

// ----- 16 specializations: (T, dir) × (transA, transB) -----

DEFINE_TROPICAL_MATMUL(tropical_matmul_f32_max_NN, float,  PairF32, NEG_INF_F32, MAX_BETTER, A_OFF_N, B_OFF_N)
DEFINE_TROPICAL_MATMUL(tropical_matmul_f32_max_NT, float,  PairF32, NEG_INF_F32, MAX_BETTER, A_OFF_N, B_OFF_T)
DEFINE_TROPICAL_MATMUL(tropical_matmul_f32_max_TN, float,  PairF32, NEG_INF_F32, MAX_BETTER, A_OFF_T, B_OFF_N)
DEFINE_TROPICAL_MATMUL(tropical_matmul_f32_max_TT, float,  PairF32, NEG_INF_F32, MAX_BETTER, A_OFF_T, B_OFF_T)

DEFINE_TROPICAL_MATMUL(tropical_matmul_f32_min_NN, float,  PairF32, POS_INF_F32, MIN_BETTER, A_OFF_N, B_OFF_N)
DEFINE_TROPICAL_MATMUL(tropical_matmul_f32_min_NT, float,  PairF32, POS_INF_F32, MIN_BETTER, A_OFF_N, B_OFF_T)
DEFINE_TROPICAL_MATMUL(tropical_matmul_f32_min_TN, float,  PairF32, POS_INF_F32, MIN_BETTER, A_OFF_T, B_OFF_N)
DEFINE_TROPICAL_MATMUL(tropical_matmul_f32_min_TT, float,  PairF32, POS_INF_F32, MIN_BETTER, A_OFF_T, B_OFF_T)

DEFINE_TROPICAL_MATMUL(tropical_matmul_f64_max_NN, double, PairF64, NEG_INF_F64, MAX_BETTER, A_OFF_N, B_OFF_N)
DEFINE_TROPICAL_MATMUL(tropical_matmul_f64_max_NT, double, PairF64, NEG_INF_F64, MAX_BETTER, A_OFF_N, B_OFF_T)
DEFINE_TROPICAL_MATMUL(tropical_matmul_f64_max_TN, double, PairF64, NEG_INF_F64, MAX_BETTER, A_OFF_T, B_OFF_N)
DEFINE_TROPICAL_MATMUL(tropical_matmul_f64_max_TT, double, PairF64, NEG_INF_F64, MAX_BETTER, A_OFF_T, B_OFF_T)

DEFINE_TROPICAL_MATMUL(tropical_matmul_f64_min_NN, double, PairF64, POS_INF_F64, MIN_BETTER, A_OFF_N, B_OFF_N)
DEFINE_TROPICAL_MATMUL(tropical_matmul_f64_min_NT, double, PairF64, POS_INF_F64, MIN_BETTER, A_OFF_N, B_OFF_T)
DEFINE_TROPICAL_MATMUL(tropical_matmul_f64_min_TN, double, PairF64, POS_INF_F64, MIN_BETTER, A_OFF_T, B_OFF_N)
DEFINE_TROPICAL_MATMUL(tropical_matmul_f64_min_TT, double, PairF64, POS_INF_F64, MIN_BETTER, A_OFF_T, B_OFF_T)

// ============================================================================
// Naive kernel: one thread per output cell.
// ============================================================================

#define COUNTING_GEMM(NAME, T, PAIR, INIT_VAL, BETTER)                         \
extern "C" __global__ void NAME(                                               \
    const PAIR* __restrict__ pair_a,                                           \
    const PAIR* __restrict__ pair_b,                                           \
    T*    __restrict__ value_c, int* __restrict__ count_c,                     \
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
        PAIR a = pair_a[OFFSET_ROW(i, k, K)];                                  \
        PAIR b = pair_b[OFFSET_ROW(k, j, N)];                                  \
        T            pv = a.val + b.val;                                       \
        unsigned int ca = (unsigned int)a.cnt;                                 \
        unsigned int cb = (unsigned int)b.cnt;                                 \
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

COUNTING_GEMM(counting_gemm_f32_max, float,  PairF32, NEG_INF_F32, MAX_BETTER)
COUNTING_GEMM(counting_gemm_f32_min, float,  PairF32, POS_INF_F32, MIN_BETTER)
COUNTING_GEMM(counting_gemm_f64_max, double, PairF64, NEG_INF_F64, MAX_BETTER)
COUNTING_GEMM(counting_gemm_f64_min, double, PairF64, POS_INF_F64, MIN_BETTER)

// ============================================================================
// Warp-K-reduction kernel: 32 threads cooperate on each output cell.
//
// Block: blockDim = (32, 4, 1). threadIdx.x = lane, threadIdx.y picks one of
// the 4 output rows the warp computes. Grid: (N, ceil(M/4), 1).
//
// Memory access:
//   A: lanes at one step read 32 contiguous A elements -> coalesced.
//   B: lanes at one step read 32 elements at fixed col j, strided by N
//      -> non-coalesced. Acceptable in the small-M*N regime where this
//      kernel is dispatched; catastrophic at large N (host avoids it).
// ============================================================================

#define COUNTING_GEMM_WARPK(NAME, T, PAIR, INIT_VAL, BETTER)                   \
extern "C" __global__ void NAME(                                               \
    const PAIR* __restrict__ pair_a,                                           \
    const PAIR* __restrict__ pair_b,                                           \
    T*    __restrict__ value_c, int* __restrict__ count_c,                     \
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
        PAIR a = pair_a[OFFSET_ROW(i, k, K)];                                  \
        PAIR b = pair_b[OFFSET_ROW(k, j, N)];                                  \
        T            pv = a.val + b.val;                                       \
        unsigned int ca = (unsigned int)a.cnt;                                 \
        unsigned int cb = (unsigned int)b.cnt;                                 \
        unsigned long long prod = (unsigned long long)ca * (unsigned long long)cb; \
        unsigned long long pc = barrett_mod(prod, Pull, MU);                   \
        bool win = BETTER(pv, acc_val);                                        \
        bool tie = (pv == acc_val);                                            \
        acc_val  = win ? pv : acc_val;                                         \
        acc_cnt  = win ? pc                                                    \
                       : (tie ? barrett_mod(acc_cnt + pc, Pull, MU) : acc_cnt);\
    }                                                                          \
                                                                               \
    /* 5-step warp shuffle tree reduction. u64 acc_cnt shuffled hi/lo. */      \
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

COUNTING_GEMM_WARPK(counting_gemm_f32_max_warpk, float,  PairF32, NEG_INF_F32, MAX_BETTER)
COUNTING_GEMM_WARPK(counting_gemm_f32_min_warpk, float,  PairF32, POS_INF_F32, MIN_BETTER)
COUNTING_GEMM_WARPK(counting_gemm_f64_max_warpk, double, PairF64, NEG_INF_F64, MAX_BETTER)
COUNTING_GEMM_WARPK(counting_gemm_f64_min_warpk, double, PairF64, POS_INF_F64, MIN_BETTER)

// ============================================================================
// Spec G — ones-specialized variants.
//
// All input counts are 1 (the entry-point case for count_ground_states_gpu).
// Drops the count buffers, count multiply, and per-step Barrett. The output
// count at cell (i, j) is the number of k positions tied at the optimum,
// bounded by K — fits in u32 for any K <= 2^32. Single Barrett at the end.
//
// Inner loop work: 2 fp loads + 1 fp add + 1 compare + 1 conditional set/inc.
// That's the irreducible work for tropical-add-with-counting on ones inputs.
// ============================================================================

#define COUNTING_GEMM_ONES(NAME, T, INIT_VAL, BETTER)                          \
extern "C" __global__ void NAME(                                               \
    const T* __restrict__ value_a,                                             \
    const T* __restrict__ value_b,                                             \
    T*       __restrict__ value_c, int* __restrict__ count_c,                  \
    int M, int N, int K, int P, unsigned long long MU                          \
) {                                                                            \
    int i = blockIdx.y * blockDim.y + threadIdx.y;                             \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                             \
    if (i >= M || j >= N) return;                                              \
                                                                               \
    T            acc_val = (INIT_VAL);                                         \
    unsigned int acc_cnt = 0;                                                  \
                                                                               \
    for (int k = 0; k < K; ++k) {                                              \
        T va = value_a[OFFSET_ROW(i, k, K)];                                   \
        T vb = value_b[OFFSET_ROW(k, j, N)];                                   \
        T pv = va + vb;                                                        \
        bool win = BETTER(pv, acc_val);                                        \
        bool tie = (pv == acc_val);                                            \
        acc_val  = win ? pv : acc_val;                                         \
        acc_cnt  = win ? 1u : (tie ? (acc_cnt + 1u) : acc_cnt);                \
    }                                                                          \
    value_c[OFFSET_ROW(i, j, N)] = acc_val;                                    \
    count_c[OFFSET_ROW(i, j, N)] = (int)barrett_mod(                           \
        (unsigned long long)acc_cnt, (unsigned long long)P, MU);               \
}

COUNTING_GEMM_ONES(counting_gemm_f32_max_ones, float,  NEG_INF_F32, MAX_BETTER)
COUNTING_GEMM_ONES(counting_gemm_f32_min_ones, float,  POS_INF_F32, MIN_BETTER)
COUNTING_GEMM_ONES(counting_gemm_f64_max_ones, double, NEG_INF_F64, MAX_BETTER)
COUNTING_GEMM_ONES(counting_gemm_f64_min_ones, double, POS_INF_F64, MIN_BETTER)

// Spec H: warpk_ones reads B in TRANSPOSED layout (N × K row-major). Lanes
// share j and stride k by 32, so B^T[j, 32s+lane] is 32 contiguous elements
// per warp-step → coalesced. Driver uploads the appropriate B layout based
// on dispatch (transposed for warpk, untransposed for naive).
#define COUNTING_GEMM_WARPK_ONES(NAME, T, INIT_VAL, BETTER)                    \
extern "C" __global__ void NAME(                                               \
    const T* __restrict__ value_a,    /* M × K row-major  */                   \
    const T* __restrict__ value_b_t,  /* N × K row-major (transposed) */       \
    T*       __restrict__ value_c, int* __restrict__ count_c,                  \
    int M, int N, int K, int P, unsigned long long MU                          \
) {                                                                            \
    int lane = threadIdx.x;                                                    \
    int i    = blockIdx.y * blockDim.y + threadIdx.y;                          \
    int j    = blockIdx.x;                                                     \
    if (i >= M) return;                                                        \
                                                                               \
    T            acc_val = (INIT_VAL);                                         \
    unsigned int acc_cnt = 0;                                                  \
                                                                               \
    for (int k = lane; k < K; k += 32) {                                       \
        T va = value_a  [OFFSET_ROW(i, k, K)];                                 \
        T vb = value_b_t[OFFSET_ROW(j, k, K)];                                 \
        T pv = va + vb;                                                        \
        bool win = BETTER(pv, acc_val);                                        \
        bool tie = (pv == acc_val);                                            \
        acc_val  = win ? pv : acc_val;                                         \
        acc_cnt  = win ? 1u : (tie ? (acc_cnt + 1u) : acc_cnt);                \
    }                                                                          \
                                                                               \
    /* 5-step warp shuffle tree reduction. acc_cnt is u32 -> single shuffle */ \
    /* per step (no hi/lo split needed). */                                    \
    for (int off = 16; off > 0; off >>= 1) {                                   \
        T            ov = __shfl_xor_sync(0xffffffff, acc_val, off);           \
        unsigned int oc = __shfl_xor_sync(0xffffffff, acc_cnt, off);           \
        bool win = BETTER(ov, acc_val);                                        \
        bool tie = (ov == acc_val);                                            \
        acc_val = win ? ov : acc_val;                                          \
        acc_cnt = win ? oc : (tie ? (acc_cnt + oc) : acc_cnt);                 \
    }                                                                          \
                                                                               \
    if (lane == 0) {                                                           \
        value_c[OFFSET_ROW(i, j, N)] = acc_val;                                \
        count_c[OFFSET_ROW(i, j, N)] = (int)barrett_mod(                       \
            (unsigned long long)acc_cnt, (unsigned long long)P, MU);           \
    }                                                                          \
}

COUNTING_GEMM_WARPK_ONES(counting_gemm_f32_max_warpk_ones, float,  NEG_INF_F32, MAX_BETTER)
COUNTING_GEMM_WARPK_ONES(counting_gemm_f32_min_warpk_ones, float,  POS_INF_F32, MIN_BETTER)
COUNTING_GEMM_WARPK_ONES(counting_gemm_f64_max_warpk_ones, double, NEG_INF_F64, MAX_BETTER)
COUNTING_GEMM_WARPK_ONES(counting_gemm_f64_min_warpk_ones, double, POS_INF_F64, MIN_BETTER)
