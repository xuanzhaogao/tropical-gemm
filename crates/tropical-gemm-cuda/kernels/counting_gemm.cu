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
