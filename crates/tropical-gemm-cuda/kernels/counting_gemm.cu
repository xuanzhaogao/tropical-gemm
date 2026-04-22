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
