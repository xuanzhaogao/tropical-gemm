// Counting Tropical GEMM CUDA Kernels (spec D tiled + spec E Barrett optimization).
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
//
// Optimization: the per-step `% P` is software-emulated division on most
// GPUs (~50 cycles). We replace it with Barrett reduction using a host-
// precomputed reciprocal `mu = floor(2^64 / P)`. For x < 2^60 and P < 2^30:
//     q = __umul64hi(x, mu)             // ~4-8 cycles
//     r = x - q * P                     // in [0, 2P)
//     if (r >= P) r -= P                // correction
// Also defers the tie-branch modular reduction: count accumulator stays
// in u64, un-reduced, until write-back. For K <= 2^31 iterations, the
// accumulator is bounded by K * P < 2^61 — safe in u64.

// Infinity sentinels for MaxPlus / MinPlus tropical zero.
#define NEG_INF_F32 __int_as_float(0xff800000)
#define POS_INF_F32 __int_as_float(0x7f800000)
#define NEG_INF_F64 __longlong_as_double(0xfff0000000000000LL)
#define POS_INF_F64 __longlong_as_double(0x7ff0000000000000LL)

// Row-major offset.
#define OFFSET_ROW(row, col, ncols) ((row) * (ncols) + (col))

#define MAX_BETTER(a, b) ((a) > (b))
#define MIN_BETTER(a, b) ((a) < (b))

// Barrett reduction. Returns x mod P given x in [0, 2^60) and P in (0, 2^30).
// `mu = floor(2^64 / P)` is precomputed on the host and passed as a kernel arg.
// Result is in [0, P).
__device__ __forceinline__ unsigned long long barrett_mod(
    unsigned long long x, unsigned long long P, unsigned long long mu)
{
    unsigned long long q = __umul64hi(x, mu);
    unsigned long long r = x - q * P;
    if (r >= P) r -= P;
    return r;
}

// ============================================================================
// F32 COUNTING KERNEL MACRO
// ============================================================================
// Block sizes: 64x32x64, Thread sizes: 4x4. Threads per block: 256 (16x16).

#define COUNTING_GEMM_F32(NAME, INIT_VAL, BETTER)                              \
extern "C" __global__ __launch_bounds__(1024, 1) void NAME(                    \
    const float* __restrict__ value_a, const int* __restrict__ count_a,        \
    const float* __restrict__ value_b, const int* __restrict__ count_b,        \
    float* __restrict__ value_c, int* __restrict__ count_c,                    \
    int M, int N, int K, int P, unsigned long long MU                          \
) {                                                                            \
    const int BLOCK_SIZE_M = 64;                                               \
    const int BLOCK_SIZE_K = 32;                                               \
    const int BLOCK_SIZE_N = 64;                                               \
    const int THREAD_SIZE_M = 2;                                               \
    const int THREAD_SIZE_N = 2;                                               \
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
    __shared__ float As_val[BLOCK_SIZE_M * BLOCK_SIZE_K];                      \
    __shared__ int   As_cnt[BLOCK_SIZE_M * BLOCK_SIZE_K];                      \
    __shared__ float Bs_val[BLOCK_SIZE_K * BLOCK_SIZE_N];                      \
    __shared__ int   Bs_cnt[BLOCK_SIZE_K * BLOCK_SIZE_N];                      \
                                                                               \
    float              val_accum[THREAD_SIZE_M * THREAD_SIZE_N];               \
    unsigned long long cnt_accum[THREAD_SIZE_M * THREAD_SIZE_N];               \
    float regs_a_val[THREAD_SIZE_M];                                           \
    unsigned int regs_a_cnt[THREAD_SIZE_M];                                    \
    float regs_b_val[THREAD_SIZE_N];                                           \
    unsigned int regs_b_cnt[THREAD_SIZE_N];                                    \
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
    const unsigned long long Pull = (unsigned long long)P;                     \
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
                regs_a_cnt[tm] = (unsigned int)As_cnt[OFFSET_ROW(r, k, BLOCK_SIZE_K)]; \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                       \
                int c = threadIdx.x * THREAD_SIZE_N + tn;                      \
                regs_b_val[tn] = Bs_val[OFFSET_ROW(k, c, BLOCK_SIZE_N)];       \
                regs_b_cnt[tn] = (unsigned int)Bs_cnt[OFFSET_ROW(k, c, BLOCK_SIZE_N)]; \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                _Pragma("unroll")                                              \
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                   \
                    int idx = tm * THREAD_SIZE_N + tn;                         \
                    float              pv = regs_a_val[tm] + regs_b_val[tn];   \
                    unsigned long long prod =                                  \
                        (unsigned long long)regs_a_cnt[tm]                     \
                        * (unsigned long long)regs_b_cnt[tn];                  \
                    unsigned long long pc = barrett_mod(prod, Pull, MU);       \
                    /* Branchless tropical_add: reduces warp divergence. */   \
                    bool win = BETTER(pv, val_accum[idx]);                     \
                    bool tie = (pv == val_accum[idx]);                         \
                    val_accum[idx] = win ? pv : val_accum[idx];                \
                    unsigned long long added = cnt_accum[idx] + pc;            \
                    cnt_accum[idx] = win ? pc                                  \
                                   : (tie ? added : cnt_accum[idx]);           \
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
                /* Final reduction: acc < K * P < 2^61, safe to Barrett. */    \
                unsigned long long red = barrett_mod(cnt_accum[idx], Pull, MU);\
                count_c[OFFSET_ROW(row, col, N)] = (int)red;                   \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

// ============================================================================
// F64 COUNTING KERNEL MACRO
// ============================================================================
// Block sizes: 32x16x32, Thread sizes: 4x4. Threads per block: 64 (8x8).

#define COUNTING_GEMM_F64(NAME, INIT_VAL, BETTER)                              \
extern "C" __global__ void NAME(                                               \
    const double* __restrict__ value_a, const int* __restrict__ count_a,       \
    const double* __restrict__ value_b, const int* __restrict__ count_b,       \
    double* __restrict__ value_c, int* __restrict__ count_c,                   \
    int M, int N, int K, int P, unsigned long long MU                          \
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
    double             val_accum[THREAD_SIZE_M * THREAD_SIZE_N];               \
    unsigned long long cnt_accum[THREAD_SIZE_M * THREAD_SIZE_N];               \
    double regs_a_val[THREAD_SIZE_M];                                          \
    unsigned int regs_a_cnt[THREAD_SIZE_M];                                    \
    double regs_b_val[THREAD_SIZE_N];                                          \
    unsigned int regs_b_cnt[THREAD_SIZE_N];                                    \
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
    const unsigned long long Pull = (unsigned long long)P;                     \
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
                regs_a_cnt[tm] = (unsigned int)As_cnt[OFFSET_ROW(r, k, BLOCK_SIZE_K)]; \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                       \
                int c = threadIdx.x * THREAD_SIZE_N + tn;                      \
                regs_b_val[tn] = Bs_val[OFFSET_ROW(k, c, BLOCK_SIZE_N)];       \
                regs_b_cnt[tn] = (unsigned int)Bs_cnt[OFFSET_ROW(k, c, BLOCK_SIZE_N)]; \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int tm = 0; tm < THREAD_SIZE_M; ++tm) {                       \
                _Pragma("unroll")                                              \
                for (int tn = 0; tn < THREAD_SIZE_N; ++tn) {                   \
                    int idx = tm * THREAD_SIZE_N + tn;                         \
                    double             pv = regs_a_val[tm] + regs_b_val[tn];   \
                    unsigned long long prod =                                  \
                        (unsigned long long)regs_a_cnt[tm]                     \
                        * (unsigned long long)regs_b_cnt[tn];                  \
                    unsigned long long pc = barrett_mod(prod, Pull, MU);       \
                    /* Branchless tropical_add: reduces warp divergence. */   \
                    bool win = BETTER(pv, val_accum[idx]);                     \
                    bool tie = (pv == val_accum[idx]);                         \
                    val_accum[idx] = win ? pv : val_accum[idx];                \
                    unsigned long long added = cnt_accum[idx] + pc;            \
                    cnt_accum[idx] = win ? pc                                  \
                                   : (tie ? added : cnt_accum[idx]);           \
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
                unsigned long long red = barrett_mod(cnt_accum[idx], Pull, MU);\
                count_c[OFFSET_ROW(row, col, N)] = (int)red;                   \
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
