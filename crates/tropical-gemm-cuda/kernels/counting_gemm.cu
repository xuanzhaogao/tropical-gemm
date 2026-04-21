// Counting Tropical GEMM CUDA Kernels (spec C).
//
// SoA element layout: each logical element = (value: T, count: int32).
// Matrices are passed as parallel value and count pointers.
//
// Semiring operations (direction D):
//   tropical_mul: (va, ca) * (vb, cb) = (va + vb, (ca * cb) mod P)
//   tropical_add: strictly-better value wins; on tie, counts add mod P.

// Infinity sentinels for MaxPlus / MinPlus tropical zero.
#define NEG_INF_F32 __int_as_float(0xff800000)
#define POS_INF_F32 __int_as_float(0x7f800000)
#define NEG_INF_F64 __longlong_as_double(0xfff0000000000000LL)
#define POS_INF_F64 __longlong_as_double(0x7ff0000000000000LL)

// Row-major addressing: matrices are uploaded row-major from host, so
// A[i, kk] with ncols=k has offset i * k + kk.
#define OFFSET_ROW(row, col, ncols) ((row) * (ncols) + (col))

// One thread computes one C[i, j] cell. Grid is 2D (m x n threads).
#define COUNTING_KERNEL(name, T, D_ZERO, D_BETTER)                          \
extern "C" __global__ void name(                                            \
    const T*   value_a, const int* count_a,                                 \
    const T*   value_b, const int* count_b,                                 \
    T*         value_c, int*       count_c,                                 \
    int m, int n, int k, int P)                                             \
{                                                                           \
    int i = blockIdx.y * blockDim.y + threadIdx.y;                          \
    int j = blockIdx.x * blockDim.x + threadIdx.x;                          \
    if (i >= m || j >= n) return;                                           \
                                                                            \
    T   acc_val = (D_ZERO);                                                 \
    int acc_cnt = 0;                                                        \
                                                                            \
    for (int kk = 0; kk < k; ++kk) {                                        \
        T   va = value_a[OFFSET_ROW(i, kk, k)];                             \
        int ca = count_a[OFFSET_ROW(i, kk, k)];                             \
        T   vb = value_b[OFFSET_ROW(kk, j, n)];                             \
        int cb = count_b[OFFSET_ROW(kk, j, n)];                             \
        T   pv = va + vb;                                                   \
        int pc = (int)(((long long)ca * (long long)cb) % (long long)P);     \
        if (D_BETTER(pv, acc_val)) {                                        \
            acc_val = pv;                                                   \
            acc_cnt = pc;                                                   \
        } else if (D_BETTER(acc_val, pv)) {                                 \
            /* keep current accumulator */                                  \
        } else {                                                            \
            acc_cnt = (int)(((long long)acc_cnt + (long long)pc)            \
                            % (long long)P);                                \
        }                                                                   \
    }                                                                       \
                                                                            \
    value_c[OFFSET_ROW(i, j, n)] = acc_val;                                 \
    count_c[OFFSET_ROW(i, j, n)] = acc_cnt;                                 \
}

#define MAX_BETTER(a, b) ((a) > (b))
#define MIN_BETTER(a, b) ((a) < (b))

COUNTING_KERNEL(counting_gemm_f32_max, float,  NEG_INF_F32, MAX_BETTER)
COUNTING_KERNEL(counting_gemm_f32_min, float,  POS_INF_F32, MIN_BETTER)
COUNTING_KERNEL(counting_gemm_f64_max, double, NEG_INF_F64, MAX_BETTER)
COUNTING_KERNEL(counting_gemm_f64_min, double, POS_INF_F64, MIN_BETTER)
