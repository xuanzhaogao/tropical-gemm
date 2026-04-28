# Spec M Implementation Plan: Cleanup + N/T flags

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strip CRT/BigInt/host-wrapper accumulated surface and ship a single column-major BLAS-style mod-P counting tropical GEMM (`tropical_matmul(tA::Char, tB::Char, A::CuMatrix, B::CuMatrix)`) with N/T flags.

**Architecture:** Three phases. **Phase A (Tasks 1–3):** add new CUDA kernels (16 column-major NN/NT/TN/TT specializations × 2 dtypes × 2 dirs), new Rust driver, new C ABI — all coexisting with the old surface. **Phase B (Task 4):** add new Julia `tropical_matmul`/`tropical_matmul!` consuming the new C ABI. **Phase C (Tasks 5–8):** delete the old surface (Julia first, then Rust, then `crt.rs`). **Phase D (Tasks 9–10):** rewrite the bench, update memory, final validation. Every commit between phases is green.

**Tech Stack:** Rust 1.x with cudarc 0.12.1 (NVRTC + primary CUDA context), CUDA C++ kernel via NVRTC, Julia 1.10+ with CUDA.jl 5.x.

---

## File structure (after cleanup)

```
crates/tropical-gemm-cuda/
  src/
    c_api.rs              4 entries: tg_tropical_matmul_<T>_<D>; thread-local errors; version
    context.rs            primary CUDA ctx, lazy global; updated COUNTING_KERNEL_NAMES (16)
    counting_kernel.rs    one fn launch_tropical_matmul<T, D>(ctx, tA, tB, M, K, N, ptrs, p)
    error.rs              kept; ERR_BOUND_TOO_LARGE removed
    gpu_mat.rs            kept (used by tropical_matmul_gpu)
    kernels.rs            kept (used by tropical_matmul_gpu)
    lib.rs                kept (tropical_matmul_gpu retained for tropical-gemm-python)
    matmul_mod.rs         one driver: tropical_matmul_kernel
    memory.rs             kept
    pair.rs               PairF32, PairF64 + DeviceRepr/ValidAsZeroBits only
  kernels/
    counting_gemm.cu      column-major macro + 16 instantiations
    tropical_gemm.cu      kept (tropical_matmul_gpu)

CountingTropicalGEMM.jl/
  src/CountingTropicalGEMM.jl
                          ~250 LOC: ModCountingTropical[Min] types,
                          tropical_matmul, tropical_matmul!, error type
  test/runtests.jl        ~150 LOC: type tests, 4 transpose-combo tests,
                          edge cases, error tests
  bench/bench_mul.jl      device-only benchmark over the new API
  Project.toml            deps: CUDA, Libdl
```

---

## Task 1: New CUDA kernels (column-major, AoS output, 16 specializations)

**Files:**
- Modify: `crates/tropical-gemm-cuda/kernels/counting_gemm.cu`
- Modify: `crates/tropical-gemm-cuda/src/context.rs` (extend COUNTING_KERNEL_NAMES)

- [ ] **Step 1: Read the existing kernel file**

```bash
cat /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm/crates/tropical-gemm-cuda/kernels/counting_gemm.cu
```

Note: `barrett_mod` helper, `PairF32`/`PairF64` device structs, `MAX_BETTER`/`MIN_BETTER`, `NEG_INF_*`/`POS_INF_*` already exist.

- [ ] **Step 2: Append column-major NN/NT/TN/TT macro template**

In `crates/tropical-gemm-cuda/kernels/counting_gemm.cu`, just before the line that says `#define COUNTING_GEMM(NAME, T, PAIR, INIT_VAL, BETTER)`, add:

```cpp
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
```

(This adds **alongside** the existing `COUNTING_GEMM` / `COUNTING_GEMM_WARPK` macros — don't touch them yet.)

- [ ] **Step 3: Register the 16 new kernel names**

In `crates/tropical-gemm-cuda/src/context.rs`, find `COUNTING_KERNEL_NAMES` (around line 75). Append the 16 new names to the end of the array (keep the existing entries):

```rust
const COUNTING_KERNEL_NAMES: &[&str] = &[
    // (existing entries unchanged) ...
    "counting_gemm_f64_max_warpk_ones",
    "counting_gemm_f64_min_warpk_ones",
    // Spec M: column-major NN/NT/TN/TT specializations.
    "tropical_matmul_f32_max_NN",
    "tropical_matmul_f32_max_NT",
    "tropical_matmul_f32_max_TN",
    "tropical_matmul_f32_max_TT",
    "tropical_matmul_f32_min_NN",
    "tropical_matmul_f32_min_NT",
    "tropical_matmul_f32_min_TN",
    "tropical_matmul_f32_min_TT",
    "tropical_matmul_f64_max_NN",
    "tropical_matmul_f64_max_NT",
    "tropical_matmul_f64_max_TN",
    "tropical_matmul_f64_max_TT",
    "tropical_matmul_f64_min_NN",
    "tropical_matmul_f64_min_NT",
    "tropical_matmul_f64_min_TN",
    "tropical_matmul_f64_min_TT",
];
```

- [ ] **Step 4: Verify NVRTC compiles**

Run:
```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo test -p tropical-gemm-cuda --lib --no-run 2>&1 | tail -10'
```

Then run the existing tests to confirm the kernels load:
```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo test -p tropical-gemm-cuda --lib 2>&1 | tail -3'
```
Expected: PASS, all 73 lib tests still pass; the new kernel symbols compile but aren't yet called.

If NVRTC fails to compile a kernel, read the error and fix the macro. Common issues: missing `;` in macro body, undefined `INIT_VAL`, etc.

- [ ] **Step 5: Commit**

```bash
git add crates/tropical-gemm-cuda/kernels/counting_gemm.cu \
        crates/tropical-gemm-cuda/src/context.rs
git commit -m "$(cat <<'EOF'
Spec M Task 1: add column-major NN/NT/TN/TT counting matmul kernels

Sixteen new CUDA kernel symbols (tropical_matmul_<T>_<D>_<NN|NT|TN|TT>)
co-exist with the existing counting_gemm_* family. AoS output (single
PairT store per cell) and column-major operand indexing. Macros
A_OFF_N / A_OFF_T / B_OFF_N / B_OFF_T select the four address patterns.
Existing kernels untouched; warpk path stays for now.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: New Rust driver `tropical_matmul_kernel<T, D>`

**Files:**
- Modify: `crates/tropical-gemm-cuda/src/counting_kernel.rs` (add `launch_tropical_matmul`)
- Modify: `crates/tropical-gemm-cuda/src/matmul_mod.rs` (add `tropical_matmul_kernel`)

- [ ] **Step 1: Write the failing test**

In `crates/tropical-gemm-cuda/src/matmul_mod.rs`, append to the existing `mod tests` block:

```rust
    #[test]
    fn tropical_matmul_nn_2x2_max_p7() {
        // Column-major A (2×2): A[0,0]=1, A[1,0]=3, A[0,1]=2, A[1,1]=4.
        // Bytes flat (col-major): [1, 3, 2, 4].
        // Column-major B (2×2): B[0,0]=5, B[1,0]=7, B[0,1]=6, B[1,1]=8.
        // Bytes flat (col-major): [5, 7, 6, 8].
        // Max-plus C = A * B. Counts all 1.
        // C[0,0] = max(1+5, 2+7) = 9   (k=1 wins)
        // C[1,0] = max(3+5, 4+7) = 11  (k=1 wins)
        // C[0,1] = max(1+6, 2+8) = 10  (k=1 wins)
        // C[1,1] = max(3+6, 4+8) = 12  (k=1 wins)
        // Output column-major: [9, 11, 10, 12].
        use crate::pair::PairF32;
        use tropical_gemm::types::Max;
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let pair_a_host = vec![
            PairF32::new(1.0, 1), PairF32::new(3.0, 1),
            PairF32::new(2.0, 1), PairF32::new(4.0, 1),
        ];
        let pair_b_host = vec![
            PairF32::new(5.0, 1), PairF32::new(7.0, 1),
            PairF32::new(6.0, 1), PairF32::new(8.0, 1),
        ];

        let pair_a_dev = ctx.device().htod_copy(pair_a_host).unwrap();
        let pair_b_dev = ctx.device().htod_copy(pair_b_host).unwrap();
        let out_dev = ctx.device().alloc_zeros::<PairF32>(4).unwrap();

        use cudarc::driver::DevicePtr;
        let a_ptr = *pair_a_dev.device_ptr();
        let b_ptr = *pair_b_dev.device_ptr();
        let out_ptr = *out_dev.device_ptr();

        tropical_matmul_kernel::<f32, Max>(
            ctx, 'N', 'N', 2, 2, 2, a_ptr, b_ptr, 7, out_ptr,
        ).expect("kernel ok");

        let out = ctx.device().dtoh_sync_copy(&out_dev).unwrap();
        assert_eq!(out[0].val, 9.0);  assert_eq!(out[0].cnt, 1);
        assert_eq!(out[1].val, 11.0); assert_eq!(out[1].cnt, 1);
        assert_eq!(out[2].val, 10.0); assert_eq!(out[2].cnt, 1);
        assert_eq!(out[3].val, 12.0); assert_eq!(out[3].cnt, 1);
    }
```

- [ ] **Step 2: Verify it fails to compile**

Run:
```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo test -p tropical-gemm-cuda tropical_matmul_nn_2x2_max_p7 2>&1 | tail -8'
```
Expected: compile error — `tropical_matmul_kernel` not defined.

- [ ] **Step 3: Add `launch_tropical_matmul` in counting_kernel.rs**

Append to `crates/tropical-gemm-cuda/src/counting_kernel.rs`, after the existing definitions:

```rust
/// Spec M: column-major NN/NT/TN/TT counting matmul launch.
/// Operates on raw device pointers (caller-owned, e.g. CUDA.jl).
/// `ctx.device().synchronize()` is called before launch to coordinate
/// with CUDA.jl's stream.
pub fn launch_tropical_matmul<T, D>(
    ctx: &CudaContext,
    tA: char,
    tB: char,
    m: usize,
    k: usize,
    n: usize,
    a_dev_ptr: u64,
    b_dev_ptr: u64,
    p: i32,
    out_dev_ptr: u64,
) -> Result<()>
where
    T: tropical_gemm::types::TropicalScalar
        + cudarc::driver::DeviceRepr
        + cudarc::driver::ValidAsZeroBits
        + Default + Clone + Copy
        + crate::pair::PackPair
        + 'static,
    D: tropical_gemm::types::TropicalDirection,
    (T, D): TropicalMatmulKernelName<T, D>,
{
    let suffix = match (tA, tB) {
        ('N', 'N') => "NN",
        ('N', 'T') => "NT",
        ('T', 'N') => "TN",
        ('T', 'T') => "TT",
        _ => return Err(crate::error::CudaError::InvalidState(format!(
            "tA/tB must be in {{'N','T'}}, got tA={}, tB={}", tA, tB
        ))),
    };
    let kernel_name_owned: String = format!("{}_{}", <(T, D) as TropicalMatmulKernelName<T, D>>::BASE_NAME, suffix);
    // cudarc requires &'static str for kernel lookup. We leak the formatted
    // name once per (T, D, tA, tB) at runtime — 16 leaks total, bounded.
    let kernel_name: &'static str = Box::leak(kernel_name_owned.into_boxed_str());
    let kernel = ctx.get_kernel(kernel_name)?;

    // Block 16x16, grid covers M, N (column-major: blockIdx.y is row, blockIdx.x is col).
    let block: (u32, u32, u32) = (16, 16, 1);
    let grid: (u32, u32, u32) = (
        ((n + 15) / 16) as u32,
        ((m + 15) / 16) as u32,
        1,
    );
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: grid,
        block_dim: block,
        shared_mem_bytes: 0,
    };

    // Barrett mu for the kernel.
    let mu: u64 = if p > 1 {
        ((1u128 << 64) / p as u128) as u64
    } else {
        0
    };

    // Pre-launch sync: wait for any pending CUDA.jl uploads on caller's stream.
    ctx.device().synchronize()?;

    // Pass raw device pointers via DevPtr (impls DeviceRepr).
    let a_dp = crate::matmul_mod::DevPtr(a_dev_ptr);
    let b_dp = crate::matmul_mod::DevPtr(b_dev_ptr);
    let out_dp = crate::matmul_mod::DevPtr(out_dev_ptr);

    unsafe {
        use cudarc::driver::LaunchAsync;
        kernel.launch(
            cfg,
            (
                a_dp, b_dp, out_dp,
                m as i32, n as i32, k as i32, p, mu,
            ),
        )?;
    }

    ctx.device().synchronize()?;
    Ok(())
}

/// Trait providing the base kernel name for each (T, D) combo. The runtime
/// dispatch to NN/NT/TN/TT appends a suffix.
pub trait TropicalMatmulKernelName<T, D> {
    const BASE_NAME: &'static str;
}
impl TropicalMatmulKernelName<f32, tropical_gemm::types::Max> for (f32, tropical_gemm::types::Max) {
    const BASE_NAME: &'static str = "tropical_matmul_f32_max";
}
impl TropicalMatmulKernelName<f32, tropical_gemm::types::Min> for (f32, tropical_gemm::types::Min) {
    const BASE_NAME: &'static str = "tropical_matmul_f32_min";
}
impl TropicalMatmulKernelName<f64, tropical_gemm::types::Max> for (f64, tropical_gemm::types::Max) {
    const BASE_NAME: &'static str = "tropical_matmul_f64_max";
}
impl TropicalMatmulKernelName<f64, tropical_gemm::types::Min> for (f64, tropical_gemm::types::Min) {
    const BASE_NAME: &'static str = "tropical_matmul_f64_min";
}
```

(Existing imports at the top of the file should already include `LaunchAsync`, `LaunchConfig`, etc. If not, add them.)

- [ ] **Step 4: Add `tropical_matmul_kernel` driver in matmul_mod.rs**

Append to `crates/tropical-gemm-cuda/src/matmul_mod.rs`, just before the `#[cfg(test)] mod tests` block:

```rust
/// Spec M: column-major counting tropical matmul.
/// Caller owns device buffers (e.g. allocated via CUDA.jl).
/// Validates flags, P, dims; launches the right kernel for (T, D, tA, tB).
pub fn tropical_matmul_kernel<T, D>(
    ctx: &CudaContext,
    tA: char,
    tB: char,
    m: usize,
    k: usize,
    n: usize,
    a_dev_ptr: u64,
    b_dev_ptr: u64,
    p: i32,
    out_dev_ptr: u64,
) -> Result<()>
where
    T: tropical_gemm::types::TropicalScalar
        + DeviceRepr
        + ValidAsZeroBits
        + Default + Clone + Copy
        + PackPair
        + 'static,
    D: TropicalDirection,
    (T, D): crate::counting_kernel::TropicalMatmulKernelName<T, D>,
    (T, D): CountingCudaKernel<T, D>,
{
    if tA != 'N' && tA != 'T' {
        return Err(CudaError::InvalidState(format!(
            "tA must be 'N' or 'T', got {:?}", tA
        )));
    }
    if tB != 'N' && tB != 'T' {
        return Err(CudaError::InvalidState(format!(
            "tB must be 'N' or 'T', got {:?}", tB
        )));
    }
    if p < P_MIN {
        return Err(CudaError::InvalidState(format!(
            "modulus must satisfy {} <= p < 2^31, got {}", P_MIN, p
        )));
    }
    if m == 0 || k == 0 || n == 0 {
        return Err(CudaError::InvalidState(
            "dimensions must be non-zero".into(),
        ));
    }
    crate::counting_kernel::launch_tropical_matmul::<T, D>(
        ctx, tA, tB, m, k, n, a_dev_ptr, b_dev_ptr, p, out_dev_ptr,
    )
}
```

(`P_MIN`, `DevPtr`, `CudaContext`, `Result`, `CudaError`, `DeviceRepr`, `ValidAsZeroBits`, `PackPair`, `TropicalDirection`, `CountingCudaKernel` should already be imported at the top of the file from prior tasks. Check; add if missing.)

- [ ] **Step 5: Run the test**

Run:
```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo test -p tropical-gemm-cuda tropical_matmul_nn_2x2_max_p7 -- --nocapture 2>&1 | tail -15'
```
Expected: PASS.

- [ ] **Step 6: Add transpose-flag and Min-direction tests**

Append to the same `mod tests` block:

```rust
    #[test]
    fn tropical_matmul_tt_2x2_max_p7() {
        // 'T','T': A buffer is K×M = 2×2 col-major; B buffer is N×K = 2×2 col-major.
        // Same logical A and B as the NN test. Verify TT path.
        // Logical A (M=2, K=2): same as NN test.
        // Storage 'T' for A: K×M col-major. Bytes flat: [A^T(0,0), A^T(1,0), A^T(0,1), A^T(1,1)]
        //                                              = [A(0,0),  A(0,1),  A(1,0),  A(1,1)]
        //                                              = [1, 2, 3, 4].
        // Storage 'T' for B: N×K col-major. Bytes flat: [B(0,0), B(0,1), B(1,0), B(1,1)] = [5, 6, 7, 8].
        use crate::pair::PairF32;
        use tropical_gemm::types::Max;
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let pair_a_host = vec![
            PairF32::new(1.0, 1), PairF32::new(2.0, 1),
            PairF32::new(3.0, 1), PairF32::new(4.0, 1),
        ];
        let pair_b_host = vec![
            PairF32::new(5.0, 1), PairF32::new(6.0, 1),
            PairF32::new(7.0, 1), PairF32::new(8.0, 1),
        ];

        let pair_a_dev = ctx.device().htod_copy(pair_a_host).unwrap();
        let pair_b_dev = ctx.device().htod_copy(pair_b_host).unwrap();
        let out_dev = ctx.device().alloc_zeros::<PairF32>(4).unwrap();

        use cudarc::driver::DevicePtr;
        let a_ptr = *pair_a_dev.device_ptr();
        let b_ptr = *pair_b_dev.device_ptr();
        let out_ptr = *out_dev.device_ptr();

        tropical_matmul_kernel::<f32, Max>(
            ctx, 'T', 'T', 2, 2, 2, a_ptr, b_ptr, 7, out_ptr,
        ).expect("kernel ok");

        let out = ctx.device().dtoh_sync_copy(&out_dev).unwrap();
        // Same logical A * B as NN; output column-major [9, 11, 10, 12].
        assert_eq!(out[0].val, 9.0);
        assert_eq!(out[1].val, 11.0);
        assert_eq!(out[2].val, 10.0);
        assert_eq!(out[3].val, 12.0);
    }

    #[test]
    fn tropical_matmul_nn_4x4_min_random_p11() {
        // Min direction f64 NN with discrete inputs forcing ties; reference cross-check.
        use crate::pair::PairF64;
        use tropical_gemm::types::Min;
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let m = 4usize; let k = 6usize; let n = 4usize;
        let p = 11i32;
        let a_val: Vec<f64> = (0..m*k).map(|i| (i % 5) as f64).collect();
        let a_cnt: Vec<i32> = (0..m*k).map(|i| ((i + 1) % 11) as i32).collect();
        let b_val: Vec<f64> = (0..k*n).map(|i| (i % 4) as f64).collect();
        let b_cnt: Vec<i32> = (0..k*n).map(|i| ((i + 2) % 11) as i32).collect();

        // Pack to PairF64 in column-major order: A is M×K, idx = i + j*M.
        let mut pair_a_host = vec![PairF64::default(); m * k];
        for i in 0..m { for j in 0..k {
            // Logical A[i, j] from row-major source a_val[i*k + j].
            pair_a_host[i + j * m] = PairF64::new(a_val[i * k + j], a_cnt[i * k + j]);
        }}
        let mut pair_b_host = vec![PairF64::default(); k * n];
        for i in 0..k { for j in 0..n {
            pair_b_host[i + j * k] = PairF64::new(b_val[i * n + j], b_cnt[i * n + j]);
        }}

        let pair_a_dev = ctx.device().htod_copy(pair_a_host).unwrap();
        let pair_b_dev = ctx.device().htod_copy(pair_b_host).unwrap();
        let out_dev = ctx.device().alloc_zeros::<PairF64>(m * n).unwrap();

        use cudarc::driver::DevicePtr;
        tropical_matmul_kernel::<f64, Min>(
            ctx, 'N', 'N', m, k, n,
            *pair_a_dev.device_ptr(),
            *pair_b_dev.device_ptr(),
            p,
            *out_dev.device_ptr(),
        ).expect("kernel ok");

        let out = ctx.device().dtoh_sync_copy(&out_dev).unwrap();

        // Reference: row-major scan, then check column-major output.
        for i in 0..m {
            for j in 0..n {
                let mut best = f64::INFINITY;
                let mut acc: i64 = 0;
                for kk in 0..k {
                    let v = a_val[i*k + kk] + b_val[kk*n + j];
                    let c = (a_cnt[i*k + kk] as i64) * (b_cnt[kk*n + j] as i64) % (p as i64);
                    if v < best { best = v; acc = c; }
                    else if v == best { acc = (acc + c) % (p as i64); }
                }
                let cell = out[i + j * m];
                assert_eq!(cell.val, best, "value mismatch at ({},{})", i, j);
                assert_eq!(cell.cnt as i64, acc, "count mismatch at ({},{})", i, j);
            }
        }
    }

    #[test]
    fn tropical_matmul_rejects_bad_flag() {
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let err = tropical_matmul_kernel::<f32, tropical_gemm::types::Max>(
            ctx, 'X', 'N', 2, 2, 2, 0, 0, 7, 0,
        );
        assert!(err.is_err(), "bad flag must be rejected");
        let msg = format!("{:?}", err.unwrap_err());
        assert!(msg.contains("tA must be"), "got: {}", msg);
    }

    #[test]
    fn tropical_matmul_rejects_p_one() {
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let err = tropical_matmul_kernel::<f32, tropical_gemm::types::Max>(
            ctx, 'N', 'N', 2, 2, 2, 0, 0, 1, 0,
        );
        assert!(err.is_err(), "p=1 must be rejected");
    }
```

- [ ] **Step 7: Run all new tests**

```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo test -p tropical-gemm-cuda tropical_matmul_ -- --nocapture 2>&1 | tail -20'
```
Expected: 4 tests pass.

- [ ] **Step 8: Run full Rust crate test sweep**

```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo test -p tropical-gemm-cuda --lib 2>&1 | tail -3'
```
Expected: lib goes from 73 → 77, all green.

- [ ] **Step 9: Commit**

```bash
git add crates/tropical-gemm-cuda/src/counting_kernel.rs \
        crates/tropical-gemm-cuda/src/matmul_mod.rs
git commit -m "$(cat <<'EOF'
Spec M Task 2: tropical_matmul_kernel Rust driver + dispatch

Adds tropical_matmul_kernel<T, D>(ctx, tA, tB, M, K, N, ptrs, p) and
launch_tropical_matmul plus a TropicalMatmulKernelName trait giving
the per-(T, D) base kernel name. Dispatches to the matching NN/NT/TN/TT
kernel from Task 1 by formatting the suffix at runtime. Pre-launch
device sync for CUDA.jl coordination. Validates flags, p, dims.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: New C ABI `tg_tropical_matmul_<T>_<D>`

**Files:**
- Modify: `crates/tropical-gemm-cuda/src/c_api.rs`

- [ ] **Step 1: Write the failing test**

Append to the `tests` module in `c_api.rs`:

```rust
    #[test]
    fn tg_tropical_matmul_f32_max_nn_smoke() {
        use crate::pair::PairF32;
        let ctx = crate::get_global_context().expect("CUDA ctx");
        let pair_a_host = vec![
            PairF32::new(1.0, 1), PairF32::new(3.0, 1),
            PairF32::new(2.0, 1), PairF32::new(4.0, 1),
        ];
        let pair_b_host = vec![
            PairF32::new(5.0, 1), PairF32::new(7.0, 1),
            PairF32::new(6.0, 1), PairF32::new(8.0, 1),
        ];
        let pair_a_dev = ctx.device().htod_copy(pair_a_host).unwrap();
        let pair_b_dev = ctx.device().htod_copy(pair_b_host).unwrap();
        let out_dev = ctx.device().alloc_zeros::<PairF32>(4).unwrap();

        use cudarc::driver::DevicePtr;
        let code = tg_tropical_matmul_f32_max(
            b'N' as i8, b'N' as i8,
            2, 2, 2,
            *pair_a_dev.device_ptr(),
            *pair_b_dev.device_ptr(),
            7,
            *out_dev.device_ptr(),
        );
        assert_eq!(code, OK);
        let out = ctx.device().dtoh_sync_copy(&out_dev).unwrap();
        assert_eq!(out[0].val, 9.0);
        assert_eq!(out[1].val, 11.0);
        assert_eq!(out[2].val, 10.0);
        assert_eq!(out[3].val, 12.0);
    }

    #[test]
    fn tg_tropical_matmul_rejects_bad_flag() {
        let code = tg_tropical_matmul_f32_max(
            b'X' as i8, b'N' as i8, 2, 2, 2, 0, 0, 7, 0,
        );
        assert_eq!(code, ERR_INVALID_INPUT);
    }
```

- [ ] **Step 2: Verify failure**

```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo test -p tropical-gemm-cuda tg_tropical_matmul 2>&1 | tail -8'
```
Expected: compile error — function not defined.

- [ ] **Step 3: Add the C ABI entries**

In `crates/tropical-gemm-cuda/src/c_api.rs`, add the import near the existing `use crate::matmul_mod::*;` line:

```rust
use crate::matmul_mod::tropical_matmul_kernel;
```

After the last `cabi_matmul_mod_p_pair_dev!` invocation (search for the existing 4 `cabi_matmul_mod_p_pair_dev!` lines), append:

```rust
// ---------------------------------------------------------------------------
// Spec M: column-major BLAS-style mod-P counting tropical matmul.
// Caller-owned device buffers; flags select the (transA, transB) kernel.
// ---------------------------------------------------------------------------

fn run_tropical_matmul<T, D>(
    tA: i8, tB: i8,
    m: usize, k: usize, n: usize,
    a_dev: u64, b_dev: u64,
    p: i32,
    out_dev: u64,
) -> i32
where
    T: tropical_gemm::types::TropicalScalar
        + cudarc::driver::DeviceRepr
        + cudarc::driver::ValidAsZeroBits
        + Default + Clone + Copy
        + crate::pair::PackPair
        + 'static,
    D: tropical_gemm::types::TropicalDirection,
    (T, D): crate::counting_kernel::TropicalMatmulKernelName<T, D>,
    (T, D): crate::counting_kernel::CountingCudaKernel<T, D>,
{
    let tA_char = tA as u8 as char;
    let tB_char = tB as u8 as char;
    if tA_char != 'N' && tA_char != 'T' {
        store_error(format!("tA must be 'N' or 'T', got {:?}", tA_char));
        return ERR_INVALID_INPUT;
    }
    if tB_char != 'N' && tB_char != 'T' {
        store_error(format!("tB must be 'N' or 'T', got {:?}", tB_char));
        return ERR_INVALID_INPUT;
    }
    if p < 2 {
        store_error(format!("modulus must be >= 2, got {}", p));
        return ERR_INVALID_INPUT;
    }
    if m == 0 || k == 0 || n == 0 {
        store_error("dimensions must be non-zero");
        return ERR_INVALID_INPUT;
    }
    if a_dev == 0 || b_dev == 0 || out_dev == 0 {
        store_error("null device pointer");
        return ERR_INVALID_INPUT;
    }

    let ctx = match get_global_context() {
        Ok(c) => c,
        Err(e) => {
            store_error(format!("CUDA context init failed: {}", e));
            return ERR_CUDA;
        }
    };

    match tropical_matmul_kernel::<T, D>(
        ctx, tA_char, tB_char, m, k, n, a_dev, b_dev, p, out_dev,
    ) {
        Ok(()) => OK,
        Err(e) => { store_error(format!("{}", e)); ERR_CUDA }
    }
}

macro_rules! cabi_tropical_matmul {
    ($name:ident, $T:ty, $D:ty) => {
        #[no_mangle]
        pub extern "C" fn $name(
            tA: i8, tB: i8,
            m: usize, k: usize, n: usize,
            a_dev: u64, b_dev: u64,
            p: i32,
            out_dev: u64,
        ) -> c_int {
            let res = catch_unwind(AssertUnwindSafe(|| {
                run_tropical_matmul::<$T, $D>(tA, tB, m, k, n, a_dev, b_dev, p, out_dev)
            }));
            match res {
                Ok(code) => code,
                Err(_) => { store_error("Rust panic across FFI boundary"); ERR_INTERNAL }
            }
        }
    };
}

cabi_tropical_matmul!(tg_tropical_matmul_f32_max, f32, Max);
cabi_tropical_matmul!(tg_tropical_matmul_f32_min, f32, Min);
cabi_tropical_matmul!(tg_tropical_matmul_f64_max, f64, Max);
cabi_tropical_matmul!(tg_tropical_matmul_f64_min, f64, Min);
```

(`Max`, `Min`, `OK`, `ERR_INVALID_INPUT`, `ERR_CUDA`, `ERR_INTERNAL`, `c_int`, `catch_unwind`, `AssertUnwindSafe`, `get_global_context`, `store_error` should already be in scope from the top of the file.)

Note: `tA: i8` because Cchar in C is `char` which Julia/CUDA-side passes via `Cchar`/`Int8`. Pass ASCII codes (`b'N' = 78`, `b'T' = 84`).

- [ ] **Step 4: Run new tests**

```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo test -p tropical-gemm-cuda tg_tropical_matmul -- --nocapture 2>&1 | tail -15'
```
Expected: 2 PASS.

- [ ] **Step 5: Run full Rust crate suite**

```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo test -p tropical-gemm-cuda 2>&1 | grep "test result"'
```
Expected: lib 79 (was 77), integration 21, all green.

- [ ] **Step 6: Build the cdylib**

```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo build --release -p tropical-gemm-cuda 2>&1 | tail -3'
```
Expected: success.

- [ ] **Step 7: Commit**

```bash
git add crates/tropical-gemm-cuda/src/c_api.rs
git commit -m "$(cat <<'EOF'
Spec M Task 3: tg_tropical_matmul_<T>_<D> C ABI

Four extern C entries (f32/f64 x Max/Min) wrapping
tropical_matmul_kernel. Caller passes (tA, tB) as i8 ASCII char codes,
device pointers as u64, logical (M, K, N). Validates flags ('N' or 'T'),
p >= 2, dims > 0, non-null pointers. catch_unwind panic guard.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Julia `tropical_matmul` + `tropical_matmul!` (new API alongside old)

**Files:**
- Modify: `CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl`
- Modify: `CountingTropicalGEMM.jl/test/runtests.jl`

- [ ] **Step 1: Append failing tests**

Append to `CountingTropicalGEMM.jl/test/runtests.jl`, before the final `end`:

```julia
    @testset "Spec M tropical_matmul NN f32 Max" begin
        P = 7
        Random.seed!(101)
        # Construct host matrices, pack to ModCountingTropical.
        A_host = [ModCountingTropical{Float32, P}(
                    Float32(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:5, _ in 1:8]
        B_host = [ModCountingTropical{Float32, P}(
                    Float32(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:8, _ in 1:6]
        A = CuArray(A_host); B = CuArray(B_host)

        # Reference: pure Julia max-plus + count multiply mod P.
        ref = Matrix{ModCountingTropical{Float32, P}}(undef, 5, 6)
        for i in 1:5, j in 1:6
            best_n = -Inf32; best_c = Int32(0)
            for kk in 1:8
                v = A_host[i, kk].n + B_host[kk, j].n
                c = Int32(mod(Int64(A_host[i, kk].c) * Int64(B_host[kk, j].c), Int64(P)))
                if v > best_n
                    best_n = v; best_c = c
                elseif v == best_n
                    best_c = Int32(mod(Int64(best_c) + Int64(c), Int64(P)))
                end
            end
            ref[i, j] = ModCountingTropical{Float32, P}(best_n, best_c)
        end

        C_dev = tropical_matmul('N', 'N', A, B)
        C_host = Array(C_dev)
        @test C_host == ref
    end

    @testset "Spec M tropical_matmul TT f64 Min" begin
        P = 11
        Random.seed!(102)
        A_host = [ModCountingTropicalMin{Float64, P}(
                    Float64(rand(0:4)), Int32(rand(0:P-1))) for _ in 1:6, _ in 1:9]
        B_host = [ModCountingTropicalMin{Float64, P}(
                    Float64(rand(0:4)), Int32(rand(0:P-1))) for _ in 1:9, _ in 1:7]
        # Build the transposed inputs A_T (Julia 9×6) and B_T (Julia 7×9).
        AT_host = [A_host[i, j] for j in 1:9, i in 1:6]
        BT_host = [B_host[i, j] for j in 1:7, i in 1:9]
        AT = CuArray(AT_host); BT = CuArray(BT_host)

        # Reference computed using algebraic A * B (the SAME logical product the
        # 'T','T' call should produce, since op_user('T', AT) = AT^T = A and same for B).
        ref = Matrix{ModCountingTropicalMin{Float64, P}}(undef, 6, 7)
        for i in 1:6, j in 1:7
            best_n = Inf; best_c = Int32(0)
            for kk in 1:9
                v = A_host[i, kk].n + B_host[kk, j].n
                c = Int32(mod(Int64(A_host[i, kk].c) * Int64(B_host[kk, j].c), Int64(P)))
                if v < best_n
                    best_n = v; best_c = c
                elseif v == best_n
                    best_c = Int32(mod(Int64(best_c) + Int64(c), Int64(P)))
                end
            end
            ref[i, j] = ModCountingTropicalMin{Float64, P}(best_n, best_c)
        end

        C_dev = tropical_matmul('T', 'T', AT, BT)
        @test Array(C_dev) == ref
    end

    @testset "Spec M tropical_matmul NT, TN f32 Max" begin
        P = 13
        Random.seed!(103)
        A_host = [ModCountingTropical{Float32, P}(
                    Float32(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:4, _ in 1:5]
        B_host = [ModCountingTropical{Float32, P}(
                    Float32(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:5, _ in 1:6]

        # Reference C = A * B (Julia algebraic).
        ref = Matrix{ModCountingTropical{Float32, P}}(undef, 4, 6)
        for i in 1:4, j in 1:6
            best_n = -Inf32; best_c = Int32(0)
            for kk in 1:5
                v = A_host[i, kk].n + B_host[kk, j].n
                c = Int32(mod(Int64(A_host[i, kk].c) * Int64(B_host[kk, j].c), Int64(P)))
                if v > best_n
                    best_n = v; best_c = c
                elseif v == best_n
                    best_c = Int32(mod(Int64(best_c) + Int64(c), Int64(P)))
                end
            end
            ref[i, j] = ModCountingTropical{Float32, P}(best_n, best_c)
        end

        # NT path: pass A as 'N' (no transpose) and B^T as 'T'.
        BT_host = [B_host[i, j] for j in 1:6, i in 1:5]
        A = CuArray(A_host); BT = CuArray(BT_host)
        C_NT = tropical_matmul('N', 'T', A, BT)
        @test Array(C_NT) == ref

        # TN path: pass A^T as 'T' and B as 'N'.
        AT_host = [A_host[i, j] for j in 1:5, i in 1:4]
        AT = CuArray(AT_host); B = CuArray(B_host)
        C_TN = tropical_matmul('T', 'N', AT, B)
        @test Array(C_TN) == ref
    end

    @testset "Spec M tropical_matmul! reuse" begin
        P = 7
        Random.seed!(104)
        A_host = [ModCountingTropical{Float32, P}(
                    Float32(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:4, _ in 1:5]
        B_host = [ModCountingTropical{Float32, P}(
                    Float32(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:5, _ in 1:6]
        A = CuArray(A_host); B = CuArray(B_host)
        C = CuArray{ModCountingTropical{Float32, P}}(undef, 4, 6)
        ref = Array(tropical_matmul('N', 'N', A, B))
        tropical_matmul!('N', 'N', A, B, C)
        @test Array(C) == ref

        # Reuse with different inputs.
        A2 = CuArray([ModCountingTropical{Float32, P}(
                        Float32(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:4, _ in 1:5])
        ref2 = Array(tropical_matmul('N', 'N', A2, B))
        tropical_matmul!('N', 'N', A2, B, C)
        @test Array(C) == ref2
    end

    @testset "Spec M tropical_matmul errors" begin
        P = 7
        A = CuArray([ModCountingTropical{Float32, P}(0.0f0, Int32(1)) for _ in 1:2, _ in 1:3])
        B = CuArray([ModCountingTropical{Float32, P}(0.0f0, Int32(1)) for _ in 1:3, _ in 1:4])
        Bbad = CuArray([ModCountingTropical{Float32, P}(0.0f0, Int32(1)) for _ in 1:5, _ in 1:4])

        # Bad flag.
        @test_throws ArgumentError tropical_matmul('X', 'N', A, B)
        @test_throws ArgumentError tropical_matmul('N', 'C', A, B)
        # Bad P (P=1 in element type).
        Aone = CuArray([ModCountingTropical{Float32, 1}(0.0f0, Int32(0)) for _ in 1:2, _ in 1:2])
        @test_throws ArgumentError tropical_matmul('N', 'N', Aone, Aone)
        # K mismatch.
        @test_throws DimensionMismatch tropical_matmul('N', 'N', A, Bbad)
        # Mismatched directions.
        Bmin = CuArray([ModCountingTropicalMin{Float32, P}(0.0f0, Int32(1)) for _ in 1:3, _ in 1:4])
        @test_throws MethodError tropical_matmul('N', 'N', A, Bmin)
    end
```

Ensure `using CUDA, Random` are at the top of `runtests.jl` (likely already imported from prior tasks; add if missing).

- [ ] **Step 2: Verify failure**

```
bash -c 'module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm/CountingTropicalGEMM.jl && julia --project=. -e "using Pkg; Pkg.test()" 2>&1 | tail -15'
```
Expected: error — `tropical_matmul` does not have a method matching `(Char, Char, CuMatrix, CuMatrix)` (the existing methods are for host `Matrix`).

- [ ] **Step 3: Add the new methods**

In `CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl`, just before the closing `end # module`, append:

```julia
# ---------------------------------------------------------------------------
# Spec M: column-major BLAS-style mod-P counting tropical GEMM.
# tropical_matmul(tA, tB, A::CuMatrix, B::CuMatrix) and the in-place variant.
# Inputs and output are device-resident CuMatrix; element type encodes
# direction (ModCountingTropical = max-plus; ModCountingTropicalMin = min-plus)
# and modulus P. Inferred from element type, no separate args.
# ---------------------------------------------------------------------------

# Per-(T, dir) ccall thunk symbols.
const _SPEC_M_SYMS = Dict(
    (Float32, :max) => :tg_tropical_matmul_f32_max,
    (Float32, :min) => :tg_tropical_matmul_f32_min,
    (Float64, :max) => :tg_tropical_matmul_f64_max,
    (Float64, :min) => :tg_tropical_matmul_f64_min,
)

# Build per-(T, dir) thunk fns at module-load via @eval. Each takes raw u64
# device pointers + (M, K, N, P, tA, tB) and ccalls the right C ABI entry.
for (T, sym_max) in ((Float32, :tg_tropical_matmul_f32_max),
                     (Float64, :tg_tropical_matmul_f64_max))
    sym_min = T === Float32 ? :tg_tropical_matmul_f32_min : :tg_tropical_matmul_f64_min
    for (dir_sym, sym) in ((:max, sym_max), (:min, sym_min))
        thunk_name = Symbol("_tg_tropical_matmul_", T, "_", dir_sym)
        @eval function $thunk_name(tA::Cchar, tB::Cchar,
                                   m::Csize_t, k::Csize_t, n::Csize_t,
                                   a_dev::UInt64, b_dev::UInt64,
                                   p::Int32,
                                   out_dev::UInt64)
            _check_version()
            code = ccall(($(QuoteNode(sym)), _libpath()), Cint,
                (Cchar, Cchar, Csize_t, Csize_t, Csize_t, UInt64, UInt64, Int32, UInt64),
                tA, tB, m, k, n, a_dev, b_dev, p, out_dev)
            if code != Int32(0)
                _throw_for(Int32(code))
            end
            return nothing
        end
    end
end

@inline function _validate_flag(flag::Char)
    flag == 'N' || flag == 'T' || throw(ArgumentError(
        "tA/tB must be 'N' or 'T', got $(flag)"))
end

@inline function _validate_p(P::Integer)
    2 <= P < (Int64(1) << 31) || throw(ArgumentError(
        "modulus P must satisfy 2 <= P < 2^31, got $P"))
end

# Compute logical (M, K, N) from CuMatrix shapes and flags.
@inline function _logical_dims(tA::Char, tB::Char, sA::NTuple{2,Int}, sB::NTuple{2,Int})
    rA, cA = sA; rB, cB = sB
    M = (tA == 'N') ? rA : cA
    Kused = (tA == 'N') ? cA : rA
    Kchk = (tB == 'N') ? rB : cB
    N = (tB == 'N') ? cB : rB
    Kused == Kchk || throw(DimensionMismatch(
        "inner K mismatch: op($tA, A) gives K=$Kused, op($tB, B) gives K=$Kchk"))
    return M, Kused, N
end

# Extract device pointer as UInt64 from a CuMatrix.
@inline _u64ptr(A::CuArray) = UInt64(UInt(pointer(A)))

# Public: tropical_matmul (max-plus on ModCountingTropical).
function tropical_matmul(tA::Char, tB::Char,
                         A::CuMatrix{ModCountingTropical{T, P}},
                         B::CuMatrix{ModCountingTropical{T, P}}
                        ) where {T <: Union{Float32, Float64}, P}
    _validate_flag(tA); _validate_flag(tB); _validate_p(P)
    M, K, N = _logical_dims(tA, tB, size(A), size(B))
    out = CuArray{ModCountingTropical{T, P}}(undef, M, N)
    _spec_m_dispatch(T, :max, tA, tB, M, K, N, A, B, out)
    return out
end

# Public: tropical_matmul (min-plus on ModCountingTropicalMin).
function tropical_matmul(tA::Char, tB::Char,
                         A::CuMatrix{ModCountingTropicalMin{T, P}},
                         B::CuMatrix{ModCountingTropicalMin{T, P}}
                        ) where {T <: Union{Float32, Float64}, P}
    _validate_flag(tA); _validate_flag(tB); _validate_p(P)
    M, K, N = _logical_dims(tA, tB, size(A), size(B))
    out = CuArray{ModCountingTropicalMin{T, P}}(undef, M, N)
    _spec_m_dispatch(T, :min, tA, tB, M, K, N, A, B, out)
    return out
end

# Public: in-place tropical_matmul!.
function tropical_matmul!(tA::Char, tB::Char,
                          A::CuMatrix{ModCountingTropical{T, P}},
                          B::CuMatrix{ModCountingTropical{T, P}},
                          C::CuMatrix{ModCountingTropical{T, P}}
                         ) where {T <: Union{Float32, Float64}, P}
    _validate_flag(tA); _validate_flag(tB); _validate_p(P)
    M, K, N = _logical_dims(tA, tB, size(A), size(B))
    size(C) == (M, N) || throw(DimensionMismatch(
        "C is $(size(C)) but op($tA,A)*op($tB,B) is $((M, N))"))
    _spec_m_dispatch(T, :max, tA, tB, M, K, N, A, B, C)
    return C
end

function tropical_matmul!(tA::Char, tB::Char,
                          A::CuMatrix{ModCountingTropicalMin{T, P}},
                          B::CuMatrix{ModCountingTropicalMin{T, P}},
                          C::CuMatrix{ModCountingTropicalMin{T, P}}
                         ) where {T <: Union{Float32, Float64}, P}
    _validate_flag(tA); _validate_flag(tB); _validate_p(P)
    M, K, N = _logical_dims(tA, tB, size(A), size(B))
    size(C) == (M, N) || throw(DimensionMismatch(
        "C is $(size(C)) but op($tA,A)*op($tB,B) is $((M, N))"))
    _spec_m_dispatch(T, :min, tA, tB, M, K, N, A, B, C)
    return C
end

# Internal: dispatch to the right (T, dir) ccall thunk.
function _spec_m_dispatch(::Type{T}, dir_sym::Symbol,
                          tA::Char, tB::Char,
                          M::Int, K::Int, N::Int,
                          A::CuArray, B::CuArray, C::CuArray) where {T}
    a_ptr = _u64ptr(A); b_ptr = _u64ptr(B); c_ptr = _u64ptr(C)
    if T === Float32 && dir_sym === :max
        _tg_tropical_matmul_Float32_max(Cchar(tA), Cchar(tB),
            Csize_t(M), Csize_t(K), Csize_t(N), a_ptr, b_ptr, Int32(P_for(A)), c_ptr)
    elseif T === Float32 && dir_sym === :min
        _tg_tropical_matmul_Float32_min(Cchar(tA), Cchar(tB),
            Csize_t(M), Csize_t(K), Csize_t(N), a_ptr, b_ptr, Int32(P_for(A)), c_ptr)
    elseif T === Float64 && dir_sym === :max
        _tg_tropical_matmul_Float64_max(Cchar(tA), Cchar(tB),
            Csize_t(M), Csize_t(K), Csize_t(N), a_ptr, b_ptr, Int32(P_for(A)), c_ptr)
    elseif T === Float64 && dir_sym === :min
        _tg_tropical_matmul_Float64_min(Cchar(tA), Cchar(tB),
            Csize_t(M), Csize_t(K), Csize_t(N), a_ptr, b_ptr, Int32(P_for(A)), c_ptr)
    else
        error("unreachable: T=$T, dir=$dir_sym")
    end
    return nothing
end

# Extract the modulus P parameter from a CuArray's element type.
@inline P_for(::CuArray{ModCountingTropical{T, P}}) where {T, P} = P
@inline P_for(::CuArray{ModCountingTropicalMin{T, P}}) where {T, P} = P

# Add new exports (do NOT remove existing exports yet; cleanup is Task 5).
# tropical_matmul is already exported from prior tasks; tropical_matmul! is new.
# Check the existing export block at the top of the module and ensure it includes
# tropical_matmul!. If not present, add it to the existing `export` block.
```

Find the existing `export tropical_matmul, tropical_matmul_min` line near the top of the module. Adjust to:

```julia
export tropical_matmul, tropical_matmul!, tropical_matmul_min
```

(`tropical_matmul!` is new for Spec M; the others stay until Task 5 cleanup.)

- [ ] **Step 4: Run new tests**

```
bash -c 'module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm/CountingTropicalGEMM.jl && julia --project=. -e "using Pkg; Pkg.test()" 2>&1 | tail -25'
```
Expected: 5 new testsets pass. The earlier 79+ tests still pass.

- [ ] **Step 5: Commit**

```bash
git add CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl \
        CountingTropicalGEMM.jl/test/runtests.jl
git commit -m "$(cat <<'EOF'
Spec M Task 4: tropical_matmul + tropical_matmul! Julia methods

Adds the new BLAS-style API on top of the Spec M C ABI. Methods specialize
on element type (ModCountingTropical for max, ModCountingTropicalMin for
min) so direction mismatches raise MethodError. Validates 'N'/'T' flags,
P range, K-match. Pointer extraction via pointer(::CuArray) -> UInt64
(no Julia-level reinterpret). Allocates output CuMatrix; writes AoS bytes
directly. Tests cover all four NN/NT/TN/TT combos vs reference and the
common error paths.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Delete Julia old surface

**Files:**
- Modify: `CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl` (large deletions)
- Modify: `CountingTropicalGEMM.jl/test/runtests.jl` (drop old testsets)
- Modify: `CountingTropicalGEMM.jl/Project.toml` (drop deps)

- [ ] **Step 1: Read the Julia source to map deletions**

```bash
grep -n "^function\|^export\|^using\|^const\|^struct\|@testset\|^module" \
  /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm/CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl \
  /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm/CountingTropicalGEMM.jl/test/runtests.jl
```

- [ ] **Step 2: Delete from `src/CountingTropicalGEMM.jl`**

Edit the file to:

a) **Update the `using` block** at top of module to:
```julia
using CUDA
using Libdl
```
(Remove `using LinearAlgebra`, `using Mods`, `using TropicalNumbers`.)

b) **Update the `export` block** to:
```julia
export ModCountingTropical, ModCountingTropicalMin
export tropical_matmul, tropical_matmul!
export CountingTropicalGEMMError
```

(Remove `Max`, `Min`, `CountedMatU64`, `TropicalMatrix`, `count_ground_states_gpu_u64`, `bench_kernel_only_u64`, `BoundTooLargeError`, `tropical_matmul_min`, `tropical_matmul_dev`, `tropical_matmul_dev_min`, `cuda_pair_buffer`, `CountingTropicalMin`.)

c) **Delete from the module body**:
- The `Max`/`Min` empty-struct definitions and their sections.
- `CountedMatU64{T}` struct.
- `BoundTooLargeError` struct + `Base.showerror` + `ERR_BOUND_TOO_LARGE` constant.
- The `count_ground_states_gpu_u64` declaration and the `for (T, sym_max, sym_min) in (...)` loop that defines its methods.
- The `bench_kernel_only_u64` declaration and corresponding `@eval` loop.
- The `_rowmajor`, `_from_rowmajor` helpers (used only by `count_ground_states_gpu_u64`).
- The `TropicalMatrix` struct + all its methods (`Base.size`, `Base.getindex`, `Base.IndexStyle`, `Base.:*`, etc.).
- The `CountingTropicalMin{T, CT}` struct (the local Mods-interop one — distinct from `ModCountingTropicalMin`!) + its `Base.zero/one/+/*/==/showerror` methods.
- `tropical_matmul_min` (the one taking `Matrix{CountingTropicalMin{T, Mod{P}}}`).
- The `tropical_matmul(A::AbstractMatrix{ModCountingTropical{T,P}}, B::...)` host-Matrix method, the host-Matrix `tropical_matmul_min`, and any host-Matrix wrappers that allocate `CuArray` internally and route through old dev API.
- `LinearAlgebra.mul!` overloads for ModCountingTropical / ModCountingTropicalMin.
- `tropical_matmul_dev`, `tropical_matmul_dev_min`, `cuda_pair_buffer`.
- `Base.convert(::Type{ModCountingTropical{T,P}}, ::CountingTropical{T, Mod{P}})` (and Min variant) interop converters.
- `_FFI_SYMS`, `_MOD_FAST_SYMS` Dict definitions.
- `_pair_type`, `_modulus`, `_check_mod_p`, `_row_major_pair`, `_zip_to_modct`, `_tropical_matmul_core`, `_ensure_cuda_jl_context`, `_tg_mod_pair_ccall` — review each: keep if used by Spec M code (Task 4), else delete.

**Keep:**
- `ModCountingTropical{T, P}`, `ModCountingTropicalMin{T, P}` structs + `+`, `*`, `zero`, `one`, `==`.
- `PairF32`, `PairF64` struct definitions (still used by Rust ABI on the other side; not strictly needed in Julia post-cleanup, but harmless to keep — they're internal).
- `CountingTropicalGEMMError` struct + `Base.showerror`.
- `ERR_INVALID_INPUT`, `ERR_CUDA`, `ERR_INTERNAL` constants. (Drop `ERR_BOUND_TOO_LARGE`.)
- `EXPECTED_API_VERSION`, `LIB_PATH`, `_resolve_library`, `__init__`, `_libpath`, `_check_version`, `_last_error_message`, `_throw_for`.
- The Spec M code from Task 4 (thunks, `_validate_flag`, `_validate_p`, `_logical_dims`, `_u64ptr`, `_spec_m_dispatch`, `P_for`, public `tropical_matmul` / `tropical_matmul!` methods).

After editing, the module file should be ~250 LOC (down from ~828).

- [ ] **Step 3: Update `_throw_for`** to drop the `ERR_BOUND_TOO_LARGE` branch:

```julia
function _throw_for(code::Int32)
    msg = _last_error_message()
    throw(CountingTropicalGEMMError(code, msg))
end
```

- [ ] **Step 4: Update `Project.toml`**

Edit `CountingTropicalGEMM.jl/Project.toml` `[deps]` block to:
```toml
[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
Libdl = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
```

Drop `LinearAlgebra`, `Mods`, `TropicalNumbers` from `[deps]`. Keep:
```toml
[compat]
CUDA = "5"
julia = "1.10"
```

Drop `Mods`, `TropicalNumbers` from `[compat]`. Keep `Random` and `Test` in `[extras]` and the `test` target.

- [ ] **Step 5: Delete from `test/runtests.jl`**

Drop these `@testset` blocks (entire block including the testset name and body):
- `"f32 Max small"`, `"f64 Min vs reference (randomized)"`, `"all-ties large K (Max, f32)"`, `"BoundTooLargeError"`, `"DimensionMismatch"` (the old count_ground_states tests)
- `"TropicalMatrix * TropicalMatrix (...)"` testsets (4 of them)
- `"tropical_matmul slow path (Mod{P, Int}, f32 Max)"`, `"tropical_matmul fast path (Mod{P, Int32}, f64 Max)"`, `"tropical_matmul dispatch on U"`
- `"mul! over ModCountingTropical"`, `"mul! over ModCountingTropicalMin"`
- `"tropical_matmul_min cross-check (...)"`, `"tropical_matmul edge cases"`, `"tropical_matmul errors"`, `"convert from CountingTropical{T, Mod{P}}"`
- `"Spec L: device-pointer matmul"` (replaced by Spec M's API)
- `"tropical_matmul direct on CountingTropical"` if present

**Keep:**
- `"ModCountingTropical type and semiring"` (Task 6 type tests).
- All `"Spec M ..."` testsets from Task 4.

Drop the `using TropicalNumbers, Mods` line if it appears anywhere in the test file.

After editing, the test file should be ~150 LOC (down from ~456).

- [ ] **Step 6: Run the test suite**

```
bash -c 'module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm/CountingTropicalGEMM.jl && julia --project=. -e "using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.test()" 2>&1 | tail -15'
```
Expected: only ModCountingTropical type tests + Spec M tests run; ~30 tests total, all green.

- [ ] **Step 7: Commit**

```bash
git add CountingTropicalGEMM.jl/src/CountingTropicalGEMM.jl \
        CountingTropicalGEMM.jl/test/runtests.jl \
        CountingTropicalGEMM.jl/Project.toml \
        CountingTropicalGEMM.jl/Manifest.toml
git commit -m "$(cat <<'EOF'
Spec M Task 5: delete Julia old surface

Drops count_ground_states_gpu_u64, bench_kernel_only_u64, CountedMatU64,
BoundTooLargeError, TropicalMatrix, CountingTropicalMin (local
Mods-interop type), Max/Min tag types, host-Matrix tropical_matmul[_min],
mul! overloads, tropical_matmul_dev[_min], cuda_pair_buffer, the
Mods/TropicalNumbers Base.convert interop, and the corresponding test
sets. Drops LinearAlgebra/Mods/TropicalNumbers deps. Keeps the new
Spec M tropical_matmul / tropical_matmul! and the
ModCountingTropical[Min] type machinery.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Delete old Rust C ABI entries

**Files:**
- Modify: `crates/tropical-gemm-cuda/src/c_api.rs`

- [ ] **Step 1: Identify deletions**

In `crates/tropical-gemm-cuda/src/c_api.rs`, identify the following sections and delete them:

a) The `count_ground_states_gpu_u64` family:
- `run_u64<T, D>(...)` function
- `cabi_count_ground_states_u64!` macro
- `tg_count_ground_states_gpu_u64_f32_max`, `_f32_min`, `_f64_max`, `_f64_min` invocations
- The `classify_cuda_error` body's u64-bound branch (rewrite to always return `ERR_CUDA`)

b) The `bench_kernel_only` family:
- `bench_kernel_only_impl<T, D>(...)` function
- `cabi_bench_kernel_only!` macro
- `tg_bench_kernel_only_u64_*` (4 entries)

c) The `matmul_mod_p` family (slow path):
- `run_matmul_mod_p<T, D>(...)`
- `cabi_matmul_mod_p!` macro
- `tg_matmul_mod_p_*` (4 entries)

d) The `matmul_mod_p_pair` family (host-pair fast path):
- `run_matmul_mod_p_pair<T, D>(...)`
- `cabi_matmul_mod_p_pair!` macro
- `tg_matmul_mod_p_pair_*` (4 entries)

e) The `matmul_mod_p_pair_dev` family (device-pointer Spec L):
- `run_matmul_mod_p_pair_dev<T, D>(...)`
- `cabi_matmul_mod_p_pair_dev!` macro
- `tg_matmul_mod_p_pair_dev_*` (4 entries)

f) Drop `ERR_BOUND_TOO_LARGE` constant from the `const` block at the top.

g) Drop the imports that are now unused: `count_ground_states_gpu_u64`, `matmul_mod_p`, `matmul_mod_p_pair`, `matmul_mod_p_kernel_only`, `choose_primes_u64`, `CRT_PRIMES` (verify with `cargo check` after deletion; add back any still-needed imports).

h) Drop the corresponding tests in the `mod tests` block:
- `null_input_returns_invalid_input`
- `zero_dim_returns_invalid_input`
- `matmul_mod_p_f32_max_smoke`, `matmul_mod_p_invalid_p_returns_invalid`
- `matmul_mod_p_pair_f32_max_smoke`
- Any `matmul_mod_p_pair_dev_*` tests

**Keep:**
- `TG_API_VERSION`, `tg_api_version`, `tg_last_error_message`
- `OK`, `ERR_INVALID_INPUT`, `ERR_CUDA`, `ERR_INTERNAL` constants
- `LAST_ERROR` thread-local + `store_error`
- `version_is_one` test
- The Spec M Task 3 additions: `run_tropical_matmul`, `cabi_tropical_matmul!`, the 4 `tg_tropical_matmul_*` entries, `tg_tropical_matmul_*_smoke` and `tg_tropical_matmul_rejects_bad_flag` tests

- [ ] **Step 2: Update `classify_cuda_error`**

Replace the body with the simpler form (no u64-bound branch):

```rust
fn classify_cuda_error(_e: &CudaError) -> i32 {
    ERR_CUDA
}
```

Or, if `classify_cuda_error` was only ever called from `run_u64`, drop the function entirely.

- [ ] **Step 3: Verify the cdylib + Rust tests still build and pass**

```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo build --release -p tropical-gemm-cuda 2>&1 | tail -5'
```
Expected: success.

```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo test -p tropical-gemm-cuda 2>&1 | grep "test result"'
```
Expected: lib tests drop (probably ~60 down from 79; Spec M new tests remain), integration tests drop (the Spec K/L/J integration tests will now fail because their C ABI symbols are gone). **The integration tests need to be deleted in this same task.**

Find the failing integration test files:
```bash
ls /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm/crates/tropical-gemm-cuda/tests/
```

For each integration test file referencing any of the deleted C ABI symbols (`count_ground_states_gpu_u64`, `matmul_mod_p`, `matmul_mod_p_pair`, `matmul_mod_p_pair_dev`, `bench_kernel_only`), delete the file outright. Examples likely include `counting_gpu.rs`, `mod_p_*.rs`, etc. Use `git rm <path>`.

Re-run:
```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo test -p tropical-gemm-cuda 2>&1 | grep "test result"'
```
Expected: lib tests pass (~60 — Spec M + retained pre-existing tests like `version_is_one`); integration tests pass (probably 0 or small remaining set).

- [ ] **Step 4: Verify the Julia test still passes**

```
bash -c 'module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm/CountingTropicalGEMM.jl && julia --project=. -e "using Pkg; Pkg.test()" 2>&1 | tail -10'
```
Expected: ~30 tests, all green. The cdylib now exports only `tg_tropical_matmul_*` + version + last-error.

- [ ] **Step 5: Commit**

```bash
git add crates/tropical-gemm-cuda/src/c_api.rs
git rm crates/tropical-gemm-cuda/tests/<deleted-files>
git commit -m "$(cat <<'EOF'
Spec M Task 6: delete old Rust C ABI entries

Drops 16 of the 20 pre-Spec-M C ABI entries:
  - tg_count_ground_states_gpu_u64_<T>_<D> (4)
  - tg_bench_kernel_only_u64_<T>_<D>     (4)
  - tg_matmul_mod_p_<T>_<D>              (4, slow path)
  - tg_matmul_mod_p_pair_<T>_<D>         (4, host pair)
  - tg_matmul_mod_p_pair_dev_<T>_<D>     (4, Spec L device path)
Plus the supporting run_*, cabi_* macros, ERR_BOUND_TOO_LARGE, and
related tests / integration test files. Keeps tg_tropical_matmul_<T>_<D>
(Spec M, 4 entries) + version + last-error helpers.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Delete old Rust drivers and old kernels

**Files:**
- Modify: `crates/tropical-gemm-cuda/src/matmul_mod.rs`
- Modify: `crates/tropical-gemm-cuda/src/counting_kernel.rs`
- Modify: `crates/tropical-gemm-cuda/kernels/counting_gemm.cu`
- Modify: `crates/tropical-gemm-cuda/src/context.rs` (drop dead kernel names)

- [ ] **Step 1: Delete drivers in `matmul_mod.rs`**

In `crates/tropical-gemm-cuda/src/matmul_mod.rs`, delete:
- `matmul_mod_p<T, D>`
- `matmul_mod_p_pair<T, D>`
- `matmul_mod_p_kernel_only<T, D>`
- `run_packed<T, D>`
- All tests for those functions (`matmul_mod_p_2x2_max_p7`, `matmul_mod_p_observable_reduction`, `matmul_mod_p_rejects_p_one`, `matmul_mod_p_pair_2x2_max_p7`, `matmul_mod_p_4x4_min_random_p11`, `matmul_mod_p_pair_4x4_min_random_p11`, `matmul_mod_p_kernel_only_round_trip_max_p7`)

**Keep:**
- `DevPtr` struct + `DeviceRepr` impl (used by Spec M).
- `tropical_matmul_kernel<T, D>` (Spec M Task 2).
- `P_MIN` constant.
- The Spec M Task 2 tests.

After editing, this file should be ~120 LOC.

- [ ] **Step 2: Delete in `counting_kernel.rs`**

Delete:
- `launch_counting_gemm<T, D>` (replaced by `launch_tropical_matmul`)
- `launch_counting_gemm_ones<T, D>` (CRT path)
- `launch_counting_gemm_dev_ptr<T, D>` (Spec L path)
- The `CountingCudaKernel` trait (now only used by deleted code) — verify with `cargo check` first; if any of the kept code still references it (e.g. as a trait bound), keep it.
- The `KERNEL_NAME_WARPK` and warpk dispatch logic.
- Any helper for transposed-B uploads.

**Keep:**
- `launch_tropical_matmul<T, D>` (Spec M Task 2)
- `TropicalMatmulKernelName<T, D>` trait + impls

After editing, this file should be ~120 LOC.

- [ ] **Step 3: Delete old kernels from `counting_gemm.cu`**

In `crates/tropical-gemm-cuda/kernels/counting_gemm.cu`, delete:
- The `COUNTING_GEMM` macro and its 4 instantiations (`counting_gemm_f32_max`, `_f32_min`, `_f64_max`, `_f64_min`).
- The `COUNTING_GEMM_WARPK` macro and its 4 instantiations.
- The ones-specialized macros and their 4 + 4 instantiations.

**Keep:**
- The `barrett_mod` helper, `PairF32`/`PairF64` structs, `MAX_BETTER`/`MIN_BETTER`, `NEG_INF_*`/`POS_INF_*`, `OFFSET_ROW`.
- The Spec M `A_OFF_*`/`B_OFF_*`, `TROPICAL_MATMUL_BODY`, `DEFINE_TROPICAL_MATMUL` macros + 16 instantiations.

After editing, this file should be ~150 LOC.

- [ ] **Step 4: Update `context.rs::COUNTING_KERNEL_NAMES`**

Drop the old kernel names; keep only the 16 Spec M entries:

```rust
const COUNTING_KERNEL_NAMES: &[&str] = &[
    "tropical_matmul_f32_max_NN",
    "tropical_matmul_f32_max_NT",
    "tropical_matmul_f32_max_TN",
    "tropical_matmul_f32_max_TT",
    "tropical_matmul_f32_min_NN",
    "tropical_matmul_f32_min_NT",
    "tropical_matmul_f32_min_TN",
    "tropical_matmul_f32_min_TT",
    "tropical_matmul_f64_max_NN",
    "tropical_matmul_f64_max_NT",
    "tropical_matmul_f64_max_TN",
    "tropical_matmul_f64_max_TT",
    "tropical_matmul_f64_min_NN",
    "tropical_matmul_f64_min_NT",
    "tropical_matmul_f64_min_TN",
    "tropical_matmul_f64_min_TT",
];
```

Also drop `COUNTING_WARPK_K_THRESHOLD` and `COUNTING_WARPK_MN_CEILING` constants (no longer used).

- [ ] **Step 5: Verify build + tests**

```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo build --release -p tropical-gemm-cuda 2>&1 | tail -5'
```
Expected: success.

```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo test -p tropical-gemm-cuda 2>&1 | grep "test result"'
```
Expected: only Spec M Rust tests + `version_is_one`, all pass.

- [ ] **Step 6: Verify Julia tests**

```
bash -c 'module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm/CountingTropicalGEMM.jl && julia --project=. -e "using Pkg; Pkg.test()" 2>&1 | tail -5'
```
Expected: ~30 tests, all green.

- [ ] **Step 7: Commit**

```bash
git add crates/tropical-gemm-cuda/src/matmul_mod.rs \
        crates/tropical-gemm-cuda/src/counting_kernel.rs \
        crates/tropical-gemm-cuda/kernels/counting_gemm.cu \
        crates/tropical-gemm-cuda/src/context.rs
git commit -m "$(cat <<'EOF'
Spec M Task 7: delete old Rust drivers and old CUDA kernels

Drops matmul_mod_p / matmul_mod_p_pair / matmul_mod_p_kernel_only /
run_packed Rust drivers; launch_counting_gemm / _ones / _dev_ptr;
the COUNTING_GEMM / _WARPK / _ones / _ones_warpk CUDA kernel families
(8 base kernels). COUNTING_KERNEL_NAMES now just the 16 Spec M kernels.
COUNTING_WARPK_* dispatch knobs removed. Keeps DevPtr,
TropicalMatmulKernelName, launch_tropical_matmul, tropical_matmul_kernel.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Delete `crt.rs`, `lib.rs::pub mod crt`, and `pair.rs::PackPair`

**Files:**
- Delete: `crates/tropical-gemm-cuda/src/crt.rs`
- Modify: `crates/tropical-gemm-cuda/src/lib.rs`
- Modify: `crates/tropical-gemm-cuda/src/pair.rs`

- [ ] **Step 1: Delete the file and module declaration**

```bash
git rm crates/tropical-gemm-cuda/src/crt.rs
```

In `crates/tropical-gemm-cuda/src/lib.rs`, remove the `pub mod crt;` line.

- [ ] **Step 2: Drop `PackPair` trait + helpers from `pair.rs`**

In `crates/tropical-gemm-cuda/src/pair.rs`, delete:
- The `PackPair` trait declaration.
- `impl PackPair for f32` and `impl PackPair for f64`.
- `pack_f32`, `pack_f64`, `pack_f32_ones`, `pack_f64_ones` free functions.

**Keep:**
- `PairF32`, `PairF64` struct definitions + `DeviceRepr`/`ValidAsZeroBits` impls + their `new()` constructors.
- Inline tests for layout/alignment.

After editing, this file should be ~70 LOC.

- [ ] **Step 3: Fix anything that referenced PackPair**

Run `cargo check` to find leftover references:

```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo check -p tropical-gemm-cuda 2>&1 | tail -20'
```

If `tropical_matmul_kernel` or `launch_tropical_matmul` still has `PackPair` in its trait bounds and that bound is unnecessary (it was for the host pack helpers, which are gone), remove the bound. Specifically: in `matmul_mod.rs::tropical_matmul_kernel<T, D>` and `counting_kernel.rs::launch_tropical_matmul<T, D>` where-clauses, remove `+ crate::pair::PackPair` and `+ PackPair` if present.

After fixing the bounds, `cargo check` should succeed.

- [ ] **Step 4: Verify full Rust build + tests**

```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo test -p tropical-gemm-cuda 2>&1 | grep "test result"'
```
Expected: still ~10 lib tests + integration, all green.

- [ ] **Step 5: Verify Julia tests**

```
bash -c 'module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm/CountingTropicalGEMM.jl && julia --project=. -e "using Pkg; Pkg.test()" 2>&1 | tail -5'
```
Expected: ~30 tests, all green.

- [ ] **Step 6: Commit**

```bash
git add crates/tropical-gemm-cuda/src/lib.rs \
        crates/tropical-gemm-cuda/src/pair.rs \
        crates/tropical-gemm-cuda/src/matmul_mod.rs \
        crates/tropical-gemm-cuda/src/counting_kernel.rs
git commit -m "$(cat <<'EOF'
Spec M Task 8: delete crt.rs and PackPair trait

CRT/BigInt/multi-prime infrastructure is no longer reachable. Removes
the entire crt.rs file and pub mod crt declaration, plus the PackPair
trait + pack_f32/pack_f64/pack_*_ones helpers. PairF32/PairF64 structs
stay (used by Spec M kernels). tropical-gemm parent crate and
tropical-gemm-python are untouched per spec scope.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Rewrite bench, drop obsolete bench files

**Files:**
- Delete: `CountingTropicalGEMM.jl/bench/bench.jl`
- Delete: `CountingTropicalGEMM.jl/bench/bench_huge.jl`
- Modify: `CountingTropicalGEMM.jl/bench/bench_mul.jl`

- [ ] **Step 1: Delete obsolete benches**

```bash
git rm CountingTropicalGEMM.jl/bench/bench.jl
git rm CountingTropicalGEMM.jl/bench/bench_huge.jl
```

- [ ] **Step 2: Rewrite `bench/bench_mul.jl`**

Replace the entire file content with:

```julia
# Benchmark Spec M tropical_matmul on ModCountingTropical{Float32, 7}.
# Inputs and outputs stay on device; measures kernel + sync time per
# call (no host upload/download in the loop).
#
# Run from workspace root:
#   julia --project=CountingTropicalGEMM.jl CountingTropicalGEMM.jl/bench/bench_mul.jl

get!(ENV, "JULIA_CUDA_USE_COMPAT", "false")

using CountingTropicalGEMM
using CUDA
using Printf
using Random

const T = Float32
const P = 7
const ELT = ModCountingTropical{T, P}

function rand_matrix(rows, cols)
    [ELT(T(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:rows, _ in 1:cols]
end

function warmup()
    A = CuArray(rand_matrix(64, 64)); B = CuArray(rand_matrix(64, 64))
    tropical_matmul('N', 'N', A, B); CUDA.synchronize()
    tropical_matmul('T', 'T', A, B); CUDA.synchronize()
    return nothing
end

function bench_combo(tA::Char, tB::Char, M, K, N; iters)
    A_rows, A_cols = (tA == 'N') ? (M, K) : (K, M)
    B_rows, B_cols = (tB == 'N') ? (K, N) : (N, K)
    A = CuArray(rand_matrix(A_rows, A_cols))
    B = CuArray(rand_matrix(B_rows, B_cols))
    # Warm.
    tropical_matmul(tA, tB, A, B); CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:iters
        tropical_matmul(tA, tB, A, B)
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - t0) / 1e6 / iters
    ops = 2.0 * M * N * K
    gops = ops / (elapsed_ms * 1e-3) / 1e9
    return elapsed_ms, gops
end

function main()
    Random.seed!(0)
    @printf "Spec M tropical_matmul bench, ModCountingTropical{Float32, 7}\n"
    @printf "GPU: %s\n" CUDA.name(CUDA.device())
    @printf "%s\n" "-"^85

    warmup()

    @printf "\n%-12s %8s %8s %14s\n" "shape" "flag" "ms" "G tropical-ops/s"
    @printf "%s\n" "-"^85
    for s in (128, 256, 512, 1024, 2048, 4096)
        iters = s <= 256 ? 30 : (s <= 1024 ? 10 : 3)
        for combo in (('N','N'), ('N','T'), ('T','N'), ('T','T'))
            ms, gops = bench_combo(combo[1], combo[2], s, s, s; iters)
            @printf "M=N=K=%-7d %s%s %8.3f %14.1f\n" s combo[1] combo[2] ms gops
        end
    end
end

isinteractive() || main()
```

- [ ] **Step 3: Smoke-run the bench at one small size to verify it loads**

```
bash -c 'module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && julia --project=CountingTropicalGEMM.jl -e "
get!(ENV, \"JULIA_CUDA_USE_COMPAT\", \"false\")
include(\"CountingTropicalGEMM.jl/bench/bench_mul.jl\")
" 2>&1 | tail -30'
```
Expected: bench prints results for the configured sizes; no errors.

- [ ] **Step 4: Commit**

```bash
git add CountingTropicalGEMM.jl/bench/bench_mul.jl
git commit -m "$(cat <<'EOF'
Spec M Task 9: rewrite bench for tropical_matmul, drop obsolete benches

bench_mul.jl now exercises the new BLAS-style tropical_matmul over all
four (tA, tB) combinations, with persistent device buffers (no PCIe in
the inner loop). bench.jl (count_ground_states_gpu_u64 path) and
bench_huge.jl removed — both used the deleted Spec K/L API.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: Final validation + memory update

**Files:**
- Modify: `/mnt/home/xgao1/.claude/projects/-mnt-home-xgao1-work-better-gpu-gemm-tropical-gemm/memory/project_counting_status.md`

- [ ] **Step 1: Final Rust + Julia test sweep**

```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo test -p tropical-gemm-cuda 2>&1 | grep "test result"'
```
Expected: all green; lib tests are Spec M tests + version test (~10), integration tests pass (likely 0 remaining).

```
bash -c 'module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm/CountingTropicalGEMM.jl && julia --project=. -e "using Pkg; Pkg.test()" 2>&1 | tail -5'
```
Expected: ~30 tests, all green.

Also confirm the parent `tropical-gemm` crate still passes (no Spec M side effects):

```
bash -c 'export PATH="$HOME/.cargo/bin:$PATH" && module load cuda && cd /mnt/home/xgao1/work/better_gpu_gemm/tropical-gemm && cargo test -p tropical-gemm 2>&1 | grep "test result"'
```
Expected: all green.

- [ ] **Step 2: Update memory**

Append to `/mnt/home/xgao1/.claude/projects/-mnt-home-xgao1-work-better-gpu-gemm-tropical-gemm/memory/project_counting_status.md`:

```markdown

## Spec M — cleanup + N/T flags (commits ?? .. ??, 2026-04-28)

Stripped CRT/BigInt/multi-prime/host-wrapper accumulated surface from
tropical-gemm-cuda and CountingTropicalGEMM.jl. Replaced with a single
column-major BLAS-style mod-P counting tropical GEMM:

```julia
tropical_matmul(tA::Char, tB::Char, A::CuMatrix, B::CuMatrix) -> CuMatrix
tropical_matmul!(tA::Char, tB::Char, A::CuMatrix, B::CuMatrix, C::CuMatrix) -> C
```

Direction inferred from element type (ModCountingTropical = max-plus,
ModCountingTropicalMin = min-plus). Modulus P inferred from element
type's `P` parameter. `tA`, `tB ∈ {'N', 'T'}` follow column-major
BLAS convention: ('N','N') on Julia col-major inputs computes algebraic A*B.

Sixteen NVRTC-specialized CUDA kernels (NN/NT/TN/TT × f32/f64 × Max/Min),
column-major-natural indexing, AoS PairT input AND output. Eager-compiled
at first context init (~25-30 s on first call, paid once).

Deleted (no back-compat):
- All count_ground_states_gpu / _u64 / bench_kernel_only / matmul_mod_p
  C ABI entries (16 of 20 dropped).
- crt.rs file, multi-prime CRT machinery, BigInt path, ones-specialized
  kernels, warpk path, transposed-B helper, PackPair trait.
- Julia: count_ground_states_gpu_u64, bench_kernel_only_u64,
  CountedMatU64, BoundTooLargeError, TropicalMatrix, CountingTropicalMin
  (Mods-interop), Max/Min tags, host-Matrix tropical_matmul[_min], mul!,
  tropical_matmul_dev[_min], cuda_pair_buffer, Mods/TropicalNumbers
  Base.convert interop. LinearAlgebra/Mods/TropicalNumbers deps dropped.

Test counts (after cleanup): Rust ~10 (was 94). Julia ~30 (was 103).

Spec doc: docs/superpowers/specs/2026-04-28-cleanup-and-tn-flags-design.md
Plan doc: docs/superpowers/plans/2026-04-28-cleanup-and-tn-flags-plan.md

The Spec K/L/J entries in this file are historically-relevant but
the code paths they describe no longer exist.
```

Replace the placeholders `?? .. ??` with the actual first/last commit SHA range from the implementation.

- [ ] **Step 3: Final commit**

```bash
git status   # ensure tree is clean
# (memory.md is in a different repo; if there are uncommitted changes there, leave them)
```

If anything else is dirty, commit it now. Otherwise this task is just a final sweep.

---

## Spec coverage check

| Spec section | Implemented in |
|---|---|
| Public Julia API (tropical_matmul, !) | Task 4 |
| 16 NVRTC-specialized kernels (column-major) | Task 1 |
| Rust driver `tropical_matmul_kernel<T,D>` | Task 2 |
| C ABI `tg_tropical_matmul_<T>_<D>` (4 entries) | Task 3 |
| Element-type dispatch (separate methods → MethodError) | Task 4 |
| Validation (P, flags, K-match) | Tasks 2, 3, 4 |
| Pointer extraction without reinterpret | Task 4 |
| AoS output | Task 1 |
| Drop CRT/BigInt | Task 8 |
| Drop u64 / count_ground_states | Tasks 5, 6 |
| Drop matmul_mod_p slow/fast/dev families | Tasks 6, 7 |
| Drop TropicalMatrix, CountingTropicalMin (interop) | Task 5 |
| Drop host-Matrix wrappers, mul!, dev API | Task 5 |
| Drop CUDA.jl interop converters | Task 5 |
| Drop warpk + ones kernels | Task 7 |
| Keep tropical_matmul_gpu (parent / Python) | not deleted (verified in spec) |
| Test coverage: 4 transpose combos, edge cases, errors | Task 4 |
| Bench rewrite | Task 9 |
| Memory update | Task 10 |
