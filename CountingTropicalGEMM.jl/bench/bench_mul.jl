# Benchmark mul! for Matrix{ModCountingTropical{Float32, 7}} and the
# device-to-device tropical_matmul_dev path. Reports per-call wall time
# (averaged over multiple iterations) and effective throughput in
# G tropical-ops/s (one tropical-op = one max-plus update with count combine
# under the mod-P semiring).
#
# Run from workspace root:
#   julia --project=CountingTropicalGEMM.jl CountingTropicalGEMM.jl/bench/bench_mul.jl

# Spec L env workaround: cluster's libcuda is older than CUDA.jl's bundled
# forward-compat artifact.
get!(ENV, "JULIA_CUDA_USE_COMPAT", "false")

using CountingTropicalGEMM
using CUDA
using LinearAlgebra
using Printf
using Random

const T = Float32
const P = 7
const ELT = ModCountingTropical{T, P}

function rand_matrix(rows, cols)
    [ELT(T(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:rows, _ in 1:cols]
end

# Warm up the runtime: NVRTC kernel compile + CUDA.jl context retain happen
# on first call. Without warmup the first measurement is dominated by ~7 s
# of one-time setup.
function warmup()
    A = rand_matrix(64, 64); B = rand_matrix(64, 64)
    C = Matrix{ELT}(undef, 64, 64)
    mul!(C, A, B)
    A_dev = cuda_pair_buffer(A); B_dev = cuda_pair_buffer(B)
    tropical_matmul_dev(A_dev, 64, 64, B_dev, 64, P)
    CUDA.synchronize()
    return nothing
end

# Bench mul! end-to-end: includes host pack, upload, kernel, download, zip.
# Each iteration creates a fresh result Matrix{ELT}; mul! writes into it.
function bench_mul(M, K, N; iters)
    A = rand_matrix(M, K); B = rand_matrix(K, N)
    C = Matrix{ELT}(undef, M, N)
    # Single warm call to bring caches up.
    mul!(C, A, B)
    t0 = time_ns()
    for _ in 1:iters
        mul!(C, A, B)
    end
    elapsed_ms = (time_ns() - t0) / 1e6 / iters
    ops = 2.0 * M * N * K
    gops = ops / (elapsed_ms * 1e-3) / 1e9
    return elapsed_ms, gops
end

# Bench tropical_matmul_dev: device buffers persist; only the kernel +
# output allocation is measured. Models a hot loop where A, B stay on GPU.
function bench_dev(M, K, N; iters)
    A = rand_matrix(M, K); B = rand_matrix(K, N)
    A_dev = cuda_pair_buffer(A); B_dev = cuda_pair_buffer(B)
    # Warm.
    tropical_matmul_dev(A_dev, M, K, B_dev, N, P); CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:iters
        out_v, out_c = tropical_matmul_dev(A_dev, M, K, B_dev, N, P)
    end
    CUDA.synchronize()
    elapsed_ms = (time_ns() - t0) / 1e6 / iters
    ops = 2.0 * M * N * K
    gops = ops / (elapsed_ms * 1e-3) / 1e9
    return elapsed_ms, gops
end

function main()
    Random.seed!(0)
    @printf "Benchmark: mul! and tropical_matmul_dev on ModCountingTropical{Float32, 7}\n"
    @printf "GPU: %s\n" CUDA.name(CUDA.device())
    @printf "%s\n" "-"^85

    warmup()

    @printf "\n%-25s %12s %14s %12s %14s\n" "shape" "mul! ms" "mul! G-ops/s" "dev ms" "dev G-ops/s"
    @printf "%s\n" "-"^85
    for s in (128, 256, 512, 1024, 2048, 4096)
        iters = s <= 256 ? 30 : (s <= 1024 ? 10 : 3)
        mul_ms, mul_g = bench_mul(s, s, s; iters)
        dev_ms, dev_g = bench_dev(s, s, s; iters)
        @printf "M=N=K=%-19d %12.3f %14.1f %12.3f %14.1f\n" s mul_ms mul_g dev_ms dev_g
    end

    @printf "\nThin shapes (M=N small, K large) — exercises warpk path:\n"
    @printf "%-25s %12s %14s %12s %14s\n" "shape" "mul! ms" "mul! G-ops/s" "dev ms" "dev G-ops/s"
    @printf "%s\n" "-"^85
    for (m, k, n) in ((64, 4096, 64), (128, 4096, 128), (256, 4096, 256))
        iters = 10
        mul_ms, mul_g = bench_mul(m, k, n; iters)
        dev_ms, dev_g = bench_dev(m, k, n; iters)
        shape_str = @sprintf "M=%d K=%d N=%d" m k n
        @printf "%-25s %12.3f %14.1f %12.3f %14.1f\n" shape_str mul_ms mul_g dev_ms dev_g
    end
end

isinteractive() || main()
