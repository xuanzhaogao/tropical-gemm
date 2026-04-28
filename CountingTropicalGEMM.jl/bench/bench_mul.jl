# Spec M benchmark: tropical_matmul on ModCountingTropical{Float32, 7}.
# Inputs and output stay on device; measures per-call wall time
# (kernel + sync) over all four (tA, tB) combinations.
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

    @printf "\n%-15s %5s %10s %18s\n" "shape" "flag" "ms/call" "G tropical-ops/s"
    @printf "%s\n" "-"^85
    for s in (128, 256, 512, 1024, 2048, 4096)
        iters = s <= 256 ? 30 : (s <= 1024 ? 10 : 3)
        for combo in (('N','N'), ('N','T'), ('T','N'), ('T','T'))
            ms, gops = bench_combo(combo[1], combo[2], s, s, s; iters)
            @printf "M=N=K=%-9d %c%c  %10.3f %18.1f\n" s combo[1] combo[2] ms gops
        end
    end
end

isinteractive() || main()
