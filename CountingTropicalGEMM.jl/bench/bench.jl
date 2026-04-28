# Kernel-only bench: upload once, time N launches via the C ABI's
# tg_bench_kernel_only_* path. This isolates the GPU kernel runtime from
# all wrapper / transfer overhead.
#
# Run from workspace root:
#   julia --project=CountingTropicalGEMM.jl CountingTropicalGEMM.jl/bench/bench.jl

using CountingTropicalGEMM
using Printf

function bench(::Type{D}, T, m::Int, k::Int, n::Int; iters::Int) where {D}
    A = T.(rand(0:6, m, k))
    B = T.(rand(0:4, k, n))
    bound = UInt64(k)
    ms = bench_kernel_only_u64(D, A, B, bound; iters)
    ops = 2.0 * m * n * k
    gops = ops / (ms * 1e-3) / 1e9
    return ms, gops
end

function main()
    println("CountingTropicalGEMM.jl kernel-only bench (A100, f32 Max, u64 path)")
    println("-"^85)

    println("\nNaive path (large square):")
    for s in (128, 256, 512, 1024, 2048, 4096)
        iters = s <= 256 ? 30 : (s <= 1024 ? 10 : 5)
        ms, gops = bench(Max, Float32, s, s, s; iters)
        @printf "size=%5d  kernel %9.3f ms  (%6.1f G tropical-ops/s)\n" s ms gops
    end

    println("\nWarpk path (small M*N, large K, transposed B):")
    for (m, k, n) in ((32, 4096, 32), (64, 4096, 64), (128, 4096, 128),
                      (256, 4096, 256), (512, 4096, 512), (1024, 4096, 1024))
        iters = 10
        ms, gops = bench(Max, Float32, m, k, n; iters)
        @printf "M=%5d K=%5d N=%5d  kernel %8.3f ms  (%6.1f G tropical-ops/s)\n" m k n ms gops
    end
end

main()
