# End-to-end Julia bench for count_ground_states_gpu_u64.
#
# Exercises the same shapes as the Rust bench_e2e_u64 example.
# Each timing wraps the full Julia call (transpose, ccall, kernel,
# residue download, u64 reconstruction, output reshape).
#
# Run from the workspace root:
#   julia --project=CountingTropicalGEMM.jl CountingTropicalGEMM.jl/bench/bench.jl

using CountingTropicalGEMM
using Printf

function bench_one(::Type{D}, T, m::Int, k::Int, n::Int; iters::Int) where {D}
    # Discrete inputs to mimic the Rust bench's data distribution.
    A = T.(rand(0:6, m, k))
    B = T.(rand(0:4, k, n))
    bound = UInt64(k)

    # Warmup (also pays the NVRTC compile cost on first call).
    _ = count_ground_states_gpu_u64(D, A, B, bound)

    t = @elapsed for _ in 1:iters
        _ = count_ground_states_gpu_u64(D, A, B, bound)
    end
    ms = t * 1000 / iters
    ops = 2.0 * m * n * k
    gops = ops / (ms * 1e-3) / 1e9
    return ms, gops
end

function main()
    println("CountingTropicalGEMM.jl end-to-end bench (A100, f32 Max, u64 fast-path)")
    println("-"^85)
    for s in (256, 512, 1024, 2048, 4096)
        iters = s <= 512 ? 5 : (s <= 2048 ? 3 : 2)
        ms, gops = bench_one(Max, Float32, s, s, s; iters)
        @printf "size=%5d  e2e %9.3f ms  (%6.1f G tropical-ops/s)\n" s ms gops
    end

    println()
    println("Tall-skinny / warpk regime:")
    println("-"^85)
    for (m, k, n) in ((32, 4096, 32), (64, 4096, 64), (128, 4096, 128), (256, 4096, 256))
        ms, gops = bench_one(Max, Float32, m, k, n; iters = 10)
        @printf "M=%4d K=%5d N=%4d  e2e %9.3f ms  (%6.1f G tropical-ops/s)\n" m k n ms gops
    end
end

main()
