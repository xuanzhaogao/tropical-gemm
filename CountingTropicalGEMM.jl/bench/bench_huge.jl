using CountingTropicalGEMM
using Printf

# M=N=K=2^14 = 16384. Working set:
#   A, B: 16384² × 4 B = 1 GB each f32
#   value_c, count_c on device: 1 GB + 1 GB
#   total device: ~4 GB (well within 80 GB)
# Compute: 2 * 16384^3 ≈ 8.8 × 10^12 tropical-ops.

const S = 16384
println("M = N = K = ", S)
println("Generating ", S, "×", S, " inputs...")
A = Float32.(rand(0:6, S, S))
B = Float32.(rand(0:4, S, S))
println("done. Calling kernel (warmup + 2 iters)...")

ms = bench_kernel_only_u64(Max, A, B, UInt64(S); iters = 2)
ops = 2.0 * S * S * S
gops = ops / (ms * 1e-3) / 1e9
@printf "kernel %9.3f ms  (%6.1f G tropical-ops/s)\n" ms gops
