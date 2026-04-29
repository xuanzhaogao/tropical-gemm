# Direct kernel-vs-kernel benchmark: our CountingTropicalGEMM vs the
# auto-compiled Julia `mul!` on `CuArray{CountingTropical{T, Int}}`.

get!(ENV, "JULIA_CUDA_USE_COMPAT", "false")

using CUDA
using LinearAlgebra
using GenericTensorNetworks: CountingTropical
using CountingTropicalGEMM # our type + kernel
using Random
using Printf

function bench_ours(M, K, N, nreps; P=2965819)
    T = ModCountingTropical{Float32, P}
    Ah = [T(rand()*100f0, Int32(rand(1:Int(P)-1))) for _ in 1:M, _ in 1:K]
    Bh = [T(rand()*100f0, Int32(rand(1:Int(P)-1))) for _ in 1:K, _ in 1:N]
    A = CuArray(Ah); B = CuArray(Bh)
    Ah = Bh = nothing; GC.gc()
    C = tropical_matmul('N', 'N', A, B); CUDA.synchronize()  # warmup
    times = Float64[]
    for _ in 1:nreps
        t = CUDA.@elapsed begin
            C = tropical_matmul('N', 'N', A, B)
        end
        push!(times, t)
    end
    A = B = C = nothing; GC.gc(); CUDA.reclaim()
    minimum(times)
end

function bench_julia_auto(M, K, N, nreps)
    T = CountingTropical{Float64, Int64}
    Ah = [T(rand()*100.0, rand(1:1_000_000)) for _ in 1:M, _ in 1:K]
    Bh = [T(rand()*100.0, rand(1:1_000_000)) for _ in 1:K, _ in 1:N]
    A = CuArray(Ah); B = CuArray(Bh)
    Ah = Bh = nothing; GC.gc()
    C = CUDA.zeros(T, M, N)
    LinearAlgebra.mul!(C, A, B); CUDA.synchronize()  # warmup
    times = Float64[]
    for _ in 1:nreps
        t = CUDA.@elapsed begin
            LinearAlgebra.mul!(C, A, B)
        end
        push!(times, t)
    end
    A = B = C = nothing; GC.gc(); CUDA.reclaim()
    minimum(times)
end

gflops(M, K, N, t) = 2.0 * M * K * N / t / 1e9

println("[init] using-blocks done"); flush(stdout)
@printf "GPU: %s\n" CUDA.name(CUDA.device()); flush(stdout)
@printf "%-10s %-12s %-10s %-12s %-10s %-8s\n" "Shape" "ours_ms" "ours_G" "julia_ms" "julia_G" "speedup"; flush(stdout)
for (sz, nreps) in [(512, 5), (1024, 5), (2048, 3), (4096, 3), (8192, 2), (16384, 2), (32768, 1)]
    M = K = N = sz
    bytes_ours  = 8.0  * (M*K + K*N + M*N)
    bytes_julia = 16.0 * (M*K + K*N + M*N)
    free, total = CUDA.Mem.info()
    if max(bytes_ours, bytes_julia) > 0.85 * total
        @printf "%-10s SKIP (need %.1f GiB, have %.1f GiB)\n" "$(M)³" max(bytes_ours,bytes_julia)/2^30 total/2^30; flush(stdout)
        continue
    end
    println("[$sz] starting ours"); flush(stdout)
    t_ours  = bench_ours(M, K, N, nreps)
    println("[$sz] ours done in $(round(t_ours*1000, digits=2)) ms; starting julia auto"); flush(stdout)
    t_julia = bench_julia_auto(M, K, N, nreps)
    @printf "%-10s %-12.2f %-10.1f %-12.2f %-10.1f %-8.2fx\n" "$(M)³" t_ours*1000 gflops(M,K,N,t_ours) t_julia*1000 gflops(M,K,N,t_julia) t_julia/t_ours
    flush(stdout)
end
