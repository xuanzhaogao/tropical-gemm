# Probe: u32 acc_c at P=7 (small-P fast path).
get!(ENV, "JULIA_CUDA_USE_COMPAT", "false")
using CUDA, CountingTropicalGEMM, Random, Printf

function bench_ours(sz, nreps, P)
    T = ModCountingTropical{Float32, P}
    Ah = [T(rand()*100f0, Int32(rand(1:P-1))) for _ in 1:sz, _ in 1:sz]
    Bh = [T(rand()*100f0, Int32(rand(1:P-1))) for _ in 1:sz, _ in 1:sz]
    A = CuArray(Ah); B = CuArray(Bh)
    tropical_matmul('T','T', A, B); CUDA.synchronize()  # warmup
    times = Float64[]
    for _ in 1:nreps
        push!(times, CUDA.@elapsed begin
            tropical_matmul('T','T', A, B)
        end)
    end
    A=B=nothing; GC.gc(); CUDA.reclaim()
    minimum(times)
end

println("GPU: $(CUDA.name(CUDA.device()))"); flush(stdout)
@printf "%-8s %-10s %-10s %-12s\n" "P" "Shape" "ms" "G/s"; flush(stdout)
for P in (7, 2965819)
    for (sz, nreps) in [(1024, 5), (2048, 3), (4096, 3), (8192, 2)]
        t = bench_ours(sz, nreps, P)
        @printf "%-8d %-10s %-10.3f %-12.1f\n" P "$(sz)³" t*1000 2*sz^3/t/1e9
        flush(stdout)
    end
end
