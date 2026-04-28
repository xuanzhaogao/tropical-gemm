#!/usr/bin/env julia
# Julia CountingTropical GPU matmul benchmark — the fair comparison point
# for our Rust tropical-gemm-cuda counting kernel.
#
# CuTropicalGEMM.jl does NOT specialize for CountingTropical, so Julia uses
# CUDA.jl's generic broadcast-based matmul fallback (GPUArrays.jl's reduction
# machinery). That's what GTN's `contractx` with `CountingTropical` + `usecuda=true`
# would invoke internally.

using Pkg
Pkg.activate(temp = true)
Pkg.add(["CUDA", "TropicalNumbers"])
Pkg.add(path = "/mnt/home/xgao1/work/better_gpu_gemm/GenericTensorNetworks.jl")

using CUDA
using TropicalNumbers
using GenericTensorNetworks
using Random
using LinearAlgebra
using Printf

# Access vendored Mods.jl.
const Mods = GenericTensorNetworks.Mods
const Mod = Mods.Mod

# Pick one 30-bit prime matching our Rust CRT_PRIMES[0].
const P = Int32(1_073_741_789)

function random_ct(size::Int, seed::Integer)
    Random.seed!(seed)
    T = CountingTropical{Float32, Mod{P, Int32}}
    [T(Float32(rand(0:6)), Mod{P, Int32}(1)) for _ in 1:size*size]
end

function bench_gpu(sz::Int, iters::Int = 3)
    a_host = reshape(random_ct(sz, 0xaaaa), sz, sz)
    b_host = reshape(random_ct(sz, 0xbbbb), sz, sz)
    a = CuArray(a_host)
    b = CuArray(b_host)

    # Warm-up (also JIT-compiles the generic kernel).
    CUDA.@sync a * b

    t = @elapsed CUDA.@sync begin
        for _ in 1:iters
            _ = a * b
        end
    end
    ms = t * 1000.0 / iters
    ops = 2.0 * sz^3  # tropical add + mul per inner step
    gops = ops / (ms * 1e-3) / 1e9
    @printf("%6d   %10.2f ms   %6.2f G tropical-ops/s\n", sz, ms, gops)
end

using Printf

println("Julia CUDA.jl generic CountingTropical GPU matmul")
println("  Element: CountingTropical{Float32, Mod{$P, Int32}}")
println("  Device : ", CUDA.name(CUDA.device()))
println("-"^60)
println("  size         GPU ms        throughput")
bench_gpu(128)
bench_gpu(256)
bench_gpu(512)
bench_gpu(1024)
bench_gpu(2048)
bench_gpu(4096, 2)
bench_gpu(8192, 1)
