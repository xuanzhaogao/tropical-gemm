using Pkg
Pkg.activate(temp = true)
Pkg.add(["CUDA", "TropicalNumbers"])
Pkg.add(path = "/mnt/home/xgao1/work/better_gpu_gemm/GenericTensorNetworks.jl")

using CUDA, TropicalNumbers, GenericTensorNetworks, Printf
const Mods = GenericTensorNetworks.Mods
const Mod = Mods.Mod
const P = Int32(1_073_741_789)
T = CountingTropical{Float32, Mod{P, Int32}}

# A = [2 3], B = [3; 2]. Both k give 2+3=5 and 3+2=5 → tie.
# Count sum = 1+1 = 2.
a = T[T(2f0, Mod{P,Int32}(1)) T(3f0, Mod{P,Int32}(1))]
b = reshape(T[T(3f0, Mod{P,Int32}(1)), T(2f0, Mod{P,Int32}(1))], 2, 1)

c_cpu = a * b
println("CPU: ", c_cpu)

ag = CuArray(a); bg = CuArray(b)
c_gpu = Array(ag * bg)
println("GPU: ", c_gpu)

# Larger sanity: 8x8 * 8x8 all-ones.
a2 = fill(T(0f0, Mod{P,Int32}(1)), 8, 8)
b2 = fill(T(0f0, Mod{P,Int32}(1)), 8, 8)
c2 = Array(CuArray(a2) * CuArray(b2))
# Expected: every C[i,j] = CountingTropical(0, 8) since 8 tied paths.
println("8x8 zeros, C[1,1] = ", c2[1,1])
@assert c2[1,1].n == 0f0
@assert c2[1,1].c.val == 8
println("OK")

# Time just the kernel, separating from allocation.
mk(m, n) = CuArray(T[T(Float32(rand(0:6)), Mod{P,Int32}(1)) for _ in 1:m, _ in 1:n])
a3 = mk(2048, 2048); b3 = mk(2048, 2048)
CUDA.@sync a3 * b3  # warm
t0 = time()
for _ in 1:3
    CUDA.@sync a3 * b3
end
t = (time() - t0) / 3
@printf("\n2048x2048 a*b on GPU: %.2f ms (%.1f G tropical-ops/s)\n",
    t*1e3, 2*2048^3 / t / 1e9)
