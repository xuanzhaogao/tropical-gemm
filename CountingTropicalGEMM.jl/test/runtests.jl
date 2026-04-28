using Test
using CountingTropicalGEMM

# Reference implementation for cross-checking. Tropical max/min-plus matmul
# with all-ones input counts.
function reference(::Type{Max}, A::Matrix{T}, B::Matrix{T}) where {T}
    m, k = size(A); _, n = size(B)
    V = fill(typemin(T), m, n)
    C = zeros(UInt64, m, n)
    for i in 1:m, j in 1:n
        for kk in 1:k
            x = A[i, kk] + B[kk, j]
            if x > V[i, j]
                V[i, j] = x; C[i, j] = UInt64(1)
            elseif x == V[i, j]
                C[i, j] += UInt64(1)
            end
        end
    end
    V, C
end

function reference(::Type{Min}, A::Matrix{T}, B::Matrix{T}) where {T}
    m, k = size(A); _, n = size(B)
    V = fill(typemax(T), m, n)
    C = zeros(UInt64, m, n)
    for i in 1:m, j in 1:n
        for kk in 1:k
            x = A[i, kk] + B[kk, j]
            if x < V[i, j]
                V[i, j] = x; C[i, j] = UInt64(1)
            elseif x == V[i, j]
                C[i, j] += UInt64(1)
            end
        end
    end
    V, C
end

@testset "CountingTropicalGEMM.jl" begin

    @testset "f32 Max small" begin
        # Verifiable by hand. A = [1 2; 3 4], B = [5 6; 7 8].
        # max-plus: C[i,j] = max_k (A[i,k] + B[k,j]).
        # C[1,1] = max(1+5, 2+7) = 9 (from k=2).
        # C[1,2] = max(1+6, 2+8) = 10 (k=2).
        # C[2,1] = max(3+5, 4+7) = 11 (k=2).
        # C[2,2] = max(3+6, 4+8) = 12 (k=2).
        # All counts = 1 (no ties).
        A = Float32[1 2; 3 4]
        B = Float32[5 6; 7 8]
        res = count_ground_states_gpu_u64(Max, A, B, UInt64(2))
        @test res.values == Float32[9 10; 11 12]
        @test res.counts == UInt64[1 1; 1 1]
    end

    @testset "f64 Min vs reference (randomized)" begin
        # Discrete inputs to maximize the chance of ties.
        A = Float64.(rand(0:4, 12, 17))
        B = Float64.(rand(0:4, 17, 11))
        ref_v, ref_c = reference(Min, A, B)
        res = count_ground_states_gpu_u64(Min, A, B, UInt64(17))
        @test res.values == ref_v
        @test res.counts == ref_c
    end

    @testset "all-ties large K (Max, f32)" begin
        m, k, n = 5, 200, 7
        A = zeros(Float32, m, k)
        B = zeros(Float32, k, n)
        res = count_ground_states_gpu_u64(Max, A, B, UInt64(k))
        @test all(res.values .== 0.0f0)
        @test all(res.counts .== UInt64(k))
    end

    @testset "BoundTooLargeError" begin
        A = Float32[1.0;;]; B = Float32[2.0;;]
        @test_throws BoundTooLargeError count_ground_states_gpu_u64(
            Max, A, B, UInt64(1) << 62)
    end

    @testset "DimensionMismatch" begin
        A = Float32[1 2; 3 4]
        B = Float32[1 2 3; 4 5 6; 7 8 9]   # 3×3 — A's K is 2, mismatch.
        @test_throws DimensionMismatch count_ground_states_gpu_u64(
            Max, A, B, UInt64(8))
    end

    @testset "TropicalMatrix * TropicalMatrix (Max, f32)" begin
        A = Float32.(rand(0:3, 8, 13))
        B = Float32.(rand(0:3, 13, 9))
        ref_v, ref_c = reference(Max, A, B)
        TA = TropicalMatrix(Max, A)
        TB = TropicalMatrix(Max, B)
        @test size(TA) == (8, 13)
        @test TA[1, 1] == A[1, 1]
        res = TA * TB
        @test res.values == ref_v
        @test res.counts == ref_c
    end

    @testset "TropicalMatrix * TropicalMatrix (Min, f64)" begin
        A = Float64.(rand(0:4, 6, 11))
        B = Float64.(rand(0:4, 11, 7))
        ref_v, ref_c = reference(Min, A, B)
        res = TropicalMatrix(Min, A) * TropicalMatrix(Min, B)
        @test res.values == ref_v
        @test res.counts == ref_c
    end

    @testset "TropicalMatrix direction mismatch" begin
        A = TropicalMatrix(Max, Float32[1 2; 3 4])
        B = TropicalMatrix(Min, Float32[5 6; 7 8])
        @test_throws ArgumentError A * B
    end

    @testset "TropicalMatrix custom bound" begin
        A = TropicalMatrix(Max, Float32[1 2; 3 4]; bound = 2)
        B = TropicalMatrix(Max, Float32[5 6; 7 8])
        res = A * B
        @test res.values == Float32[9 10; 11 12]
        @test res.counts == UInt64[1 1; 1 1]
    end

    @testset "ModCountingTropical type and semiring" begin
        # Layout: must be byte-compatible with the Rust PairT.
        @test sizeof(ModCountingTropical{Float32, 7}) == 8
        @test sizeof(ModCountingTropical{Float64, 7}) == 16
        @test sizeof(ModCountingTropicalMin{Float32, 7}) == 8
        @test sizeof(ModCountingTropicalMin{Float64, 7}) == 16

        # Semiring identities.
        Z = zero(ModCountingTropical{Float32, 7})
        O = one(ModCountingTropical{Float32, 7})
        @test Z.n == typemin(Float32) && Z.c == Int32(0)
        @test O.n == 0.0f0 && O.c == Int32(1)

        # Max-plus addition: take greater n; sum counts on tie mod P.
        a = ModCountingTropical{Float32, 7}(1.0f0, Int32(3))
        b = ModCountingTropical{Float32, 7}(2.0f0, Int32(5))
        c = ModCountingTropical{Float32, 7}(1.0f0, Int32(6))
        @test a + b == b
        @test b + a == b
        @test a + c == ModCountingTropical{Float32, 7}(1.0f0, Int32(2))   # 3+6=9 mod 7 = 2

        # Tropical mul: sum n, multiply counts mod P.
        d = ModCountingTropical{Float32, 7}(2.0f0, Int32(4))
        e = ModCountingTropical{Float32, 7}(3.0f0, Int32(5))
        @test d * e == ModCountingTropical{Float32, 7}(5.0f0, Int32(6))   # 4*5=20 mod 7 = 6

        # Min-plus mirror.
        Zm = zero(ModCountingTropicalMin{Float64, 11})
        Om = one(ModCountingTropicalMin{Float64, 11})
        @test Zm.n == typemax(Float64) && Zm.c == Int32(0)
        @test Om.n == 0.0 && Om.c == Int32(1)

        am = ModCountingTropicalMin{Float64, 11}(1.0, Int32(3))
        bm = ModCountingTropicalMin{Float64, 11}(2.0, Int32(5))
        cm = ModCountingTropicalMin{Float64, 11}(1.0, Int32(9))
        @test am + bm == am          # min picks 1.0
        @test am + cm == ModCountingTropicalMin{Float64, 11}(1.0, Int32(1))  # 3+9=12 mod 11 = 1
        @test am * bm == ModCountingTropicalMin{Float64, 11}(3.0, Int32(4))  # 3*5=15 mod 11 = 4

        # Int32 overflow guard: counts near (2^31 - 1) must stay correct via Int64 path.
        big1 = ModCountingTropical{Float32, 2_000_000_011}(0.0f0, Int32(2_000_000_000))
        big2 = ModCountingTropical{Float32, 2_000_000_011}(0.0f0, Int32(2_000_000_000))
        expected = Int32(mod(Int64(2_000_000_000)^2, Int64(2_000_000_011)))
        @test (big1 * big2).c == expected
    end
end
