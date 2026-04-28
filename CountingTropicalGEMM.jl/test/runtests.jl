using Test
using Random
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

    @testset "tropical_matmul (Max f32)" begin
        P = 7
        Random.seed!(1)
        A = [ModCountingTropical{Float32, P}(
                Float32(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:5, _ in 1:8]
        B = [ModCountingTropical{Float32, P}(
                Float32(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:8, _ in 1:6]
        # Reference: pure Julia max-plus + count multiply mod P.
        ref = Matrix{ModCountingTropical{Float32, P}}(undef, 5, 6)
        for i in 1:5, j in 1:6
            best_n = -Inf32
            best_c = Int32(0)
            for kk in 1:8
                v = A[i, kk].n + B[kk, j].n
                c = Int32(mod(Int64(A[i, kk].c) * Int64(B[kk, j].c), Int64(P)))
                if v > best_n
                    best_n = v; best_c = c
                elseif v == best_n
                    best_c = Int32(mod(Int64(best_c) + Int64(c), Int64(P)))
                end
            end
            ref[i, j] = ModCountingTropical{Float32, P}(best_n, best_c)
        end

        C = tropical_matmul(A, B)
        @test C == ref
    end

    @testset "tropical_matmul_min (Min f64)" begin
        P = 11
        Random.seed!(2)
        A = [ModCountingTropicalMin{Float64, P}(
                Float64(rand(0:4)), Int32(rand(0:P-1))) for _ in 1:6, _ in 1:9]
        B = [ModCountingTropicalMin{Float64, P}(
                Float64(rand(0:4)), Int32(rand(0:P-1))) for _ in 1:9, _ in 1:7]
        ref = Matrix{ModCountingTropicalMin{Float64, P}}(undef, 6, 7)
        for i in 1:6, j in 1:7
            best_n = Inf
            best_c = Int32(0)
            for kk in 1:9
                v = A[i, kk].n + B[kk, j].n
                c = Int32(mod(Int64(A[i, kk].c) * Int64(B[kk, j].c), Int64(P)))
                if v < best_n
                    best_n = v; best_c = c
                elseif v == best_n
                    best_c = Int32(mod(Int64(best_c) + Int64(c), Int64(P)))
                end
            end
            ref[i, j] = ModCountingTropicalMin{Float64, P}(best_n, best_c)
        end

        C = tropical_matmul_min(A, B)
        @test C == ref
    end

    @testset "mul! over ModCountingTropical" begin
        using LinearAlgebra
        P = 7
        Random.seed!(42)
        A = [ModCountingTropical{Float32, P}(
                Float32(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:4, _ in 1:5]
        B = [ModCountingTropical{Float32, P}(
                Float32(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:5, _ in 1:6]

        # In-place vs functional must agree.
        ref = tropical_matmul(A, B)
        C = Matrix{ModCountingTropical{Float32, P}}(undef, 4, 6)
        mul!(C, A, B)
        @test C == ref

        # Reuse the same C buffer with different inputs.
        A2 = [ModCountingTropical{Float32, P}(
                 Float32(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:4, _ in 1:5]
        ref2 = tropical_matmul(A2, B)
        mul!(C, A2, B)
        @test C == ref2

        # Wrong-size C → DimensionMismatch.
        Cbad = Matrix{ModCountingTropical{Float32, P}}(undef, 3, 6)  # rows wrong
        @test_throws DimensionMismatch mul!(Cbad, A, B)
    end

    @testset "mul! over ModCountingTropicalMin" begin
        using LinearAlgebra
        P = 11
        Random.seed!(43)
        A = [ModCountingTropicalMin{Float64, P}(
                Float64(rand(0:4)), Int32(rand(0:P-1))) for _ in 1:5, _ in 1:7]
        B = [ModCountingTropicalMin{Float64, P}(
                Float64(rand(0:4)), Int32(rand(0:P-1))) for _ in 1:7, _ in 1:4]
        ref = tropical_matmul_min(A, B)
        C = Matrix{ModCountingTropicalMin{Float64, P}}(undef, 5, 4)
        mul!(C, A, B)
        @test C == ref
    end

    @testset "tropical_matmul edge cases" begin
        # 1×1×1.
        A11 = reshape([ModCountingTropical{Float32, 7}(2.0f0, Int32(3))], 1, 1)
        B11 = reshape([ModCountingTropical{Float32, 7}(5.0f0, Int32(4))], 1, 1)
        C11 = tropical_matmul(A11, B11)
        @test C11[1, 1].n == 7.0f0
        @test C11[1, 1].c == Int32(mod(3 * 4, 7))   # 12 mod 7 = 5

        # K = 1 (no ties possible).
        A1 = [ModCountingTropical{Float32, 7}(Float32(i), Int32(2)) for i in 1:3, _ in 1:1]
        B1 = [ModCountingTropical{Float32, 7}(Float32(j), Int32(3)) for _ in 1:1, j in 1:4]
        C1 = tropical_matmul(A1, B1)
        @test size(C1) == (3, 4)
        for i in 1:3, j in 1:4
            @test C1[i, j].n == Float32(i + j)
            @test C1[i, j].c == Int32(mod(2 * 3, 7))   # 6
        end

        # P = 2 (smallest valid prime). All-tie input with K=5, counts=1:
        # 5 ties of (1*1)=1, sum mod 2 = 5 mod 2 = 1.
        A2 = [ModCountingTropical{Float32, 2}(0.0f0, Int32(1)) for _ in 1:3, _ in 1:5]
        B2 = [ModCountingTropical{Float32, 2}(0.0f0, Int32(1)) for _ in 1:5, _ in 1:3]
        C2 = tropical_matmul(A2, B2)
        @test all(c -> c.n == 0.0f0, C2)
        @test all(c -> c.c == Int32(1), C2)

        # All-tie reduction triggers: P=17, K=5, counts 2*3=6 each, sum 30 mod 17 = 13.
        A3 = [ModCountingTropical{Float64, 17}(0.0, Int32(2)) for _ in 1:2, _ in 1:5]
        B3 = [ModCountingTropical{Float64, 17}(0.0, Int32(3)) for _ in 1:5, _ in 1:2]
        C3 = tropical_matmul(A3, B3)
        @test all(c -> c.n == 0.0, C3)
        @test all(c -> c.c == Int32(13), C3)

        # Largest valid prime fitting i32 positive: 2^31 - 1 = 2147483647.
        # All-ties with input counts 1, K=3, P=2147483647: sum = 3, mod = 3.
        Phuge = 2147483647
        Ah = [ModCountingTropical{Float32, Phuge}(0.0f0, Int32(1)) for _ in 1:2, _ in 1:3]
        Bh = [ModCountingTropical{Float32, Phuge}(0.0f0, Int32(1)) for _ in 1:3, _ in 1:2]
        Ch = tropical_matmul(Ah, Bh)
        @test all(c -> c.n == 0.0f0, Ch)
        @test all(c -> c.c == Int32(3), Ch)
    end

    @testset "tropical_matmul errors" begin
        # K-mismatch.
        A = [ModCountingTropical{Float32, 7}(0.0f0, Int32(1)) for _ in 1:2, _ in 1:3]
        Bbad = [ModCountingTropical{Float32, 7}(0.0f0, Int32(1)) for _ in 1:4, _ in 1:2]
        @test_throws DimensionMismatch tropical_matmul(A, Bbad)

        # P out of range — must be checked at the Julia layer (ArgumentError).
        # Julia's typeparam can hold any Int, so we can construct types with bad P.
        # P = 1: invalid (must be >= 2).
        A1 = [ModCountingTropical{Float32, 1}(0.0f0, Int32(0)) for _ in 1:2, _ in 1:2]
        @test_throws ArgumentError tropical_matmul(A1, A1)

        # P = 2^31 (just past the i32 positive max): also invalid.
        Pover = Int(1) << 31
        Aover = [ModCountingTropical{Float32, Pover}(0.0f0, Int32(0)) for _ in 1:2, _ in 1:2]
        @test_throws ArgumentError tropical_matmul(Aover, Aover)

        # Mismatched moduli (different P) → MethodError (no method matches).
        Aother = [ModCountingTropical{Float32, 11}(0.0f0, Int32(0)) for _ in 1:2, _ in 1:3]
        @test_throws MethodError tropical_matmul(A, Aother)

        # Mismatched directions (Max vs Min) → MethodError.
        Amax = [ModCountingTropical{Float32, 7}(0.0f0, Int32(1)) for _ in 1:2, _ in 1:3]
        Bmin = [ModCountingTropicalMin{Float32, 7}(0.0f0, Int32(1)) for _ in 1:3, _ in 1:2]
        @test_throws MethodError tropical_matmul(Amax, Bmin)
    end

    @testset "convert from CountingTropical{T, Mod{P}}" begin
        using TropicalNumbers, Mods
        P = 7
        # CountingTropical from TropicalNumbers.jl with Mods.jl v2 Mod{P}.
        x = CountingTropical{Float32, Mod{P}}(2.5f0, Mod{P}(3))
        y = convert(ModCountingTropical{Float32, P}, x)
        @test y.n == 2.5f0
        @test y.c == Int32(3)

        # Out-of-range count value (count > 2^31 - 1) — would be caught by
        # Int32 conversion. With Mod{P} the value is always in [0, P) and
        # P < 2^31 by our contract, so this is a defensive check; we
        # assert the converter rejects a deliberately-crafted bad value.
        # (Skipped — Mod{P} invariant guarantees the value fits, so the
        # converter is total within its domain.)
    end
end
