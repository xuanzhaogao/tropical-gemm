# Spec L: cudarc and CUDA.jl share the primary CUDA context, but if
# CUDA.jl loads its bundled forward-compat libcuda artifact (newer than the
# host kernel module), cudarc's cuInit fails with
# CUDA_ERROR_OPERATING_SYSTEM. Set this env var *before* CUDA.jl loads to
# force the process to open the system libcuda that matches the kernel
# driver. Must precede any code path that imports CUDA (transitively
# CountingTropicalGEMM does).
get!(ENV, "JULIA_CUDA_USE_COMPAT", "false")

using Test
using Random
using CUDA
using CountingTropicalGEMM

@testset "CountingTropicalGEMM.jl" begin

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

    @testset "Spec M tropical_matmul NN f32 Max" begin
        P = 7
        Random.seed!(101)
        A_host = [ModCountingTropical{Float32, P}(
                    Float32(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:5, _ in 1:8]
        B_host = [ModCountingTropical{Float32, P}(
                    Float32(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:8, _ in 1:6]
        A = CuArray(A_host); B = CuArray(B_host)

        ref = Matrix{ModCountingTropical{Float32, P}}(undef, 5, 6)
        for i in 1:5, j in 1:6
            best_n = -Inf32; best_c = Int32(0)
            for kk in 1:8
                v = A_host[i, kk].n + B_host[kk, j].n
                c = Int32(mod(Int64(A_host[i, kk].c) * Int64(B_host[kk, j].c), Int64(P)))
                if v > best_n
                    best_n = v; best_c = c
                elseif v == best_n
                    best_c = Int32(mod(Int64(best_c) + Int64(c), Int64(P)))
                end
            end
            ref[i, j] = ModCountingTropical{Float32, P}(best_n, best_c)
        end

        C_dev = tropical_matmul('N', 'N', A, B)
        @test Array(C_dev) == ref
    end

    @testset "Spec M tropical_matmul TT f64 Min" begin
        P = 11
        Random.seed!(102)
        A_host = [ModCountingTropicalMin{Float64, P}(
                    Float64(rand(0:4)), Int32(rand(0:P-1))) for _ in 1:6, _ in 1:9]
        B_host = [ModCountingTropicalMin{Float64, P}(
                    Float64(rand(0:4)), Int32(rand(0:P-1))) for _ in 1:9, _ in 1:7]
        # Build the transposed inputs A_T (Julia 9×6) and B_T (Julia 7×9).
        AT_host = [A_host[i, j] for j in 1:9, i in 1:6]
        BT_host = [B_host[i, j] for j in 1:7, i in 1:9]
        AT = CuArray(AT_host); BT = CuArray(BT_host)

        ref = Matrix{ModCountingTropicalMin{Float64, P}}(undef, 6, 7)
        for i in 1:6, j in 1:7
            best_n = Inf; best_c = Int32(0)
            for kk in 1:9
                v = A_host[i, kk].n + B_host[kk, j].n
                c = Int32(mod(Int64(A_host[i, kk].c) * Int64(B_host[kk, j].c), Int64(P)))
                if v < best_n
                    best_n = v; best_c = c
                elseif v == best_n
                    best_c = Int32(mod(Int64(best_c) + Int64(c), Int64(P)))
                end
            end
            ref[i, j] = ModCountingTropicalMin{Float64, P}(best_n, best_c)
        end

        C_dev = tropical_matmul('T', 'T', AT, BT)
        @test Array(C_dev) == ref
    end

    @testset "Spec M tropical_matmul NT, TN f32 Max" begin
        P = 13
        Random.seed!(103)
        A_host = [ModCountingTropical{Float32, P}(
                    Float32(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:4, _ in 1:5]
        B_host = [ModCountingTropical{Float32, P}(
                    Float32(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:5, _ in 1:6]

        ref = Matrix{ModCountingTropical{Float32, P}}(undef, 4, 6)
        for i in 1:4, j in 1:6
            best_n = -Inf32; best_c = Int32(0)
            for kk in 1:5
                v = A_host[i, kk].n + B_host[kk, j].n
                c = Int32(mod(Int64(A_host[i, kk].c) * Int64(B_host[kk, j].c), Int64(P)))
                if v > best_n
                    best_n = v; best_c = c
                elseif v == best_n
                    best_c = Int32(mod(Int64(best_c) + Int64(c), Int64(P)))
                end
            end
            ref[i, j] = ModCountingTropical{Float32, P}(best_n, best_c)
        end

        # NT path: A as 'N', B^T as 'T'.
        BT_host = [B_host[i, j] for j in 1:6, i in 1:5]
        A = CuArray(A_host); BT = CuArray(BT_host)
        C_NT = tropical_matmul('N', 'T', A, BT)
        @test Array(C_NT) == ref

        # TN path: A^T as 'T', B as 'N'.
        AT_host = [A_host[i, j] for j in 1:5, i in 1:4]
        AT = CuArray(AT_host); B = CuArray(B_host)
        C_TN = tropical_matmul('T', 'N', AT, B)
        @test Array(C_TN) == ref
    end

    @testset "Spec M tropical_matmul! reuse" begin
        P = 7
        Random.seed!(104)
        A_host = [ModCountingTropical{Float32, P}(
                    Float32(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:4, _ in 1:5]
        B_host = [ModCountingTropical{Float32, P}(
                    Float32(rand(0:3)), Int32(rand(0:P-1))) for _ in 1:5, _ in 1:6]
        A = CuArray(A_host); B = CuArray(B_host)
        C = CuArray{ModCountingTropical{Float32, P}}(undef, 4, 6)
        ref = Array(tropical_matmul('N', 'N', A, B))
        tropical_matmul!('N', 'N', A, B, C)
        @test Array(C) == ref
    end

    @testset "Spec M tropical_matmul errors" begin
        P = 7
        A = CuArray([ModCountingTropical{Float32, P}(0.0f0, Int32(1)) for _ in 1:2, _ in 1:3])
        B = CuArray([ModCountingTropical{Float32, P}(0.0f0, Int32(1)) for _ in 1:3, _ in 1:4])
        Bbad = CuArray([ModCountingTropical{Float32, P}(0.0f0, Int32(1)) for _ in 1:5, _ in 1:4])

        @test_throws ArgumentError tropical_matmul('X', 'N', A, B)
        @test_throws ArgumentError tropical_matmul('N', 'C', A, B)

        Aone = CuArray([ModCountingTropical{Float32, 1}(0.0f0, Int32(0)) for _ in 1:2, _ in 1:2])
        @test_throws ArgumentError tropical_matmul('N', 'N', Aone, Aone)

        @test_throws DimensionMismatch tropical_matmul('N', 'N', A, Bbad)

        Bmin = CuArray([ModCountingTropicalMin{Float32, P}(0.0f0, Int32(1)) for _ in 1:3, _ in 1:4])
        @test_throws MethodError tropical_matmul('N', 'N', A, Bmin)
    end
end
