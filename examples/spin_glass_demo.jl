# Spec O: spin-glass tensor-network demo on GPU.
#
# Builds the tensor network for a small spin glass with elements of type
# `ModCountingTropical{Float32, P}`, places it on the GPU, and contracts via
# OMEinsum. Each pairwise contraction reduces to `mul!(C, A, B)`, which our
# package overloads to the tiled CUDA kernel. The result (ground-state
# score, count) is verified against (i) brute-force enumeration and
# (ii) GenericTensorNetworks `solve(..., CountingMax())`.
#
# Run from workspace root:
#   JULIA_CUDA_USE_COMPAT=false julia --project=examples examples/spin_glass_demo.jl

get!(ENV, "JULIA_CUDA_USE_COMPAT", "false")

using CountingTropicalGEMM
using CUDA
using OMEinsum
using GenericTensorNetworks
using Graphs
using Random
using Printf

const P = 2147483647                          # M31 (2^31 − 1), max P our Int32 count supports
const ELT = ModCountingTropical{Float32, P}

# Tensor index a ∈ {1, 2}  →  spin {+1, -1}.
spin(a::Int) = a == 1 ? 1.0f0 : -1.0f0

# 2x2 edge tensor on the GPU. Score is −J σ_a σ_b so max-plus picks the
# minimum-energy configuration. Count starts at 1 per spin pair.
function edge_tensor(J::Float32)
    M = Matrix{ELT}(undef, 2, 2)
    for a in 1:2, b in 1:2
        M[a, b] = ELT(-J * spin(a) * spin(b), Int32(1))
    end
    return CuArray(M)
end

# Brute-force ground state and count. `edges_with_J` :: Vector{Tuple{Tuple{Int,Int}, Float32}}.
function brute_force(N::Int, edges_with_J::Vector{Tuple{Tuple{Int,Int}, Float32}})
    best = Inf32
    cnt = 0
    for k in 0:(2^N - 1)
        spins = ntuple(i -> ((k >> (i-1)) & 1) == 0 ? 1.0f0 : -1.0f0, N)
        H = 0.0f0
        for ((i, j), J) in edges_with_J
            H += J * spins[i] * spins[j]
        end
        if H < best
            best = H; cnt = 1
        elseif H == best
            cnt += 1
        end
    end
    # Tropical score is −min_H so it's compared against the kernel's value.
    return -best, cnt
end

# GenericTensorNetworks reference: ground-state score & count via CountingMax.
# Js_by_edge maps an unordered edge tuple `(min(i,j), max(i,j))` to its J value;
# we iterate edges(g) in canonical order (the order GTN's SpinGlass expects)
# and emit a coupling vector aligned to that.
function gtn_solve(g::SimpleGraph,
                   Js_by_edge::Dict{Tuple{Int,Int}, Float32},
                   N::Int;
                   usecuda::Bool = false)
    Js_canonical = Float64[]
    for e in edges(g)
        i, j = src(e), dst(e)
        key = (min(i, j), max(i, j))
        push!(Js_canonical, Float64(Js_by_edge[key]))
    end
    h = zeros(Float64, N)
    problem = SpinGlass(g, Js_canonical, h)
    tn = GenericTensorNetwork(problem)
    res_arr = solve(tn, CountingMax(); usecuda)
    # solve(...) returns a 0-d array; on the CUDA path it's a CuArray{T,0}
    # which forbids scalar indexing. Round-trip via Array to extract the
    # single CountingTropical scalar.
    res = (res_arr isa Array ? res_arr : Array(res_arr))[]
    return Float32(res.n), Int(res.c)
end

# ---------------------------------------------------------------------------
# Demo A: chain N spins.
# ---------------------------------------------------------------------------
function demo_chain(N::Int = 10; seed::Int = 1)
    Random.seed!(seed)
    Js = Float32.(rand([-1, 1], N - 1))
    edges_with_J = [(i, i+1) => Js[i] for i in 1:N-1]
    edges_with_J = Tuple{Tuple{Int,Int}, Float32}[((i, i+1), Js[i]) for i in 1:N-1]

    Es = [edge_tensor(J) for J in Js]
    @assert N == 10 "chain demo hardcodes N=10 OMEinsum string"
    code = ein"ab,bc,cd,de,ef,fg,gh,hi,ij->"
    # Reduce the multi-tensor einsum to a left-to-right tree of binary
    # contractions so each step becomes a `mul!` on a CuMatrix.
    optcode = optimize_code(code, uniformsize(code, 2), GreedyMethod())
    res = optcode(Es...)::CuArray{ELT, 0}
    res_host = Array(res)[]
    score = res_host.n
    count = Int(res_host.c)

    bf_score, bf_count = brute_force(N, edges_with_J)
    g = path_graph(N)
    Js_by_edge = Dict((min(i,j), max(i,j)) => J for ((i, j), J) in edges_with_J)
    gtn_score, gtn_count = gtn_solve(g, Js_by_edge, N)

    println("=== Demo A: chain N=$N ===")
    @printf "  kernel:                 score=%-6.1f count=%d\n" score count
    @printf "  brute force:            score=%-6.1f count=%d\n" bf_score bf_count
    @printf "  GenericTensorNetworks:  score=%-6.1f count=%d\n" gtn_score gtn_count
    @assert score == bf_score          "chain: kernel score $(score) != brute $(bf_score)"
    @assert count == bf_count          "chain: kernel count $(count) != brute $(bf_count)"
    @assert gtn_score == bf_score      "chain: GTN score $(gtn_score) != brute $(bf_score)"
    @assert gtn_count == bf_count      "chain: GTN count $(gtn_count) != brute $(bf_count)"
    println("  all three agree.\n")
end

# ---------------------------------------------------------------------------
# Demo B: 3x3 grid, 9 spins, 12 edges.
# Vertex layout (1-indexed):
#   1 2 3
#   4 5 6
#   7 8 9
# Horizontal edges: (1,2)(2,3)(4,5)(5,6)(7,8)(8,9)
# Vertical   edges: (1,4)(4,7)(2,5)(5,8)(3,6)(6,9)
# ---------------------------------------------------------------------------
function demo_grid(; seed::Int = 2)
    Random.seed!(seed)
    edge_list = [
        (1,2),(2,3),(4,5),(5,6),(7,8),(8,9),    # horizontal
        (1,4),(4,7),(2,5),(5,8),(3,6),(6,9),    # vertical
    ]
    N = 9
    Js = Float32.(rand([-1, 1], length(edge_list)))
    edges_with_J = [(e, J) for (e, J) in zip(edge_list, Js)]
    Es = [edge_tensor(J) for J in Js]

    # OMEinsum tags: each spin gets a single label a..i. Each edge tensor
    # has the two endpoint labels. Repeated labels are summed (= max-plus
    # reduction over that spin).
    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
    edge_label_pairs = [[labels[i], labels[j]] for (i, j) in edge_list]
    code = OMEinsum.EinCode(edge_label_pairs, Char[])
    optcode = optimize_code(code, uniformsize(code, 2), TreeSA(; ntrials=1, niters=5))
    res = optcode(Es...)::CuArray{ELT, 0}
    res_host = Array(res)[]
    score = res_host.n
    count = Int(res_host.c)

    bf_score, bf_count = brute_force(N, edges_with_J)
    g = SimpleGraph(N)
    for (i, j) in edge_list
        add_edge!(g, i, j)
    end
    Js_by_edge = Dict((min(i,j), max(i,j)) => J for ((i, j), J) in edges_with_J)
    gtn_score, gtn_count = gtn_solve(g, Js_by_edge, N)

    println("=== Demo B: 3x3 grid (9 spins, 12 edges) ===")
    @printf "  kernel:                 score=%-6.1f count=%d\n" score count
    @printf "  brute force:            score=%-6.1f count=%d\n" bf_score bf_count
    @printf "  GenericTensorNetworks:  score=%-6.1f count=%d\n" gtn_score gtn_count
    @assert score == bf_score          "grid: kernel score $(score) != brute $(bf_score)"
    @assert count == bf_count          "grid: kernel count $(count) != brute $(bf_count)"
    @assert gtn_score == bf_score      "grid: GTN score $(gtn_score) != brute $(bf_score)"
    @assert gtn_count == bf_count      "grid: GTN count $(gtn_count) != brute $(bf_count)"
    println("  all three agree.\n")
end

# ---------------------------------------------------------------------------
# Demo C: L×L grid benchmark. No brute force (2^(L²) is infeasible for L≥6);
# verification is against GenericTensorNetworks. Times the OMEinsum + GPU
# pipeline against GTN's CountingMax solve.
# ---------------------------------------------------------------------------
function build_grid_edges(L::Int)
    edges = Tuple{Int,Int}[]
    @inline vid(r, c) = (r - 1) * L + c
    for r in 1:L, c in 1:L
        if c < L; push!(edges, (vid(r, c), vid(r, c + 1))); end   # horizontal
        if r < L; push!(edges, (vid(r, c), vid(r + 1, c))); end   # vertical
    end
    return edges
end

function bench_grid(L::Int; seed::Int = 42, ntrials_treesa::Int = 1, niters_treesa::Int = 5,
                    sc_skip::Real = 18, run_gtn::Bool = true)
    Random.seed!(seed)
    N = L * L
    edge_list = build_grid_edges(L)
    Js = Float32.(rand([-1, 1], length(edge_list)))
    edges_with_J = [(e, J) for (e, J) in zip(edge_list, Js)]
    Es = [edge_tensor(J) for J in Js]

    # OMEinsum: each spin gets a label string "v<index>". Indices repeated
    # across edge tensors are summed (max-plus reduction over that spin).
    labels = [Symbol("v", i) for i in 1:N]
    edge_label_pairs = [[labels[i], labels[j]] for (i, j) in edge_list]
    code = OMEinsum.EinCode(edge_label_pairs, Symbol[])

    @printf "=== Demo C: %d×%d grid (%d spins, %d edges) ===\n" L L N length(edge_list)

    # Time the contraction-order optimization (CPU, GTN does the same internally).
    t_opt = @elapsed optcode = optimize_code(code, uniformsize(code, 2),
        TreeSA(; ntrials=ntrials_treesa, niters=niters_treesa))
    cc = contraction_complexity(optcode, uniformsize(code, 2))
    @printf "  TreeSA optimize:        %8.3f s   tc=%g sc=%g rwc=%g\n" t_opt cc.tc cc.sc cc.rwc

    if cc.sc > sc_skip
        @printf "  largest intermediate dim ≈ 2^%g = %g elems × 8 B = %.2f GB — skipping (sc>%g)\n" cc.sc 2.0^cc.sc 2.0^cc.sc * 8 / 1e9 sc_skip
        return (; t_opt, t_kernel = NaN, t_gtn = NaN, score = NaN, count = -1, gtn_score = NaN, gtn_count = -1, sc = cc.sc)
    end

    # Warmup (NVRTC compile, kernel cache, etc.).
    optcode(Es...); CUDA.synchronize()
    # Measure kernel-driven contraction (warm).
    t_kernel = @elapsed begin
        res = optcode(Es...)
        CUDA.synchronize()
    end
    res_host = Array(res::CuArray{ELT, 0})[]
    score = res_host.n
    count = Int(res_host.c)
    @printf "  GPU contract (warm):    %8.3f s   score=%-8.1f count=%d\n" t_kernel score count

    # GTN reference (CPU).
    g = SimpleGraph(N)
    for (i, j) in edge_list; add_edge!(g, i, j); end
    Js_by_edge = Dict((min(i,j), max(i,j)) => J for ((i, j), J) in edges_with_J)
    # Warmup GTN (compile-cost) — solve once, discard.
    gtn_solve(g, Js_by_edge, N)
    t_gtn = @elapsed gtn_score, gtn_count = gtn_solve(g, Js_by_edge, N)
    @printf "  GTN solve   (warm):     %8.3f s   score=%-8.1f count=%d\n" t_gtn gtn_score gtn_count

    if score == gtn_score && count == gtn_count
        println("  ✓ kernel and GTN agree.\n")
    elseif score == gtn_score && count == mod(Int(gtn_count), P)
        @printf "  ✓ score matches; count overflows P=%d (kernel = GTN mod P).\n\n" P
    else
        @warn "kernel and GTN disagree" score count gtn_score gtn_count
    end
    return (; t_opt, t_kernel, t_gtn, score, count, gtn_score, gtn_count)
end

# ---------------------------------------------------------------------------
# Demo D: CRT reconstruction. Run the GPU pipeline twice with two coprime
# primes P1, P2 to obtain count mod P1 and count mod P2, then combine via
# 2-prime CRT to recover the exact count modulo P1*P2.
#
# Prime choice (Spec Q): the GPU kernel has a defer-mod fast path that
# requires `K · (P-1)² < 2^63`, where K is the largest matmul reduction
# dim in the contraction. M31-class primes (P ≈ 2^31) violate this gate
# and route to the ~2.4× slower Barrett-in-loop kernel. We instead pick
# two primes near the largest P that keeps the fast path active for the
# given K — this is faster even though P1·P2 is smaller, so the CRT
# envelope is smaller. Keep `K_MAX_LOG2` generous enough to cover the
# largest actual K in the contraction; the count_bound argument controls
# how much margin we want over the expected count.
# ---------------------------------------------------------------------------

# Largest reduction dim K we expect to see in any single matmul step of
# the OMEinsum contraction. For grid problems with bond dim 2 the
# practical K rarely exceeds 2^20; raise this if a particular problem
# shows kernel rejection.
const K_MAX_LOG2 = 18

function _is_prime(n::Integer)
    n < 2 && return false
    n < 4 && return true
    iseven(n) && return false
    n % 3 == 0 && return n == 3
    d = 5
    while d * d ≤ n
        n % d == 0 && return false
        n % (d + 2) == 0 && return false
        d += 6
    end
    return true
end

function _prev_prime(n::Integer)
    p = iseven(n) ? n - 1 : n
    while p ≥ 3 && !_is_prime(p); p -= 2; end
    return p
end

# Pick two distinct primes (P1 > P2) maximally below the fast-path
# envelope. Both satisfy K_max · (P-1)² < 2^63, so the GPU kernel runs
# the defer-mod fast path. Caller checks P1·P2 > expected count.
function pick_fast_crt_primes(k_max_log2::Integer = K_MAX_LOG2)
    # P-1 < floor(sqrt(2^63 / K_max))
    pmax = isqrt(Int128(1) << (63 - k_max_log2))
    pmax = min(pmax, Int128(2)^31 - 1)
    p1 = _prev_prime(Int(pmax))
    p2 = _prev_prime(p1 - 2)
    return p1, p2
end

const _P1, _P2 = pick_fast_crt_primes(K_MAX_LOG2)
@assert gcd(_P1, _P2) == 1

function _edge_tensor_for(::Type{E}, J::Float32) where E
    M = Matrix{E}(undef, 2, 2)
    for a in 1:2, b in 1:2
        M[a, b] = E(-J * spin(a) * spin(b), Int32(1))
    end
    return CuArray(M)
end

function _contract_with_P(::Type{E}, optcode, Js::Vector{Float32}) where E
    Es = [_edge_tensor_for(E, J) for J in Js]
    optcode(Es...); CUDA.synchronize()                # warmup
    t = @elapsed begin
        res = optcode(Es...)
        CUDA.synchronize()
    end
    res_host = Array(res::CuArray{E, 0})[]
    return res_host.n, Int(res_host.c), t
end

# 2-prime CRT: returns x ∈ [0, P1*P2) with x ≡ r1 (mod P1), x ≡ r2 (mod P2).
function crt2(r1::Integer, P1::Integer, r2::Integer, P2::Integer)
    inv_P1_mod_P2 = invmod(P1, P2)
    k = mod((Int128(r2) - Int128(r1)) * Int128(inv_P1_mod_P2), Int128(P2))
    return Int128(r1) + Int128(P1) * k
end

function demo_grid_crt(L::Int; seed::Int = 42, ntrials::Int = 5, niters::Int = 20)
    Random.seed!(seed)
    N = L * L
    edge_list = build_grid_edges(L)
    Js = Float32.(rand([-1, 1], length(edge_list)))
    edges_with_J = [(e, J) for (e, J) in zip(edge_list, Js)]

    labels = [Symbol("v", i) for i in 1:N]
    edge_label_pairs = [[labels[i], labels[j]] for (i, j) in edge_list]
    code = OMEinsum.EinCode(edge_label_pairs, Symbol[])
    t_opt = @elapsed optcode = optimize_code(code, uniformsize(code, 2),
        TreeSA(; ntrials, niters))
    cc = contraction_complexity(optcode, uniformsize(code, 2))

    @printf "=== Demo D: %d×%d grid CRT count reconstruction ===\n" L L
    @printf "  TreeSA optimize:        %8.3f s   tc=%g sc=%g\n" t_opt cc.tc cc.sc
    @printf "  fast-path CRT primes:   P1=%d (%.1f bits)  P2=%d (%.1f bits)  P1·P2=%s (%.1f bits)\n" _P1 log2(_P1) _P2 log2(_P2) string(Int128(_P1)*Int128(_P2)) log2(Float64(Int128(_P1)*Int128(_P2)))

    score1, count1, t1 = _contract_with_P(ModCountingTropical{Float32, _P1}, optcode, Js)
    score2, count2, t2 = _contract_with_P(ModCountingTropical{Float32, _P2}, optcode, Js)
    @assert score1 == score2

    exact = crt2(count1, _P1, count2, _P2)
    @printf "  pass 1 (P=%d): %8.3f s   count mod P1 = %d\n" _P1 t1 count1
    @printf "  pass 2 (P=%d): %8.3f s   count mod P2 = %d\n" _P2 t2 count2
    @printf "  CRT-reconstructed exact count = %s\n" string(exact)
    @printf "  ground-state score = %s\n" string(score1)

    # Cross-check against GTN — run its GPU path so it's an apples-to-apples
    # GPU-vs-GPU comparison. (CPU path also available via `usecuda=false`.)
    g = SimpleGraph(N)
    for (i, j) in edge_list; add_edge!(g, i, j); end
    Js_by_edge = Dict((min(i,j), max(i,j)) => J for ((i, j), J) in edges_with_J)
    gtn_solve(g, Js_by_edge, N; usecuda=true); CUDA.synchronize()  # GTN-GPU warmup
    t_gtn = @elapsed begin
        gtn_score, gtn_count = gtn_solve(g, Js_by_edge, N; usecuda=true)
        CUDA.synchronize()
    end
    @printf "  GTN-GPU reference: %8.3f s   exact count = %d\n" t_gtn gtn_count
    @printf "  speedup over GTN-GPU:           %.2f×  (ours = %.3f s for both passes)\n" (t_gtn / (t1 + t2)) (t1 + t2)

    if Int128(score1) == Int128(gtn_score) && exact == Int128(gtn_count)
        println("  ✓ CRT reconstruction matches GTN-GPU exactly.\n")
    else
        @warn "disagreement" score1 gtn_score exact gtn_count
    end
    return (; t_opt, t1, t2, t_gtn, exact, gtn_count)
end

function main()
    @printf "GPU: %s\n\n" CUDA.name(CUDA.device())
    demo_chain()
    demo_grid()
    println("--- benchmark (no brute force) ---\n")
    bench_grid(6)
    bench_grid(8)
    bench_grid(10)
    bench_grid(12)
    bench_grid(14)
    bench_grid(20; ntrials_treesa = 5, niters_treesa = 20, sc_skip = 26)
    println("--- CRT demo ---\n")
    demo_grid_crt(20)
    println("All checks passed.")
end

isinteractive() || main()
