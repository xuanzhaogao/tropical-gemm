# Spec O: Spin Glass Tensor-Network Demo via OMEinsum + `mul!` overload

## Goal

End-to-end demo: compute the ground-state energy and ground-state
**count** of a small spin glass by building its tensor network manually
(`ModCountingTropical{Float32, P}` element type, on GPU as `CuArray`),
contracting with `OMEinsum`, and verifying against (i) brute force and
(ii) `GenericTensorNetworks.solve(..., CountingMax())`.

Mechanism: overload `LinearAlgebra.mul!` for
`CuMatrix{ModCountingTropical{T, P}}` (and the `Min` variant) so
OMEinsum's pairwise contractions transparently dispatch onto our tiled
`tropical_matmul!`.

## Two demo cases

### Demo A — Chain (N = 10)

Spin glass on a path graph `1 — 2 — … — 10`. Each edge `(i, i+1)` carries
random `J_i ∈ {±1}`. No fields. Spin variables `σ_i ∈ {+1, −1}`.

Energy: `H(σ) = Σ_i J_i σ_i σ_{i+1}`. Score `S = −H` (max-plus tropical
finds argmax-S, i.e. min-energy ground state).

Tensor network: 9 edge tensors, each shape `(2, 2)`:
```
E_i[a, b] = ModCountingTropical{Float32, P}( -J_i * spin(a) * spin(b), Int32(1) )
```
where `spin(1) = +1`, `spin(2) = −1`.

OMEinsum contraction:
```julia
ein"ab,bc,cd,de,ef,fg,gh,hi,ij->"(E_1, E_2, ..., E_9)
```
Returns a 0-rank `CountingTropical` containing `(score, count)`.

### Demo B — 3×3 grid (9 spins, 12 edges)

Vertices `a..i` laid out:
```
a b c
d e f
g h i
```
12 edges total (6 horizontal + 6 vertical). Each carries random `J_e ∈ {±1}`.
No fields.

OMEinsum contraction:
```julia
ein"ab,bc,de,ef,gh,hi,ad,dg,be,eh,cf,fi->"(E_ab, E_bc, ..., E_fi)
```

### Verification (both)

1. **Brute force.** Enumerate all `2^N` spin assignments, compute `H(σ)`,
   record `min_H` and `count = #{σ : H(σ) = min_H}`. `N ≤ 10` so this is
   1024 configs at most — runs in milliseconds.
2. **GenericTensorNetworks.** Build the same SpinGlass as
   `SpinGlass(g, J, zeros(N))`, call
   `solve(GenericTensorNetwork(problem), CountingMax())`. Returns
   `CountingTropical{Float64, Float64}(score, count)`. Compare scalar
   `score == −min_H` and `count` matches our result.

The script asserts equality and prints a per-demo summary.

## Components

### 1. `LinearAlgebra.mul!` overload (`src/CountingTropicalGEMM.jl`)

```julia
import LinearAlgebra: mul!

function LinearAlgebra.mul!(
    C::CuMatrix{ModCountingTropical{T, P}},
    A::CuMatrix{ModCountingTropical{T, P}},
    B::CuMatrix{ModCountingTropical{T, P}},
) where {T <: Union{Float32, Float64}, P}
    tropical_matmul!('N', 'N', A, B, C)
    return C
end

function LinearAlgebra.mul!(
    C::CuMatrix{ModCountingTropicalMin{T, P}},
    A::CuMatrix{ModCountingTropicalMin{T, P}},
    B::CuMatrix{ModCountingTropicalMin{T, P}},
) where {T <: Union{Float32, Float64}, P}
    tropical_matmul!('N', 'N', A, B, C)
    return C
end
```

**Why only the `(C, A, B)` 3-arg form.** OMEinsum's binary contraction
applies `permutedims` upfront and then calls a plain matrix multiply on
the reshaped operands; transpose wrappers do not reach `mul!`. If
profiling later shows OMEinsum's permute-then-multiply is wasteful, we
can add overloads on `Transpose`/`Adjoint` wrappers in a follow-on spec.

**5-argument `mul!(C, A, B, α, β)`** is *not* implemented; counting
tropical has no scalar `α/β` semantics. If OMEinsum reaches the 5-arg
form, the call falls through to the generic Julia method and is correct
but slow — we'll accept that for now.

A unit test mul!-overload sanity check (small CuMatrix case) lands in
`test/runtests.jl`.

### 2. `examples/` subproject

```
examples/
├── Project.toml          # OMEinsum, GenericTensorNetworks, Graphs, CUDA, CountingTropicalGEMM (devved)
├── Manifest.toml         # generated
├── README.md             # how to run
└── spin_glass_demo.jl    # the script
```

`examples/Project.toml`:
```toml
name = "Examples"
uuid = "12345678-1234-1234-1234-123456789abc"

[deps]
CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
CountingTropicalGEMM = "3f1a8e2b-7d9c-4e5a-9f3b-2c8a1b6d4e7f"
GenericTensorNetworks = "3521c873-ad32-4bb4-b63d-f4f178f42b49"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
OMEinsum = "ebe7aa44-baf0-506c-a96f-8464559b3922"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
```

Modulus choice: `P = 65537` (prime, ≫ 2^10 ≥ any ground-state count for
N ≤ 16).

### 3. `examples/spin_glass_demo.jl` outline

```julia
using CountingTropicalGEMM, CUDA, OMEinsum
using GenericTensorNetworks, Graphs
using Random

const P = 65537
const ELT = ModCountingTropical{Float32, P}

# spin(s) maps tensor-index 1 -> +1, 2 -> -1.
spin(s::Int) = s == 1 ? 1.0f0 : -1.0f0

function edge_tensor(J::Float32)
    M = Matrix{ELT}(undef, 2, 2)
    for a in 1:2, b in 1:2
        M[a, b] = ELT(-J * spin(a) * spin(b), Int32(1))
    end
    return CuArray(M)
end

# --- Demo A: chain N=10 ---
function demo_chain(N::Int = 10, seed::Int = 1)
    Random.seed!(seed)
    Js = Float32.(rand([-1, 1], N - 1))
    Es = [edge_tensor(J) for J in Js]
    code = ein"ab,bc,cd,de,ef,fg,gh,hi,ij->"
    res = code(Es...)                        # 0-rank Array{ELT}
    score = res[].n; count = Int(res[].c)
    bf_score, bf_count = brute_force_chain(Js)
    # Build same problem in GenericTensorNetworks
    g = path_graph(N)
    gtn_score, gtn_count = gtn_solve(g, Js, N)
    @assert score ≈ bf_score
    @assert count == bf_count
    @assert gtn_score ≈ bf_score
    @assert gtn_count == bf_count
    println("Chain N=$N: score=$score count=$count  (verified)")
end

# --- Demo B: 3x3 grid ---
function demo_grid(seed::Int = 2) ... end

# --- Brute force ---
function brute_force_energy(spins, edges_with_J)
    H = 0.0f0
    for ((i, j), J) in edges_with_J
        H += J * spins[i] * spins[j]
    end
    return H
end
function brute_force_count(N, edges_with_J)
    best = Inf32; cnt = 0
    for k in 0:(2^N - 1)
        spins = [((k >> (i-1)) & 1) == 0 ? 1.0f0 : -1.0f0 for i in 1:N]
        H = brute_force_energy(spins, edges_with_J)
        if H < best; best = H; cnt = 1
        elseif H == best; cnt += 1; end
    end
    return -best, cnt
end

# --- GenericTensorNetworks reference ---
function gtn_solve(g, Js, N)
    problem = SpinGlass(g, Js, zeros(Float32, N))
    tn = GenericTensorNetwork(problem)
    res = solve(tn, CountingMax())[]
    return Float32(res.n), Int(res.c)
end

demo_chain()
demo_grid()
println("All checks passed.")
```

(Skeleton; full code lands in implementation.)

## Out of scope

- Field terms `h_v`. No field in either demo.
- Larger graphs / performance benchmarking. The demo is correctness-only.
- `mul!` overloads on `Transpose`/`Adjoint` wrappers. OMEinsum's permute
  + matmul path doesn't need them; revisit only if profiling shows the
  permutedims is the dominant cost.
- 5-arg `mul!(C, A, B, α, β)`.
- CPU fallback. The demo requires a GPU.

## Risks

- **OMEinsum may not call `mul!` for tiny 2×2 contractions.** For some
  paths it inlines a generic loop. If so, the chain demo runs slow but
  correct on GPU. Mitigation: log when `mul!` is invoked (transient
  println in our overload during development) and verify it fires at
  least once for the grid case.
- **`permutedims` on `CuArray{ModCountingTropical{T, P}}`.** CUDA.jl
  generic permutedims kernel must handle this isbits 8-byte struct.
  Standard CUDA.jl supports `isbits` element types fine; if not, the
  demo errors loudly and we add `Base.isbitstype` or a custom permute.
- **Modulus P collision.** True ground-state count for N=10 chain or
  3×3 grid never exceeds 2^N ≤ 1024 ≪ 65537. Safe.
- **GenericTensorNetworks compatibility.** `SpinGlass(g, Js, h)`
  signature varies across versions. Pin a version in
  `examples/Project.toml`. If the installed version's API differs,
  fall back to `IsingSolver` or hand-build the score on the same graph;
  the brute-force check is the canonical reference.

## Test plan

1. `JULIA_CUDA_USE_COMPAT=false julia --project=CountingTropicalGEMM.jl
   CountingTropicalGEMM.jl/test/runtests.jl` — existing tests + the new
   `mul!`-overload sanity test pass.
2. `JULIA_CUDA_USE_COMPAT=false julia --project=examples
   examples/spin_glass_demo.jl` — both demos run, all four `@assert`
   checks pass, prints the score/count summary.
