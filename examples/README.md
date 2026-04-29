# Examples

End-to-end demos using `CountingTropicalGEMM.jl`.

## `spin_glass_demo.jl`

Computes the **ground-state energy and ground-state count** of two small
spin glasses by building their tensor networks manually, contracting via
[`OMEinsum.jl`](https://github.com/under-Peter/OMEinsum.jl), and verifying
the result against brute-force enumeration and
[`GenericTensorNetworks.jl`](https://github.com/QuEraComputing/GenericTensorNetworks.jl)'s
`solve(..., CountingMax())`.

Each pairwise contraction is performed by `LinearAlgebra.mul!`, which the
package overloads on `CuMatrix{ModCountingTropical{T, P}}` (and the `Min`
variant) to dispatch onto the tiled CUDA kernel — so OMEinsum drives the
computation and our GPU code does the work.

### Cases

- **Demo A — chain of 10 spins**, random `J ∈ {±1}` on each of the 9 edges.
- **Demo B — 3×3 grid** (9 spins, 12 edges), random `J ∈ {±1}` per edge.

Modulus `P = 65537` is well above any count we'll encounter (≤ 2¹⁰).

### Run

From the workspace root:

```bash
# 1. Build the Rust shared library once (matches the kernel sources).
cargo build --release -p tropical-gemm-cuda

# 2. Instantiate the example environment (first time only).
julia --project=examples -e 'using Pkg; Pkg.develop(path="CountingTropicalGEMM.jl"); Pkg.instantiate()'

# 3. Run the demo.
JULIA_CUDA_USE_COMPAT=false julia --project=examples examples/spin_glass_demo.jl
```

Expected output (J's depend on the seeds in the script):

```
GPU: Quadro RTX 6000

=== Demo A: chain N=10 ===
  kernel:                 score=9.0    count=2
  brute force:            score=9.0    count=2
  GenericTensorNetworks:  score=9.0    count=2
  all three agree.

=== Demo B: 3x3 grid (9 spins, 12 edges) ===
  kernel:                 score=10.0   count=4
  brute force:            score=10.0   count=4
  GenericTensorNetworks:  score=10.0   count=4
  all three agree.

All checks passed.
```

### How the math maps to tensors

For a spin glass `H(σ) = Σ_{(i,j) ∈ E} J_{ij} σ_i σ_j` (no fields), every
edge `(i, j)` becomes a 2×2 tensor

```
E_{ij}[a, b] = ModCountingTropical{Float32, P}( -J_{ij} · spin(a) · spin(b), 1 )
```

with `spin(1) = +1`, `spin(2) = -1`. Tropical multiplication adds scores
and multiplies counts mod `P`; tropical addition (max-plus) keeps the
larger score and sums counts on a tie. Contracting all edges with each
spin appearing in two tensors (one per incident edge) gives a rank-0
result whose `(value, count)` is the ground-state score and the number of
ground-state configurations.
