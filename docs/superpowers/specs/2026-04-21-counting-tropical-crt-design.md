# CountingTropical + CRT GEMM backend — design

**Date:** 2026-04-21
**Status:** design approved, pending implementation plan
**Scope:** CPU/SIMD backend first. CUDA is a follow-up phase.

## Goal

Enable exact counting of ground-state (or optimal) configurations via tropical
matmul, where the count is too large for fixed-width integers. Mirrors the
`CountingAll` / `big_integer_solve` pattern in GenericTensorNetworks.jl:
run the same `CountingTropical` GEMM over several prime moduli and
reconstruct the count field to `BigInt` via the Chinese Remainder Theorem.

## Design decisions

1. **Backend phase.** CPU scalar kernel first; SIMD and CUDA in follow-ups.
2. **Tropical direction.** Parameterized: `Max` or `Min` selected via a marker
   trait, so the same type serves longest-path and ground-state (min-energy)
   problems without negation tricks.
3. **Count field.** `Mod<const P: i32>` only. No direct `u64` path, no direct
   `BigInt` path. Exact large counts are obtained by running the GEMM once per
   prime and CRT-reconstructing host-side.
4. **Matrix layout.** AoS `Mat<CountingTropical<T, C, D>>` for storage and
   public API; the inner micro-kernel operates on SoA tiles produced by the
   packing stage. Keeps the `Mat<T>` user surface uniform and the kernel
   SIMD/GPU-ready.

## Types

Location: `crates/tropical-gemm/src/types/`.

- `direction.rs` — `pub trait TropicalDirection` with associated function
  `zero_value<T: TropicalScalar>() -> T` and `prefers(a, b) -> Ordering`
  (or a `winner` helper). Unit structs `Max` and `Min` implement it. `Max`
  returns `T::neg_infinity()` and prefers the larger value; `Min` the
  opposite.
- `counting.rs` — refactor `CountingTropical<T, C>` to
  `CountingTropical<T, C, D: TropicalDirection = Max>`. `tropical_zero`,
  `tropical_add`, and `tropical_add_argmax` dispatch via `D`. Existing tests
  continue to compile by relying on the default.
- `modp.rs` — `#[repr(transparent)] pub struct Mod<const P: i32>(pub i32)`.
  - Invariant: inner value is reduced, i.e. `0 <= self.0 < P`.
  - `scalar_add`: `((a + b) as i64) % (P as i64)` cast back to `i32`.
  - `scalar_mul`: `((a as i64) * (b as i64)) % (P as i64)` cast back.
  - `scalar_zero = Mod(0)`, `scalar_one = Mod(1)`.
  - `pos_infinity` / `neg_infinity` are unused for `Mod` (it appears only
    in the count field, not as a tropical value). The trait requires them;
    we implement both as `unreachable!("Mod<P> is a count field, not a \
    tropical value")` so a misuse fails loudly.
  - Prime table: `const CRT_PRIMES: [i32; 16]` of 30-bit primes fitting in
    `i32`, chosen so squared products fit in `i64`. Concrete values picked
    at implementation time (prevprime walk from `1 << 30`).

## GEMM path

Location: `crates/tropical-gemm/src/core/` and `simd/`.

- `packing.rs`: when packing `CountingTropical<T, C, D>` tiles, produce two
  contiguous slices (value stream, count stream) so the micro-kernel consumes
  SoA while storage stays AoS. AoS→SoA conversion happens once per tile
  during packing; SoA→AoS happens once per micro-tile on the write-back.
- `kernel.rs`: add a `CountingKernel<T, C, D>` micro-kernel. Scalar inner
  loop: load `(va, ca)` and `(vb, cb)`; product `(va ⊗_T vb, ca *_mod cb)`;
  fold into accumulator with direction-aware compare — on strict win replace,
  on tie add counts mod p.
- `simd/dispatch.rs`: for v1, dispatch `CountingTropical` to the scalar
  kernel (no vectorized lanes). Leave a `TODO` for AVX2/AVX-512/NEON lanes.
- Existing `MaxPlus`/`MinPlus`/`MaxMul` paths untouched.

## CRT driver

Location: new `crates/tropical-gemm/src/crt.rs`, re-exported at crate root.

```text
count_ground_states::<T, D>(
    a_values: MatRef<T>,         // tropical values of A
    b_values: MatRef<T>,         // tropical values of B
    max_primes: usize,           // adaptive cap, default 8
) -> CountedMat<T>               // { values: Mat<T>, counts: Vec<BigInt> }
```

- The input count field is implicitly `1` everywhere (multiplicity of the
  literal value at each A[i,k] / B[k,j] entry). API takes plain value
  matrices and never asks users to construct `CountingTropical` by hand.
- Algorithm:
  1. For `k = 1..=max_primes`: pick `p_k = CRT_PRIMES[k-1]`; build
     `Mat<CountingTropical<T, Mod<p_k>, D>>` where every count is `Mod(1)`;
     run `matmul`; extract `value` and `count` fields.
  2. Assert the `value` field is identical to iteration 1 (flag numerical
     drift; for integer `T` this is exact equality, for float `T` we use
     exact `==` mirroring GTN — ground-state problems over integer/rational
     weights produce bit-identical floats in practice).
  3. After `k >= 2`, CRT-combine `(count_1, p_1), …, (count_k, p_k)` per
     cell to `BigInt`. Stop when iteration `k`'s reconstruction equals
     iteration `k-1`'s.
  4. If no convergence by `max_primes`, return the last reconstruction and
     log a warning.
- Dependencies: `num-bigint`, `num-integer`.

## Python API

Location: `crates/tropical-gemm-python/src/lib.rs`.

```python
count_ground_states(a, b, direction='max', max_primes=8)
    -> (values: np.ndarray, counts: np.ndarray[object])
```

Returns a NumPy value matrix and a matching object-dtype array of Python
`int` (unbounded). `direction` accepts `'max'` or `'min'`.

## Testing

- **Unit.** `Mod<P>` axioms (commutativity, associativity, distributivity,
  inverse of `scalar_one` under `scalar_mul`). `CountingTropical` semiring
  laws for both `D=Max` and `D=Min`. Direction marker round-trip.
- **Packing.** `CountingTropical` tile: AoS → SoA pack → micro-kernel →
  SoA → AoS write-back produces identical output to a reference AoS kernel.
- **Correctness (small).** Graphs with analytically known ground-state
  count (2×N ladder MIS, Ising chain degeneracy). Assert `count_ground_states`
  equals a reference `BigInt` computation done with a naive direct-BigInt
  matmul on the same inputs.
- **Large-count regression.** A case where the true count provably exceeds
  `2^64`. Assert CRT result matches an oracle, and assert a naive `u64`
  count path would have wrapped (sanity on the motivation).
- **Convergence.** Verify the adaptive stop triggers after `k=2` when
  counts fit within one prime, and after more iterations when they don't.

## Out of scope for v1

- CUDA kernel (phase B). The `Mod<P>` scalar is already GPU-ready; the CRT
  driver does not change between CPU and GPU.
- SIMD vectorization (AVX2 / AVX-512 / NEON) of the counting kernel.
- Argmax composition with counting.
- Negative moduli / signed arithmetic in `Mod` beyond representing residues
  as `i32`.
- MinPlus-specific optimization beyond the direction marker.

## Follow-ups (phase B / later)

1. CUDA kernel for `CountingTropical<f32, Mod<p>, D>` in
   `tropical-gemm-cuda/src/kernels.rs`. Prime loop is host-side, one
   kernel launch per prime. Streams can overlap launches with CRT work.
2. SIMD lanes: AVX2 and AVX-512 vectorized compares + fused mod-mul using
   Barrett reduction with compile-time `P`.
3. Argmax tracking for counting: record the `k` index of the first optimal
   pair plus the full count. Aligns with existing `MatWithArgmax`.
