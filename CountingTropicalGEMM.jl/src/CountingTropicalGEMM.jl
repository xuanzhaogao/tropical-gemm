"""
    CountingTropicalGEMM

Julia bindings for the `tropical-gemm-cuda` Rust crate's
`count_ground_states_gpu_u64` entry point. Computes per-cell
(value, count) for tropical (max-plus or min-plus) matrix multiplication
on a CUDA GPU, returning counts as `UInt64` (the u64 fast-path; bound
must satisfy `count_upper_bound < 2^60`).

Backed by the C ABI in `crates/tropical-gemm-cuda/src/c_api.rs`.

# Example

```julia
using CountingTropicalGEMM
A = rand(Float32, 64, 128)
B = rand(Float32, 128, 64)
res = count_ground_states_gpu_u64(Max, A, B, UInt64(128))
res.values  # 64×64 Matrix{Float32}: tropical-max value at each cell
res.counts  # 64×64 Matrix{UInt64} : count of ground states per cell
```
"""
module CountingTropicalGEMM

using Libdl

export Max, Min, CountedMatU64, TropicalMatrix
export count_ground_states_gpu_u64, bench_kernel_only_u64
export CountingTropicalGEMMError, BoundTooLargeError

# ---------------------------------------------------------------------------
# Direction tags. Match the Rust `Max` / `Min` marker types.
# ---------------------------------------------------------------------------
struct Max end
struct Min end

# ---------------------------------------------------------------------------
# Result + error types.
# ---------------------------------------------------------------------------
struct CountedMatU64{T}
    values::Matrix{T}
    counts::Matrix{UInt64}
end

abstract type CountingTropicalGEMMException <: Exception end

"""
    CountingTropicalGEMMError(code, msg)

Generic FFI / CUDA error. `code` matches the C ABI return code:
1=invalid input, 3=CUDA error, 4=internal/panic.
"""
struct CountingTropicalGEMMError <: CountingTropicalGEMMException
    code::Int32
    msg::String
end

"""
    BoundTooLargeError(msg)

`count_upper_bound` exceeds the u64 fast-path envelope (would need ≥ 3
of the 30-bit CRT primes). Caller should fall back to the Rust BigInt
entry point or decompose the problem.
"""
struct BoundTooLargeError <: CountingTropicalGEMMException
    msg::String
end

Base.showerror(io::IO, e::CountingTropicalGEMMError) =
    print(io, "CountingTropicalGEMMError(code=", e.code, "): ", e.msg)
Base.showerror(io::IO, e::BoundTooLargeError) =
    print(io, "BoundTooLargeError: ", e.msg)

const ERR_INVALID_INPUT   = Int32(1)
const ERR_BOUND_TOO_LARGE = Int32(2)
const ERR_CUDA            = Int32(3)
const ERR_INTERNAL        = Int32(4)

const EXPECTED_API_VERSION = Int32(1)

# ---------------------------------------------------------------------------
# Library resolution.
# ---------------------------------------------------------------------------
const LIB_PATH = Ref{String}("")

function _resolve_library()::String
    # 1. Explicit override.
    env = get(ENV, "TROPICAL_GEMM_LIB", "")
    if !isempty(env) && isfile(env)
        return env
    end
    # 2. Workspace dev-build output, relative to this package.
    candidate = normpath(joinpath(@__DIR__, "..", "..", "target", "release",
                                  "libtropical_gemm_cuda." * Libdl.dlext))
    if isfile(candidate)
        return candidate
    end
    # 3. Standard search paths.
    found = Libdl.find_library("tropical_gemm_cuda")
    if !isempty(found)
        return found
    end
    return ""  # Defer error to first ccall.
end

function __init__()
    LIB_PATH[] = _resolve_library()
end

function _libpath()::String
    if isempty(LIB_PATH[])
        error("libtropical_gemm_cuda not found. Set ENV[\"TROPICAL_GEMM_LIB\"] " *
              "to the absolute path, or run `cargo build --release -p " *
              "tropical-gemm-cuda` in the workspace root.")
    end
    return LIB_PATH[]
end

function _check_version()
    v = ccall((:tg_api_version, _libpath()), Cint, ())
    if v != EXPECTED_API_VERSION
        error("CountingTropicalGEMM ABI version mismatch: expected ",
              EXPECTED_API_VERSION, ", library reports ", v)
    end
end

# ---------------------------------------------------------------------------
# Error handling: map a non-zero return code into a typed Julia exception.
# ---------------------------------------------------------------------------
function _last_error_message()::String
    ptr = ccall((:tg_last_error_message, _libpath()), Cstring, ())
    ptr == C_NULL && return "(no message)"
    return unsafe_string(ptr)
end

function _throw_for(code::Int32)
    msg = _last_error_message()
    if code == ERR_BOUND_TOO_LARGE
        throw(BoundTooLargeError(msg))
    else
        throw(CountingTropicalGEMMError(code, msg))
    end
end

# ---------------------------------------------------------------------------
# Row-major helpers. Julia's Matrix{T} is column-major; the kernel expects
# row-major. We materialize transposed copies at the FFI boundary.
# ---------------------------------------------------------------------------
@inline _rowmajor(A::Matrix{T}) where {T} = collect(transpose(A))  # row-major Vector after collect

# Reshape a row-major vector back into a Julia (column-major) Matrix with
# the original logical shape (rows, cols). The buffer holds rows*cols
# elements in row-major order, so transpose-after-reshape recovers the
# correct column-major matrix.
@inline function _from_rowmajor(buf::Vector{T}, rows::Int, cols::Int) where {T}
    return collect(transpose(reshape(buf, cols, rows)))
end

# ---------------------------------------------------------------------------
# Public entry: dispatch on direction tag and scalar element type.
# ---------------------------------------------------------------------------
"""
    count_ground_states_gpu_u64(dir, A, B, bound) -> CountedMatU64

`dir` is `Max` or `Min`. `A` is `M×K`, `B` is `K×N`, both `Matrix{T}`
with `T ∈ {Float32, Float64}`. `bound` is the per-cell count upper
bound (u64; must be `< 2^60` for the u64 fast-path). Returns
`CountedMatU64{T}` with `values::Matrix{T}` and `counts::Matrix{UInt64}`,
both `M×N`.

Throws `BoundTooLargeError` if the bound exceeds the u64 envelope.
Throws `CountingTropicalGEMMError` on other failures.
"""
function count_ground_states_gpu_u64 end

for (T, sym_max, sym_min) in (
    (Float32, :tg_count_ground_states_gpu_u64_f32_max, :tg_count_ground_states_gpu_u64_f32_min),
    (Float64, :tg_count_ground_states_gpu_u64_f64_max, :tg_count_ground_states_gpu_u64_f64_min),
)
    for (dir, sym) in ((:Max, sym_max), (:Min, sym_min))
        @eval function count_ground_states_gpu_u64(::Type{$dir},
                                                   A::Matrix{$T}, B::Matrix{$T},
                                                   bound::Unsigned)::CountedMatU64{$T}
            m, k = size(A)
            k2, n = size(B)
            k == k2 || throw(DimensionMismatch(string(
                "A is ", size(A), " but B is ", size(B), "; inner dims must match")))

            a_rm = _rowmajor(A)
            b_rm = _rowmajor(B)
            out_v = Vector{$T}(undef, m * n)
            out_c = Vector{UInt64}(undef, m * n)

            _check_version()
            code = ccall(($(QuoteNode(sym)), _libpath()), Cint,
                         (Ptr{$T}, Csize_t, Csize_t,
                          Ptr{$T}, Csize_t,
                          UInt64,
                          Ptr{$T}, Ptr{UInt64}),
                         a_rm, m, k, b_rm, n,
                         convert(UInt64, bound),
                         out_v, out_c)
            if code != Int32(0)
                _throw_for(Int32(code))
            end
            CountedMatU64{$T}(_from_rowmajor(out_v, m, n),
                              _from_rowmajor(out_c, m, n))
        end
    end
end

"""
    bench_kernel_only_u64(dir, A, B, bound; iters=10) -> avg_ms::Float64

Time the counting kernel itself: upload `A` and `B` once, run the kernel
`iters` times, return average per-launch wall time in milliseconds.
Bypasses CRT combine and u64 reconstruction. Use this to measure the
kernel's intrinsic GPU speed without wrapper overhead.
"""
function bench_kernel_only_u64 end

for (T, sym_max, sym_min) in (
    (Float32, :tg_bench_kernel_only_u64_f32_max, :tg_bench_kernel_only_u64_f32_min),
    (Float64, :tg_bench_kernel_only_u64_f64_max, :tg_bench_kernel_only_u64_f64_min),
)
    for (dir, sym) in ((:Max, sym_max), (:Min, sym_min))
        @eval function bench_kernel_only_u64(::Type{$dir},
                                             A::Matrix{$T}, B::Matrix{$T},
                                             bound::Unsigned;
                                             iters::Integer = 10)::Float64
            m, k = size(A); k2, n = size(B)
            k == k2 || throw(DimensionMismatch(string(
                "A is ", size(A), " but B is ", size(B), "; inner dims must match")))
            iters > 0 || throw(ArgumentError("iters must be positive"))

            # Inputs need to be in row-major byte order for the kernel; do
            # the same boundary transpose as count_ground_states_gpu_u64.
            a_rm = _rowmajor(A)
            b_rm = _rowmajor(B)
            avg = Ref{Float64}(0.0)

            _check_version()
            code = ccall(($(QuoteNode(sym)), _libpath()), Cint,
                         (Ptr{$T}, Csize_t, Csize_t,
                          Ptr{$T}, Csize_t,
                          UInt64, UInt32, Ptr{Float64}),
                         a_rm, m, k, b_rm, n,
                         convert(UInt64, bound), convert(UInt32, iters), avg)
            if code != Int32(0)
                _throw_for(Int32(code))
            end
            avg[]
        end
    end
end

# ---------------------------------------------------------------------------
# Standard matmul interface: TropicalMatrix{T,D} * TropicalMatrix{T,D}.
# Wraps a value matrix with a direction tag (Max/Min) and an implicit
# all-ones count. `*` dispatches to count_ground_states_gpu_u64 and returns
# CountedMatU64{T}.
# ---------------------------------------------------------------------------
"""
    TropicalMatrix{T,D}(data; bound = 0)
    TropicalMatrix(D, data; bound = 0)

Matrix in the (counting) tropical semiring. `D` is `Max` or `Min`,
`T <: Union{Float32, Float64}`. Input counts are implicitly 1 per cell.

`bound` is the per-cell `count_upper_bound` forwarded to the kernel.
`bound = 0` (default) auto-selects `K = size(A, 2)` at multiply time —
the maximum possible count for a single matmul of all-ones counts.
Override only if you know a tighter or looser envelope is needed.
"""
struct TropicalMatrix{T<:Union{Float32,Float64}, D} <: AbstractMatrix{T}
    data::Matrix{T}
    bound::UInt64
end

TropicalMatrix{T,D}(data::Matrix{T}; bound::Integer = 0) where {T,D} =
    TropicalMatrix{T,D}(data, UInt64(bound))

TropicalMatrix(::Type{D}, data::Matrix{T}; bound::Integer = 0) where {T,D} =
    TropicalMatrix{T,D}(data, UInt64(bound))

Base.size(M::TropicalMatrix) = size(M.data)
Base.getindex(M::TropicalMatrix, I::Vararg{Int,N}) where {N} = getindex(M.data, I...)
Base.IndexStyle(::Type{<:TropicalMatrix}) = IndexLinear()
Base.getindex(M::TropicalMatrix, i::Int) = getindex(M.data, i)

function Base.:*(A::TropicalMatrix{T,D}, B::TropicalMatrix{T,D}) where {T,D}
    K = size(A, 2)
    bound = A.bound == 0 ? UInt64(K) : A.bound
    count_ground_states_gpu_u64(D, A.data, B.data, bound)
end

function Base.:*(A::TropicalMatrix{T,DA}, B::TropicalMatrix{T,DB}) where {T,DA,DB}
    throw(ArgumentError("cannot multiply TropicalMatrix with mismatched directions: $DA vs $DB"))
end

# ---------------------------------------------------------------------------
# Min-plus counterpart of TropicalNumbers.jl's CountingTropical.
# CountingTropical is max-plus; we define a parallel min-plus type so the
# same FFI driver can serve both directions.
# ---------------------------------------------------------------------------
"""
    CountingTropicalMin{T, CT}(n, c)

Min-plus counting tropical number: `n::T` is the value, `c::CT` is the
ground-state multiplicity. Semiring operations: `+` takes the smaller
`n` (sum counts on tie); `*` adds `n` and multiplies counts.
"""
struct CountingTropicalMin{T, CT}
    n::T
    c::CT
end

Base.zero(::Type{CountingTropicalMin{T, CT}}) where {T, CT} =
    CountingTropicalMin{T, CT}(typemax(T), zero(CT))
Base.one(::Type{CountingTropicalMin{T, CT}}) where {T, CT} =
    CountingTropicalMin{T, CT}(zero(T), one(CT))

function Base.:+(a::CountingTropicalMin{T, CT}, b::CountingTropicalMin{T, CT}) where {T, CT}
    if a.n < b.n
        a
    elseif b.n < a.n
        b
    else
        CountingTropicalMin{T, CT}(a.n, a.c + b.c)
    end
end

Base.:*(a::CountingTropicalMin{T, CT}, b::CountingTropicalMin{T, CT}) where {T, CT} =
    CountingTropicalMin{T, CT}(a.n + b.n, a.c * b.c)

Base.:(==)(a::CountingTropicalMin, b::CountingTropicalMin) =
    a.n == b.n && a.c == b.c

export CountingTropicalMin

end # module
