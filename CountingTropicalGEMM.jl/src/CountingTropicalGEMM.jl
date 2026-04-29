"""
    CountingTropicalGEMM

Julia bindings for the `tropical-gemm-cuda` Rust crate's BLAS-style mod-P
counting tropical matmul on the GPU. Inputs and output are device-resident
`CuMatrix` of `ModCountingTropical{T, P}` (max-plus) or
`ModCountingTropicalMin{T, P}` (min-plus); element type encodes both
direction and modulus. Per-operand `'N'`/`'T'` flags follow the BLAS
column-major convention.

# Example

```julia
using CountingTropicalGEMM, CUDA

P = 7
A = CuArray([ModCountingTropical{Float32, P}(rand(Float32), Int32(rand(0:P-1)))
             for _ in 1:M, _ in 1:K])
B = CuArray([ModCountingTropical{Float32, P}(rand(Float32), Int32(rand(0:P-1)))
             for _ in 1:K, _ in 1:N])
C = tropical_matmul('N', 'N', A, B)   # M x N CuMatrix
```
"""
module CountingTropicalGEMM

using CUDA
using Libdl

export ModCountingTropical, ModCountingTropicalMin
export tropical_matmul, tropical_matmul!
export CountingTropicalGEMMError

# ---------------------------------------------------------------------------
# Error type.
# ---------------------------------------------------------------------------
"""
    CountingTropicalGEMMError(code, msg)

Generic FFI / CUDA error. `code` matches the C ABI return code:
1=invalid input, 3=CUDA error, 4=internal/panic.
"""
struct CountingTropicalGEMMError <: Exception
    code::Int32
    msg::String
end

Base.showerror(io::IO, e::CountingTropicalGEMMError) =
    print(io, "CountingTropicalGEMMError(code=", e.code, "): ", e.msg)

const ERR_INVALID_INPUT   = Int32(1)
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
    # CUDA.jl + cudarc interop note: both runtimes target the same CUDA
    # *primary* context (cudarc retains it via cuDevicePrimaryCtxRetain;
    # CUDA.jl uses the same context by default), so device pointers are
    # interchangeable across the two runtimes. However, if CUDA.jl loads
    # its bundled forward-compat libcuda artifact (newer than the host
    # kernel module), cudarc's `cuInit(0)` fails with
    # CUDA_ERROR_OPERATING_SYSTEM. If you see that error, set
    # `ENV["JULIA_CUDA_USE_COMPAT"] = "false"` *before* `using CUDA` so the
    # process opens the system libcuda that matches the kernel driver.
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
    throw(CountingTropicalGEMMError(code, msg))
end

# ---------------------------------------------------------------------------
# AoS counting tropical element types matching the Rust PairT layout exactly.
# ---------------------------------------------------------------------------
"""
    ModCountingTropical{T, P}(n::T, c::Int32)

Max-plus counting tropical number with count reduced mod `P`. Layout
matches the Rust `PairT` struct exactly: `n::T` followed by `c::Int32`
(plus 4 B padding when `T == Float64`).

`T ∈ {Float32, Float64}`. `P` is the modulus (compile-time `Int`,
must satisfy 2 <= P < 2^31). Counts are stored as `Int32` in `[0, P)`.
"""
struct ModCountingTropical{T, P}
    n::T
    c::Int32
end

"""
    ModCountingTropicalMin{T, P}(n::T, c::Int32)

Min-plus counterpart of `ModCountingTropical`. Same layout and
constraints; `+` takes the smaller `n`.
"""
struct ModCountingTropicalMin{T, P}
    n::T
    c::Int32
end

# --- Max-plus semiring ---

Base.zero(::Type{ModCountingTropical{T, P}}) where {T, P} =
    ModCountingTropical{T, P}(typemin(T), Int32(0))
Base.one(::Type{ModCountingTropical{T, P}}) where {T, P} =
    ModCountingTropical{T, P}(zero(T), Int32(1))

function Base.:+(a::ModCountingTropical{T, P}, b::ModCountingTropical{T, P}) where {T, P}
    if a.n > b.n
        a
    elseif b.n > a.n
        b
    else
        ModCountingTropical{T, P}(a.n,
            Int32(mod(Int64(a.c) + Int64(b.c), Int64(P))))
    end
end

function Base.:*(a::ModCountingTropical{T, P}, b::ModCountingTropical{T, P}) where {T, P}
    ModCountingTropical{T, P}(a.n + b.n,
        Int32(mod(Int64(a.c) * Int64(b.c), Int64(P))))
end

Base.:(==)(a::ModCountingTropical, b::ModCountingTropical) =
    a.n == b.n && a.c == b.c

# --- Min-plus semiring ---

# Tuple-style display with unicode subscripts: value carries '+' or '-'
# (max-plus / min-plus); count carries the modulus P.
# Examples: (3.0₊, 2₇)   (1.5₋, 4₁₁)
const _SUBSCRIPT_DIGITS = ('₀','₁','₂','₃','₄','₅','₆','₇','₈','₉')
_subscript_int(n::Integer) = join(_SUBSCRIPT_DIGITS[d - '0' + 1] for d in string(n))

function Base.show(io::IO, x::ModCountingTropical{T, P}) where {T, P}
    print(io, "(", x.n, "₊, ", x.c, _subscript_int(P), ")")
end
Base.show(io::IO, ::MIME"text/plain", x::ModCountingTropical) = show(io, x)

Base.zero(::Type{ModCountingTropicalMin{T, P}}) where {T, P} =
    ModCountingTropicalMin{T, P}(typemax(T), Int32(0))
Base.one(::Type{ModCountingTropicalMin{T, P}}) where {T, P} =
    ModCountingTropicalMin{T, P}(zero(T), Int32(1))

function Base.:+(a::ModCountingTropicalMin{T, P}, b::ModCountingTropicalMin{T, P}) where {T, P}
    if a.n < b.n
        a
    elseif b.n < a.n
        b
    else
        ModCountingTropicalMin{T, P}(a.n,
            Int32(mod(Int64(a.c) + Int64(b.c), Int64(P))))
    end
end

function Base.:*(a::ModCountingTropicalMin{T, P}, b::ModCountingTropicalMin{T, P}) where {T, P}
    ModCountingTropicalMin{T, P}(a.n + b.n,
        Int32(mod(Int64(a.c) * Int64(b.c), Int64(P))))
end

Base.:(==)(a::ModCountingTropicalMin, b::ModCountingTropicalMin) =
    a.n == b.n && a.c == b.c

function Base.show(io::IO, x::ModCountingTropicalMin{T, P}) where {T, P}
    print(io, "(", x.n, "₋, ", x.c, _subscript_int(P), ")")
end
Base.show(io::IO, ::MIME"text/plain", x::ModCountingTropicalMin) = show(io, x)

# ---------------------------------------------------------------------------
# Spec M: column-major BLAS-style mod-P counting tropical GEMM.
# Inputs and output are device-resident CuMatrix; element type encodes
# direction (ModCountingTropical = max-plus, ModCountingTropicalMin = min-plus)
# and modulus P.
# ---------------------------------------------------------------------------

# Per-(T, dir) ccall thunks. Each takes (tA, tB, M, K, N, a_dev, b_dev, p, out_dev).
for (T, sym_max, sym_min) in (
    (Float32, :tg_tropical_matmul_f32_max, :tg_tropical_matmul_f32_min),
    (Float64, :tg_tropical_matmul_f64_max, :tg_tropical_matmul_f64_min),
)
    for (dir_sym, sym) in ((:max, sym_max), (:min, sym_min))
        thunk = Symbol("_spec_m_thunk_", T, "_", dir_sym)
        @eval function $thunk(tA::Cchar, tB::Cchar,
                              m::Csize_t, k::Csize_t, n::Csize_t,
                              a_dev::UInt64, b_dev::UInt64,
                              p::Int32,
                              out_dev::UInt64)
            _check_version()
            code = ccall(($(QuoteNode(sym)), _libpath()), Cint,
                (Cchar, Cchar, Csize_t, Csize_t, Csize_t, UInt64, UInt64, Int32, UInt64),
                tA, tB, m, k, n, a_dev, b_dev, p, out_dev)
            if code != Int32(0)
                _throw_for(Int32(code))
            end
            return nothing
        end
    end
end

@inline function _spec_m_validate_flag(flag::Char)
    flag == 'N' || flag == 'T' || throw(ArgumentError(
        "tA/tB must be 'N' or 'T', got $(flag)"))
end

@inline function _spec_m_validate_p(P::Integer)
    2 <= P < (Int64(1) << 31) || throw(ArgumentError(
        "modulus P must satisfy 2 <= P < 2^31, got $P"))
end

@inline function _spec_m_logical_dims(tA::Char, tB::Char, sA::NTuple{2,Int}, sB::NTuple{2,Int})
    rA, cA = sA; rB, cB = sB
    M = (tA == 'N') ? rA : cA
    Kused = (tA == 'N') ? cA : rA
    Kchk = (tB == 'N') ? rB : cB
    N = (tB == 'N') ? cB : rB
    Kused == Kchk || throw(DimensionMismatch(
        "inner K mismatch: op($tA, A) gives K=$Kused, op($tB, B) gives K=$Kchk"))
    return M, Kused, N
end

@inline _spec_m_u64ptr(A::CuArray) = UInt64(UInt(pointer(A)))

# Internal: dispatch to the right (T, dir) thunk.
function _spec_m_call(::Type{T}, ::Val{dir}, tA::Char, tB::Char,
                      M::Int, K::Int, N::Int, P::Integer,
                      a_ptr::UInt64, b_ptr::UInt64, out_ptr::UInt64
                     ) where {T, dir}
    if T === Float32 && dir === :max
        _spec_m_thunk_Float32_max(Cchar(tA), Cchar(tB),
            Csize_t(M), Csize_t(K), Csize_t(N), a_ptr, b_ptr, Int32(P), out_ptr)
    elseif T === Float32 && dir === :min
        _spec_m_thunk_Float32_min(Cchar(tA), Cchar(tB),
            Csize_t(M), Csize_t(K), Csize_t(N), a_ptr, b_ptr, Int32(P), out_ptr)
    elseif T === Float64 && dir === :max
        _spec_m_thunk_Float64_max(Cchar(tA), Cchar(tB),
            Csize_t(M), Csize_t(K), Csize_t(N), a_ptr, b_ptr, Int32(P), out_ptr)
    elseif T === Float64 && dir === :min
        _spec_m_thunk_Float64_min(Cchar(tA), Cchar(tB),
            Csize_t(M), Csize_t(K), Csize_t(N), a_ptr, b_ptr, Int32(P), out_ptr)
    else
        error("unreachable: T=$T, dir=$dir")
    end
end

# Public: max-plus on CuMatrix{ModCountingTropical{T, P}}.
function tropical_matmul(tA::Char, tB::Char,
                         A::CuMatrix{ModCountingTropical{T, P}},
                         B::CuMatrix{ModCountingTropical{T, P}}
                        ) where {T <: Union{Float32, Float64}, P}
    _spec_m_validate_flag(tA); _spec_m_validate_flag(tB); _spec_m_validate_p(P)
    M, K, N = _spec_m_logical_dims(tA, tB, size(A), size(B))
    out = CuArray{ModCountingTropical{T, P}}(undef, M, N)
    _spec_m_call(T, Val(:max), tA, tB, M, K, N, P,
                 _spec_m_u64ptr(A), _spec_m_u64ptr(B), _spec_m_u64ptr(out))
    return out
end

# Public: min-plus on CuMatrix{ModCountingTropicalMin{T, P}}.
function tropical_matmul(tA::Char, tB::Char,
                         A::CuMatrix{ModCountingTropicalMin{T, P}},
                         B::CuMatrix{ModCountingTropicalMin{T, P}}
                        ) where {T <: Union{Float32, Float64}, P}
    _spec_m_validate_flag(tA); _spec_m_validate_flag(tB); _spec_m_validate_p(P)
    M, K, N = _spec_m_logical_dims(tA, tB, size(A), size(B))
    out = CuArray{ModCountingTropicalMin{T, P}}(undef, M, N)
    _spec_m_call(T, Val(:min), tA, tB, M, K, N, P,
                 _spec_m_u64ptr(A), _spec_m_u64ptr(B), _spec_m_u64ptr(out))
    return out
end

# Public: in-place tropical_matmul! (max-plus).
function tropical_matmul!(tA::Char, tB::Char,
                          A::CuMatrix{ModCountingTropical{T, P}},
                          B::CuMatrix{ModCountingTropical{T, P}},
                          C::CuMatrix{ModCountingTropical{T, P}}
                         ) where {T <: Union{Float32, Float64}, P}
    _spec_m_validate_flag(tA); _spec_m_validate_flag(tB); _spec_m_validate_p(P)
    M, K, N = _spec_m_logical_dims(tA, tB, size(A), size(B))
    size(C) == (M, N) || throw(DimensionMismatch(
        "C is $(size(C)) but op($tA,A)*op($tB,B) is $((M, N))"))
    _spec_m_call(T, Val(:max), tA, tB, M, K, N, P,
                 _spec_m_u64ptr(A), _spec_m_u64ptr(B), _spec_m_u64ptr(C))
    return C
end

# Public: in-place tropical_matmul! (min-plus).
function tropical_matmul!(tA::Char, tB::Char,
                          A::CuMatrix{ModCountingTropicalMin{T, P}},
                          B::CuMatrix{ModCountingTropicalMin{T, P}},
                          C::CuMatrix{ModCountingTropicalMin{T, P}}
                         ) where {T <: Union{Float32, Float64}, P}
    _spec_m_validate_flag(tA); _spec_m_validate_flag(tB); _spec_m_validate_p(P)
    M, K, N = _spec_m_logical_dims(tA, tB, size(A), size(B))
    size(C) == (M, N) || throw(DimensionMismatch(
        "C is $(size(C)) but op($tA,A)*op($tB,B) is $((M, N))"))
    _spec_m_call(T, Val(:min), tA, tB, M, K, N, P,
                 _spec_m_u64ptr(A), _spec_m_u64ptr(B), _spec_m_u64ptr(C))
    return C
end

end # module
