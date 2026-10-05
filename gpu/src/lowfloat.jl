# Software model of narrow floating-point formats (Float8, Float4, ...).
#
# GPUs have no scalar arithmetic units for Float8/Float4 (those formats exist
# only inside tensor-core matrix multiplies), so their *accuracy* is studied by
# emulation: a LowFloat stores a Float32 and EVERY operation result is rounded
# (to nearest, ties to even) to the target format -- the model of an ALU of
# that precision. Transcendentals are evaluated in Float32 and rounded once
# (a correctly rounded low-precision libm). Not for speed.

"""
    LowFloat{E, M, FN} <: AbstractFloat

Emulated binary floating point with `E` exponent and `M` mantissa bits.
`FN = false`: IEEE-like (Inf on overflow; Float16, BFloat16, E5M2).
`FN = true`: "finite" variants without Inf, overflow saturates (E4M3FN, E2M1).
"""
struct LowFloat{E, M, FN} <: AbstractFloat
    x::Float32
    global _rawlf(::Type{LowFloat{E, M, FN}}, x::Float32) where {E, M, FN} = new{E, M, FN}(x)
end

const Float8_E4M3 = LowFloat{4, 3, true}     # OCP FP8 E4M3FN: max 448
const Float8_E5M2 = LowFloat{5, 2, false}    # OCP FP8 E5M2: max 57344
const Float4_E2M1 = LowFloat{2, 1, true}     # FP4 E2M1: values 0, .5, 1, 1.5, 2, 3, 4, 6
const Float16_emu = LowFloat{5, 10, false}   # IEEE half (to cross-check the native one)
const BFloat16_emu = LowFloat{8, 7, false}   # bfloat16

@inline _bias(::Type{LowFloat{E, M, FN}}) where {E, M, FN} = 2^(E - 1) - 1
@inline _emin(F::Type{<:LowFloat}) = 1 - _bias(F)
@inline _emax(::Type{LowFloat{E, M, FN}}) where {E, M, FN} = FN ? 2^(E - 1) : 2^(E - 1) - 1
# largest finite value: IEEE (2 - 2^-M)·2^emax; E4M3FN reserves the all-ones
# mantissa at the top exponent for NaN: (2 - 2^(1-M))·2^emax; E2M1 has no NaN
@inline function _fmax(F::Type{LowFloat{E, M, FN}}) where {E, M, FN}
    m = (FN && M >= 2) ? 2.0f0 - 2.0f0^(1 - M) : 2.0f0 - 2.0f0^(-M)
    return m * 2.0f0^_emax(F)
end

@inline function _round(F::Type{LowFloat{E, M, FN}}, x::Float32) where {E, M, FN}
    (isfinite(x) & !iszero(x)) || return x
    e = max(exponent(x), _emin(F))             # subnormals share emin
    q = ldexp(1.0f0, e - M)                    # spacing of representable values
    r = round(x / q) * q                       # ties to even
    fm = _fmax(F)
    abs(r) > fm && return FN ? copysign(fm, x) : copysign(Inf32, x)
    return r
end

(::Type{F})(x::Float32) where {F <: LowFloat} = _rawlf(F, _round(F, x))
(::Type{F})(x::Real) where {F <: LowFloat} = F(Float32(x))
(::Type{F})(x::F) where {F <: LowFloat} = x
Base.Float32(a::LowFloat) = a.x
Base.Float64(a::LowFloat) = Float64(a.x)
Base.float(a::LowFloat) = a
Base.convert(::Type{F}, x::Real) where {F <: LowFloat} = F(x)
Base.convert(::Type{F}, x::F) where {F <: LowFloat} = x
Base.promote_rule(::Type{F}, ::Type{<:Integer}) where {F <: LowFloat} = F
Base.promote_rule(::Type{F}, ::Type{Float32}) where {F <: LowFloat} = Float32
Base.promote_rule(::Type{F}, ::Type{Float64}) where {F <: LowFloat} = Float64

for op in (:+, :-, :*, :/)
    @eval @inline Base.$op(a::F, b::F) where {F <: LowFloat} = F($op(a.x, b.x))
end
@inline Base.:-(a::F) where {F <: LowFloat} = _rawlf(F, -a.x)
@inline Base.:^(a::F, n::Integer) where {F <: LowFloat} = F(a.x^n)
@inline Base.inv(a::F) where {F <: LowFloat} = F(inv(a.x))
for f in (:exp, :log, :sin, :cos, :sqrt, :cbrt, :atan, :tan, :expm1, :log1p)
    @eval @inline Base.$f(a::F) where {F <: LowFloat} = F($f(a.x))
end
@inline Base.atan(a::F, b::F) where {F <: LowFloat} = F(atan(a.x, b.x))
@inline Base.sincos(a::F) where {F <: LowFloat} = ((s, c) = sincos(a.x); (F(s), F(c)))
@inline Base.abs(a::F) where {F <: LowFloat} = _rawlf(F, abs(a.x))
@inline Base.copysign(a::F, b::F) where {F <: LowFloat} = _rawlf(F, copysign(a.x, b.x))
@inline Base.flipsign(a::F, b::F) where {F <: LowFloat} = _rawlf(F, flipsign(a.x, b.x))
@inline Base.signbit(a::LowFloat) = signbit(a.x)
@inline Base.sign(a::F) where {F <: LowFloat} = _rawlf(F, sign(a.x))
for f in (:isnan, :isinf, :isfinite, :iszero, :isone, :isinteger)
    @eval @inline Base.$f(a::LowFloat) = $f(a.x)
end
for op in (:<, :<=, :(==), :isless, :isequal)
    @eval @inline Base.$op(a::F, b::F) where {F <: LowFloat} = $op(a.x, b.x)
end
Base.zero(::Type{F}) where {F <: LowFloat} = _rawlf(F, 0.0f0)
Base.one(::Type{F}) where {F <: LowFloat} = _rawlf(F, 1.0f0)
Base.eps(::Type{LowFloat{E, M, FN}}) where {E, M, FN} = _rawlf(LowFloat{E, M, FN}, 2.0f0^(-M))
Base.floatmax(F::Type{<:LowFloat}) = _rawlf(F, _fmax(F))
Base.floatmin(F::Type{<:LowFloat}) = _rawlf(F, 2.0f0^_emin(F))
Base.typemax(F::Type{LowFloat{E, M, FN}}) where {E, M, FN} = FN ? floatmax(F) : _rawlf(F, Inf32)
Base.typemin(F::Type{<:LowFloat}) = -typemax(F)
Base.hash(a::LowFloat, h::UInt) = hash(a.x, h)
Base.show(io::IO, a::LowFloat{E, M, FN}) where {E, M, FN} = print(io, "LowFloat{$E,$M}(", a.x, ")")

"largest finite value and smallest normal value of a float type (for scaling)"
format_range(::Type{F}) where {F <: AbstractFloat} = (Float64(floatmax(F)), Float64(floatmin(F)))
