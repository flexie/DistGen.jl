# Layer 0: Device abstraction via KernelAbstractions.jl
#
# Provides GPU-portable array allocation and copy operations.
# Falls back to CPU when no GPU backend is loaded.

"""
    device_array(backend, T, dims...)

Allocate an uninitialized array of type `T` on the given KernelAbstractions backend.
Falls back to a regular `Array` for `CPU()`.
"""
function device_array(::CPU, ::Type{T}, dims::Integer...) where {T}
    return Array{T}(undef, dims...)
end

function device_array(backend::Backend, ::Type{T}, dims::Integer...) where {Backend, T}
    return KernelAbstractions.zeros(backend, T, dims...)
end

"""
    get_backend(x::AbstractArray)

Return the KernelAbstractions backend for the given array.
"""
get_backend(x::AbstractArray) = KernelAbstractions.get_backend(x)
get_backend(::Array) = CPU()

"""
    device_copy!(dst, dst_range, src, src_range)

Copy a region from `src` to `dst`, working on any backend.
"""
function device_copy!(dst::AbstractArray, dst_range::Tuple, src::AbstractArray, src_range::Tuple)
    dst_view = view(dst, dst_range...)
    src_view = view(src, src_range...)
    copyto!(dst_view, src_view)
    return dst
end

# ─── Zygote-compatible, device-aware tensor creation ─────────────────────────
#
# These helpers create constant tensors (zeros, ones) on the same device as a
# reference array. They are marked @non_differentiable so Zygote treats them
# as constants — correct since no gradient should flow through the creation of
# a zero/one-filled pad or normalization parameter.

"""
    _zeros_like(x::AbstractArray, sz...) -> AbstractArray

Create a zero-filled array on the same device as `x` with shape `sz`.
Non-differentiable (gradient is always zero).
"""
_zeros_like(x::AbstractArray{T}, sz...) where {T} = fill!(similar(x, sz...), zero(T))

"""
    _ones_like(x::AbstractArray, sz...) -> AbstractArray

Create a one-filled array on the same device as `x` with shape `sz`.
Non-differentiable (gradient is always zero).
"""
_ones_like(x::AbstractArray{T}, sz...) where {T} = fill!(similar(x, sz...), one(T))

ChainRulesCore.@non_differentiable _zeros_like(::Any, ::Any...)
ChainRulesCore.@non_differentiable _ones_like(::Any, ::Any...)
