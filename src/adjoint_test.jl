# Adjoint test verification for all communication primitives.
#
# From arXiv-2006.03108v1, Section 5:
#   |<Fx, y> - <x, F*y>| / max(||Fx||·||y||, ||x||·||F*y||) < eps
#
# This is the dot-product test for verifying that the ChainRules rrules
# correctly implement the adjoint of each primitive.

"""
    adjoint_test(forward_fn, adjoint_fn, x, y; atol=1e-10, rtol=1e-8) -> (passed, relative_error)

Verify that `adjoint_fn` is the correct adjoint of `forward_fn` using the dot-product test.

Computes:
- `Fx = forward_fn(x)`
- `Fᵀy = adjoint_fn(y)`
- Tests: `|<Fx, y> - <x, Fᵀy>| / max(||Fx||·||y||, ||x||·||Fᵀy||) < atol`

# Arguments
- `forward_fn`: The forward operation `x → Fx`
- `adjoint_fn`: The adjoint operation `y → F*y`
- `x`: Random input in the domain
- `y`: Random input in the range
- `atol`: Absolute tolerance for the test
- `rtol`: Relative tolerance

# Returns
- `(passed::Bool, relative_error::Float64)`
"""
function adjoint_test(
    forward_fn::Function,
    adjoint_fn::Function,
    x::AbstractArray{T},
    y::AbstractArray{T};
    atol::Float64 = 1e-10,
    rtol::Float64 = 1e-8,
    comm::MPI.Comm = MPI.COMM_WORLD
) where {T <: AbstractFloat}
    # Forward application
    Fx = forward_fn(x)

    # Adjoint application
    Fᵀy = adjoint_fn(y)

    # Local dot products
    local_Fx_y = dot_product(Fx, y)
    local_x_Fᵀy = dot_product(x, Fᵀy)

    # Global dot products (sum across all ranks)
    global_Fx_y = MPI.Allreduce(local_Fx_y, MPI.SUM, comm)
    global_x_Fᵀy = MPI.Allreduce(local_x_Fᵀy, MPI.SUM, comm)

    # Norms for normalization
    local_Fx_norm = sum(abs2, Fx)
    local_y_norm = sum(abs2, y)
    local_x_norm = sum(abs2, x)
    local_Fᵀy_norm = sum(abs2, Fᵀy)

    global_Fx_norm = sqrt(MPI.Allreduce(local_Fx_norm, MPI.SUM, comm))
    global_y_norm = sqrt(MPI.Allreduce(local_y_norm, MPI.SUM, comm))
    global_x_norm = sqrt(MPI.Allreduce(local_x_norm, MPI.SUM, comm))
    global_Fᵀy_norm = sqrt(MPI.Allreduce(local_Fᵀy_norm, MPI.SUM, comm))

    # Relative error
    numerator = abs(global_Fx_y - global_x_Fᵀy)
    denominator = max(global_Fx_norm * global_y_norm, global_x_norm * global_Fᵀy_norm)

    if denominator < eps(T)
        # Both sides essentially zero
        relative_error = zero(T)
    else
        relative_error = numerator / denominator
    end

    passed = relative_error < max(atol, rtol * denominator)

    return (passed, Float64(relative_error))
end

"""
    dot_product(a::AbstractArray, b::AbstractArray) -> scalar

Local dot product (sum of element-wise products). Handles shape mismatches
by computing over the minimum shared elements.
"""
function dot_product(a::AbstractArray{T}, b::AbstractArray{T}) where {T}
    if size(a) == size(b)
        return sum(a .* b)
    else
        # Shape mismatch — one side may have ghost regions
        # Compute over overlapping region (both from start)
        min_shape = ntuple(d -> min(size(a, d), size(b, d)), max(ndims(a), ndims(b)))
        ranges = ntuple(d -> 1:min_shape[d], length(min_shape))
        a_view = view(a, ranges...)
        b_view = view(b, ranges...)
        return sum(a_view .* b_view)
    end
end

"""
    adjoint_test_halo_exchange(P::CartesianPartition, halo_sizes, local_dims; T=Float64) -> (passed, error)

Convenience function to test halo exchange adjoint.
"""
function adjoint_test_halo_exchange(
    P::CartesianPartition{N},
    halo_sizes::Vector{Tuple{Int, Int}},
    local_dims::NTuple{M, Int};
    T::Type = Float64
) where {N, M}
    !P.active && return (true, 0.0)

    info = compute_halo_info(P, halo_sizes)

    # Tensor with halo space included
    padded_dims = ntuple(M) do d
        if d <= N
            local_dims[d] + halo_sizes[d][1] + halo_sizes[d][2]
        else
            local_dims[d]
        end
    end

    x = randn(T, padded_dims...)
    y = randn(T, padded_dims...)

    forward_fn = z -> halo_exchange(z, info)

    # Adjoint via ChainRules
    adjoint_fn = function(z)
        _, pullback = ChainRulesCore.rrule(halo_exchange, x, info)
        _, x̄, _ = pullback(z)
        return x̄
    end

    return adjoint_test(forward_fn, adjoint_fn, x, y; comm=P.comm)
end
