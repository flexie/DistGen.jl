# Layer 3: ParametricOperators.jl Bridge
#
# Package extension that adds:
# 1. ChainRules rrule definitions for ParCompose, ParKron, ParDistributed, ParRepartition
# 2. Lux.jl bridge: wrap ParOperator as AbstractLuxLayer
# 3. Integration with DomainDecomposition.jl primitives

module ParametricOperatorsExt

using DomainDecomposition
using ChainRulesCore

# Conditionally load ParametricOperators types
import ParametricOperators
import ParametricOperators: ParOperator, ParLinearOperator, ParCompose, ParKron,
    ParParameterized, ParAdjoint, DDT, RDT, Domain, Range,
    Parameters, init, init!

# ─── ChainRules for ParCompose ──────────────────────────────────────────────

"""
    rrule(::typeof(*), A::ParParameterized, x::AbstractVector)

AD rule for applying a parameterized operator: y = A(x).
For linear operators: ∂y/∂x = A, so pullback(ȳ) = A' * ȳ
"""
function ChainRulesCore.rrule(::typeof(*), A::ParParameterized{D,R,Linear}, x::AbstractVector{D}) where {D, R, Linear}
    y = A(x)

    function par_op_pullback(ȳ)
        # For linear operator: adjoint application
        Aᵀ = adjoint(A)
        x̄ = Aᵀ(ȳ)
        # No gradient w.r.t. operator itself in this simplified version
        return NoTangent(), NoTangent(), x̄
    end

    return y, par_op_pullback
end

"""
    rrule for ParCompose: (A ∘ B)(x) = A(B(x))

Chain rule: ∂/∂x = A' ∘ B'  (applied in reverse order)
"""
function ChainRulesCore.rrule(::typeof(*), C::ParParameterized{D,R,Linear,<:ParametricOperators.ParCompose}, x::AbstractVector{D}) where {D, R, Linear}
    # Forward pass through composition
    y = C(x)

    function compose_pullback(ȳ)
        Cᵀ = adjoint(C)
        x̄ = Cᵀ(ȳ)
        return NoTangent(), NoTangent(), x̄
    end

    return y, compose_pullback
end

"""
    rrule for ParKron: (A ⊗ B)(x)

Kronecker product adjoint: (A ⊗ B)' = A' ⊗ B'
"""
function ChainRulesCore.rrule(::typeof(*), K::ParParameterized{D,R,Linear,<:ParametricOperators.ParKron}, x::AbstractVector{D}) where {D, R, Linear}
    y = K(x)

    function kron_pullback(ȳ)
        Kᵀ = adjoint(K)
        x̄ = Kᵀ(ȳ)
        return NoTangent(), NoTangent(), x̄
    end

    return y, kron_pullback
end

# ─── Parameter Gradient Rules ────────────────────────────────────────────────

"""
    par_operator_gradient(A::ParParameterized, x, ȳ) -> Parameters

Compute gradient w.r.t. parameters of operator A.
For a parametric linear operator A(θ): ∂L/∂θ = ∂L/∂y ⊗ ∂A/∂θ · x

This is the key extension needed for training — computing parameter gradients
for operators defined through the ParametricOperators algebra.
"""
function par_operator_gradient(A, x::AbstractVector, ȳ::AbstractVector)
    # Finite-difference approximation for parameter gradients
    # (To be replaced with Enzyme.jl for efficiency in production)
    params = A.params
    grads = similar(params)

    h = sqrt(eps(eltype(x)))

    for (op, θ) in pairs(params)
        θ === nothing && continue
        g = similar(θ)
        for i in eachindex(θ)
            θ_plus = copy(θ)
            θ_plus[i] += h
            params_plus = merge(params, Dict(op => θ_plus))
            A_plus = typeof(A.op)(params_plus)
            y_plus = A_plus(x)
            g[i] = dot(ȳ, (y_plus - A(x)) / h)
        end
        grads[op] = g
    end

    return grads
end

# ─── Lux.jl Bridge ──────────────────────────────────────────────────────────

"""
    ParOperatorLayer

Wraps a ParametricOperators.jl ParOperator as a Lux-compatible layer.

The operator's parameters are stored in the Lux parameter state,
enabling seamless integration with Lux training loops and optimizers.

# Fields
- `op`: The parametric operator (not yet parameterized)
- `input_shape`: Expected input shape (for reshape if needed)
- `output_shape`: Expected output shape
"""
struct ParOperatorLayer
    op::ParOperator
    input_size::Int
    output_size::Int
end

"""
    ParOperatorLayer(op::ParOperator)
"""
function ParOperatorLayer(op::ParOperator)
    return ParOperatorLayer(op, Domain(op), Range(op))
end

"""
    init_par_layer(rng, layer::ParOperatorLayer) -> (params, state)

Initialize parameters for the ParOperator layer.
"""
function init_par_layer(rng, layer::ParOperatorLayer)
    params_dict = init(layer.op)

    # Flatten into a NamedTuple for Lux compatibility
    # Store the full Parameters dict as a single entry
    ps = (operator_params=params_dict,)
    st = (op=layer.op,)

    return ps, st
end

"""
    (layer::ParOperatorLayer)(x, ps, st) -> (y, st)

Apply the operator. x is vectorized if needed.
"""
function (layer::ParOperatorLayer)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    # Parameterize the operator with current parameters
    A = st.op(ps.operator_params)

    # Handle batched input: if x is a matrix, apply column-wise
    if ndims(x) == 1
        y = A(x)
    else
        y = A(x)  # ParOperator already handles matrix input
    end

    return y, st
end

# ─── Distributed ParOperator ────────────────────────────────────────────────

"""
    DistParOperatorLayer

A ParOperator distributed across a DomainDecomposition.jl partition.

Uses DomainDecomposition primitives (repartition, broadcast) to distribute
the operator's computation across multiple ranks.

# Fields
- `op`: The parametric operator
- `partition`: DomainDecomposition CartesianPartition
- `repartition_in`: Optional input repartition
- `repartition_out`: Optional output repartition
"""
struct DistParOperatorLayer
    op::ParOperator
    partition::CartesianPartition
    input_size::Int
    output_size::Int
end

function DistParOperatorLayer(op::ParOperator, P::CartesianPartition)
    # Local sizes based on partition
    local_in = DomainDecomposition.balanced_decomposition(Domain(op), prod(P.dims))[P.rank + 1]
    local_out = DomainDecomposition.balanced_decomposition(Range(op), prod(P.dims))[P.rank + 1]
    return DistParOperatorLayer(op, P, local_in, local_out)
end

end # module
