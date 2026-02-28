# All-reduce primitive: in-place sum across all workers in a partition.
#
# Forward: allreduce(SUM) across partition.
# Adjoint: self-adjoint — allreduce(SUM) on gradients (from arXiv-2006.03108v1).

"""
    AllReduceInfo

Pre-computed metadata for an all-reduce operation.

# Fields
- `partition`: The partition over which to all-reduce
- `op`: MPI reduction operation (default: SUM)
"""
struct AllReduceInfo
    partition::CartesianPartition
    op::MPI.Op
end

"""
    setup_all_reduce(P::CartesianPartition; op=MPI.SUM) -> AllReduceInfo
"""
function setup_all_reduce(P::CartesianPartition; op=MPI.SUM)
    return AllReduceInfo(P, op)
end

"""
    all_reduce_op(x::AbstractArray, info::AllReduceInfo) -> AbstractArray

All-reduce `x` across all workers in the partition. Each worker receives the sum.
"""
function all_reduce_op(x::AbstractArray{T}, info::AllReduceInfo) where {T}
    !info.partition.active && return x
    y = similar(x)
    MPI.Allreduce!(x, y, info.op, info.partition.comm)
    return y
end

"""
    all_reduce_op!(x::AbstractArray, info::AllReduceInfo) -> AbstractArray

In-place all-reduce variant.
"""
function all_reduce_op!(x::AbstractArray{T}, info::AllReduceInfo) where {T}
    !info.partition.active && return x
    MPI.Allreduce!(MPI.IN_PLACE, x, info.op, info.partition.comm)
    return x
end

# ─── ChainRules rrule ────────────────────────────────────────────────────────

"""
    rrule(::typeof(all_reduce_op), x, info)

All-reduce with SUM is self-adjoint: backward pass is the same allreduce(SUM).
"""
function ChainRulesCore.rrule(::typeof(all_reduce_op), x::AbstractArray{T}, info::AllReduceInfo) where {T}
    y = all_reduce_op(x, info)

    function all_reduce_pullback(ȳ)
        if !info.partition.active
            return NoTangent(), ȳ, NoTangent()
        end

        # Self-adjoint: apply same all-reduce to gradient
        x̄ = similar(ȳ)
        MPI.Allreduce!(ȳ, x̄, info.op, info.partition.comm)

        return NoTangent(), x̄, NoTangent()
    end

    return y, all_reduce_pullback
end
