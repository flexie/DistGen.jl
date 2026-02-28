# Sum-reduce primitive: sum N partitions to one.
#
# Forward: reduce (sum) tensors from all workers in source partition to root.
# Adjoint: broadcast gradient from root to all source workers (from arXiv-2006.03108v1).

"""
    SumReduceInfo

Pre-computed metadata for a sum-reduce operation.

# Fields
- `src_partition`: Source partition (all workers contribute)
- `dst_partition`: Destination partition (root receives sum)
- `union_comm`: Communicator spanning both partitions
- `root_rank`: Rank in union_comm that receives the sum
- `src_active`: Whether this rank is in the source partition
- `dst_active`: Whether this rank is in the destination partition
"""
struct SumReduceInfo
    src_partition::CartesianPartition
    dst_partition::CartesianPartition
    union_comm::MPI.Comm
    root_rank::Int
    src_active::Bool
    dst_active::Bool
end

"""
    setup_sum_reduce(src::CartesianPartition, dst::CartesianPartition, parent_comm::MPI.Comm) -> SumReduceInfo

Set up sum-reduce from `src` partition to `dst` partition (root).
"""
function setup_sum_reduce(src::CartesianPartition, dst::CartesianPartition, parent_comm::MPI.Comm)
    union_comm = create_partition_union(src, dst, parent_comm)
    root_rank = 0  # Root of destination partition in union_comm
    return SumReduceInfo(src, dst, union_comm, root_rank, src.active, dst.active)
end

"""
    sum_reduce_op(x::AbstractArray, info::SumReduceInfo) -> AbstractArray

Sum-reduce `x` from all source partition workers to the destination root.
Non-root workers receive a zero-filled result.
"""
function sum_reduce_op(x::AbstractArray{T}, info::SumReduceInfo) where {T}
    if info.union_comm == MPI.COMM_NULL
        return x
    end

    y = similar(x)
    MPI.Reduce!(x, y, MPI.SUM, info.root_rank, info.union_comm)

    if !info.dst_active
        fill!(y, zero(T))
    end

    return y
end

# ─── ChainRules rrule ────────────────────────────────────────────────────────

"""
    rrule(::typeof(sum_reduce_op), x, info)

Adjoint of sum-reduce is broadcast: gradient at root is broadcast to all source workers.
"""
function ChainRulesCore.rrule(::typeof(sum_reduce_op), x::AbstractArray{T}, info::SumReduceInfo) where {T}
    y = sum_reduce_op(x, info)

    function sum_reduce_pullback(ȳ)
        ȳ_val = ChainRulesCore.unthunk(ȳ)
        if info.union_comm == MPI.COMM_NULL
            return NoTangent(), ȳ_val, NoTangent()
        end

        # Broadcast gradient from root to all source workers
        x̄ = copy(ȳ_val)
        MPI.Bcast!(x̄, info.root_rank, info.union_comm)

        return NoTangent(), x̄, NoTangent()
    end

    return y, sum_reduce_pullback
end
