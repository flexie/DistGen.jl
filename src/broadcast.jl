# Broadcast primitive: replicate data from one partition to many.
#
# Forward: root broadcasts tensor to all workers in target partition.
# Adjoint: sum-reduce gradients back to root (from arXiv-2006.03108v1).

"""
    BroadcastInfo

Pre-computed metadata for a broadcast operation between two partitions.

# Fields
- `src_partition`: Source partition (1 worker holds data)
- `dst_partition`: Destination partition (receives copies)
- `union_comm`: Communicator spanning both partitions
- `root_rank`: Rank in union_comm that holds the source data
- `src_active`: Whether this rank is in the source partition
- `dst_active`: Whether this rank is in the destination partition
"""
struct BroadcastInfo
    src_partition::CartesianPartition
    dst_partition::CartesianPartition
    union_comm::MPI.Comm
    root_rank::Int
    src_active::Bool
    dst_active::Bool
end

"""
    setup_broadcast(src::CartesianPartition, dst::CartesianPartition, parent_comm::MPI.Comm) -> BroadcastInfo

Set up broadcast from `src` partition (single worker or root) to `dst` partition.
"""
function setup_broadcast(src::CartesianPartition, dst::CartesianPartition, parent_comm::MPI.Comm)
    union_comm = create_partition_union(src, dst, parent_comm)
    # Root is rank 0 in the source partition, translated to union_comm rank
    # For simplicity, the source partition root is the rank with coords all-zero
    root_rank = 0  # Will be determined via group translation in practice

    return BroadcastInfo(src, dst, union_comm, root_rank, src.active, dst.active)
end

"""
    broadcast_op(x::AbstractArray, info::BroadcastInfo) -> AbstractArray

Broadcast `x` from source partition root to all workers in the destination partition.
Workers not in the source partition provide a zero-filled tensor of matching shape.
"""
function broadcast_op(x::AbstractArray{T}, info::BroadcastInfo) where {T}
    if info.union_comm == MPI.COMM_NULL
        return x  # Not participating
    end

    # Broadcast the shape first so all ranks know the tensor size
    shape_buf = info.src_active ? collect(Int, size(x)) : zeros(Int, ndims(x))
    ndim_buf = Int[info.src_active ? ndims(x) : 0]
    MPI.Bcast!(ndim_buf, info.root_rank, info.union_comm)
    nd = ndim_buf[1]

    if !info.src_active
        shape_buf = zeros(Int, nd)
    end
    MPI.Bcast!(shape_buf, info.root_rank, info.union_comm)
    shape = Tuple(shape_buf)

    # Broadcast the data
    if info.src_active
        y = copy(x)
    else
        y = zeros(T, shape...)
    end
    MPI.Bcast!(y, info.root_rank, info.union_comm)

    return y
end

# ─── ChainRules rrule ────────────────────────────────────────────────────────

"""
    rrule(::typeof(broadcast_op), x, info)

Adjoint of broadcast is sum-reduce: gradients from all destination workers
are summed and returned to the source worker.
"""
function ChainRulesCore.rrule(::typeof(broadcast_op), x::AbstractArray{T}, info::BroadcastInfo) where {T}
    y = broadcast_op(x, info)

    function broadcast_pullback(ȳ)
        ȳ_val = ChainRulesCore.unthunk(ȳ)
        if info.union_comm == MPI.COMM_NULL
            return NoTangent(), ȳ_val, NoTangent()
        end

        # Sum-reduce gradients back to root
        x̄ = similar(ȳ_val)
        MPI.Reduce!(ȳ_val, x̄, MPI.SUM, info.root_rank, info.union_comm)

        if !info.src_active
            x̄ = ZeroTangent()
        end

        return NoTangent(), x̄, NoTangent()
    end

    return y, broadcast_pullback
end
