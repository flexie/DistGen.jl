# Repartition primitive: all-to-all redistribution between different partition topologies.
#
# Forward: redistribute data from source partition (P_x) to destination partition (P_y).
# Adjoint: reverse repartition (P_y → P_x) with same overlap structure.
#
# Reference: distdl/src/distdl/backends/mpi/functional/repartition.py
#            ParametricOperators.jl/src/ParRepartition.jl

"""
    OverlapEntry

A single send/receive overlap between two workers.

# Fields
- `partner_rank`: Rank in the union communicator to send to / receive from
- `global_range`: Overlap region in global index space
- `local_range`: Corresponding range in local tensor coordinates
- `shape`: Size of the overlap region
"""
struct OverlapEntry{N}
    partner_rank::Int
    global_range::NTuple{N, UnitRange{Int}}
    local_range::NTuple{N, UnitRange{Int}}
    shape::NTuple{N, Int}
end

"""
    RepartitionInfo{N}

Pre-computed metadata for repartition between two Cartesian partitions.

# Fields
- `src_partition`: Source partition
- `dst_partition`: Destination partition
- `union_comm`: Communicator spanning union of src and dst
- `global_shape`: Global tensor shape
- `sends`: Overlap entries for data this rank must send (in dst coords)
- `recvs`: Overlap entries for data this rank must receive (in src coords)
- `src_active`: Whether this rank is in source partition
- `dst_active`: Whether this rank is in destination partition
"""
struct RepartitionInfo{N}
    src_partition::CartesianPartition{N}
    dst_partition::CartesianPartition{N}
    union_comm::MPI.Comm
    global_shape::NTuple{N, Int}
    sends::Vector{OverlapEntry{N}}
    recvs::Vector{OverlapEntry{N}}
    src_active::Bool
    dst_active::Bool
end

"""
    setup_repartition(
        src::CartesianPartition{N}, dst::CartesianPartition{N},
        global_shape::NTuple{N, Int}, parent_comm::MPI.Comm
    ) -> RepartitionInfo{N}

Pre-compute all overlap regions between source and destination partitions.
Each worker identifies which destination workers need parts of its local data (sends)
and which source workers provide parts of its local data (recvs).
"""
function setup_repartition(
    src::CartesianPartition{N}, dst::CartesianPartition{N},
    global_shape::NTuple{N, Int}, parent_comm::MPI.Comm
) where {N}
    union_comm = create_partition_union(src, dst, parent_comm)

    sends = OverlapEntry{N}[]
    recvs = OverlapEntry{N}[]

    if union_comm == MPI.COMM_NULL
        return RepartitionInfo{N}(src, dst, union_comm, global_shape, sends, recvs, false, false)
    end

    # Gather partition info: each rank broadcasts its identity to the union
    union_rank = MPI.Comm_rank(union_comm)
    union_size = MPI.Comm_size(union_comm)

    # All-gather source coords and dst coords in the union
    # Pack: [is_src, src_coords..., is_dst, dst_coords...]
    local_info = zeros(Int, 2 + 2 * N)
    if src.active
        local_info[1] = 1
        for d in 1:N
            local_info[1 + d] = src.coords[d]
        end
    end
    if dst.active
        local_info[N + 2] = 1
        for d in 1:N
            local_info[N + 2 + d] = dst.coords[d]
        end
    end

    all_info = MPI.Allgather(local_info, union_comm)
    info_matrix = reshape(all_info, 2 + 2 * N, union_size)

    # Build sends: if this rank is in src, find all dst workers whose subtensor
    # overlaps with our local subtensor
    if src.active
        my_src_ranges = subtensor_indices(global_shape, src.dims, src.coords)

        for r in 0:(union_size - 1)
            if info_matrix[N + 2, r + 1] == 1  # r is in dst
                dst_coords_r = ntuple(d -> info_matrix[N + 2 + d, r + 1], N)
                overlap = compute_overlaps(
                    global_shape,
                    src.dims, src.coords,
                    dst.dims, dst_coords_r
                )
                if overlap !== nothing
                    local_range = ntuple(N) do d
                        global_to_local(overlap[d], first(my_src_ranges[d]))
                    end
                    shape = ntuple(d -> length(overlap[d]), N)
                    push!(sends, OverlapEntry{N}(r, overlap, local_range, shape))
                end
            end
        end
    end

    # Build recvs: if this rank is in dst, find all src workers whose subtensor
    # overlaps with our local subtensor
    if dst.active
        my_dst_ranges = subtensor_indices(global_shape, dst.dims, dst.coords)

        for r in 0:(union_size - 1)
            if info_matrix[1, r + 1] == 1  # r is in src
                src_coords_r = ntuple(d -> info_matrix[1 + d, r + 1], N)
                overlap = compute_overlaps(
                    global_shape,
                    src.dims, src_coords_r,
                    dst.dims, dst.coords
                )
                if overlap !== nothing
                    local_range = ntuple(N) do d
                        global_to_local(overlap[d], first(my_dst_ranges[d]))
                    end
                    shape = ntuple(d -> length(overlap[d]), N)
                    push!(recvs, OverlapEntry{N}(r, overlap, local_range, shape))
                end
            end
        end
    end

    return RepartitionInfo{N}(src, dst, union_comm, global_shape, sends, recvs, src.active, dst.active)
end

"""
    repartition_op(x::AbstractArray{T}, info::RepartitionInfo{N}) -> AbstractArray{T}

Redistribute data from source partition to destination partition via point-to-point
all-to-all communication.

Workers in the source partition send their local data to all overlapping destination workers.
Workers in the destination partition receive and assemble their local data from overlapping source workers.
"""
function repartition_op(x::AbstractArray{T}, info::RepartitionInfo{N}) where {T, N}
    if info.union_comm == MPI.COMM_NULL
        return x
    end

    union_rank = MPI.Comm_rank(info.union_comm)

    # Allocate output tensor for dst workers
    if info.dst_active
        out_shape = local_shape(info.global_shape, info.dst_partition.dims, info.dst_partition.coords)
        y = zeros(T, out_shape...)
    else
        y = zeros(T, ntuple(_ -> 0, N)...)
    end

    reqs = MPI.Request[]

    # Post all receives first (non-blocking)
    recv_bufs = Vector{Array{T}}(undef, length(info.recvs))
    for (i, entry) in enumerate(info.recvs)
        recv_bufs[i] = Array{T}(undef, entry.shape...)
        if entry.partner_rank == union_rank
            # Self-copy: will handle separately
            continue
        end
        push!(reqs, MPI.Irecv!(recv_bufs[i], info.union_comm; source=entry.partner_rank, tag=_overlap_tag(entry)))
    end

    # Post all sends (non-blocking)
    send_bufs = Vector{Array{T}}(undef, length(info.sends))
    for (i, entry) in enumerate(info.sends)
        send_bufs[i] = collect(view(x, entry.local_range...))
        if entry.partner_rank == union_rank
            # Self-copy: find matching recv and copy directly
            for (j, rentry) in enumerate(info.recvs)
                if rentry.partner_rank == union_rank && rentry.global_range == entry.global_range
                    copyto!(recv_bufs[j], send_bufs[i])
                    break
                end
            end
            continue
        end
        push!(reqs, MPI.Isend(send_bufs[i], info.union_comm; dest=entry.partner_rank, tag=_overlap_tag(entry)))
    end

    # Wait for all non-self communications
    MPI.Waitall(reqs)

    # Unpack received data into output tensor
    if info.dst_active
        for (i, entry) in enumerate(info.recvs)
            view(y, entry.local_range...) .= recv_bufs[i]
        end
    end

    return y
end

"""
Deterministic tag from a global overlap range. Both sender and receiver compute the
same tag for the same overlap region, ensuring MPI messages match correctly.
"""
function _overlap_tag(entry::OverlapEntry{N}) where {N}
    h = 0
    for d in 1:N
        h = h * 10000 + first(entry.global_range[d]) * 100 + last(entry.global_range[d])
    end
    return mod(abs(h), 32000) + 1  # MPI tags must be positive, < MPI_TAG_UB
end

# ─── ChainRules rrule ────────────────────────────────────────────────────────

"""
    rrule(::typeof(repartition_op), x, info)

Adjoint of repartition is the reverse repartition: swap sends ↔ recvs,
swap src ↔ dst partitions.
"""
function ChainRulesCore.rrule(::typeof(repartition_op), x::AbstractArray{T}, info::RepartitionInfo{N}) where {T, N}
    y = repartition_op(x, info)

    function repartition_pullback(ȳ)
        # Reverse repartition: dst → src
        reverse_info = RepartitionInfo{N}(
            info.dst_partition,
            info.src_partition,
            info.union_comm,
            info.global_shape,
            info.recvs,   # old recvs become sends
            info.sends,   # old sends become recvs
            info.dst_active,
            info.src_active,
        )
        x̄ = repartition_op(ȳ, reverse_info)
        return NoTangent(), x̄, NoTangent()
    end

    return y, repartition_pullback
end
