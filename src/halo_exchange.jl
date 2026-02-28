# Halo exchange primitive for domain-decomposed computation.
#
# Exchanges ghost (halo) regions between neighboring workers on a Cartesian partition.
# Follows distdl's halo_exchange.py: processes dimensions sequentially, uses non-blocking
# MPI with tag-based routing.
#
# Adjoint: reverse exchange direction + accumulate into bulk (from arXiv-2006.03108v1).

"""
    HaloInfo{N}

Pre-computed halo exchange metadata for an N-dimensional domain.

# Fields
- `partition`: The Cartesian partition
- `neighbors`: Neighbor ranks per dimension `[(left, right), ...]`
- `halo_sizes`: Halo widths per dimension `[(left, right), ...]`
- `send_slices`: Bulk slices to pack for sending, per dimension `[(left_bulk, right_bulk), ...]`
- `recv_slices`: Ghost slices to unpack into, per dimension `[(left_ghost, right_ghost), ...]`
"""
struct HaloInfo{N}
    partition::CartesianPartition{N}
    neighbors::Vector{Tuple{Int, Int}}
    halo_sizes::Vector{Tuple{Int, Int}}
    # Computed at exchange time based on tensor shape + halo sizes
end

"""
    compute_halo_info(P::CartesianPartition{N}, halo_sizes::Vector{Tuple{Int,Int}}) -> HaloInfo{N}

Pre-compute halo exchange info. `halo_sizes[d] = (left_halo, right_halo)` for each
spatial dimension.
"""
function compute_halo_info(P::CartesianPartition{N}, halo_sizes::Vector{Tuple{Int, Int}}) where {N}
    neighbors = neighbor_ranks(P)
    return HaloInfo{N}(P, neighbors, halo_sizes)
end

"""
    _compute_slices(shape::NTuple{M, Int}, dim::Int, left_halo::Int, right_halo::Int)

Compute the send (bulk) and receive (ghost) index ranges for dimension `dim`.

Returns `(left_send, left_recv, right_send, right_recv)` as `UnitRange{Int}` tuples,
where each tuple has ranges for all dimensions.
"""
function _compute_slices(shape::NTuple{M, Int}, dim::Int, left_halo::Int, right_halo::Int) where {M}
    n = shape[dim]

    # Left ghost: indices 1:left_halo
    # Left bulk (to send right neighbor's left ghost): indices (left_halo+1):(2*left_halo)
    # Wait — following distdl's convention:
    # The tensor is: [left_ghost | bulk | right_ghost]
    # left_ghost = 1:left_halo
    # bulk = (left_halo+1):(n - right_halo)
    # right_ghost = (n - right_halo + 1):n
    #
    # Send regions (from bulk near boundaries):
    # left_send (send to left neighbor): (left_halo+1):(left_halo + left_halo)  → left neighbor's right ghost
    # right_send (send to right neighbor): (n - right_halo - right_halo + 1):(n - right_halo) → right neighbor's left ghost

    # Make full-dimensional slice (Colon for all dims except `dim`)
    function make_range(range_for_dim)
        return ntuple(M) do d
            d == dim ? range_for_dim : Colon()
        end
    end

    if left_halo > 0
        left_recv = make_range(1:left_halo)
        left_send = make_range((left_halo + 1):(2 * left_halo))
    else
        left_recv = nothing
        left_send = nothing
    end

    if right_halo > 0
        right_recv = make_range((n - right_halo + 1):n)
        right_send = make_range((n - 2 * right_halo + 1):(n - right_halo))
    else
        right_recv = nothing
        right_send = nothing
    end

    return (left_send, left_recv, right_send, right_recv)
end

"""
    halo_exchange!(x::AbstractArray, info::HaloInfo)

In-place halo exchange: fill ghost regions of `x` with data from neighbors.

Processes dimensions sequentially. For each dimension:
1. Pack bulk boundary data into send buffers
2. Post non-blocking sends and receives
3. Wait for completion and unpack into ghost regions

The tensor `x` is assumed to already include space for halos (ghost regions).
"""
function halo_exchange!(x::AbstractArray{T, M}, info::HaloInfo{N}) where {T, M, N}
    !info.partition.active && return x

    for d in 1:N
        left_halo, right_halo = info.halo_sizes[d]
        (left_halo == 0 && right_halo == 0) && continue

        left_rank, right_rank = info.neighbors[d]

        left_send, left_recv, right_send, right_recv = _compute_slices(size(x), d, left_halo, right_halo)

        reqs = MPI.Request[]

        # Post receives first (non-blocking)
        if left_halo > 0 && left_rank != MPI.PROC_NULL
            left_recv_buf = similar(x, size(view(x, left_recv...)))
            push!(reqs, MPI.Irecv!(left_recv_buf, info.partition.comm; source=left_rank, tag=1))
        else
            left_recv_buf = nothing
        end

        if right_halo > 0 && right_rank != MPI.PROC_NULL
            right_recv_buf = similar(x, size(view(x, right_recv...)))
            push!(reqs, MPI.Irecv!(right_recv_buf, info.partition.comm; source=right_rank, tag=0))
        else
            right_recv_buf = nothing
        end

        # Pack and send (non-blocking)
        if left_halo > 0 && left_rank != MPI.PROC_NULL
            left_send_buf = copyto!(similar(x, size(view(x, left_send...))...), view(x, left_send...))  # contiguous device-aware copy
            push!(reqs, MPI.Isend(left_send_buf, info.partition.comm; dest=left_rank, tag=0))
        end

        if right_halo > 0 && right_rank != MPI.PROC_NULL
            right_send_buf = copyto!(similar(x, size(view(x, right_send...))...), view(x, right_send...))  # contiguous device-aware copy
            push!(reqs, MPI.Isend(right_send_buf, info.partition.comm; dest=right_rank, tag=1))
        end

        # Wait for all communications to complete
        MPI.Waitall(reqs)

        # Unpack received data into ghost regions
        if left_recv_buf !== nothing
            view(x, left_recv...) .= left_recv_buf
        end
        if right_recv_buf !== nothing
            view(x, right_recv...) .= right_recv_buf
        end
    end

    return x
end

"""
    halo_exchange(x::AbstractArray, info::HaloInfo) -> AbstractArray

Out-of-place halo exchange (copies input, then exchanges in-place).
Suitable for use with AD (non-mutating).
"""
function halo_exchange(x::AbstractArray, info::HaloInfo)
    y = copy(x)
    halo_exchange!(y, info)
    return y
end

# ─── ChainRules rrule ────────────────────────────────────────────────────────

"""
    rrule(::typeof(halo_exchange), x, info)

Adjoint of halo exchange:
- Forward: exchange ghost regions (bulk → neighbor's ghost)
- Backward: reverse exchange (ghost gradients → neighbor's bulk) + accumulate

From arXiv-2006.03108v1: the adjoint of a halo exchange is the reverse exchange
with accumulation into the bulk gradient.
"""
function ChainRulesCore.rrule(::typeof(halo_exchange), x::AbstractArray{T, M}, info::HaloInfo{N}) where {T, M, N}
    y = halo_exchange(x, info)

    function halo_exchange_pullback(ȳ)
        x̄ = copy(ChainRulesCore.unthunk(ȳ))

        if info.partition.active
            # Adjoint: send ghost gradients to neighbors, accumulate into bulk
            for d in N:-1:1  # Reverse dimension order (following distdl convention)
                left_halo, right_halo = info.halo_sizes[d]
                (left_halo == 0 && right_halo == 0) && continue

                left_rank, right_rank = info.neighbors[d]

                left_send, left_recv, right_send, right_recv = _compute_slices(size(x̄), d, left_halo, right_halo)

                reqs = MPI.Request[]

                # In adjoint: ghosts become sends, bulks become receives
                # Receive into temporary buffers for bulk accumulation
                if left_halo > 0 && left_rank != MPI.PROC_NULL
                    left_bulk_recv = similar(x̄, size(view(x̄, left_send...)))
                    push!(reqs, MPI.Irecv!(left_bulk_recv, info.partition.comm; source=left_rank, tag=0))
                else
                    left_bulk_recv = nothing
                end

                if right_halo > 0 && right_rank != MPI.PROC_NULL
                    right_bulk_recv = similar(x̄, size(view(x̄, right_send...)))
                    push!(reqs, MPI.Irecv!(right_bulk_recv, info.partition.comm; source=right_rank, tag=1))
                else
                    right_bulk_recv = nothing
                end

                # Send ghost gradients to neighbors
                if left_halo > 0 && left_rank != MPI.PROC_NULL
                    left_ghost_send = copyto!(similar(x̄, size(view(x̄, left_recv...))...), view(x̄, left_recv...))
                    push!(reqs, MPI.Isend(left_ghost_send, info.partition.comm; dest=left_rank, tag=1))
                end

                if right_halo > 0 && right_rank != MPI.PROC_NULL
                    right_ghost_send = copyto!(similar(x̄, size(view(x̄, right_recv...))...), view(x̄, right_recv...))
                    push!(reqs, MPI.Isend(right_ghost_send, info.partition.comm; dest=right_rank, tag=0))
                end

                MPI.Waitall(reqs)

                # Accumulate received gradients into bulk (+=, not overwrite)
                if left_bulk_recv !== nothing
                    view(x̄, left_send...) .+= left_bulk_recv
                end
                if right_bulk_recv !== nothing
                    view(x̄, right_send...) .+= right_bulk_recv
                end

                # Zero out ghost gradients only where a neighbor overwrote them in
                # the forward pass. At boundaries (PROC_NULL), the forward was
                # identity for the ghost region, so the adjoint must also be identity.
                if left_recv !== nothing && left_rank != MPI.PROC_NULL
                    view(x̄, left_recv...) .= zero(T)
                end
                if right_recv !== nothing && right_rank != MPI.PROC_NULL
                    view(x̄, right_recv...) .= zero(T)
                end
            end
        end

        return NoTangent(), x̄, NoTangent()
    end

    return y, halo_exchange_pullback
end
