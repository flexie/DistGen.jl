# CartesianPartition: wraps MPI Cartesian communicators for domain decomposition.
#
# Mirrors distdl's MPICartesianPartition but leverages Julia's MPI.jl.
# Encodes neighbor relationships needed for halo exchange.

"""
    CartesianPartition

A partition of workers arranged on a Cartesian grid over an MPI communicator.

# Fields
- `comm`: MPI Cartesian communicator
- `comm_cart`: The underlying Cartesian communicator (may equal `comm`)
- `dims`: Number of workers along each dimension `(P_1, P_2, ..., P_N)`
- `coords`: 0-based Cartesian coordinates of this rank `(c_1, c_2, ..., c_N)`
- `rank`: MPI rank in the Cartesian communicator
- `size`: Total number of ranks
- `active`: Whether this rank participates in computation
"""
struct CartesianPartition{N}
    comm::MPI.Comm
    dims::NTuple{N, Int}
    coords::NTuple{N, Int}
    rank::Int
    size::Int
    active::Bool
end

"""
    ndims(P::CartesianPartition) -> Int

Number of dimensions in the partition grid.
"""
Base.ndims(::CartesianPartition{N}) where {N} = N

"""
    size(P::CartesianPartition) -> NTuple{N, Int}

Shape of the partition grid (workers per dimension).
"""
Base.size(P::CartesianPartition) = P.dims

"""
    length(P::CartesianPartition) -> Int

Total number of workers in the partition.
"""
Base.length(P::CartesianPartition) = P.size

"""
    create_cartesian_topology(comm::MPI.Comm, dims::NTuple{N, Int}; periodic=ntuple(_->false, N)) -> CartesianPartition

Create a Cartesian partition from an existing communicator.
`dims` specifies workers per dimension. Ranks beyond `prod(dims)` are inactive.

# Example
```julia
MPI.Init()
P = create_cartesian_topology(MPI.COMM_WORLD, (2, 2, 2))
```
"""
function create_cartesian_topology(
    comm::MPI.Comm,
    dims::NTuple{N, Int};
    periodic::NTuple{N, Bool} = ntuple(_ -> false, N)
) where {N}
    total = prod(dims)

    # MPI_Cart_create is collective: ALL ranks in comm must call it.
    # Ranks beyond prod(dims) receive MPI_COMM_NULL.
    comm_cart = MPI.Cart_create(comm, Cint.(collect(dims)); periodic=collect(periodic), reorder=true)

    active = comm_cart != MPI.COMM_NULL

    if active
        cart_rank = MPI.Comm_rank(comm_cart)
        coords_vec = MPI.Cart_coords(comm_cart)
        coords = ntuple(i -> Int(coords_vec[i]), N)
    else
        cart_rank = -1
        coords = ntuple(_ -> -1, N)
    end

    return CartesianPartition{N}(
        active ? comm_cart : MPI.COMM_NULL,
        dims,
        coords,
        active ? cart_rank : -1,
        total,
        active,
    )
end

"""
    neighbor_ranks(P::CartesianPartition) -> Vector{Tuple{Int, Int}}

For each dimension, return `(left_rank, right_rank)` neighbor pair.
Returns `MPI.PROC_NULL` for non-existent neighbors (boundaries in non-periodic grids).
"""
function neighbor_ranks(P::CartesianPartition{N}) where {N}
    !P.active && return Tuple{Int, Int}[]
    neighbors = Vector{Tuple{Int, Int}}(undef, N)
    for d in 1:N
        # MPI.Cart_shift uses 0-based dimension index
        left, right = MPI.Cart_shift(P.comm, d - 1, 1)
        neighbors[d] = (Int(left), Int(right))
    end
    return neighbors
end

"""
    create_subpartition(P::CartesianPartition, keep_dims::NTuple{M, Int}) -> CartesianPartition

Create a lower-dimensional sub-partition by keeping only the specified dimensions.
`keep_dims` are 1-based dimension indices.
"""
function create_subpartition(P::CartesianPartition{N}, keep_dims::NTuple{M, Int}) where {N, M}
    !P.active && return CartesianPartition{M}(
        MPI.COMM_NULL,
        ntuple(i -> P.dims[keep_dims[i]], M),
        ntuple(_ -> -1, M),
        -1, prod(P.dims[d] for d in keep_dims), false
    )
    remain = zeros(Cint, N)
    for d in keep_dims
        remain[d] = Cint(1)
    end
    sub_comm = MPI.Cart_sub(P.comm, remain)
    sub_rank = MPI.Comm_rank(sub_comm)
    sub_coords_vec = MPI.Cart_coords(sub_comm)
    sub_dims = ntuple(i -> P.dims[keep_dims[i]], M)
    sub_coords = ntuple(i -> Int(sub_coords_vec[i]), M)

    return CartesianPartition{M}(
        sub_comm,
        sub_dims,
        sub_coords,
        sub_rank,
        prod(sub_dims),
        true,
    )
end

"""
    create_partition_union(P1::CartesianPartition, P2::CartesianPartition, parent_comm::MPI.Comm) -> MPI.Comm

Create a communicator that spans the union of two partitions.
`parent_comm` must contain all ranks in both partitions.
Returns `MPI.COMM_NULL` for ranks not in either partition.
"""
function create_partition_union(P1::CartesianPartition, P2::CartesianPartition, parent_comm::MPI.Comm)
    # Determine which ranks are in P1, P2, or both
    in_p1 = P1.active ? 1 : 0
    in_p2 = P2.active ? 1 : 0
    in_union = (in_p1 | in_p2) > 0

    # Use MPI_Comm_split: color=0 for union members, MPI_UNDEFINED for others
    color = in_union ? Cint(0) : MPI.API.MPI_UNDEFINED[]
    key = MPI.Comm_rank(parent_comm)

    union_comm = MPI.Comm_split(parent_comm, color, key)
    return union_comm
end

"""
    coords_to_rank(P::CartesianPartition, coords::NTuple{N, Int}) -> Int

Convert 0-based Cartesian coordinates to a rank in the partition communicator.
"""
function coords_to_rank(P::CartesianPartition{N}, coords::NTuple{N, Int}) where {N}
    !P.active && return -1
    return MPI.Cart_rank(P.comm, collect(Cint, coords))
end
