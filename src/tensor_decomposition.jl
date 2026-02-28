# Tensor decomposition utilities for domain decomposition.
#
# Provides balanced decomposition of global tensors across Cartesian partitions,
# following distdl's tensor_decomposition.py patterns.

"""
    balanced_decomposition(global_size::Int, n_partitions::Int) -> Vector{Int}

Compute balanced local sizes for splitting `global_size` elements across `n_partitions`.
Extra elements are distributed to the first `global_size % n_partitions` ranks.

# Example
```julia
balanced_decomposition(10, 3)  # [4, 3, 3]
balanced_decomposition(12, 4)  # [3, 3, 3, 3]
```
"""
function balanced_decomposition(global_size::Int, n_partitions::Int)
    base = global_size ÷ n_partitions
    remainder = global_size % n_partitions
    return [base + (i <= remainder ? 1 : 0) for i in 1:n_partitions]
end

"""
    subtensor_indices(global_shape::NTuple{N,Int}, partition_shape::NTuple{N,Int}, coords::NTuple{N,Int}) -> NTuple{N, UnitRange{Int}}

Compute the global index ranges for the subtensor owned by the worker at `coords`
(0-based) in a partition of shape `partition_shape`.

# Example
```julia
subtensor_indices((100, 100), (2, 2), (0, 0))  # (1:50, 1:50)
subtensor_indices((100, 100), (2, 2), (1, 1))  # (51:100, 51:100)
```
"""
function subtensor_indices(
    global_shape::NTuple{N, Int},
    partition_shape::NTuple{N, Int},
    coords::NTuple{N, Int}  # 0-based
) where {N}
    ranges = ntuple(N) do d
        sizes = balanced_decomposition(global_shape[d], partition_shape[d])
        start = sum(sizes[1:coords[d]]; init=0) + 1  # 1-based Julia indexing
        stop = start + sizes[coords[d] + 1] - 1
        start:stop
    end
    return ranges
end

"""
    local_shape(global_shape::NTuple{N,Int}, partition_shape::NTuple{N,Int}, coords::NTuple{N,Int}) -> NTuple{N, Int}

Compute the local tensor shape for the worker at `coords`.
"""
function local_shape(
    global_shape::NTuple{N, Int},
    partition_shape::NTuple{N, Int},
    coords::NTuple{N, Int}
) where {N}
    indices = subtensor_indices(global_shape, partition_shape, coords)
    return ntuple(d -> length(indices[d]), N)
end

"""
    compute_overlaps(
        global_shape::NTuple{N,Int},
        src_partition::NTuple{N,Int}, src_coords::NTuple{N,Int},
        dst_partition::NTuple{N,Int}, dst_coords::NTuple{N,Int}
    ) -> Union{Nothing, NTuple{N, UnitRange{Int}}}

Compute the intersection (in global indices) between a source subtensor and a
destination subtensor. Returns `nothing` if there is no overlap.
"""
function compute_overlaps(
    global_shape::NTuple{N, Int},
    src_partition::NTuple{N, Int}, src_coords::NTuple{N, Int},
    dst_partition::NTuple{N, Int}, dst_coords::NTuple{N, Int}
) where {N}
    src_ranges = subtensor_indices(global_shape, src_partition, src_coords)
    dst_ranges = subtensor_indices(global_shape, dst_partition, dst_coords)

    overlap = ntuple(N) do d
        lo = max(first(src_ranges[d]), first(dst_ranges[d]))
        hi = min(last(src_ranges[d]), last(dst_ranges[d]))
        lo:hi
    end

    # Check if overlap is non-empty in all dimensions
    for d in 1:N
        if length(overlap[d]) <= 0
            return nothing
        end
    end

    return overlap
end

"""
    global_to_local(global_range::UnitRange{Int}, subtensor_start::Int) -> UnitRange{Int}

Convert global index range to local (0-offset) range relative to subtensor start.
"""
function global_to_local(global_range::UnitRange{Int}, subtensor_start::Int)
    return (first(global_range) - subtensor_start + 1):(last(global_range) - subtensor_start + 1)
end

"""
    compute_halo_sizes(kernel_size::Int, stride::Int=1, dilation::Int=1) -> Tuple{Int, Int}

Compute left and right halo (ghost region) sizes for a convolution.
Returns `(left_halo, right_halo)`.
"""
function compute_halo_sizes(kernel_size::Int; stride::Int=1, dilation::Int=1)
    effective_kernel = (kernel_size - 1) * dilation + 1
    pad_total = effective_kernel - 1
    left = pad_total ÷ 2
    right = pad_total - left
    return (left, right)
end
