# Distributed Linear Layer: Broadcast weights → local matmul → SumReduce
#
# For channel-only partitions (e.g., bottleneck of U-Net).

"""
    DistLinear

Domain-decomposed linear (fully connected) layer.

Composition:
1. Broadcast weights from root to all workers
2. Local matrix multiplication on partitioned channels
3. Sum-reduce partial results to get global output

For channel-partitioned data, each worker holds a slice of the input channels
and computes a partial output. The sum-reduce aggregates these.

# Fields
- `partition`: Partition over channel dimension
- `in_features`: Global input features
- `out_features`: Global output features
- `use_bias`: Whether to include bias
"""
struct DistLinear
    partition::CartesianPartition
    in_features::Int
    out_features::Int
    use_bias::Bool
end

"""
    DistLinear(P, in_features, out_features; bias=true)
"""
function DistLinear(
    P::CartesianPartition,
    in_features::Int,
    out_features::Int;
    bias::Bool = true
)
    return DistLinear(P, in_features, out_features, bias)
end

"""
    dist_linear_forward(x, weight, bias, layer::DistLinear)

Forward pass:
`x` shape: (local_in_features, batch)
`weight` shape: (out_features, local_in_features) — each worker has a column slice
`bias` shape: (out_features,) — only applied after sum-reduce (at root or all)

1. Local matmul: y_partial = weight * x
2. AllReduce (sum) across channel partitions
"""
function dist_linear_forward(
    x::AbstractArray{T, 2},
    weight::AbstractArray{T, 2},
    bias::Union{AbstractArray{T, 1}, Nothing},
    layer::DistLinear
) where {T}
    # Step 1: Local matmul (partial output from local channel slice)
    y_partial = weight * x  # (out_features, batch)

    # Step 2: AllReduce to sum partial products from all channel partitions
    ar_info = setup_all_reduce(layer.partition)
    y = all_reduce_op(y_partial, ar_info)

    # Step 3: Add bias (same on all workers after allreduce)
    if bias !== nothing
        y .+= reshape(bias, :, 1)
    end

    return y
end

function init_dist_linear(rng, layer::DistLinear; T::Type=Float32)
    # Each worker gets a column slice of the weight matrix
    local_in = balanced_decomposition(layer.in_features, prod(layer.partition.dims))
    my_in = local_in[layer.partition.rank + 1]

    std = sqrt(T(2) / T(layer.in_features))
    weight = randn(rng, T, layer.out_features, my_in) .* std
    bias = layer.use_bias ? zeros(T, layer.out_features) : nothing

    return (weight=weight, bias=bias)
end

function (layer::DistLinear)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    bias = haskey(ps, :bias) ? ps.bias : nothing
    y = dist_linear_forward(x, ps.weight, bias, layer)
    return y, st
end

# ─── DistSkipConnection ─────────────────────────────────────────────────────

"""
    DistSkipConnection

Skip connection that handles partition topology alignment for U-Net.
If encoder and decoder use different partitions, a repartition is applied
before concatenation along the channel dimension.

# Fields
- `repartition_info`: Repartition metadata (nothing if topologies match)
"""
struct DistSkipConnection
    repartition_info::Union{RepartitionInfo, Nothing}
end

"""
    DistSkipConnection(encoder_P, decoder_P, global_shape, parent_comm)

If encoder and decoder partitions differ, set up repartition.
If they match, skip connection is a simple concatenation.
"""
function DistSkipConnection(
    encoder_P::CartesianPartition{N},
    decoder_P::CartesianPartition{N},
    global_shape::NTuple{N, Int},
    parent_comm::MPI.Comm
) where {N}
    if encoder_P.dims == decoder_P.dims
        return DistSkipConnection(nothing)
    else
        repart_info = setup_repartition(encoder_P, decoder_P, global_shape, parent_comm)
        return DistSkipConnection(repart_info)
    end
end

"""
    dist_skip_forward(x_encoder, x_decoder, layer::DistSkipConnection)

Concatenate encoder and decoder features along channel dimension (dim 4 for 5D).
Repartition encoder features if needed to match decoder topology.
"""
function dist_skip_forward(
    x_encoder::AbstractArray{T, 5},
    x_decoder::AbstractArray{T, 5},
    layer::DistSkipConnection
) where {T}
    if layer.repartition_info !== nothing
        x_encoder = repartition_op(x_encoder, layer.repartition_info)
    end

    # Concatenate along channel dimension (dim 4)
    return cat(x_encoder, x_decoder; dims=4)
end
