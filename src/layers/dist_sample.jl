# Distributed Downsample / Upsample with topology repartition.
#
# At U-Net level transitions, the partition topology changes:
# - Downsample: more spatial partitions → fewer spatial, more channel partitions
# - Upsample: fewer spatial, more channel → more spatial partitions
#
# Each involves: Repartition → local strided conv (down) or transposed conv (up)

"""
    DistDownsample

Distributed downsampling layer for U-Net encoder transitions.

Composition:
1. Repartition from fine spatial partition to coarser spatial + channel partition
2. Local strided convolution (stride 2) for spatial downsampling
3. Halo exchange on the new partition (for subsequent convolutions)

# Fields
- `src_partition`: Fine spatial partition (e.g., 8×8×8 spatial)
- `dst_partition`: Coarse partition (e.g., 4×4×4 spatial × 2 channel)
- `repartition_info`: Pre-computed repartition metadata
- `in_channels`: Input channels
- `out_channels`: Output channels (typically 2× input)
- `kernel_size`: Downsampling kernel size
"""
struct DistDownsample
    src_partition::CartesianPartition
    dst_partition::CartesianPartition
    repartition_info::RepartitionInfo
    in_channels::Int
    out_channels::Int
    kernel_size::NTuple{3, Int}
end

"""
    DistDownsample(src_P, dst_P, global_shape, parent_comm, in_ch, out_ch; kernel_size=(2,2,2))
"""
function DistDownsample(
    src_P::CartesianPartition{N},
    dst_P::CartesianPartition{N},
    global_shape::NTuple{N, Int},
    parent_comm::MPI.Comm,
    in_channels::Int,
    out_channels::Int;
    kernel_size::NTuple{3, Int} = (2, 2, 2)
) where {N}
    repart_info = setup_repartition(src_P, dst_P, global_shape, parent_comm)
    return DistDownsample(src_P, dst_P, repart_info, in_channels, out_channels, kernel_size)
end

"""
    dist_downsample_forward(x, weight, bias, layer::DistDownsample)

Forward pass:
1. Repartition to target topology
2. Local strided convolution (stride = kernel_size for non-overlapping downsampling)
"""
function dist_downsample_forward(
    x::AbstractArray{T, 5},
    weight::AbstractArray{T, 5},
    bias::Union{AbstractArray{T, 1}, Nothing},
    layer::DistDownsample
) where {T}
    # Step 1: Repartition
    x_repart = repartition_op(x, layer.repartition_info)

    # Step 2: Local strided convolution
    stride = collect(layer.kernel_size)
    cdims = NNlib.DenseConvDims(
        size(x_repart), size(weight);
        stride=stride,
        padding=ntuple(_ -> 0, 6)
    )
    y = NNlib.conv(x_repart, weight, cdims)

    if bias !== nothing
        y .+= reshape(bias, 1, 1, 1, :, 1)
    end

    return y
end

function init_dist_downsample(rng, layer::DistDownsample; T::Type=Float32)
    fan_in = layer.in_channels * prod(layer.kernel_size)
    std = sqrt(T(2) / T(fan_in))
    weight = randn(rng, T, layer.kernel_size..., layer.in_channels, layer.out_channels) .* std
    bias = zeros(T, layer.out_channels)
    return (weight=weight, bias=bias)
end

function (layer::DistDownsample)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    bias = haskey(ps, :bias) ? ps.bias : nothing
    y = dist_downsample_forward(x, ps.weight, bias, layer)
    return y, st
end

# ─── DistUpsample ────────────────────────────────────────────────────────────

"""
    DistUpsample

Distributed upsampling layer for U-Net decoder transitions.

Composition:
1. Local transposed convolution for spatial upsampling
2. Repartition from coarse partition back to finer spatial partition

# Fields
- `src_partition`: Coarse partition (e.g., 4×4×4 spatial × 2 channel)
- `dst_partition`: Fine spatial partition (e.g., 8×8×8 spatial)
- `repartition_info`: Pre-computed repartition metadata
- `in_channels`: Input channels
- `out_channels`: Output channels (typically 0.5× input)
- `kernel_size`: Upsampling kernel size
"""
struct DistUpsample
    src_partition::CartesianPartition
    dst_partition::CartesianPartition
    repartition_info::RepartitionInfo
    in_channels::Int
    out_channels::Int
    kernel_size::NTuple{3, Int}
end

function DistUpsample(
    src_P::CartesianPartition{N},
    dst_P::CartesianPartition{N},
    global_shape::NTuple{N, Int},
    parent_comm::MPI.Comm,
    in_channels::Int,
    out_channels::Int;
    kernel_size::NTuple{3, Int} = (2, 2, 2)
) where {N}
    repart_info = setup_repartition(src_P, dst_P, global_shape, parent_comm)
    return DistUpsample(src_P, dst_P, repart_info, in_channels, out_channels, kernel_size)
end

"""
    dist_upsample_forward(x, weight, bias, layer::DistUpsample)

Forward pass:
1. Local transposed convolution (stride = kernel_size)
2. Repartition to target (finer) topology
"""
function dist_upsample_forward(
    x::AbstractArray{T, 5},
    weight::AbstractArray{T, 5},
    bias::Union{AbstractArray{T, 1}, Nothing},
    layer::DistUpsample
) where {T}
    # Step 1: Local transposed convolution
    # For transposed conv, weight shape is (kW, kH, kD, C_out, C_in)
    stride = collect(layer.kernel_size)
    cdims = NNlib.DenseConvDims(
        size(x), size(weight);
        stride=stride,
        padding=ntuple(_ -> 0, 6)
    )
    # NNlib's ∇conv_data acts as transposed convolution
    # Output size = (input_size - 1) * stride + kernel_size
    out_spatial = ntuple(d -> (size(x, d) - 1) * stride[d] + layer.kernel_size[d], 3)
    y_size = (out_spatial..., layer.out_channels, size(x, 5))
    y = zeros(T, y_size...)

    cdims_transp = NNlib.DenseConvDims(
        y_size, size(weight);
        stride=stride,
        padding=ntuple(_ -> 0, 6)
    )
    y = NNlib.∇conv_data(x, weight, cdims_transp)

    if bias !== nothing
        y .+= reshape(bias, 1, 1, 1, :, 1)
    end

    # Step 2: Repartition to finer spatial partition
    y_repart = repartition_op(y, layer.repartition_info)

    return y_repart
end

function init_dist_upsample(rng, layer::DistUpsample; T::Type=Float32)
    # Weight for transposed conv: (kW, kH, kD, C_out, C_in)
    fan_in = layer.in_channels * prod(layer.kernel_size)
    std = sqrt(T(2) / T(fan_in))
    weight = randn(rng, T, layer.kernel_size..., layer.out_channels, layer.in_channels) .* std
    bias = zeros(T, layer.out_channels)
    return (weight=weight, bias=bias)
end

function (layer::DistUpsample)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    bias = haskey(ps, :bias) ? ps.bias : nothing
    y = dist_upsample_forward(x, ps.weight, bias, layer)
    return y, st
end
