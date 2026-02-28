# Distributed 3D Convolution: HaloExchange → local Conv3d → (optional SumReduce)
#
# Follows distdl's conv_general.py composition pattern.
# Implements Lux.jl AbstractLuxLayer interface (explicit parameters/state).

using NNlib: conv, ∇conv_data, ∇conv_filter

"""
    DistConv3d{N}

Domain-decomposed 3D convolution layer.

Composition:
1. Halo exchange to fill ghost regions for spatial convolution
2. Local Conv3d on the padded local tensor (using NNlib.conv)
3. Optional sum-reduce across channel partitions

# Fields
- `partition`: Cartesian partition for spatial decomposition
- `in_channels`: Number of input channels (global)
- `out_channels`: Number of output channels (global)
- `kernel_size`: Convolution kernel size (spatial dims)
- `stride`: Convolution stride
- `padding`: Convolution padding (handled via halo exchange, not local padding)
- `dilation`: Convolution dilation
- `halo_info`: Pre-computed halo exchange metadata
- `use_bias`: Whether to include bias
"""
struct DistConv3d
    partition::CartesianPartition
    in_channels::Int
    out_channels::Int
    kernel_size::NTuple{3, Int}
    stride::NTuple{3, Int}
    dilation::NTuple{3, Int}
    halo_info::HaloInfo
    use_bias::Bool
end

"""
    DistConv3d(P, in_ch, out_ch, kernel_size; stride=1, dilation=1, bias=true)

Construct a distributed 3D convolution layer.

The halo sizes are computed from the kernel size and dilation to ensure
each local convolution has access to the necessary neighbor data.
"""
function DistConv3d(
    P::CartesianPartition,
    in_channels::Int,
    out_channels::Int,
    kernel_size::NTuple{3, Int};
    stride::NTuple{3, Int} = (1, 1, 1),
    dilation::NTuple{3, Int} = (1, 1, 1),
    bias::Bool = true
)
    # Compute halo sizes from convolution parameters
    # For a convolution with kernel k and dilation d:
    # effective_kernel = (k - 1) * d + 1
    # halo = (effective_kernel - 1) / 2 on each side (for same-padding)
    spatial_ndims = min(ndims(P), 3)
    halo_sizes = Vector{Tuple{Int, Int}}(undef, spatial_ndims)
    for d in 1:spatial_ndims
        left, right = compute_halo_sizes(kernel_size[d]; dilation=dilation[d])
        halo_sizes[d] = (left, right)
    end

    halo_info = compute_halo_info(P, halo_sizes)

    return DistConv3d(P, in_channels, out_channels, kernel_size, stride, dilation, halo_info, bias)
end

"""
    dist_conv3d_forward(x, weight, bias, layer::DistConv3d)

Forward pass:
1. Halo exchange on spatial dimensions
2. Local convolution (NNlib.conv)
3. Trim output to correct size

`x` shape: (W, H, D, C_in, batch) — local spatial dims with halo space
`weight` shape: (kW, kH, kD, C_in, C_out) — full convolution kernel
"""
function dist_conv3d_forward(
    x::AbstractArray{T, 5},
    weight::AbstractArray{T, 5},
    bias::Union{AbstractArray{T, 1}, Nothing},
    layer::DistConv3d
) where {T}
    # Step 1: Halo exchange to fill ghost regions
    x_exchanged = halo_exchange(x, layer.halo_info)

    # Step 2: Local convolution (NNlib expects WHDC format with no padding since
    # halo exchange already provides the necessary spatial context)
    # DenseConvDims expects: (spatial..., C_in, batch) and (spatial..., C_in, C_out)
    cdims = NNlib.DenseConvDims(
        size(x_exchanged), size(weight);
        stride=collect(layer.stride),
        dilation=collect(layer.dilation),
        padding=ntuple(_ -> 0, 6)  # No additional padding — halos provide it
    )
    y = conv(x_exchanged, weight, cdims)

    # Step 3: Add bias if present
    if bias !== nothing
        # Reshape bias for broadcasting: (1, 1, 1, C_out, 1)
        y .+= reshape(bias, 1, 1, 1, :, 1)
    end

    return y
end

"""
    init_dist_conv3d(rng, layer::DistConv3d; T=Float32) -> (weight, bias)

Initialize parameters for distributed convolution (Kaiming/He initialization).
"""
function init_dist_conv3d(rng, layer::DistConv3d; T::Type=Float32)
    fan_in = layer.in_channels * prod(layer.kernel_size)
    std = sqrt(T(2) / T(fan_in))

    weight = randn(rng, T, layer.kernel_size..., layer.in_channels, layer.out_channels) .* std

    bias = layer.use_bias ? zeros(T, layer.out_channels) : nothing

    return (weight=weight, bias=bias)
end

# ─── Lux-compatible interface ────────────────────────────────────────────────
# These follow Lux.jl's (layer, x, ps, st) -> (y, st) convention.
# Full Lux integration requires `using Lux` which is optional.

"""
    (layer::DistConv3d)(x, ps, st) -> (y, st)

Lux-style forward pass. `ps` contains `:weight` and optionally `:bias`.
`st` is passed through unchanged.
"""
function (layer::DistConv3d)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    bias = haskey(ps, :bias) ? ps.bias : nothing
    y = dist_conv3d_forward(x, ps.weight, bias, layer)
    return y, st
end
