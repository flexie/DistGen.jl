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
    weight_norm::Bool       # Karras forced weight normalization
    concat_ones::Bool       # Prepend ones channel (Karras input block)
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
    bias::Bool = true,
    weight_norm::Bool = false,
    concat_ones::Bool = false
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

    return DistConv3d(P, in_channels, out_channels, kernel_size, stride, dilation,
                      halo_info, bias, weight_norm, concat_ones)
end

"""
    pad_for_halo(x, halo_info)

Auto-pad a tensor with zeros to add halo space for halo exchange.
Takes unpadded `(W,H,D,C,B)` → padded with bulk centered, zeros in ghost region.
If the tensor already has halo space (detected by matching expected padded size), returns as-is.
"""
function pad_for_halo(x::AbstractArray{T, 5}, halo_info::HaloInfo{N}) where {T, N}
    spatial_ndims = min(N, 3)
    needs_padding = false
    for d in 1:spatial_ndims
        left, right = halo_info.halo_sizes[d]
        if left > 0 || right > 0
            needs_padding = true
            break
        end
    end
    !needs_padding && return x

    # Functional zero-padding (Zygote-compatible — no mutation)
    # Pad each spatial dimension sequentially using cat with zero slabs
    y = x
    for d in 1:spatial_ndims
        left, right = halo_info.halo_sizes[d]
        sz = size(y)
        if left > 0
            pad_sz = ntuple(i -> i == d ? left : sz[i], 5)
            y = cat(_zeros_like(y, pad_sz...), y; dims=d)
        end
        if right > 0
            sz = size(y)
            pad_sz = ntuple(i -> i == d ? right : sz[i], 5)
            y = cat(y, _zeros_like(y, pad_sz...); dims=d)
        end
    end
    return y
end

"""
    dist_conv3d_forward(x, weight, bias, layer::DistConv3d; auto_pad=false)

Forward pass:
1. Optionally auto-pad input (add halo space if `auto_pad=true`)
2. Halo exchange on spatial dimensions
3. Local convolution (NNlib.conv)

`x` shape: (W, H, D, C_in, batch) — local spatial dims
  - If `auto_pad=false` (default): x must already include halo space
  - If `auto_pad=true`: x is unpadded bulk data, halo is added automatically
`weight` shape: (kW, kH, kD, C_in, C_out) — full convolution kernel
"""
function dist_conv3d_forward(
    x::AbstractArray{T, 5},
    weight::AbstractArray{T, 5},
    bias::Union{AbstractArray{T, 1}, Nothing},
    layer::DistConv3d;
    auto_pad::Bool = false
) where {T}
    # Step 0: Concat ones channel if needed (Karras input block protection)
    if layer.concat_ones
        sz = size(x)
        ones_ch = _ones_like(x, sz[1], sz[2], sz[3], 1, sz[5])
        x = cat(ones_ch, x; dims=4)
    end

    # Step 1: Auto-pad for halo if requested
    if auto_pad
        x = pad_for_halo(x, layer.halo_info)
    end

    # Step 2: Halo exchange to fill ghost regions
    x_exchanged = halo_exchange(x, layer.halo_info)

    # Step 3: Apply weight normalization if enabled (Karras Algorithm 1)
    w = if layer.weight_norm
        fan_in = T(layer.in_channels + (layer.concat_ones ? 1 : 0)) * T(prod(layer.kernel_size))
        normalize_weight(weight) ./ sqrt(fan_in)
    else
        weight
    end

    # Step 4: Local convolution (NNlib expects WHDC format with no padding since
    # halo exchange already provides the necessary spatial context)
    cdims = NNlib.DenseConvDims(
        size(x_exchanged), size(w);
        stride=collect(layer.stride),
        dilation=collect(layer.dilation),
        padding=ntuple(_ -> 0, 6)  # No additional padding — halos provide it
    )
    y = conv(x_exchanged, w, cdims)

    # Step 5: Add bias if present
    if bias !== nothing
        # Reshape bias for broadcasting: (1, 1, 1, C_out, 1)
        y = y .+ reshape(bias, 1, 1, 1, :, 1)
    end

    return y
end

"""
    init_dist_conv3d(rng, layer::DistConv3d; T=Float32) -> (weight, bias)

Initialize parameters for distributed convolution (Kaiming/He initialization).
"""
function init_dist_conv3d(rng, layer::DistConv3d; T::Type=Float32)
    c_in = layer.in_channels + (layer.concat_ones ? 1 : 0)
    fan_in = c_in * prod(layer.kernel_size)
    std = sqrt(T(2) / T(fan_in))

    weight = randn(rng, T, layer.kernel_size..., c_in, layer.out_channels) .* std

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
