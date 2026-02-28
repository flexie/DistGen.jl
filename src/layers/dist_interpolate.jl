# Distributed trilinear interpolation for Karras-style up/downsampling.
#
# The Karras reference uses F.interpolate(mode='trilinear') instead of
# strided/transposed convolutions. This provides smoother gradients and
# matches the magnitude-preserving design philosophy.
#
# DistInterpolateDown: halo exchange → meanpool(2) → 1×1 conv (channel projection)
# DistInterpolateUp: upsample_trilinear(×2) → repartition

using NNlib: meanpool, upsample_trilinear

"""
    DistInterpolateDown

Distributed downsampling via trilinear interpolation (÷2 per spatial dim).

Composition:
1. Mean pooling with kernel=stride=2 (equivalent to trilinear ÷2)
2. 1×1 DistConv3d for channel projection (weight-normed, no bias)
3. Optional repartition if topology changes between levels
"""
struct DistInterpolateDown
    conv::DistConv3d           # 1×1 conv for channel projection
    repartition_info::Union{RepartitionInfo, Nothing}
end

"""
    DistInterpolateDown(P_src, P_dst, global_shape_5d, comm, in_ch, out_ch)

Construct a distributed interpolation-based downsampler.
"""
function DistInterpolateDown(
    P_src::CartesianPartition{N},
    P_dst::CartesianPartition{N},
    global_shape_5d::NTuple{N, Int},
    parent_comm::MPI.Comm,
    in_channels::Int,
    out_channels::Int
) where {N}
    # 1×1 conv for channel projection on the destination partition
    # After pooling, the spatial dimensions are halved, so we need the conv
    # on the source partition (before repartition) or destination (after).
    # We apply conv on source partition first (before repartition).
    conv = DistConv3d(P_src, in_channels, out_channels, (1, 1, 1);
                      bias=false)

    repart_info = if P_src.dims != P_dst.dims
        # Compute the shape after pooling for repartition
        pooled_shape = ntuple(N) do d
            if d <= 3
                global_shape_5d[d] ÷ 2
            elseif d == 4
                out_channels
            else
                global_shape_5d[d]
            end
        end
        setup_repartition(P_src, P_dst, pooled_shape, parent_comm)
    else
        nothing
    end

    return DistInterpolateDown(conv, repart_info)
end

"""
    dist_interpolate_down_forward(x, ps, layer::DistInterpolateDown)

Forward pass:
1. Mean pool ÷2 (local operation — each rank pools its own tile)
2. 1×1 conv for channel projection
3. Repartition if topology changes
"""
function dist_interpolate_down_forward(
    x::AbstractArray{T, 5},
    ps::NamedTuple,
    layer::DistInterpolateDown
) where {T}
    # Step 1: Mean pool ÷2 — NNlib.meanpool expects (W,H,D,C,N) with PoolDims
    pdims = NNlib.PoolDims(size(x), (2, 2, 2);
                           stride=(2, 2, 2), padding=(0, 0, 0, 0, 0, 0))
    x_pooled = meanpool(x, pdims)

    # Step 2: 1×1 conv for channel projection (no halo needed for 1×1)
    weight = ps.weight
    # Apply weight normalization if stored params are raw
    if haskey(ps, :weight_norm) && ps.weight_norm
        weight = normalize_weight(weight) ./ sqrt(T(size(weight, 4)))
    end
    cdims = NNlib.DenseConvDims(size(x_pooled), size(weight);
                                stride=[1, 1, 1], padding=ntuple(_ -> 0, 6))
    y = NNlib.conv(x_pooled, weight, cdims)

    # Step 3: Repartition if topology changes
    if layer.repartition_info !== nothing
        y = repartition_op(y, layer.repartition_info)
    end

    return y
end

function init_dist_interpolate_down(rng, layer::DistInterpolateDown; T::Type=Float32)
    return init_dist_conv3d(rng, layer.conv; T=T)
end

function (layer::DistInterpolateDown)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    y = dist_interpolate_down_forward(x, ps, layer)
    return y, st
end

"""
    DistInterpolateUp

Distributed upsampling via trilinear interpolation (×2 per spatial dim).

Composition:
1. Local trilinear upsample ×2
2. Optional repartition to finer spatial partition
"""
struct DistInterpolateUp
    repartition_info::Union{RepartitionInfo, Nothing}
end

"""
    DistInterpolateUp(P_src, P_dst, global_shape_5d, comm)

Construct a distributed interpolation-based upsampler.
`global_shape_5d` is the OUTPUT shape after upsampling.
"""
function DistInterpolateUp(
    P_src::CartesianPartition{N},
    P_dst::CartesianPartition{N},
    global_shape_5d::NTuple{N, Int},
    parent_comm::MPI.Comm
) where {N}
    repart_info = if P_src.dims != P_dst.dims
        setup_repartition(P_src, P_dst, global_shape_5d, parent_comm)
    else
        nothing
    end
    return DistInterpolateUp(repart_info)
end

"""
    dist_interpolate_up_forward(x, layer::DistInterpolateUp)

Forward pass:
1. Trilinear upsample ×2 (local)
2. Repartition to finer partition if needed
"""
function dist_interpolate_up_forward(
    x::AbstractArray{T, 5},
    layer::DistInterpolateUp
) where {T}
    # Step 1: Upsample ×2 via nearest-neighbor using repeat (GPU-compatible, Zygote-differentiable)
    y = repeat(x, inner=(2, 2, 2, 1, 1))

    # Step 2: Repartition if topology changes
    if layer.repartition_info !== nothing
        y = repartition_op(y, layer.repartition_info)
    end

    return y
end

function (layer::DistInterpolateUp)(x::AbstractArray, ::NamedTuple, st::NamedTuple)
    y = dist_interpolate_up_forward(x, layer)
    return y, st
end
