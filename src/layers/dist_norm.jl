# Distributed Group Normalization: AllReduce for cross-partition statistics.
#
# GroupNorm normalizes over (spatial + channel_group) dimensions.
# In a domain-decomposed setting, spatial statistics must be aggregated
# across partition workers via AllReduce before normalization.

"""
    DistGroupNorm

Domain-decomposed group normalization.

Forward:
1. Compute local sum and sum-of-squares over spatial dimensions
2. AllReduce to get global mean and variance across spatial partitions
3. Normalize locally using global statistics
4. Apply learnable affine transform (scale γ, shift β)

# Fields
- `partition`: Cartesian partition (spatial decomposition)
- `all_reduce_info`: Pre-computed all-reduce metadata
- `num_channels`: Number of channels
- `num_groups`: Number of groups (must divide num_channels)
- `eps`: Epsilon for numerical stability
"""
struct DistGroupNorm
    partition::CartesianPartition
    all_reduce_info::AllReduceInfo
    num_channels::Int
    num_groups::Int
    eps::Float64
end

"""
    DistGroupNorm(P, num_channels, num_groups; eps=1e-5)
"""
function DistGroupNorm(
    P::CartesianPartition,
    num_channels::Int,
    num_groups::Int;
    eps::Float64 = 1e-5
)
    @assert num_channels % num_groups == 0 "num_channels must be divisible by num_groups"
    all_reduce_info = setup_all_reduce(P)
    return DistGroupNorm(P, all_reduce_info, num_channels, num_groups, eps)
end

"""
    dist_groupnorm_forward(x, γ, β, layer::DistGroupNorm)

Forward pass for distributed group normalization.

`x` shape: (W, H, D, C, batch) for 3D data
`γ`, `β` shape: (C,) — per-channel affine parameters
"""
function dist_groupnorm_forward(
    x::AbstractArray{T, 5},
    γ::AbstractArray{T, 1},
    β::AbstractArray{T, 1},
    layer::DistGroupNorm
) where {T}
    W, H, D, C, B = size(x)
    G = layer.num_groups
    C_per_G = C ÷ G

    # Reshape to (W, H, D, C_per_G, G, B) for group-wise statistics
    x_grouped = reshape(x, W, H, D, C_per_G, G, B)

    # Local spatial size (elements per group per batch)
    local_spatial_size = W * H * D * C_per_G

    # Step 1: Local sum and sum-of-squares over (W, H, D, C_per_G) dims
    local_sum = sum(x_grouped; dims=(1, 2, 3, 4))      # (1, 1, 1, 1, G, B)
    local_sum_sq = sum(x_grouped .^ 2; dims=(1, 2, 3, 4))

    # Step 2: AllReduce across spatial partitions to get global statistics
    global_sum = all_reduce_op(dropdims(local_sum; dims=(1, 2, 3, 4)), layer.all_reduce_info)
    global_sum_sq = all_reduce_op(dropdims(local_sum_sq; dims=(1, 2, 3, 4)), layer.all_reduce_info)

    # Total number of spatial elements across all partitions
    n_partitions = prod(layer.partition.dims)
    global_count = T(local_spatial_size * n_partitions)

    # Compute global mean and variance
    mean = global_sum ./ global_count          # (G, B)
    var = global_sum_sq ./ global_count .- mean .^ 2  # (G, B)

    # Step 3: Normalize locally
    mean_5d = reshape(mean, 1, 1, 1, 1, G, B)
    inv_std = reshape(T(1) ./ sqrt.(var .+ T(layer.eps)), 1, 1, 1, 1, G, B)

    x_norm = (x_grouped .- mean_5d) .* inv_std

    # Reshape back to (W, H, D, C, B)
    x_norm = reshape(x_norm, W, H, D, C, B)

    # Step 4: Affine transform
    y = x_norm .* reshape(γ, 1, 1, 1, :, 1) .+ reshape(β, 1, 1, 1, :, 1)

    return y
end

"""
    init_dist_groupnorm(rng, layer::DistGroupNorm; T=Float32) -> (γ, β)
"""
function init_dist_groupnorm(rng, layer::DistGroupNorm; T::Type=Float32)
    γ = ones(T, layer.num_channels)
    β = zeros(T, layer.num_channels)
    return (scale=γ, bias=β)
end

function (layer::DistGroupNorm)(x::AbstractArray, ps::NamedTuple, st::NamedTuple)
    y = dist_groupnorm_forward(x, ps.scale, ps.bias, layer)
    return y, st
end

# ─── Adaptive Group Norm for time/noise conditioning ─────────────────────────

"""
    DistAdaptiveGroupNorm

Distributed adaptive group normalization for conditioning on time/noise level.
Applies FiLM-style modulation: `y = γ_cond(t) * GroupNorm(x) + β_cond(t)`
where γ_cond, β_cond are predicted from the conditioning signal.

# Fields
- `groupnorm`: Underlying DistGroupNorm
- `cond_dim`: Dimension of the conditioning input
"""
struct DistAdaptiveGroupNorm
    groupnorm::DistGroupNorm
    cond_dim::Int
end

"""
    dist_adaptive_groupnorm_forward(x, cond, γ_proj_w, γ_proj_b, β_proj_w, β_proj_b, layer)

`x`: (W, H, D, C, B) — input tensor
`cond`: (cond_dim, B) — conditioning signal (e.g., time embedding)
"""
function dist_adaptive_groupnorm_forward(
    x::AbstractArray{T, 5},
    cond::AbstractArray{T, 2},
    ps::NamedTuple,
    layer::DistAdaptiveGroupNorm
) where {T}
    C = size(x, 4)

    # Project conditioning to per-channel scale and shift
    # γ_cond = W_γ * cond + b_γ  →  (C, B)
    # β_cond = W_β * cond + b_β  →  (C, B)
    γ_cond = ps.gamma_weight * cond .+ ps.gamma_bias
    β_cond = ps.beta_weight * cond .+ ps.beta_bias

    # Apply group norm (without learnable affine — we use conditioning instead)
    # Device-aware, non-differentiable (constants: scale=1, bias=0)
    gn_ps = (scale=_ones_like(x, C), bias=_zeros_like(x, C))
    x_norm, _ = layer.groupnorm(x, gn_ps, NamedTuple())

    # FiLM modulation
    y = x_norm .* reshape(γ_cond, 1, 1, 1, C, :) .+ reshape(β_cond, 1, 1, 1, C, :)

    return y
end

function init_dist_adaptive_groupnorm(rng, layer::DistAdaptiveGroupNorm; T::Type=Float32)
    C = layer.groupnorm.num_channels
    d = layer.cond_dim
    std = sqrt(T(2) / T(d))
    return (
        gamma_weight=randn(rng, T, C, d) .* std,
        gamma_bias=ones(T, C, 1),  # Initialize scale near 1
        beta_weight=randn(rng, T, C, d) .* std,
        beta_bias=zeros(T, C, 1),
    )
end
