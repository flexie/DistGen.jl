# Distributed 3D U-Net for FWI with time/noise conditioning.
#
# Multi-level architecture with automatic partition topology transitions:
# Level 0: 1024^3 × 20ch  — 8×8×8 spatial × 1 channel  (512 GPUs)
# Level 1:  512^3 × 64ch  — 8×8×8 spatial × 1 channel
# Level 2:  256^3 × 128ch — 4×4×4 spatial × 8 channel
# Level 3:  128^3 × 256ch — 2×2×2 spatial × 64 channel
# Level 4:   64^3 × 512ch — 1×1×1 spatial × 512 channel
#
# Training: both score-based diffusion and flow matching (shared backbone).

"""
    PartitionPlan

Describes the partition topology at each U-Net level.

# Fields
- `spatial_dims`: Spatial partition dimensions at each level
- `channel_dims`: Channel partition factor at each level
- `spatial_shape`: Global spatial shape at each level
- `channels`: Number of channels at each level
"""
struct PartitionPlan{N}
    levels::Int
    spatial_dims::Vector{NTuple{N, Int}}
    channels::Vector{Int}
    spatial_shape::Vector{NTuple{N, Int}}
end

"""
    plan_partitions(global_shape, in_channels, channel_mult, n_gpus; levels=5)

Automatically plan partition topologies for a U-Net given the global shape,
number of GPUs, and channel multipliers per level.

Strategy: at coarse levels, shift parallelism from spatial to channel dimensions
to keep all GPUs busy as spatial resolution decreases.
"""
function plan_partitions(
    global_shape::NTuple{3, Int},
    in_channels::Int,
    channel_mult::Vector{Int},
    n_gpus::Int;
    levels::Int = length(channel_mult)
)
    spatial_dims_list = NTuple{3, Int}[]
    channels_list = Int[]
    spatial_shapes = NTuple{3, Int}[]

    for l in 1:levels
        # Spatial shape at this level (halved each level)
        factor = 2^(l - 1)
        sp = ntuple(d -> global_shape[d] ÷ factor, 3)

        # Channels at this level
        ch = in_channels * channel_mult[l]

        # Determine spatial vs channel partitioning
        # Max spatial partitions: each local domain should be ≥ 32 elements per dim
        max_spatial_per_dim = ntuple(d -> max(1, sp[d] ÷ 32), 3)
        max_spatial = prod(max_spatial_per_dim)

        if max_spatial >= n_gpus
            # Enough spatial work: use pure spatial partitioning
            # Find balanced 3D factorization of n_gpus
            sp_dims = _balanced_3d_factorization(n_gpus)
        else
            # Need channel parallelism too
            sp_dims = max_spatial_per_dim
            # Remaining GPUs go to channel dim (handled by layer)
        end

        push!(spatial_dims_list, sp_dims)
        push!(channels_list, ch)
        push!(spatial_shapes, sp)
    end

    return PartitionPlan{3}(levels, spatial_dims_list, channels_list, spatial_shapes)
end

"""
Find a balanced 3D factorization of n close to a cube root.
"""
function _balanced_3d_factorization(n::Int)
    cbrt_n = round(Int, n^(1/3))
    best = (1, 1, n)
    best_ratio = n / 1.0

    for i in cbrt_n:-1:1
        n % i != 0 && continue
        remaining = n ÷ i
        sqrt_r = round(Int, sqrt(remaining))
        for j in sqrt_r:-1:1
            remaining % j != 0 && continue
            k = remaining ÷ j
            ratio = max(i, j, k) / min(i, j, k)
            if ratio < best_ratio
                best = (i, j, k)
                best_ratio = ratio
            end
        end
    end

    return best
end

# ─── U-Net Block ─────────────────────────────────────────────────────────────

"""
    DistResBlock

A residual block for the distributed U-Net: two convolutions with group norm
and conditioning injection.

    x → GroupNorm → SiLU → Conv3d → AdaptiveGroupNorm(cond) → SiLU → Conv3d → + x
"""
struct DistResBlock
    conv1::DistConv3d
    conv2::DistConv3d
    norm1::DistGroupNorm
    norm2::DistAdaptiveGroupNorm
    skip_conv::Union{DistConv3d, Nothing}  # If in_ch != out_ch
end

function DistResBlock(
    P::CartesianPartition,
    in_channels::Int,
    out_channels::Int,
    cond_dim::Int;
    num_groups::Int = min(32, in_channels)
)
    norm1 = DistGroupNorm(P, in_channels, num_groups)
    conv1 = DistConv3d(P, in_channels, out_channels, (3, 3, 3))

    gn2 = DistGroupNorm(P, out_channels, min(32, out_channels))
    norm2 = DistAdaptiveGroupNorm(gn2, cond_dim)
    conv2 = DistConv3d(P, out_channels, out_channels, (3, 3, 3))

    skip_conv = if in_channels != out_channels
        DistConv3d(P, in_channels, out_channels, (1, 1, 1); bias=false)
    else
        nothing
    end

    return DistResBlock(conv1, conv2, norm1, norm2, skip_conv)
end

# ─── Distributed U-Net ──────────────────────────────────────────────────────

"""
    DistUNet3d

Complete distributed 3D U-Net for FWI score estimation / flow matching.

# Fields
- `plan`: Partition topology plan
- `encoder_blocks`: Residual blocks for each encoder level
- `decoder_blocks`: Residual blocks for each decoder level
- `downsamplers`: Downsampling layers between encoder levels
- `upsamplers`: Upsampling layers between decoder levels
- `skip_connections`: Skip connections aligning encoder/decoder topologies
- `bottleneck`: Bottleneck residual block
- `in_conv`: Initial convolution
- `out_conv`: Final convolution
- `time_embed_dim`: Dimension of time/noise embedding
"""
struct DistUNet3d
    plan::PartitionPlan
    n_levels::Int
    in_channels::Int
    out_channels::Int
    time_embed_dim::Int
end

"""
    DistUNet3d(plan, in_channels, out_channels; time_embed_dim=256)
"""
function DistUNet3d(
    plan::PartitionPlan,
    in_channels::Int,
    out_channels::Int;
    time_embed_dim::Int = 256
)
    return DistUNet3d(plan, plan.levels, in_channels, out_channels, time_embed_dim)
end

# ─── Time Embedding ──────────────────────────────────────────────────────────

"""
    sinusoidal_embedding(t::AbstractVector{T}, dim::Int) -> Matrix{T}

Sinusoidal positional embedding for time/noise level conditioning.
`t`: (batch,) — time values in [0, 1]
Returns: (dim, batch)
"""
function sinusoidal_embedding(t::AbstractVector{T}, dim::Int) where {T}
    half_dim = dim ÷ 2
    freq = T(1) ./ (T(10000) .^ (T.(0:half_dim-1) ./ T(half_dim)))
    # t: (batch,), freq: (half_dim,)
    angles = freq * t'  # (half_dim, batch)
    return vcat(sin.(angles), cos.(angles))  # (dim, batch)
end

# ─── Training Loops ──────────────────────────────────────────────────────────

"""
    score_based_diffusion_loss(model, ps, st, x, σ_schedule; rng=Random.default_rng())

Denoising score matching loss for score-based diffusion training.

    L = E_{σ} E_{x, ε} [ σ² ||s_θ(x + σε, σ) - (-ε/σ)||² ]

where `s_θ` is the score network (model), ε ~ N(0, I), and σ is sampled from schedule.
"""
function score_based_diffusion_loss(
    model_forward::Function,
    ps::NamedTuple,
    st::NamedTuple,
    x::AbstractArray{T},
    sigma::T;
    rng = nothing
) where {T}
    # Sample noise
    ε = randn(T, size(x)...)

    # Perturbed input
    x_noisy = x .+ sigma .* ε

    # Time conditioning: embed sigma
    batch_size = size(x)[end]
    t_embed = fill(sigma, batch_size)

    # Score prediction: s_θ(x_noisy, σ)
    score_pred, st = model_forward(x_noisy, t_embed, ps, st)

    # Target score: -ε/σ
    target = -ε ./ sigma

    # Weighted MSE loss: σ² ||s_θ - target||²
    loss = sigma^2 * sum((score_pred .- target) .^ 2) / length(x)

    return loss, st
end

"""
    flow_matching_loss(model, ps, st, x0, x1, t)

Conditional flow matching (OT-CFM) loss.

    L = E_{t, x0, x1} [ ||v_θ(x_t, t) - (x1 - x0)||² ]

where x_t = (1-t)x0 + t·x1 is the interpolant, and v_θ is the velocity field.
"""
function flow_matching_loss(
    model_forward::Function,
    ps::NamedTuple,
    st::NamedTuple,
    x0::AbstractArray{T},
    x1::AbstractArray{T},
    t::T
) where {T}
    # Interpolated point
    x_t = (one(T) - t) .* x0 .+ t .* x1

    # Target velocity (constant for OT paths)
    target_v = x1 .- x0

    # Time conditioning
    batch_size = size(x0)[end]
    t_embed = fill(t, batch_size)

    # Velocity prediction
    v_pred, st = model_forward(x_t, t_embed, ps, st)

    # MSE loss
    loss = sum((v_pred .- target_v) .^ 2) / length(x0)

    return loss, st
end

"""
    langevin_sample(model_forward, ps, st, shape, sigmas; n_steps=100, step_size=1e-3)

Annealed Langevin dynamics sampling for score-based diffusion.
"""
function langevin_sample(
    model_forward::Function,
    ps::NamedTuple,
    st::NamedTuple,
    shape::NTuple{N, Int},
    sigmas::AbstractVector{T};
    n_steps::Int = 100,
    step_size::T = T(1e-3)
) where {T, N}
    x = randn(T, shape...)

    for σ in sigmas
        α = step_size * (σ / sigmas[end])^2
        batch_size = shape[end]
        t_embed = fill(σ, batch_size)

        for _ in 1:n_steps
            score, st = model_forward(x, t_embed, ps, st)
            noise = randn(T, shape...)
            x = x .+ α .* score .+ sqrt(T(2) * α) .* noise
        end
    end

    return x
end

"""
    ode_sample(model_forward, ps, st, x0, n_steps; method=:euler)

ODE-based sampling for flow matching (integrate velocity field from t=0 to t=1).
"""
function ode_sample(
    model_forward::Function,
    ps::NamedTuple,
    st::NamedTuple,
    x0::AbstractArray{T};
    n_steps::Int = 100
) where {T}
    dt = one(T) / T(n_steps)
    x = copy(x0)
    batch_size = size(x0)[end]

    for i in 0:(n_steps - 1)
        t = T(i) / T(n_steps)
        t_embed = fill(t, batch_size)
        v, st = model_forward(x, t_embed, ps, st)
        x = x .+ dt .* v  # Euler step
    end

    return x
end
