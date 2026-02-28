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

# ═══════════════════════════════════════════════════════════════════════════════
# Karras-style Encoder/Decoder blocks (magnitude-preserving, no GroupNorm)
# Reference: networks_3d.py Encoder/Decoder classes
# ═══════════════════════════════════════════════════════════════════════════════

"""
    DistKarrasEncoder

Karras-style encoder block with magnitude-preserving operations.

Forward:
    if downsample: trilinear_interp(÷2) → 1×1 conv
    pixel_norm(x) → res = x
    mp_silu → conv1
    if has_emb: x *= (linear(emb) * gain + 1)  # FiLM
    mp_silu → dropout → conv2
    x = mp_add(x, res; t)

# Fields
- `conv1`: 3×3×3 weight-normed DistConv3d
- `conv2`: 3×3×3 weight-normed DistConv3d
- `dim_in`: Input channels
- `dim_out`: Output channels
- `downsample`: Whether to downsample (÷2)
- `has_emb`: Whether this block uses time embedding
- `emb_dim`: Embedding dimension (for FiLM linear)
- `mp_add_t`: MPAdd t parameter
- `dropout`: Dropout probability
"""
struct DistKarrasEncoder
    conv1::DistConv3d
    conv2::DistConv3d
    downsample_conv::Union{DistConv3d, Nothing}  # 1×1 if downsample
    dim_in::Int
    dim_out::Int
    downsample::Bool
    has_emb::Bool
    emb_dim::Int
    mp_add_t::Float32
    dropout::Float32
end

function DistKarrasEncoder(
    P::CartesianPartition,
    dim_in::Int,
    dim_out::Int;
    emb_dim::Int = 0,
    downsample::Bool = false,
    mp_add_t::Float32 = 0.3f0,
    dropout::Float32 = 0.1f0
)
    has_emb = emb_dim > 0

    # After downsample, curr_dim becomes dim_out
    curr_dim = downsample ? dim_out : dim_in

    downsample_conv = downsample ?
        DistConv3d(P, dim_in, dim_out, (1, 1, 1); bias=false, weight_norm=true) :
        nothing

    conv1 = DistConv3d(P, curr_dim, dim_out, (3, 3, 3); bias=false, weight_norm=true)
    conv2 = DistConv3d(P, dim_out, dim_out, (3, 3, 3); bias=false, weight_norm=true)

    return DistKarrasEncoder(conv1, conv2, downsample_conv, dim_in, dim_out,
                             downsample, has_emb, emb_dim, mp_add_t, dropout)
end

"""
Initialize parameters for DistKarrasEncoder.
"""
function init_dist_karras_encoder(rng, layer::DistKarrasEncoder; T::Type=Float32)
    ps = (
        conv1 = init_dist_conv3d(rng, layer.conv1; T=T),
        conv2 = init_dist_conv3d(rng, layer.conv2; T=T),
    )

    if layer.downsample
        ps = merge(ps, (downsample_conv = init_dist_conv3d(rng, layer.downsample_conv; T=T),))
    end

    if layer.has_emb
        # FiLM: Linear(emb_dim, dim_out) + Gain (scalar, init 0)
        emb_weight = randn(rng, T, layer.dim_out, layer.emb_dim)
        # Apply weight normalization to init
        emb_weight = normalize_weight(emb_weight) ./ sqrt(T(layer.emb_dim))
        ps = merge(ps, (
            emb_linear_weight = emb_weight,
            emb_gain = zeros(T, 1),  # Gain init to 0
        ))
    end

    return ps
end

"""
    dist_karras_encoder_forward(x, emb, ps, layer::DistKarrasEncoder)

Forward pass matching Python Encoder.forward.

`x`: (W,H,D,C,B) spatial tensor
`emb`: (emb_dim, B) time embedding, or nothing
`ps`: NamedTuple of parameters
"""
function dist_karras_encoder_forward(
    x::AbstractArray{T, 5},
    emb::Union{AbstractArray{T, 2}, Nothing},
    ps::NamedTuple,
    layer::DistKarrasEncoder
) where {T}
    # Downsample if needed: meanpool ÷2 → 1×1 conv
    if layer.downsample
        pdims = NNlib.PoolDims(size(x), (2, 2, 2);
                               stride=(2, 2, 2), padding=(0, 0, 0, 0, 0, 0))
        x = NNlib.meanpool(x, pdims)
        x = dist_conv3d_forward(x, ps.downsample_conv.weight, nothing, layer.downsample_conv; auto_pad=true)
    end

    # PixelNorm
    x = pixel_norm(x; dim=4)

    # Residual
    res = x

    # Block 1: mp_silu → conv1
    x = mp_silu(x)
    x = dist_conv3d_forward(x, ps.conv1.weight, nothing, layer.conv1; auto_pad=true)

    # FiLM conditioning: x *= (Linear(emb) * Gain + 1)
    if layer.has_emb && emb !== nothing
        # emb: (emb_dim, B), weight: (dim_out, emb_dim)
        w_emb = normalize_weight(ps.emb_linear_weight) ./ sqrt(T(layer.emb_dim))
        scale = w_emb * emb  # (dim_out, B)
        scale = scale .* ps.emb_gain  # apply learnable gain
        scale = scale .+ one(T)  # +1 for residual connection
        # Reshape for broadcasting: (1,1,1,dim_out,B)
        x = x .* reshape(scale, 1, 1, 1, layer.dim_out, :)
    end

    # Block 2: mp_silu → dropout → conv2
    x = mp_silu(x)
    # Dropout: scale by 1/(1-p) during training (approx via random mask)
    # For inference-only code, we skip dropout; for training, apply stochastic mask
    if layer.dropout > 0
        # Deterministic for now — dropout applied externally or via training flag
        # In production, add a training flag. For now, no-op.
    end
    x = dist_conv3d_forward(x, ps.conv2.weight, nothing, layer.conv2; auto_pad=true)

    # Magnitude-preserving residual
    x = mp_add(x, res; t=layer.mp_add_t)

    return x
end

# ─── Karras Decoder ──────────────────────────────────────────────────────────

"""
    DistKarrasDecoder

Karras-style decoder block with magnitude-preserving operations.

Forward:
    if upsample: trilinear_interp(×2)
    res = res_conv(x)  (1×1 if dim_in ≠ dim_out, else identity)
    mp_silu → conv1
    if has_emb: x *= (linear(emb) * gain + 1)
    mp_silu → dropout → conv2
    x = mp_add(x, res; t)

# Fields
- `conv1`, `conv2`: 3×3×3 weight-normed convolutions
- `res_conv`: 1×1 conv if dim_in ≠ dim_out, else nothing (identity)
- `dim_in`, `dim_out`: Channel dimensions
- `upsample`: Whether to upsample (×2)
- `needs_skip`: Whether this block expects a skip connection input
- `has_emb`: Whether this block uses time embedding
"""
struct DistKarrasDecoder
    conv1::DistConv3d
    conv2::DistConv3d
    res_conv::Union{DistConv3d, Nothing}  # 1×1 if dim_in != dim_out
    dim_in::Int
    dim_out::Int
    upsample::Bool
    needs_skip::Bool
    has_emb::Bool
    emb_dim::Int
    mp_add_t::Float32
    dropout::Float32
end

function DistKarrasDecoder(
    P::CartesianPartition,
    dim_in::Int,
    dim_out::Int;
    emb_dim::Int = 0,
    upsample::Bool = false,
    mp_add_t::Float32 = 0.3f0,
    dropout::Float32 = 0.1f0
)
    has_emb = emb_dim > 0
    needs_skip = !upsample

    conv1 = DistConv3d(P, dim_in, dim_out, (3, 3, 3); bias=false, weight_norm=true)
    conv2 = DistConv3d(P, dim_out, dim_out, (3, 3, 3); bias=false, weight_norm=true)

    res_conv = (dim_in != dim_out) ?
        DistConv3d(P, dim_in, dim_out, (1, 1, 1); bias=false, weight_norm=true) :
        nothing

    return DistKarrasDecoder(conv1, conv2, res_conv, dim_in, dim_out,
                             upsample, needs_skip, has_emb, emb_dim,
                             mp_add_t, dropout)
end

function init_dist_karras_decoder(rng, layer::DistKarrasDecoder; T::Type=Float32)
    ps = (
        conv1 = init_dist_conv3d(rng, layer.conv1; T=T),
        conv2 = init_dist_conv3d(rng, layer.conv2; T=T),
    )

    if layer.res_conv !== nothing
        ps = merge(ps, (res_conv = init_dist_conv3d(rng, layer.res_conv; T=T),))
    end

    if layer.has_emb
        emb_weight = randn(rng, T, layer.dim_out, layer.emb_dim)
        emb_weight = normalize_weight(emb_weight) ./ sqrt(T(layer.emb_dim))
        ps = merge(ps, (
            emb_linear_weight = emb_weight,
            emb_gain = zeros(T, 1),
        ))
    end

    return ps
end

"""
    dist_karras_decoder_forward(x, emb, ps, layer::DistKarrasDecoder)

Forward pass matching Python Decoder.forward.
"""
function dist_karras_decoder_forward(
    x::AbstractArray{T, 5},
    emb::Union{AbstractArray{T, 2}, Nothing},
    ps::NamedTuple,
    layer::DistKarrasDecoder
) where {T}
    # Upsample if needed: nearest-neighbor ×2
    if layer.upsample
        W, H, D, C, B = size(x)
        x_up = similar(x, 2W, 2H, 2D, C, B)
        for b in 1:B, c in 1:C, k in 1:D, j in 1:H, i in 1:W
            val = x[i, j, k, c, b]
            x_up[2i-1, 2j-1, 2k-1, c, b] = val
            x_up[2i,   2j-1, 2k-1, c, b] = val
            x_up[2i-1, 2j,   2k-1, c, b] = val
            x_up[2i,   2j,   2k-1, c, b] = val
            x_up[2i-1, 2j-1, 2k,   c, b] = val
            x_up[2i,   2j-1, 2k,   c, b] = val
            x_up[2i-1, 2j,   2k,   c, b] = val
            x_up[2i,   2j,   2k,   c, b] = val
        end
        x = x_up
    end

    # Residual path
    if layer.res_conv !== nothing
        res = dist_conv3d_forward(x, ps.res_conv.weight, nothing, layer.res_conv; auto_pad=true)
    else
        res = x
    end

    # Block 1: mp_silu → conv1
    x = mp_silu(x)
    x = dist_conv3d_forward(x, ps.conv1.weight, nothing, layer.conv1; auto_pad=true)

    # FiLM conditioning
    if layer.has_emb && emb !== nothing
        w_emb = normalize_weight(ps.emb_linear_weight) ./ sqrt(T(layer.emb_dim))
        scale = w_emb * emb
        scale = scale .* ps.emb_gain .+ one(T)
        x = x .* reshape(scale, 1, 1, 1, layer.dim_out, :)
    end

    # Block 2: mp_silu → dropout → conv2
    x = mp_silu(x)
    x = dist_conv3d_forward(x, ps.conv2.weight, nothing, layer.conv2; auto_pad=true)

    # Magnitude-preserving residual
    x = mp_add(x, res; t=layer.mp_add_t)

    return x
end

# ═══════════════════════════════════════════════════════════════════════════════
# DistKarrasUNet3d — Complete Karras-style U-Net
# Reference: networks_3d.py KarrasUnet3D class
# ═══════════════════════════════════════════════════════════════════════════════

"""
    DistKarrasUNet3d

Complete domain-decomposed Karras-style 3D U-Net.

Architecture mirrors KarrasUnet3D from networks_3d.py:
- Input block: 3×3 conv with concat_ones, weight-normed
- Encoder: flat list of DistKarrasEncoder blocks
- Middle: 2 DistKarrasDecoder blocks at bottleneck
- Decoder: flat list of DistKarrasDecoder blocks with skip connections
- Output block: 3×3 conv + learnable Gain

# Fields
- `downs`: Vector of encoder blocks
- `mids`: Vector of middle (bottleneck) decoder blocks
- `ups`: Vector of decoder blocks
- `input_block`: Initial DistConv3d (concat_ones=true, weight_norm=true)
- `output_conv`: Final DistConv3d (weight_norm=true)
- `partition`: Spatial partition at full resolution
- `dim`: Base channel dimension
- `emb_dim`: Embedding dimension (4 * dim)
- `fourier_dim`: Fourier embedding dimension
- `mp_cat_t`: MPCat t parameter for skip connections
- `num_downsamples`: Number of downsampling stages
"""
struct DistKarrasUNet3d
    downs::Vector{DistKarrasEncoder}
    mids::Vector{DistKarrasDecoder}
    ups::Vector{DistKarrasDecoder}
    input_block::DistConv3d
    output_conv::DistConv3d
    partition::CartesianPartition
    dim::Int
    emb_dim::Int
    fourier_dim::Int
    mp_cat_t::Float32
    num_downsamples::Int
    in_channels::Int
    out_channels::Int
end

"""
    DistKarrasUNet3d(P, in_channels, out_channels; kwargs...)

Construct a Karras-style distributed U-Net.

# Keyword arguments
- `dim`: Base channel dimension (default 8)
- `dim_max`: Maximum channel dimension (default 32)
- `num_downsamples`: Number of downsampling stages (default 2)
- `num_blocks_per_stage`: Blocks per stage (default (1,1))
- `fourier_dim`: Fourier embedding dimension (default 16)
- `mp_cat_t`: MPCat t parameter (default 0.5)
- `mp_add_t`: MPAdd t parameter for residuals (default 0.3)
- `dropout`: Dropout probability (default 0.1)
"""
function DistKarrasUNet3d(
    P::CartesianPartition,
    in_channels::Int,
    out_channels::Int;
    dim::Int = 8,
    dim_max::Int = 32,
    num_downsamples::Int = 2,
    num_blocks_per_stage::Tuple{Vararg{Int}} = ntuple(_ -> 1, num_downsamples),
    fourier_dim::Int = 16,
    mp_cat_t::Float32 = 0.5f0,
    mp_add_t::Float32 = 0.3f0,
    dropout::Float32 = 0.1f0
)
    emb_dim = dim * 4

    # Input block: 3×3 conv with concat_ones
    input_block = DistConv3d(P, in_channels, dim, (3, 3, 3);
                             bias=false, weight_norm=true, concat_ones=true)

    # Output block: 3×3 conv
    output_conv = DistConv3d(P, dim, out_channels, (3, 3, 3);
                             bias=false, weight_norm=true)

    # Build encoder/decoder lists matching Python reference
    downs = DistKarrasEncoder[]
    ups = DistKarrasDecoder[]

    curr_dim = dim

    # First decoder to handle initial skip (dim*2 → dim because of MPCat doubling)
    pushfirst!(ups, DistKarrasDecoder(P, dim * 2, dim;
                                       emb_dim=emb_dim, mp_add_t=mp_add_t, dropout=dropout))

    # Expand num_blocks_per_stage to include initial stage
    # Python: if len == num_downsamples, prepend first element
    all_blocks = (num_blocks_per_stage[1], num_blocks_per_stage...)
    # all_blocks now has num_downsamples + 1 entries

    init_blocks = all_blocks[1]
    rest_blocks = all_blocks[2:end]

    # Initial stage blocks (no downsample)
    for _ in 1:init_blocks
        push!(downs, DistKarrasEncoder(P, curr_dim, curr_dim;
                                       emb_dim=emb_dim, mp_add_t=mp_add_t, dropout=dropout))
        pushfirst!(ups, DistKarrasDecoder(P, curr_dim * 2, curr_dim;
                                          emb_dim=emb_dim, mp_add_t=mp_add_t, dropout=dropout))
    end

    # Downsample stages
    for (_, stage_blocks) in enumerate(rest_blocks)
        dim_out = min(dim_max, curr_dim * 2)

        # Downsample encoder
        push!(downs, DistKarrasEncoder(P, curr_dim, dim_out;
                                       emb_dim=emb_dim, downsample=true,
                                       mp_add_t=mp_add_t, dropout=dropout))

        # Matching upsample decoder (prepended)
        pushfirst!(ups, DistKarrasDecoder(P, dim_out, curr_dim;
                                          emb_dim=emb_dim, upsample=true,
                                          mp_add_t=mp_add_t, dropout=dropout))
        # First decoder after upsample handles skip
        pushfirst!(ups, DistKarrasDecoder(P, dim_out * 2, dim_out;
                                          emb_dim=emb_dim, mp_add_t=mp_add_t, dropout=dropout))

        # Stage blocks (no downsample)
        for _ in 1:stage_blocks
            push!(downs, DistKarrasEncoder(P, dim_out, dim_out;
                                           emb_dim=emb_dim, mp_add_t=mp_add_t, dropout=dropout))
            pushfirst!(ups, DistKarrasDecoder(P, dim_out * 2, dim_out;
                                              emb_dim=emb_dim, mp_add_t=mp_add_t, dropout=dropout))
        end

        curr_dim = dim_out
    end

    # Middle blocks (2 decoders at bottleneck, no skip)
    mids = [
        DistKarrasDecoder(P, curr_dim, curr_dim;
                          emb_dim=emb_dim, upsample=false, mp_add_t=mp_add_t, dropout=dropout),
        DistKarrasDecoder(P, curr_dim, curr_dim;
                          emb_dim=emb_dim, upsample=false, mp_add_t=mp_add_t, dropout=dropout),
    ]
    # Mark mids as not needing skip (they have upsample=false but needs_skip should be false)
    # In the Decoder constructor, needs_skip = !upsample, so mids would have needs_skip=true.
    # But mids should NOT use skip. We handle this in the forward pass by not providing skips.

    return DistKarrasUNet3d(downs, mids, ups, input_block, output_conv,
                            P, dim, emb_dim, fourier_dim, mp_cat_t,
                            num_downsamples, in_channels, out_channels)
end

"""
Initialize all parameters for DistKarrasUNet3d.
Returns a nested NamedTuple compatible with Zygote.
"""
function init_dist_karras_unet(rng, model::DistKarrasUNet3d; T::Type=Float32)
    # Fourier embedding frequencies (fixed, not learned)
    half_dim = model.fourier_dim ÷ 2
    fourier_freqs = randn(rng, T, half_dim)

    # Time embedding linear: fourier_dim → emb_dim
    time_linear_weight = randn(rng, T, model.emb_dim, model.fourier_dim)
    time_linear_weight = normalize_weight(time_linear_weight) ./ sqrt(T(model.fourier_dim))

    # Input block
    input_block_ps = init_dist_conv3d(rng, model.input_block; T=T)

    # Output block + gain
    output_conv_ps = init_dist_conv3d(rng, model.output_conv; T=T)
    output_gain = zeros(T, 1)  # Gain init to 0

    # Encoder blocks
    downs_ps = ntuple(i -> init_dist_karras_encoder(rng, model.downs[i]; T=T), length(model.downs))

    # Middle blocks
    mids_ps = ntuple(i -> init_dist_karras_decoder(rng, model.mids[i]; T=T), length(model.mids))

    # Decoder blocks
    ups_ps = ntuple(i -> init_dist_karras_decoder(rng, model.ups[i]; T=T), length(model.ups))

    return (
        fourier_freqs = fourier_freqs,
        time_linear_weight = time_linear_weight,
        input_block = input_block_ps,
        output_conv = output_conv_ps,
        output_gain = output_gain,
        downs = downs_ps,
        mids = mids_ps,
        ups = ups_ps,
    )
end

"""
    dist_karras_unet_forward(x, time, ps, model::DistKarrasUNet3d)

Forward pass matching KarrasUnet3D.forward.

`x`: (W,H,D,C_in,B) input tensor
`time`: (B,) noise level / time conditioning
`ps`: NamedTuple from init_dist_karras_unet
"""
function dist_karras_unet_forward(
    x::AbstractArray{T, 5},
    time::AbstractVector{T},
    ps::NamedTuple,
    model::DistKarrasUNet3d
) where {T}
    # Time embedding: Fourier → Linear → MPSiLU
    emb = mp_fourier_embedding(time, ps.fourier_freqs)  # (fourier_dim, B)
    w_time = normalize_weight(ps.time_linear_weight) ./ sqrt(T(model.fourier_dim))
    emb = w_time * emb  # (emb_dim, B)
    emb = mp_silu(emb)

    # Input block
    x = dist_conv3d_forward(x, ps.input_block.weight, nothing, model.input_block; auto_pad=true)

    # Skip connections
    skips = AbstractArray{T, 5}[x]

    # Encoder
    for (i, enc) in enumerate(model.downs)
        x = dist_karras_encoder_forward(x, emb, ps.downs[i], enc)
        push!(skips, x)
    end

    # Middle
    for (i, mid) in enumerate(model.mids)
        x = dist_karras_decoder_forward(x, emb, ps.mids[i], mid)
    end

    # Decoder with skip connections
    for (i, dec) in enumerate(model.ups)
        if dec.needs_skip
            skip = pop!(skips)
            x = mp_cat(x, skip; t=model.mp_cat_t, dim=4)
        end
        x = dist_karras_decoder_forward(x, emb, ps.ups[i], dec)
    end

    # Output block: conv + gain
    y = dist_conv3d_forward(x, ps.output_conv.weight, nothing, model.output_conv; auto_pad=true)
    y = y .* ps.output_gain

    return y
end

# ═══════════════════════════════════════════════════════════════════════════════
# EDM Preconditioner
# Reference: networks_3d.py EDMPrecond class
# ═══════════════════════════════════════════════════════════════════════════════

"""
    edm_precond_forward(x, sigma, ps, model; sigma_data=0.5f0)

EDM preconditioner (Karras et al. 2022).

Applies sigma-dependent scaling to input and output:
    c_skip = σ_d² / (σ² + σ_d²)
    c_out  = σ · σ_d / √(σ² + σ_d²)
    c_in   = 1 / √(σ_d² + σ²)
    c_noise = log(σ) / 4
    D(x, σ) = c_skip · x + c_out · F_θ(c_in · x, c_noise)

`x`: (W,H,D,C,B) input
`sigma`: (B,) noise levels
`ps`: model parameters
`model`: DistKarrasUNet3d
"""
function edm_precond_forward(
    x::AbstractArray{T, 5},
    sigma::AbstractVector{T},
    ps::NamedTuple,
    model::DistKarrasUNet3d;
    sigma_data::T = T(0.5)
) where {T}
    B = size(x, 5)

    # Compute per-sample scaling factors
    σ² = sigma .^ 2
    σ_d² = sigma_data^2

    c_skip = σ_d² ./ (σ² .+ σ_d²)  # (B,)
    c_out  = sigma .* sigma_data ./ sqrt.(σ² .+ σ_d²)  # (B,)
    c_in   = one(T) ./ sqrt.(σ_d² .+ σ²)  # (B,)
    c_noise = log.(sigma) ./ T(4)  # (B,)

    # Scale input
    x_scaled = x .* reshape(c_in, 1, 1, 1, 1, B)

    # Run U-Net
    F_x = dist_karras_unet_forward(x_scaled, c_noise, ps, model)

    # Preconditioned output
    D_x = x .* reshape(c_skip, 1, 1, 1, 1, B) .+
          F_x .* reshape(c_out, 1, 1, 1, 1, B)

    return D_x
end
