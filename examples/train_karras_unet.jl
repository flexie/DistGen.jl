# Training example for Domain-Decomposed Karras U-Net
#
# Run with: mpiexecjl -n 4 julia --project examples/train_karras_unet.jl
#
# Demonstrates:
# 1. EDM training (score-based diffusion with Karras preconditioning)
# 2. Flow matching training
# Both use the DistKarrasUNet3d with MPI domain decomposition.

using MPI
using Random
using DomainDecomposition
using Zygote

# ─── SGD helper (must be defined before use) ────────────────────────────────

function _sgd_update(ps::NamedTuple, grads::NamedTuple, lr)
    return NamedTuple{keys(ps)}(map(keys(ps)) do k
        p = ps[k]
        g = haskey(grads, k) ? grads[k] : nothing
        if g === nothing
            p
        elseif p isa NamedTuple && g isa NamedTuple
            _sgd_update(p, g, lr)
        elseif p isa Tuple && g isa Tuple
            Tuple(_sgd_update_elem(p[i], g[i], lr) for i in eachindex(p))
        elseif p isa AbstractArray && g isa AbstractArray
            p .- lr .* g
        else
            p
        end
    end)
end

function _sgd_update_elem(p, g, lr)
    if p isa NamedTuple && g isa NamedTuple
        _sgd_update(p, g, lr)
    elseif p isa AbstractArray && g isa AbstractArray
        p .- lr .* g
    else
        p
    end
end

# ─── Main ────────────────────────────────────────────────────────────────────

MPI.Initialized() || MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

@assert nprocs >= 4 "This example requires at least 4 MPI ranks"

# ─── Configuration ───────────────────────────────────────────────────────────

const FT = Float32
const global_spatial = (32, 32, 32)
const in_ch = 2
const out_ch = 2
const base_dim = 8
const max_dim = 16
const n_downsamples = 2
const fourier_d = 8
const n_train_steps = 10
const lr = FT(1e-3)

# ─── Setup ───────────────────────────────────────────────────────────────────

P = create_cartesian_topology(comm, (2, 2, 1))

if !P.active
    MPI.Finalize()
    exit(0)
end

rank == 0 && println("=" ^ 60)
rank == 0 && println("Domain-Decomposed Karras U-Net Training Example")
rank == 0 && println("=" ^ 60)
rank == 0 && println("  MPI ranks: $nprocs")
rank == 0 && println("  Global spatial: $global_spatial")
rank == 0 && println("  Channels: $in_ch → $out_ch")
rank == 0 && println("  Model dim: $base_dim, dim_max: $max_dim")
rank == 0 && println("  Downsamples: $n_downsamples")
rank == 0 && println()

# Build model
model = DistKarrasUNet3d(P, in_ch, out_ch;
    dim=base_dim, dim_max=max_dim, num_downsamples=n_downsamples,
    num_blocks_per_stage=(1, 1), fourier_dim=fourier_d)

# Initialize parameters (same RNG seed on all ranks)
rng = MersenneTwister(42)
ps = init_dist_karras_unet(rng, model; T=FT)

rank == 0 && println("Model constructed:")
rank == 0 && println("  Encoder blocks: $(length(model.downs))")
rank == 0 && println("  Middle blocks:  $(length(model.mids))")
rank == 0 && println("  Decoder blocks: $(length(model.ups))")
rank == 0 && println()

# ─── Helper: create local padded input ───────────────────────────────────────

function make_local_input(rng_seed, global_sp, channels, batch, partition)
    # Unpadded input — auto_pad in conv handles halo internally
    x_global = randn(MersenneTwister(rng_seed), FT, global_sp..., channels, batch)
    local_ranges = subtensor_indices(global_sp, partition.dims, partition.coords)
    return x_global[local_ranges..., :, :]
end

# ─── Part 1: EDM Training ───────────────────────────────────────────────────

rank == 0 && println("Part 1: EDM Preconditioning Training")
rank == 0 && println("-" ^ 40)

ps_edm = deepcopy(ps)
ps_edm = merge(ps_edm, (output_gain = FT[0.01],))

for step in 1:n_train_steps
    x_local = make_local_input(1000 + step, global_spatial, in_ch, 1, P)
    sigma = FT[0.1 + 0.9 * rand(MersenneTwister(2000 + step))]

    loss, grads = Zygote.withgradient(ps_edm) do p
        D_x = edm_precond_forward(x_local, sigma, p, model; sigma_data=FT(0.5))
        sum((D_x .- x_local) .^ 2) / length(D_x)
    end

    global ps_edm = _sgd_update(ps_edm, grads[1], lr)
    rank == 0 && println("  Step $step/$n_train_steps — EDM loss: $(round(loss; digits=6))")
end

rank == 0 && println()

# ─── Part 2: Flow Matching Training ─────────────────────────────────────────

rank == 0 && println("Part 2: Flow Matching Training")
rank == 0 && println("-" ^ 40)

ps_fm = deepcopy(ps)
ps_fm = merge(ps_fm, (output_gain = FT[0.01],))

for step in 1:n_train_steps
    x0_local = make_local_input(3000 + step, global_spatial, in_ch, 1, P)
    x1_local = make_local_input(4000 + step, global_spatial, in_ch, 1, P)
    t = FT(rand(MersenneTwister(5000 + step)))

    x_t = (one(FT) - t) .* x0_local .+ t .* x1_local
    target_v = x1_local .- x0_local

    loss, grads = Zygote.withgradient(ps_fm) do p
        v_pred = dist_karras_unet_forward(x_t, FT[t], p, model)
        sum((v_pred .- target_v) .^ 2) / length(v_pred)
    end

    global ps_fm = _sgd_update(ps_fm, grads[1], lr)
    rank == 0 && println("  Step $step/$n_train_steps — FM loss: $(round(loss; digits=6))")
end

rank == 0 && println()
rank == 0 && println("=" ^ 60)
rank == 0 && println("Training complete!")
rank == 0 && println("=" ^ 60)
