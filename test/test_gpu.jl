# GPU test suite for DomainDecomposition.jl
#
# Requires 8 MPI ranks with one NVIDIA GPU per rank.
# Run: mpiexecjl -n 8 julia --project test/test_gpu.jl
#
# Tests verify that all domain-decomposed operations produce identical results
# on GPU (CuArray) and CPU (Array), using a (2,2,2) spatial partition.

using Test
using MPI
using Random
using NNlib
using DomainDecomposition

# ─── Conditional CUDA import ─────────────────────────────────────────────────

const CUDA_AVAILABLE = try
    using CUDA
    CUDA.functional()
catch
    false
end

if !CUDA_AVAILABLE
    @info "CUDA not available or not functional — skipping GPU tests"
    exit(0)
end

using CUDA

# ─── MPI + GPU setup ─────────────────────────────────────────────────────────

MPI.Initialized() || MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)

if nprocs < 8
    rank == 0 && @info "GPU tests require 8 MPI ranks, got $nprocs — skipping"
    MPI.Finalize()
    exit(0)
end

# Bind each rank to a GPU via local rank
local_rank = rank % length(CUDA.devices())
CUDA.device!(local_rank)
rank == 0 && @info "GPU tests: 8 ranks on $(length(CUDA.devices())) GPUs"

# ─── Helpers ──────────────────────────────────────────────────────────────────

to_gpu(x::AbstractArray) = CuArray(x)
to_cpu(x::CuArray) = Array(x)
to_cpu(x::AbstractArray) = x

function verify_gpu(x)
    @test x isa CuArray
end

"""Recursively transfer all array parameters to GPU."""
_params_to_gpu(x::AbstractArray) = to_gpu(x)
_params_to_gpu(x::Nothing) = nothing
_params_to_gpu(x::Number) = x
_params_to_gpu(x::NamedTuple) = NamedTuple{keys(x)}(map(_params_to_gpu, values(x)))
_params_to_gpu(x::Tuple) = map(_params_to_gpu, x)

# ═══════════════════════════════════════════════════════════════════════════════
# Layer 1: MPI Primitives on CuArrays
# ═══════════════════════════════════════════════════════════════════════════════

@testset "GPU: Halo Exchange" begin
    P = create_cartesian_topology(comm, (2, 2, 2))

    if P.active
        halo_sizes = [(1, 1), (1, 1), (1, 1)]
        info = DomainDecomposition.compute_halo_info(P, halo_sizes)

        # Create deterministic input on CPU, transfer to GPU
        rng = MersenneTwister(42 + rank)
        W, H, D, C, B = 6, 6, 6, 2, 1  # 4 bulk + 2 halo per spatial dim
        x_cpu = randn(rng, Float32, W, H, D, C, B)
        x_gpu = to_gpu(x_cpu)

        # CPU reference
        y_cpu = halo_exchange(x_cpu, info)

        # GPU execution
        y_gpu = halo_exchange(x_gpu, info)
        verify_gpu(y_gpu)

        # Compare
        @test to_cpu(y_gpu) ≈ y_cpu atol=1e-5
    end
end

@testset "GPU: AllReduce" begin
    P = create_cartesian_topology(comm, (2, 2, 2))

    if P.active
        ar_info = setup_all_reduce(P)

        rng = MersenneTwister(100 + rank)
        x_cpu = randn(rng, Float32, 4, 3)
        x_gpu = to_gpu(x_cpu)

        # CPU reference
        y_cpu = all_reduce_op(x_cpu, ar_info)

        # GPU execution
        y_gpu = all_reduce_op(x_gpu, ar_info)
        verify_gpu(y_gpu)

        @test to_cpu(y_gpu) ≈ y_cpu atol=1e-5
    end
end

@testset "GPU: Repartition" begin
    # Repartition from (2,2,2) to (4,2,1) — tests redistribution on GPU
    P_src = create_cartesian_topology(comm, (2, 2, 2))
    P_dst = create_cartesian_topology(comm, (4, 2, 1))

    global_shape = (16, 16, 16, 2, 1)
    repart_info = setup_repartition(P_src, P_dst, global_shape, comm)

    # Build deterministic global tensor, slice for src partition
    rng_global = MersenneTwister(777)
    x_global = randn(rng_global, Float32, global_shape...)

    if P_src.active
        src_ranges = subtensor_indices(global_shape, P_src.dims, P_src.coords)
        x_local_cpu = x_global[src_ranges...]
    else
        x_local_cpu = zeros(Float32, ntuple(_ -> 0, 5)...)
    end

    x_local_gpu = to_gpu(x_local_cpu)

    # CPU reference
    y_cpu = repartition_op(x_local_cpu, repart_info)

    # GPU execution
    y_gpu = repartition_op(x_local_gpu, repart_info)

    if P_dst.active
        verify_gpu(y_gpu)
        @test to_cpu(y_gpu) ≈ y_cpu atol=1e-5
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Layer 2: NN Layers on CuArrays
# ═══════════════════════════════════════════════════════════════════════════════

@testset "GPU: mp_ops" begin
    rng = MersenneTwister(42)
    x_cpu = randn(rng, Float32, 4, 4, 4, 4, 1)
    res_cpu = randn(MersenneTwister(43), Float32, 4, 4, 4, 4, 1)
    x_gpu = to_gpu(x_cpu)
    res_gpu = to_gpu(res_cpu)

    @testset "mp_silu" begin
        y_cpu = mp_silu(x_cpu)
        y_gpu = mp_silu(x_gpu)
        verify_gpu(y_gpu)
        @test to_cpu(y_gpu) ≈ y_cpu atol=1e-5
    end

    @testset "mp_add" begin
        y_cpu = mp_add(x_cpu, res_cpu; t=0.3f0)
        y_gpu = mp_add(x_gpu, res_gpu; t=0.3f0)
        verify_gpu(y_gpu)
        @test to_cpu(y_gpu) ≈ y_cpu atol=1e-5
    end

    @testset "pixel_norm" begin
        y_cpu = pixel_norm(x_cpu; dim=4)
        y_gpu = pixel_norm(x_gpu; dim=4)
        verify_gpu(y_gpu)
        @test to_cpu(y_gpu) ≈ y_cpu atol=1e-5
    end

    @testset "normalize_weight" begin
        w_cpu = randn(MersenneTwister(44), Float32, 3, 3, 3, 2, 4)
        w_gpu = to_gpu(w_cpu)
        y_cpu = normalize_weight(w_cpu)
        y_gpu = normalize_weight(w_gpu)
        verify_gpu(y_gpu)
        @test to_cpu(y_gpu) ≈ y_cpu atol=1e-5
    end

    @testset "mp_fourier_embedding" begin
        t_cpu = randn(MersenneTwister(45), Float32, 2)
        freqs_cpu = randn(MersenneTwister(46), Float32, 8)
        t_gpu = to_gpu(t_cpu)
        freqs_gpu = to_gpu(freqs_cpu)
        y_cpu = mp_fourier_embedding(t_cpu, freqs_cpu)
        y_gpu = mp_fourier_embedding(t_gpu, freqs_gpu)
        verify_gpu(y_gpu)
        @test to_cpu(y_gpu) ≈ y_cpu atol=1e-5
    end
end

@testset "GPU: DistConv3d" begin
    P = create_cartesian_topology(comm, (2, 2, 2))

    if P.active
        C_in, C_out = 2, 4
        kernel = (3, 3, 3)
        global_sp = (16, 16, 16)
        B = 1

        layer = DistConv3d(P, C_in, C_out, kernel; bias=false, weight_norm=true)

        rng = MersenneTwister(42)
        weight_cpu = randn(rng, Float32, kernel..., C_in, C_out)

        # Build unpadded local input
        x_global = randn(MersenneTwister(100), Float32, global_sp..., C_in, B)
        local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
        x_local_cpu = x_global[local_ranges..., :, :]

        # CPU reference
        y_cpu = dist_conv3d_forward(x_local_cpu, weight_cpu, nothing, layer; auto_pad=true)

        # GPU execution
        x_local_gpu = to_gpu(x_local_cpu)
        weight_gpu = to_gpu(weight_cpu)
        y_gpu = dist_conv3d_forward(x_local_gpu, weight_gpu, nothing, layer; auto_pad=true)

        verify_gpu(y_gpu)
        @test to_cpu(y_gpu) ≈ y_cpu atol=1e-4

        @testset "with concat_ones" begin
            layer_co = DistConv3d(P, C_in, C_out, kernel;
                                   bias=false, weight_norm=true, concat_ones=true)
            weight_co_cpu = randn(MersenneTwister(200), Float32, kernel..., C_in + 1, C_out)
            weight_co_gpu = to_gpu(weight_co_cpu)

            y_cpu2 = dist_conv3d_forward(x_local_cpu, weight_co_cpu, nothing, layer_co; auto_pad=true)
            y_gpu2 = dist_conv3d_forward(x_local_gpu, weight_co_gpu, nothing, layer_co; auto_pad=true)

            verify_gpu(y_gpu2)
            @test to_cpu(y_gpu2) ≈ y_cpu2 atol=1e-4
        end
    end
end

@testset "GPU: DistKarrasEncoder" begin
    P = create_cartesian_topology(comm, (2, 2, 2))

    if P.active
        dim, emb_dim, B = 8, 32, 1
        global_sp = (16, 16, 16)

        @testset "basic encoder (no downsample)" begin
            enc = DistKarrasEncoder(P, dim, dim; emb_dim=emb_dim)

            rng = MersenneTwister(42)
            ps_cpu = init_dist_karras_encoder(rng, enc; T=Float32)

            x_global = randn(MersenneTwister(100), Float32, global_sp..., dim, B)
            local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
            x_local_cpu = x_global[local_ranges..., :, :]
            emb_cpu = randn(MersenneTwister(200), Float32, emb_dim, B)

            # CPU reference
            y_cpu = dist_karras_encoder_forward(x_local_cpu, emb_cpu, ps_cpu, enc)

            # GPU
            x_gpu = to_gpu(x_local_cpu)
            emb_gpu = to_gpu(emb_cpu)
            ps_gpu = _params_to_gpu(ps_cpu)

            y_gpu = dist_karras_encoder_forward(x_gpu, emb_gpu, ps_gpu, enc)
            verify_gpu(y_gpu)
            @test to_cpu(y_gpu) ≈ y_cpu atol=1e-3
        end

        @testset "encoder with downsample" begin
            enc = DistKarrasEncoder(P, dim, dim * 2; emb_dim=emb_dim, downsample=true)

            rng = MersenneTwister(42)
            ps_cpu = init_dist_karras_encoder(rng, enc; T=Float32)

            x_global = randn(MersenneTwister(100), Float32, global_sp..., dim, B)
            local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
            x_local_cpu = x_global[local_ranges..., :, :]
            emb_cpu = randn(MersenneTwister(200), Float32, emb_dim, B)

            y_cpu = dist_karras_encoder_forward(x_local_cpu, emb_cpu, ps_cpu, enc)

            x_gpu = to_gpu(x_local_cpu)
            emb_gpu = to_gpu(emb_cpu)
            ps_gpu = _params_to_gpu(ps_cpu)

            y_gpu = dist_karras_encoder_forward(x_gpu, emb_gpu, ps_gpu, enc)
            verify_gpu(y_gpu)
            @test size(to_cpu(y_gpu)) == size(y_cpu)
            @test to_cpu(y_gpu) ≈ y_cpu atol=1e-3
        end
    end
end

@testset "GPU: DistKarrasDecoder" begin
    P = create_cartesian_topology(comm, (2, 2, 2))

    if P.active
        dim, emb_dim, B = 8, 32, 1
        global_sp = (16, 16, 16)

        @testset "basic decoder (no upsample)" begin
            dec = DistKarrasDecoder(P, dim * 2, dim; emb_dim=emb_dim)

            rng = MersenneTwister(42)
            ps_cpu = init_dist_karras_decoder(rng, dec; T=Float32)

            x_global = randn(MersenneTwister(100), Float32, global_sp..., dim * 2, B)
            local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
            x_local_cpu = x_global[local_ranges..., :, :]
            emb_cpu = randn(MersenneTwister(200), Float32, emb_dim, B)

            y_cpu = dist_karras_decoder_forward(x_local_cpu, emb_cpu, ps_cpu, dec)

            x_gpu = to_gpu(x_local_cpu)
            emb_gpu = to_gpu(emb_cpu)
            ps_gpu = _params_to_gpu(ps_cpu)

            y_gpu = dist_karras_decoder_forward(x_gpu, emb_gpu, ps_gpu, dec)
            verify_gpu(y_gpu)
            @test to_cpu(y_gpu) ≈ y_cpu atol=1e-3
        end

        @testset "decoder with upsample" begin
            dec = DistKarrasDecoder(P, dim, dim; emb_dim=emb_dim, upsample=true)

            rng = MersenneTwister(42)
            ps_cpu = init_dist_karras_decoder(rng, dec; T=Float32)

            coarse_sp = (8, 8, 8)
            x_global = randn(MersenneTwister(100), Float32, coarse_sp..., dim, B)
            local_ranges = subtensor_indices(coarse_sp, P.dims, P.coords)
            x_local_cpu = x_global[local_ranges..., :, :]
            emb_cpu = randn(MersenneTwister(200), Float32, emb_dim, B)

            y_cpu = dist_karras_decoder_forward(x_local_cpu, emb_cpu, ps_cpu, dec)

            x_gpu = to_gpu(x_local_cpu)
            emb_gpu = to_gpu(emb_cpu)
            ps_gpu = _params_to_gpu(ps_cpu)

            y_gpu = dist_karras_decoder_forward(x_gpu, emb_gpu, ps_gpu, dec)
            verify_gpu(y_gpu)
            @test size(to_cpu(y_gpu)) == size(y_cpu)
            @test to_cpu(y_gpu) ≈ y_cpu atol=1e-3
        end
    end
end

@testset "GPU: DistKarrasUNet3d" begin
    P = create_cartesian_topology(comm, (2, 2, 2))

    if P.active
        model = DistKarrasUNet3d(P, 2, 2;
            dim=8, dim_max=16, num_downsamples=2,
            num_blocks_per_stage=(1, 1),
            fourier_dim=8)

        rng = MersenneTwister(42)
        ps_cpu = init_dist_karras_unet(rng, model; T=Float32)

        global_sp = (32, 32, 32)
        B = 1

        x_global = randn(MersenneTwister(100), Float32, global_sp..., 2, B)
        local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
        x_local_cpu = x_global[local_ranges..., :, :]
        time_cpu = Float32[0.5]

        # CPU reference
        y_cpu = dist_karras_unet_forward(x_local_cpu, time_cpu, ps_cpu, model)

        # GPU
        x_gpu = to_gpu(x_local_cpu)
        time_gpu = to_gpu(time_cpu)
        ps_gpu = _params_to_gpu(ps_cpu)

        y_gpu = dist_karras_unet_forward(x_gpu, time_gpu, ps_gpu, model)
        verify_gpu(y_gpu)
        @test size(to_cpu(y_gpu)) == size(y_cpu)
        @test to_cpu(y_gpu) ≈ y_cpu atol=1e-2
    end
end

@testset "GPU: EDM Preconditioner" begin
    P = create_cartesian_topology(comm, (2, 2, 2))

    if P.active
        model = DistKarrasUNet3d(P, 2, 2;
            dim=8, dim_max=16, num_downsamples=2,
            num_blocks_per_stage=(1, 1),
            fourier_dim=8)

        rng = MersenneTwister(42)
        ps_cpu = init_dist_karras_unet(rng, model; T=Float32)

        global_sp = (32, 32, 32)
        B = 1
        x_global = randn(MersenneTwister(100), Float32, global_sp..., 2, B)
        local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
        x_local_cpu = x_global[local_ranges..., :, :]
        sigma_cpu = Float32[1.0]

        # CPU reference
        D_cpu = edm_precond_forward(x_local_cpu, sigma_cpu, ps_cpu, model)

        # GPU
        x_gpu = to_gpu(x_local_cpu)
        sigma_gpu = to_gpu(sigma_cpu)
        ps_gpu = _params_to_gpu(ps_cpu)

        D_gpu = edm_precond_forward(x_gpu, sigma_gpu, ps_gpu, model)
        verify_gpu(D_gpu)
        @test size(to_cpu(D_gpu)) == size(D_cpu)
        @test to_cpu(D_gpu) ≈ D_cpu atol=1e-2
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# AD: Zygote gradient on GPU
# ═══════════════════════════════════════════════════════════════════════════════

ad_available = try
    using Zygote
    true
catch
    rank == 0 && @info "Zygote not available, skipping GPU AD tests"
    false
end

if ad_available

@testset "GPU: Zygote AD — DistConv3d" begin
    P = create_cartesian_topology(comm, (2, 2, 2))

    if P.active
        C_in, C_out = 2, 4
        global_sp = (16, 16, 16)
        B = 1

        layer = DistConv3d(P, C_in, C_out, (3, 3, 3);
                           bias=false, weight_norm=true)

        rng = MersenneTwister(42)
        weight_cpu = randn(rng, Float32, 3, 3, 3, C_in, C_out)

        x_global = randn(MersenneTwister(100), Float32, global_sp..., C_in, B)
        local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
        x_local_cpu = x_global[local_ranges..., :, :]

        # CPU gradient
        g_cpu = Zygote.gradient(
            w -> sum(dist_conv3d_forward(x_local_cpu, w, nothing, layer; auto_pad=true)),
            weight_cpu
        )[1]

        # GPU gradient
        x_gpu = to_gpu(x_local_cpu)
        weight_gpu = to_gpu(weight_cpu)
        g_gpu = Zygote.gradient(
            w -> sum(dist_conv3d_forward(x_gpu, w, nothing, layer; auto_pad=true)),
            weight_gpu
        )[1]

        verify_gpu(g_gpu)
        @test to_cpu(g_gpu) ≈ g_cpu atol=1e-3
    end
end

@testset "GPU: Zygote AD — mp_ops" begin
    x_cpu = randn(Float32, 4, 4, 4, 4, 1)
    x_gpu = to_gpu(x_cpu)

    @testset "mp_silu gradient" begin
        g_cpu = Zygote.gradient(x -> sum(mp_silu(x)), x_cpu)[1]
        g_gpu = Zygote.gradient(x -> sum(mp_silu(x)), x_gpu)[1]
        verify_gpu(g_gpu)
        @test to_cpu(g_gpu) ≈ g_cpu atol=1e-5
    end

    @testset "pixel_norm gradient" begin
        g_cpu = Zygote.gradient(x -> sum(pixel_norm(x; dim=4)), x_cpu)[1]
        g_gpu = Zygote.gradient(x -> sum(pixel_norm(x; dim=4)), x_gpu)[1]
        verify_gpu(g_gpu)
        @test to_cpu(g_gpu) ≈ g_cpu atol=1e-4
    end

    @testset "normalize_weight gradient" begin
        w_cpu = randn(Float32, 3, 3, 3, 2, 4)
        w_gpu = to_gpu(w_cpu)
        g_cpu = Zygote.gradient(w -> sum(normalize_weight(w)), w_cpu)[1]
        g_gpu = Zygote.gradient(w -> sum(normalize_weight(w)), w_gpu)[1]
        verify_gpu(g_gpu)
        @test to_cpu(g_gpu) ≈ g_cpu atol=1e-4
    end
end

end # if ad_available

rank == 0 && @info "GPU tests completed"
