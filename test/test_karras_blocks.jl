using Test
using MPI
using NNlib
using Random
using DomainDecomposition

# ═══════════════════════════════════════════════════════════════════════════════
# Test: Karras Encoder Block
# ═══════════════════════════════════════════════════════════════════════════════

@testset "DistKarrasEncoder" begin
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if nprocs >= 4
        P = create_cartesian_topology(comm, (2, 2, 1))

        if P.active
            dim, emb_dim, B = 8, 32, 1
            global_sp = (8, 8, 8)

            @testset "basic encoder (no downsample)" begin
                enc = DistKarrasEncoder(P, dim, dim; emb_dim=emb_dim)

                rng = MersenneTwister(42)
                ps = init_dist_karras_encoder(rng, enc; T=Float64)

                # Build local input — UNPADDED (auto_pad handles it)
                x_global = randn(MersenneTwister(100), Float64, global_sp..., dim, B)
                local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
                local_sp = Tuple(length.(local_ranges))
                x_local = x_global[local_ranges..., :, :]

                emb = randn(MersenneTwister(200), Float64, emb_dim, B)

                y = dist_karras_encoder_forward(x_local, emb, ps, enc)

                @test ndims(y) == 5
                @test size(y, 4) == dim
                @test size(y, 5) == B
                @test size(y, 1) == local_sp[1]
                @test size(y, 2) == local_sp[2]
                @test size(y, 3) == local_sp[3]
            end

            @testset "FiLM changes output" begin
                enc = DistKarrasEncoder(P, dim, dim; emb_dim=emb_dim)

                rng = MersenneTwister(42)
                ps = init_dist_karras_encoder(rng, enc; T=Float64)
                ps = merge(ps, (emb_gain = [1.0],))

                x_global = randn(MersenneTwister(100), Float64, global_sp..., dim, B)
                local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
                x_local = x_global[local_ranges..., :, :]

                emb1 = ones(Float64, emb_dim, B)
                emb2 = 2.0 .* ones(Float64, emb_dim, B)

                y1 = dist_karras_encoder_forward(x_local, emb1, ps, enc)
                y2 = dist_karras_encoder_forward(x_local, emb2, ps, enc)
                @test !(y1 ≈ y2)
            end

            @testset "encoder with downsample" begin
                enc = DistKarrasEncoder(P, dim, dim * 2; emb_dim=emb_dim, downsample=true)
                rng = MersenneTwister(42)
                ps = init_dist_karras_encoder(rng, enc; T=Float64)

                x_global = randn(MersenneTwister(100), Float64, global_sp..., dim, B)
                local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
                local_sp = Tuple(length.(local_ranges))
                x_local = x_global[local_ranges..., :, :]

                emb = randn(MersenneTwister(200), Float64, emb_dim, B)
                y = dist_karras_encoder_forward(x_local, emb, ps, enc)

                @test size(y, 1) == local_sp[1] ÷ 2
                @test size(y, 2) == local_sp[2] ÷ 2
                @test size(y, 3) == local_sp[3] ÷ 2
                @test size(y, 4) == dim * 2
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Test: Karras Decoder Block
# ═══════════════════════════════════════════════════════════════════════════════

@testset "DistKarrasDecoder" begin
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if nprocs >= 4
        P = create_cartesian_topology(comm, (2, 2, 1))

        if P.active
            dim, emb_dim, B = 8, 32, 1
            global_sp = (8, 8, 8)

            @testset "basic decoder (needs_skip, no upsample)" begin
                dec = DistKarrasDecoder(P, dim * 2, dim; emb_dim=emb_dim)
                @test dec.needs_skip == true

                rng = MersenneTwister(42)
                ps = init_dist_karras_decoder(rng, dec; T=Float64)

                # Unpadded input
                x_global = randn(MersenneTwister(100), Float64, global_sp..., dim * 2, B)
                local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
                local_sp = Tuple(length.(local_ranges))
                x_local = x_global[local_ranges..., :, :]

                emb = randn(MersenneTwister(200), Float64, emb_dim, B)
                y = dist_karras_decoder_forward(x_local, emb, ps, dec)

                @test size(y, 4) == dim
                @test size(y, 1) == local_sp[1]
            end

            @testset "decoder with upsample" begin
                dec = DistKarrasDecoder(P, dim, dim; emb_dim=emb_dim, upsample=true)
                @test dec.needs_skip == false

                rng = MersenneTwister(42)
                ps = init_dist_karras_decoder(rng, dec; T=Float64)

                coarse_sp = (4, 4, 4)
                x_global = randn(MersenneTwister(100), Float64, coarse_sp..., dim, B)
                local_ranges = subtensor_indices(coarse_sp, P.dims, P.coords)
                local_sp = Tuple(length.(local_ranges))
                x_local = x_global[local_ranges..., :, :]

                emb = randn(MersenneTwister(200), Float64, emb_dim, B)
                y = dist_karras_decoder_forward(x_local, emb, ps, dec)

                @test size(y, 1) == local_sp[1] * 2
                @test size(y, 2) == local_sp[2] * 2
                @test size(y, 3) == local_sp[3] * 2
                @test size(y, 4) == dim
            end
        end
    end
end
