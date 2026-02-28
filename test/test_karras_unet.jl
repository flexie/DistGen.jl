using Test
using MPI
using Random
using DomainDecomposition

# ═══════════════════════════════════════════════════════════════════════════════
# Test: DistKarrasUNet3d construction + forward
# ═══════════════════════════════════════════════════════════════════════════════

@testset "DistKarrasUNet3d" begin
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if nprocs >= 4
        P = create_cartesian_topology(comm, (2, 2, 1))

        if P.active
            @testset "construction" begin
                model = DistKarrasUNet3d(P, 2, 2;
                    dim=8, dim_max=16, num_downsamples=2,
                    num_blocks_per_stage=(1, 1),
                    fourier_dim=8)

                @test model.dim == 8
                @test model.in_channels == 2
                @test model.out_channels == 2
                @test model.num_downsamples == 2
                @test length(model.downs) > 0
                @test length(model.ups) > 0
                @test length(model.mids) == 2
            end

            @testset "forward pass" begin
                model = DistKarrasUNet3d(P, 2, 2;
                    dim=8, dim_max=16, num_downsamples=2,
                    num_blocks_per_stage=(1, 1),
                    fourier_dim=8)

                rng = MersenneTwister(42)
                ps = init_dist_karras_unet(rng, model; T=Float64)

                global_sp = (32, 32, 32)
                B = 1

                # Unpadded local input (auto_pad handles halo internally)
                x_global = randn(MersenneTwister(100), Float64, global_sp..., 2, B)
                local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
                local_sp = Tuple(length.(local_ranges))
                x_local = x_global[local_ranges..., :, :]

                time = Float64[0.5]

                y = dist_karras_unet_forward(x_local, time, ps, model)

                @test size(y, 1) == local_sp[1]
                @test size(y, 2) == local_sp[2]
                @test size(y, 3) == local_sp[3]
                @test size(y, 4) == 2  # out_channels
                @test size(y, 5) == B
            end

            @testset "different times produce different outputs" begin
                model = DistKarrasUNet3d(P, 2, 2;
                    dim=8, dim_max=16, num_downsamples=2,
                    num_blocks_per_stage=(1, 1),
                    fourier_dim=8)

                rng = MersenneTwister(42)
                ps = init_dist_karras_unet(rng, model; T=Float64)
                ps = merge(ps, (output_gain = [1.0],))

                # Set emb_gain to non-zero so FiLM conditioning is active
                # (Karras initializes gain to 0, making FiLM a no-op initially)
                _set_gain(nt::NamedTuple) = begin
                    pairs = []
                    for k in keys(nt)
                        v = nt[k]
                        if k == :emb_gain && v isa AbstractArray
                            push!(pairs, k => ones(eltype(v), size(v)))
                        elseif v isa NamedTuple
                            push!(pairs, k => _set_gain(v))
                        elseif v isa Tuple
                            push!(pairs, k => Tuple(_set_gain_elem(e) for e in v))
                        else
                            push!(pairs, k => v)
                        end
                    end
                    NamedTuple{Tuple(first.(pairs))}(Tuple(last.(pairs)))
                end
                _set_gain_elem(v::NamedTuple) = _set_gain(v)
                _set_gain_elem(v) = v
                ps = _set_gain(ps)

                global_sp = (32, 32, 32)
                B = 1
                x_global = randn(MersenneTwister(100), Float64, global_sp..., 2, B)
                local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
                x_local = x_global[local_ranges..., :, :]

                y1 = dist_karras_unet_forward(x_local, Float64[0.1], ps, model)
                y2 = dist_karras_unet_forward(x_local, Float64[0.9], ps, model)

                @test !(y1 ≈ y2)
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Test: EDM Preconditioner
# ═══════════════════════════════════════════════════════════════════════════════

@testset "EDM Preconditioner" begin
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if nprocs >= 4
        P = create_cartesian_topology(comm, (2, 2, 1))

        if P.active
            model = DistKarrasUNet3d(P, 2, 2;
                dim=8, dim_max=16, num_downsamples=2,
                num_blocks_per_stage=(1, 1),
                fourier_dim=8)

            rng = MersenneTwister(42)
            ps = init_dist_karras_unet(rng, model; T=Float64)

            global_sp = (32, 32, 32)
            B = 1
            x_global = randn(MersenneTwister(100), Float64, global_sp..., 2, B)
            local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
            local_sp = Tuple(length.(local_ranges))
            x_local = x_global[local_ranges..., :, :]

            @testset "EDM forward runs" begin
                D_x = edm_precond_forward(x_local, Float64[1.0], ps, model)
                @test size(D_x) == size(x_local)
            end

            @testset "different sigma produces different output" begin
                D_x1 = edm_precond_forward(x_local, Float64[0.1], ps, model)
                D_x2 = edm_precond_forward(x_local, Float64[1.0], ps, model)
                @test !(D_x1 ≈ D_x2)
            end
        end
    end
end
