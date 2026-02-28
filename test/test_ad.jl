using Test
using MPI
using Random
using DomainDecomposition

# Only run AD tests if Zygote is available
ad_available = try
    using Zygote
    true
catch
    @info "Zygote not available, skipping AD tests"
    false
end

if ad_available

@testset "Zygote AD — MP operations" begin
    x = randn(Float32, 4, 4, 4, 4, 1)
    res = randn(Float32, 4, 4, 4, 4, 1)

    @testset "mp_silu gradient" begin
        g = Zygote.gradient(x -> sum(mp_silu(x)), x)[1]
        @test g !== nothing
        @test all(isfinite, g)
    end

    @testset "mp_add gradient" begin
        g = Zygote.gradient((x, r) -> sum(mp_add(x, r; t=0.3f0)), x, res)
        @test g[1] !== nothing
        @test g[2] !== nothing
        @test all(isfinite, g[1])
    end

    @testset "pixel_norm gradient" begin
        g = Zygote.gradient(x -> sum(pixel_norm(x; dim=4)), x)[1]
        @test g !== nothing
        @test all(isfinite, g)
    end

    @testset "normalize_weight gradient" begin
        w = randn(Float32, 3, 3, 3, 2, 4)
        g = Zygote.gradient(w -> sum(normalize_weight(w)), w)[1]
        @test g !== nothing
        @test all(isfinite, g)
    end

    @testset "mp_fourier_embedding gradient" begin
        t = randn(Float32, 2)
        freqs = randn(Float32, 4)
        g = Zygote.gradient(t -> sum(mp_fourier_embedding(t, freqs)), t)[1]
        @test g !== nothing
        @test all(isfinite, g)
    end
end

@testset "Zygote AD — DistConv3d (weight-normed)" begin
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    nprocs = MPI.Comm_size(comm)

    if nprocs >= 4
        P = create_cartesian_topology(comm, (2, 2, 1))

        if P.active
            C_in, C_out = 2, 4
            global_sp = (8, 8, 8)
            B = 1

            layer = DistConv3d(P, C_in, C_out, (3, 3, 3);
                               bias=false, weight_norm=true)

            rng = MersenneTwister(42)
            weight = randn(rng, Float32, 3, 3, 3, C_in, C_out)

            # Unpadded local input — auto_pad will handle it
            x_global = randn(MersenneTwister(100), Float32, global_sp..., C_in, B)
            local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
            x_local = x_global[local_ranges..., :, :]

            g = Zygote.gradient(w -> sum(dist_conv3d_forward(x_local, w, nothing, layer; auto_pad=true)), weight)
            @test g[1] !== nothing
            @test all(isfinite, g[1])
        end
    end
end

end # if ad_available
