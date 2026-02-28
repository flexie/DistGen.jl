using Test
using MPI
using DomainDecomposition

@testset "Repartition" begin
    MPI.Initialized() || MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if nprocs >= 4
        @testset "repartition 2→4 (1D)" begin
            P_src = create_cartesian_topology(comm, (2,))
            P_dst = create_cartesian_topology(comm, (4,))
            global_shape = (100,)

            info = DomainDecomposition.setup_repartition(P_src, P_dst, global_shape, comm)

            if P_src.active
                # Fill local data with global indices
                ranges = subtensor_indices(global_shape, P_src.dims, P_src.coords)
                x = collect(Float64, ranges[1])
            else
                x = Float64[]
            end

            y = repartition_op(x, info)

            if P_dst.active
                # Verify: local data matches expected global indices
                expected_ranges = subtensor_indices(global_shape, P_dst.dims, P_dst.coords)
                expected = collect(Float64, expected_ranges[1])
                @test length(y) == length(expected)
                @test y ≈ expected
            end
        end

        @testset "repartition 2×2 → 4×1 (2D)" begin
            P_src = create_cartesian_topology(comm, (2, 2))
            P_dst = create_cartesian_topology(comm, (4, 1))
            global_shape = (100, 100)

            info = DomainDecomposition.setup_repartition(P_src, P_dst, global_shape, comm)

            if P_src.active
                ranges = subtensor_indices(global_shape, P_src.dims, P_src.coords)
                local_sz = DomainDecomposition.local_shape(global_shape, P_src.dims, P_src.coords)
                # Fill with rank-derived pattern for verification
                x = fill(Float64(P_src.rank + 1), local_sz...)
            else
                x = zeros(Float64, 0, 0)
            end

            y = repartition_op(x, info)

            if P_dst.active
                expected_sz = DomainDecomposition.local_shape(global_shape, P_dst.dims, P_dst.coords)
                @test size(y) == expected_sz
            end
        end

        @testset "repartition roundtrip preserves data" begin
            P_src = create_cartesian_topology(comm, (2,))
            P_dst = create_cartesian_topology(comm, (4,))
            global_shape = (100,)

            info_fwd = DomainDecomposition.setup_repartition(P_src, P_dst, global_shape, comm)

            if P_src.active
                ranges = subtensor_indices(global_shape, P_src.dims, P_src.coords)
                x_orig = collect(Float64, ranges[1])
            else
                x_orig = Float64[]
            end

            # Forward: 2 → 4
            y = repartition_op(x_orig, info_fwd)

            # Backward: 4 → 2
            info_bwd = DomainDecomposition.setup_repartition(P_dst, P_src, global_shape, comm)
            x_round = repartition_op(y, info_bwd)

            if P_src.active
                @test x_round ≈ x_orig
            end
        end
    end
end
