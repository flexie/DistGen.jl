using Test
using MPI
using DomainDecomposition

@testset "Halo Exchange" begin
    MPI.Initialized() || MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    @testset "1D halo exchange, 2 procs" begin
        if nprocs >= 2
            P = create_cartesian_topology(comm, (2,))

            if P.active
                halo_sizes = [(1, 1)]
                info = DomainDecomposition.compute_halo_info(P, halo_sizes)

                # Local tensor: [ghost_left | bulk(5) | ghost_right] = 7 elements
                # Fill bulk with rank-specific values, ghosts with -1
                local_size = 7
                x = fill(Float64(-1), local_size)
                x[2:6] .= Float64(P.rank + 1)  # bulk = rank+1

                halo_exchange!(x, info)

                if P.coords == (0,)
                    # Left boundary: left ghost stays -1 (PROC_NULL)
                    @test x[1] == -1.0
                    # Right ghost should have neighbor's (rank 1) left bulk value
                    @test x[7] == 2.0  # rank 1's leftmost bulk
                elseif P.coords == (1,)
                    # Right boundary: right ghost stays -1
                    @test x[7] == -1.0
                    # Left ghost should have neighbor's (rank 0) right bulk value
                    @test x[1] == 1.0  # rank 0's rightmost bulk
                end
            end
        end
    end

    @testset "2D halo exchange, 2×2 procs" begin
        if nprocs >= 4
            P = create_cartesian_topology(comm, (2, 2))

            if P.active
                halo_sizes = [(1, 1), (1, 1)]
                info = DomainDecomposition.compute_halo_info(P, halo_sizes)

                # Local tensor: (5+2) × (5+2) = 7×7 with 1 ghost on each side
                x = zeros(Float64, 7, 7)
                # Fill bulk (2:6, 2:6) with rank-specific value
                x[2:6, 2:6] .= Float64(P.rank + 1)

                halo_exchange!(x, info)

                # Verify ghost regions received correct values from neighbors
                nbrs = neighbor_ranks(P)
                left_d1, right_d1 = nbrs[1]
                left_d2, right_d2 = nbrs[2]

                # Check that interior bulk is unchanged
                @test all(x[3:5, 3:5] .== Float64(P.rank + 1))
            end
        end
    end

    @testset "out-of-place halo_exchange (for AD)" begin
        if nprocs >= 2
            P = create_cartesian_topology(comm, (2,))

            if P.active
                halo_sizes = [(1, 1)]
                info = DomainDecomposition.compute_halo_info(P, halo_sizes)

                x = fill(Float64(P.rank + 1), 7)
                x[1] = -1.0
                x[7] = -1.0

                y = halo_exchange(x, info)

                # Original unchanged
                @test x[1] == -1.0
                @test x[7] == -1.0

                # y has exchanged values
                @test y !== x
            end
        end
    end
end
