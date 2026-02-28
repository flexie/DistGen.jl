using Test
using MPI
using DomainDecomposition

@testset "CartesianPartition" begin
    MPI.Initialized() || MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if nprocs >= 4
        @testset "create_cartesian_topology 2×2" begin
            P = create_cartesian_topology(comm, (2, 2))

            if rank < 4
                @test P.active == true
                @test P.size == 4
                @test P.dims == (2, 2)
                @test all(c -> 0 <= c <= 1, P.coords)
            else
                @test P.active == false
            end
        end

        @testset "neighbor_ranks" begin
            P = create_cartesian_topology(comm, (2, 2))

            if P.active
                nbrs = neighbor_ranks(P)
                @test length(nbrs) == 2

                # Non-periodic: boundary workers have PROC_NULL neighbors
                for (d, (left, right)) in enumerate(nbrs)
                    if P.coords[d] == 0
                        @test left == MPI.PROC_NULL
                    end
                    if P.coords[d] == P.dims[d] - 1
                        @test right == MPI.PROC_NULL
                    end
                end
            end
        end

        @testset "create_subpartition" begin
            P = create_cartesian_topology(comm, (2, 2))

            if P.active
                # Keep only dimension 1
                P_sub = create_subpartition(P, (1,))
                @test ndims(P_sub) == 1
                @test P_sub.dims == (2,)
                @test P_sub.active == true
            end
        end
    end

    if nprocs >= 2
        @testset "create_cartesian_topology 2×1" begin
            P = create_cartesian_topology(comm, (2, 1))

            if rank < 2
                @test P.active == true
                @test P.dims == (2, 1)
            end
        end

        @testset "periodic neighbors" begin
            P = create_cartesian_topology(comm, (2,); periodic=(true,))

            if P.active
                nbrs = neighbor_ranks(P)
                # With periodic, no PROC_NULL
                @test nbrs[1][1] != MPI.PROC_NULL
                @test nbrs[1][2] != MPI.PROC_NULL
            end
        end
    end

    @testset "single-rank partition" begin
        P = create_cartesian_topology(comm, (1,))

        if rank == 0
            @test P.active == true
            @test P.dims == (1,)
            @test P.coords == (0,)
        end
    end
end
