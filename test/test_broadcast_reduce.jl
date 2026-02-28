using Test
using MPI
using DomainDecomposition

@testset "Broadcast / SumReduce / AllReduce" begin
    MPI.Initialized() || MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if nprocs >= 2
        @testset "broadcast_op: 1 → 2" begin
            P_src = create_cartesian_topology(comm, (1,))
            P_dst = create_cartesian_topology(comm, (2,))

            info = DomainDecomposition.setup_broadcast(P_src, P_dst, comm)

            if P_src.active
                x = Float64[1.0, 2.0, 3.0]
            else
                x = Float64[]
            end

            if info.union_comm != MPI.COMM_NULL
                y = broadcast_op(x, info)

                if P_dst.active
                    @test y == [1.0, 2.0, 3.0]
                end
            end
        end

        @testset "all_reduce_op: sum across partition" begin
            P = create_cartesian_topology(comm, (min(nprocs, 2),))

            if P.active
                ar_info = DomainDecomposition.setup_all_reduce(P)
                x = Float64[Float64(P.rank + 1)]
                y = all_reduce_op(x, ar_info)

                # Sum of ranks: 1 + 2 = 3 for 2 procs
                expected_sum = sum(1:min(nprocs, 2))
                @test y[1] ≈ expected_sum
            end
        end

        @testset "all_reduce is self-adjoint" begin
            P = create_cartesian_topology(comm, (min(nprocs, 2),))

            if P.active
                ar_info = DomainDecomposition.setup_all_reduce(P)

                x = randn(Float64, 10)
                y = randn(Float64, 10)

                Ax = all_reduce_op(x, ar_info)
                Aty = all_reduce_op(y, ar_info)

                # <Ax, y> should equal <x, A'y>
                dot1 = sum(Ax .* y)
                dot2 = sum(x .* Aty)

                # Allreduce to get global dots
                global_dot1 = MPI.Allreduce(dot1, MPI.SUM, P.comm)
                global_dot2 = MPI.Allreduce(dot2, MPI.SUM, P.comm)

                @test global_dot1 ≈ global_dot2 atol=1e-10
            end
        end
    end
end
