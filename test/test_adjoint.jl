using Test
using MPI
using ChainRulesCore
using DomainDecomposition

@testset "Adjoint Tests (Dot Product Test)" begin
    MPI.Initialized() || MPI.Init()

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if nprocs >= 2
        @testset "halo_exchange adjoint test" begin
            P = create_cartesian_topology(comm, (2,))

            if P.active
                halo_sizes = [(2, 2)]
                info = DomainDecomposition.compute_halo_info(P, halo_sizes)

                # Padded local tensor: bulk(10) + 2 ghost on each side = 14
                local_sz = (14,)
                x = randn(Float64, local_sz...)
                y = randn(Float64, local_sz...)

                # Forward
                Fx = halo_exchange(x, info)

                # Adjoint via ChainRules rrule
                _, pullback = ChainRulesCore.rrule(halo_exchange, x, info)
                _, Fᵀy, _ = pullback(y)

                # Local dot products
                dot_Fx_y = sum(Fx .* y)
                dot_x_Fᵀy = sum(x .* Fᵀy)

                # Global dot products
                global_dot1 = MPI.Allreduce(dot_Fx_y, MPI.SUM, P.comm)
                global_dot2 = MPI.Allreduce(dot_x_Fᵀy, MPI.SUM, P.comm)

                @test global_dot1 ≈ global_dot2 atol=1e-10
            end
        end

        @testset "all_reduce adjoint test" begin
            P = create_cartesian_topology(comm, (min(nprocs, 2),))

            if P.active
                ar_info = DomainDecomposition.setup_all_reduce(P)

                x = randn(Float64, 20)
                y = randn(Float64, 20)

                # Forward
                Ax = all_reduce_op(x, ar_info)

                # Adjoint via rrule
                _, pullback = ChainRulesCore.rrule(all_reduce_op, x, ar_info)
                _, Aᵀy, _ = pullback(y)

                dot1 = sum(Ax .* y)
                dot2 = sum(x .* Aᵀy)

                global_dot1 = MPI.Allreduce(dot1, MPI.SUM, P.comm)
                global_dot2 = MPI.Allreduce(dot2, MPI.SUM, P.comm)

                @test global_dot1 ≈ global_dot2 atol=1e-10
            end
        end
    end

    if nprocs >= 4
        @testset "repartition adjoint test" begin
            P_src = create_cartesian_topology(comm, (2,))
            P_dst = create_cartesian_topology(comm, (4,))
            global_shape = (100,)

            info = DomainDecomposition.setup_repartition(P_src, P_dst, global_shape, comm)

            if P_src.active
                src_sz = DomainDecomposition.local_shape(global_shape, P_src.dims, P_src.coords)
                x = randn(Float64, src_sz...)
            else
                x = Float64[]
            end

            # Forward
            Fx = repartition_op(x, info)

            # Generate y in the range (dst partition)
            if P_dst.active
                dst_sz = DomainDecomposition.local_shape(global_shape, P_dst.dims, P_dst.coords)
                y = randn(Float64, dst_sz...)
            else
                y = Float64[]
            end

            # Adjoint via rrule
            _, pullback = ChainRulesCore.rrule(repartition_op, x, info)
            _, Fᵀy, _ = pullback(y)

            # Dot products (only active workers contribute)
            dot_Fx_y = P_dst.active ? sum(Fx .* y) : 0.0
            dot_x_Fᵀy = P_src.active ? sum(x .* Fᵀy) : 0.0

            if info.union_comm != MPI.COMM_NULL
                global_dot1 = MPI.Allreduce(dot_Fx_y, MPI.SUM, info.union_comm)
                global_dot2 = MPI.Allreduce(dot_x_Fᵀy, MPI.SUM, info.union_comm)

                @test global_dot1 ≈ global_dot2 atol=1e-8
            end
        end
    end
end
