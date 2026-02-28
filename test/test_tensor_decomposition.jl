using Test
using DomainDecomposition

@testset "Tensor Decomposition" begin
    @testset "balanced_decomposition" begin
        # Even split
        @test balanced_decomposition(12, 4) == [3, 3, 3, 3]
        @test balanced_decomposition(12, 3) == [4, 4, 4]

        # Uneven split: extra goes to first ranks
        @test balanced_decomposition(10, 3) == [4, 3, 3]
        @test balanced_decomposition(7, 3) == [3, 2, 2]
        @test balanced_decomposition(1, 1) == [1]

        # Sum always equals original
        for n in [1, 7, 10, 100, 1024], p in [1, 2, 3, 4, 8, 16]
            sizes = balanced_decomposition(n, p)
            @test sum(sizes) == n
            @test length(sizes) == p
            @test all(s -> s >= 0, sizes)
        end
    end

    @testset "subtensor_indices" begin
        # 1D: 100 elements, 4 partitions
        r = subtensor_indices((100,), (4,), (0,))
        @test r == (1:25,)
        r = subtensor_indices((100,), (4,), (3,))
        @test r == (76:100,)

        # 2D: 100×100, 2×2 partitions
        r = subtensor_indices((100, 100), (2, 2), (0, 0))
        @test r == (1:50, 1:50)
        r = subtensor_indices((100, 100), (2, 2), (1, 1))
        @test r == (51:100, 51:100)

        # Uneven: 10×10, 3×3
        r = subtensor_indices((10, 10), (3, 3), (0, 0))
        @test r == (1:4, 1:4)
        r = subtensor_indices((10, 10), (3, 3), (2, 2))
        @test r == (8:10, 8:10)

        # All subtensors cover the full domain
        for shape in [(100,), (50, 50), (10, 10, 10)]
            N = length(shape)
            pdims = ntuple(_ -> 2, N)
            covered = Set{NTuple{N, Int}}()
            for idx in Iterators.product([0:p-1 for p in pdims]...)
                coords = NTuple{N, Int}(idx)
                ranges = subtensor_indices(shape, pdims, coords)
                for pt in Iterators.product(ranges...)
                    push!(covered, NTuple{N, Int}(pt))
                end
            end
            total = prod(shape)
            @test length(covered) == total
        end
    end

    @testset "compute_overlaps" begin
        # Same partition → full overlap
        overlap = compute_overlaps((100,), (2,), (0,), (2,), (0,))
        @test overlap == (1:50,)

        # Different partitions, partial overlap
        overlap = compute_overlaps((100,), (2,), (0,), (4,), (0,))
        @test overlap == (1:25,)

        overlap = compute_overlaps((100,), (2,), (0,), (4,), (1,))
        @test overlap == (26:50,)

        # No overlap
        overlap = compute_overlaps((100,), (2,), (0,), (2,), (1,))
        @test overlap === nothing

        # 2D overlap
        overlap = compute_overlaps((10, 10), (2, 2), (0, 0), (1, 1), (0, 0))
        @test overlap == (1:5, 1:5)
    end

    @testset "compute_halo_sizes" begin
        # Kernel 3, stride 1, dilation 1 → (1, 1)
        @test compute_halo_sizes(3) == (1, 1)

        # Kernel 5 → (2, 2)
        @test compute_halo_sizes(5) == (2, 2)

        # Kernel 1 → (0, 0)
        @test compute_halo_sizes(1) == (0, 0)

        # Kernel 3, dilation 2 → effective 5, pad = (2, 2)
        @test compute_halo_sizes(3; dilation=2) == (2, 2)
    end

    @testset "global_to_local" begin
        @test DomainDecomposition.global_to_local(51:100, 51) == 1:50
        @test DomainDecomposition.global_to_local(26:50, 1) == 26:50
    end
end
