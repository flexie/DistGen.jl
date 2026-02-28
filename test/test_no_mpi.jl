# Standalone tests that don't require MPI runtime.
# Tests the pure-logic components: tensor decomposition, halo slice computation,
# partition planning, sinusoidal embedding, and balanced factorization.

using Test

# ─── Inline the functions we're testing (no MPI dependency) ──────────────────

function balanced_decomposition(global_size::Int, n_partitions::Int)
    base = global_size ÷ n_partitions
    remainder = global_size % n_partitions
    return [base + (i <= remainder ? 1 : 0) for i in 1:n_partitions]
end

function subtensor_indices(
    global_shape::NTuple{N, Int},
    partition_shape::NTuple{N, Int},
    coords::NTuple{N, Int}
) where {N}
    ranges = ntuple(N) do d
        sizes = balanced_decomposition(global_shape[d], partition_shape[d])
        start = sum(sizes[1:coords[d]]; init=0) + 1
        stop = start + sizes[coords[d] + 1] - 1
        start:stop
    end
    return ranges
end

function local_shape(
    global_shape::NTuple{N, Int},
    partition_shape::NTuple{N, Int},
    coords::NTuple{N, Int}
) where {N}
    indices = subtensor_indices(global_shape, partition_shape, coords)
    return ntuple(d -> length(indices[d]), N)
end

function compute_overlaps(
    global_shape::NTuple{N, Int},
    src_partition::NTuple{N, Int}, src_coords::NTuple{N, Int},
    dst_partition::NTuple{N, Int}, dst_coords::NTuple{N, Int}
) where {N}
    src_ranges = subtensor_indices(global_shape, src_partition, src_coords)
    dst_ranges = subtensor_indices(global_shape, dst_partition, dst_coords)
    overlap = ntuple(N) do d
        lo = max(first(src_ranges[d]), first(dst_ranges[d]))
        hi = min(last(src_ranges[d]), last(dst_ranges[d]))
        lo:hi
    end
    for d in 1:N
        length(overlap[d]) <= 0 && return nothing
    end
    return overlap
end

function global_to_local(global_range::UnitRange{Int}, subtensor_start::Int)
    return (first(global_range) - subtensor_start + 1):(last(global_range) - subtensor_start + 1)
end

function compute_halo_sizes(kernel_size::Int; stride::Int=1, dilation::Int=1)
    effective_kernel = (kernel_size - 1) * dilation + 1
    pad_total = effective_kernel - 1
    left = pad_total ÷ 2
    right = pad_total - left
    return (left, right)
end

function _compute_slices(shape::NTuple{M, Int}, dim::Int, left_halo::Int, right_halo::Int) where {M}
    n = shape[dim]
    function make_range(range_for_dim)
        return ntuple(M) do d
            d == dim ? range_for_dim : Colon()
        end
    end
    left_recv = left_halo > 0 ? make_range(1:left_halo) : nothing
    left_send = left_halo > 0 ? make_range((left_halo + 1):(2 * left_halo)) : nothing
    right_recv = right_halo > 0 ? make_range((n - right_halo + 1):n) : nothing
    right_send = right_halo > 0 ? make_range((n - 2 * right_halo + 1):(n - right_halo)) : nothing
    return (left_send, left_recv, right_send, right_recv)
end

function _balanced_3d_factorization(n::Int)
    cbrt_n = round(Int, n^(1/3))
    best = (1, 1, n)
    best_ratio = Float64(n)
    for i in cbrt_n:-1:1
        n % i != 0 && continue
        remaining = n ÷ i
        sqrt_r = round(Int, sqrt(remaining))
        for j in sqrt_r:-1:1
            remaining % j != 0 && continue
            k = remaining ÷ j
            ratio = max(i, j, k) / min(i, j, k)
            if ratio < best_ratio
                best = (i, j, k)
                best_ratio = ratio
            end
        end
    end
    return best
end

function sinusoidal_embedding(t::AbstractVector{T}, dim::Int) where {T}
    half_dim = dim ÷ 2
    freq = T(1) ./ (T(10000) .^ (T.(0:half_dim-1) ./ T(half_dim)))
    angles = freq * t'
    return vcat(sin.(angles), cos.(angles))
end

# ─── Test 1: Tensor Decomposition ───────────────────────────────────────────

@testset "Test 1: Tensor Decomposition" begin
    @testset "balanced_decomposition" begin
        @test balanced_decomposition(12, 4) == [3, 3, 3, 3]
        @test balanced_decomposition(12, 3) == [4, 4, 4]
        @test balanced_decomposition(10, 3) == [4, 3, 3]
        @test balanced_decomposition(7, 3) == [3, 2, 2]
        @test balanced_decomposition(1, 1) == [1]

        for n in [1, 7, 10, 100, 1024], p in [1, 2, 3, 4, 8, 16]
            sizes = balanced_decomposition(n, p)
            @test sum(sizes) == n
            @test length(sizes) == p
            @test all(s -> s >= 0, sizes)
        end
    end

    @testset "subtensor_indices" begin
        @test subtensor_indices((100,), (4,), (0,)) == (1:25,)
        @test subtensor_indices((100,), (4,), (3,)) == (76:100,)
        @test subtensor_indices((100, 100), (2, 2), (0, 0)) == (1:50, 1:50)
        @test subtensor_indices((100, 100), (2, 2), (1, 1)) == (51:100, 51:100)
        @test subtensor_indices((10, 10), (3, 3), (0, 0)) == (1:4, 1:4)
        @test subtensor_indices((10, 10), (3, 3), (2, 2)) == (8:10, 8:10)

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
            @test length(covered) == prod(shape)
        end
    end

    @testset "local_shape" begin
        @test local_shape((100,), (4,), (0,)) == (25,)
        @test local_shape((100, 100), (2, 2), (0, 0)) == (50, 50)
        @test local_shape((10,), (3,), (2,)) == (3,)
    end

    @testset "compute_overlaps" begin
        @test compute_overlaps((100,), (2,), (0,), (2,), (0,)) == (1:50,)
        @test compute_overlaps((100,), (2,), (0,), (4,), (0,)) == (1:25,)
        @test compute_overlaps((100,), (2,), (0,), (4,), (1,)) == (26:50,)
        @test compute_overlaps((100,), (2,), (0,), (2,), (1,)) === nothing
        @test compute_overlaps((10, 10), (2, 2), (0, 0), (1, 1), (0, 0)) == (1:5, 1:5)
    end

    @testset "global_to_local" begin
        @test global_to_local(51:100, 51) == 1:50
        @test global_to_local(26:50, 1) == 26:50
        @test global_to_local(1:25, 1) == 1:25
    end
end

# ─── Test 2: Halo Size Computation ──────────────────────────────────────────

@testset "Test 2: Halo Size Computation" begin
    @test compute_halo_sizes(3) == (1, 1)
    @test compute_halo_sizes(5) == (2, 2)
    @test compute_halo_sizes(1) == (0, 0)
    @test compute_halo_sizes(7) == (3, 3)
    @test compute_halo_sizes(3; dilation=2) == (2, 2)
    @test compute_halo_sizes(3; dilation=3) == (3, 3)

    # Halo slice computation for a 1D tensor of size 14 with halo=2
    left_send, left_recv, right_send, right_recv = _compute_slices((14,), 1, 2, 2)
    @test left_recv == (1:2,)
    @test left_send == (3:4,)
    @test right_recv == (13:14,)
    @test right_send == (11:12,)

    # 2D tensor (10, 8), halo in dim 1 only
    ls, lr, rs, rr = _compute_slices((10, 8), 1, 1, 1)
    @test ls == (2:2, Colon())
    @test lr == (1:1, Colon())
    @test rs == (9:9, Colon())
    @test rr == (10:10, Colon())

    # Zero halo
    ls, lr, rs, rr = _compute_slices((10,), 1, 0, 0)
    @test ls === nothing
    @test lr === nothing
    @test rs === nothing
    @test rr === nothing
end

# ─── Test 3: Balanced 3D Factorization ──────────────────────────────────────

@testset "Test 3: Balanced 3D Factorization" begin
    # Perfect cube
    f = _balanced_3d_factorization(8)
    @test prod(f) == 8
    @test f == (2, 2, 2)

    # 512 = 8^3
    f = _balanced_3d_factorization(512)
    @test prod(f) == 512
    @test f == (8, 8, 8)

    # 64 = 4^3
    f = _balanced_3d_factorization(64)
    @test prod(f) == 64
    @test f == (4, 4, 4)

    # Non-cube: 12 should give something balanced
    f = _balanced_3d_factorization(12)
    @test prod(f) == 12
    # Ratio should be reasonable (e.g., 2×2×3)
    @test max(f...) / min(f...) <= 3

    # Prime: 7 can only be 1×1×7
    f = _balanced_3d_factorization(7)
    @test prod(f) == 7

    # 1 GPU
    f = _balanced_3d_factorization(1)
    @test f == (1, 1, 1)

    # All results must multiply to n
    for n in [1, 2, 4, 6, 8, 12, 16, 24, 27, 32, 64, 128, 256, 512]
        f = _balanced_3d_factorization(n)
        @test prod(f) == n
        @test all(x -> x >= 1, f)
    end
end

# ─── Test 4: Sinusoidal Embedding ───────────────────────────────────────────

@testset "Test 4: Sinusoidal Embedding" begin
    t = Float32[0.0, 0.5, 1.0]
    dim = 16

    emb = sinusoidal_embedding(t, dim)

    # Output shape: (dim, batch)
    @test size(emb) == (16, 3)

    # t=0 should give sin(0)=0 for all sin components, cos(0)=1 for all cos
    @test all(abs.(emb[1:8, 1]) .< 1e-5)    # sin(0) ≈ 0
    @test all(abs.(emb[9:16, 1] .- 1.0) .< 1e-5)  # cos(0) ≈ 1

    # Values should be in [-1, 1]
    @test all(-1.0 .<= emb .<= 1.0)

    # Different times should produce different embeddings
    @test emb[:, 1] != emb[:, 2]
    @test emb[:, 2] != emb[:, 3]

    # Larger dim
    emb64 = sinusoidal_embedding(Float64[0.25], 64)
    @test size(emb64) == (64, 1)
    @test all(-1.0 .<= emb64 .<= 1.0)
end

# ─── Test 5: Repartition Overlap Computation ────────────────────────────────

@testset "Test 5: Repartition Overlap Computation" begin
    # Simulate a 2→4 repartition on a 1D domain of size 100
    global_shape = (100,)
    src_dims = (2,)
    dst_dims = (4,)

    # From src worker 0 (owns 1:50), find overlaps with all dst workers
    overlaps_from_0 = []
    for dc in 0:3
        ov = compute_overlaps(global_shape, src_dims, (0,), dst_dims, (dc,))
        push!(overlaps_from_0, (dc, ov))
    end

    # src[0] owns 1:50. dst workers own: 0→1:25, 1→26:50, 2→51:75, 3→76:100
    @test overlaps_from_0[1] == (0, (1:25,))     # overlap with dst 0
    @test overlaps_from_0[2] == (1, (26:50,))     # overlap with dst 1
    @test overlaps_from_0[3] == (2, nothing)       # no overlap with dst 2
    @test overlaps_from_0[4] == (3, nothing)       # no overlap with dst 3

    # From src worker 1 (owns 51:100)
    overlaps_from_1 = []
    for dc in 0:3
        ov = compute_overlaps(global_shape, src_dims, (1,), dst_dims, (dc,))
        push!(overlaps_from_1, (dc, ov))
    end

    @test overlaps_from_1[1] == (0, nothing)
    @test overlaps_from_1[2] == (1, nothing)
    @test overlaps_from_1[3] == (2, (51:75,))
    @test overlaps_from_1[4] == (3, (76:100,))

    # 2D: 100×100, src 2×2 → dst 4×1
    global_shape_2d = (100, 100)
    # src (0,0) owns (1:50, 1:50), dst (0,0) owns (1:25, 1:100)
    ov = compute_overlaps(global_shape_2d, (2, 2), (0, 0), (4, 1), (0, 0))
    @test ov == (1:25, 1:50)

    ov = compute_overlaps(global_shape_2d, (2, 2), (0, 0), (4, 1), (1, 0))
    @test ov == (26:50, 1:50)

    # No overlap: src (0,0)=(1:50,1:50) vs dst (2,0)=(51:75, 1:100) in dim 1
    ov = compute_overlaps(global_shape_2d, (2, 2), (0, 0), (4, 1), (2, 0))
    @test ov === nothing

    # Verify all overlaps from all src workers cover each dst worker's full domain
    for dc in 0:3
        dst_ranges = subtensor_indices((100,), (4,), (dc,))
        dst_size = length(dst_ranges[1])

        # Collect all global indices sent to this dst worker
        received = Set{Int}()
        for sc in 0:1
            ov = compute_overlaps((100,), (2,), (sc,), (4,), (dc,))
            ov === nothing && continue
            for idx in ov[1]
                push!(received, idx)
            end
        end
        @test length(received) == dst_size
    end
end

println("\nAll 5 tests passed!")
