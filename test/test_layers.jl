using Test
using MPI
using NNlib
using DomainDecomposition

# ─── Helper: gather spatially-partitioned array to rank 0 ────────────────────

"""
    gather_to_rank0(y_local, P::CartesianPartition, global_spatial_shape)

Gather a 5D spatially-partitioned tensor (W,H,D,C,B) to rank 0.
Only spatial dims (1:3) are partitioned; C and B are replicated.
Returns the assembled global array on rank 0, nothing on others.
"""
function gather_to_rank0(
    y_local::AbstractArray{T, 5},
    P::CartesianPartition{3},
    global_spatial_shape::NTuple{3, Int}
) where {T}
    C = size(y_local, 4)
    B = size(y_local, 5)
    rank = P.rank
    comm = P.comm

    if rank == 0
        global_y = zeros(T, global_spatial_shape..., C, B)
        # Place own data
        ranges = subtensor_indices(global_spatial_shape, P.dims, P.coords)
        global_y[ranges..., :, :] .= y_local

        # Receive from all other ranks
        for r in 1:(prod(P.dims) - 1)
            r_coords = Tuple(MPI.Cart_coords(comm, r))
            r_ranges = subtensor_indices(global_spatial_shape, P.dims, r_coords)
            r_shape = Tuple(length.(r_ranges))
            buf = Array{T}(undef, r_shape..., C, B)
            MPI.Recv!(buf, comm; source=r, tag=r)
            global_y[r_ranges..., :, :] .= buf
        end
        return global_y
    else
        MPI.Send(collect(y_local), comm; dest=0, tag=rank)
        return nothing
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Test 1: DistConv3d
# ═══════════════════════════════════════════════════════════════════════════════

@testset "DistConv3d" begin
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if nprocs >= 4
        P = create_cartesian_topology(comm, (2, 2, 1))

        if P.active
            C_in, C_out = 2, 3
            kernel = (3, 3, 3)
            global_sp = (8, 8, 8)
            B = 1

            layer = DistConv3d(P, C_in, C_out, kernel)

            @testset "construction" begin
                @test layer.in_channels == C_in
                @test layer.out_channels == C_out
                @test layer.kernel_size == kernel
                @test layer.use_bias == true
            end

            # Shared deterministic weight/bias (same on all ranks)
            using Random
            rng = MersenneTwister(42)
            weight = randn(rng, Float64, kernel..., C_in, C_out)
            bias = zeros(Float64, C_out)

            # Build global input
            rng2 = MersenneTwister(123)
            x_global = randn(rng2, Float64, global_sp..., C_in, B)

            # Build local input with halo padding
            local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
            local_sp = Tuple(length.(local_ranges))
            halo = 1  # (3-1)/2 = 1 per side for kernel=3
            padded_sp = Tuple(local_sp[d] + 2 * halo for d in 1:3)

            # Extract local tile with halo from zero-padded global
            x_padded_global = zeros(Float64, global_sp[1] + 2*halo, global_sp[2] + 2*halo, global_sp[3] + 2*halo, C_in, B)
            x_padded_global[(1+halo):(global_sp[1]+halo), (1+halo):(global_sp[2]+halo), (1+halo):(global_sp[3]+halo), :, :] .= x_global

            padded_ranges = Tuple((first(local_ranges[d])):(last(local_ranges[d]) + 2*halo) for d in 1:3)
            x_local = x_padded_global[padded_ranges..., :, :]

            ps = (weight=weight, bias=bias)
            st = NamedTuple()

            @testset "Lux interface" begin
                y, st2 = layer(x_local, ps, st)
                @test st2 === st
                @test ndims(y) == 5
            end

            @testset "output shape" begin
                y, _ = layer(x_local, ps, st)
                # stride=1 with halo → output spatial = local spatial
                @test size(y, 1) == local_sp[1]
                @test size(y, 2) == local_sp[2]
                @test size(y, 3) == local_sp[3]
                @test size(y, 4) == C_out
                @test size(y, 5) == B
            end

            @testset "distributed vs serial" begin
                y_local_out, _ = layer(x_local, ps, st)
                y_gathered = gather_to_rank0(y_local_out, P, global_sp)

                if rank == 0
                    # Serial conv with same-padding on global tensor
                    cdims = NNlib.DenseConvDims(
                        size(x_padded_global), size(weight);
                        stride=[1,1,1], padding=ntuple(_->0, 6)
                    )
                    y_serial = NNlib.conv(x_padded_global, weight, cdims)
                    y_serial .+= reshape(bias, 1, 1, 1, :, 1)

                    @test size(y_gathered) == size(y_serial)
                    @test y_gathered ≈ y_serial atol=1e-10
                end
            end

            @testset "no-bias variant" begin
                layer_nb = DistConv3d(P, C_in, C_out, kernel; bias=false)
                @test layer_nb.use_bias == false
                ps_nb = (weight=weight,)
                y, _ = layer_nb(x_local, ps_nb, st)
                @test size(y, 4) == C_out
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Test 2: DistGroupNorm
# ═══════════════════════════════════════════════════════════════════════════════

@testset "DistGroupNorm" begin
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if nprocs >= 4
        P = create_cartesian_topology(comm, (2, 2, 1))

        if P.active
            C, G, B = 4, 2, 2
            global_sp = (8, 8, 4)

            layer = DistGroupNorm(P, C, G)

            @testset "construction" begin
                @test layer.num_channels == C
                @test layer.num_groups == G
                @test layer.eps == 1e-5
            end

            @testset "non-divisible channels asserts" begin
                @test_throws AssertionError DistGroupNorm(P, 5, 2)
            end

            # Shared deterministic input
            using Random
            rng = MersenneTwister(77)
            x_global = randn(rng, Float64, global_sp..., C, B)

            # Slice local spatial
            local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
            x_local = x_global[local_ranges..., :, :]

            γ = ones(Float64, C)
            β = zeros(Float64, C)
            ps = (scale=γ, bias=β)
            st = NamedTuple()

            @testset "Lux interface + shape" begin
                y, st2 = layer(x_local, ps, st)
                @test st2 === st
                @test size(y) == size(x_local)
            end

            @testset "distributed vs serial" begin
                y_local_out, _ = layer(x_local, ps, st)
                y_gathered = gather_to_rank0(y_local_out, P, global_sp)

                if rank == 0
                    # Serial GroupNorm on global tensor
                    W, H, D = global_sp
                    C_per_G = C ÷ G
                    x_g = reshape(x_global, W, H, D, C_per_G, G, B)
                    spatial_size = W * H * D * C_per_G
                    local_sum = sum(x_g; dims=(1,2,3,4))
                    local_sum_sq = sum(x_g .^ 2; dims=(1,2,3,4))
                    μ = dropdims(local_sum; dims=(1,2,3,4)) ./ spatial_size
                    σ² = dropdims(local_sum_sq; dims=(1,2,3,4)) ./ spatial_size .- μ .^ 2
                    μ_6d = reshape(μ, 1, 1, 1, 1, G, B)
                    inv_std = reshape(1.0 ./ sqrt.(σ² .+ 1e-5), 1, 1, 1, 1, G, B)
                    y_serial = reshape((x_g .- μ_6d) .* inv_std, W, H, D, C, B)

                    @test y_gathered ≈ y_serial atol=1e-10
                end
            end

            @testset "affine parameters" begin
                γ2 = fill(2.0, C)
                β2 = fill(0.5, C)
                ps_id = (scale=ones(Float64, C), bias=zeros(Float64, C))
                ps_aff = (scale=γ2, bias=β2)
                y_id, _ = layer(x_local, ps_id, st)
                y_aff, _ = layer(x_local, ps_aff, st)
                @test y_aff ≈ 2.0 .* y_id .+ 0.5 atol=1e-12
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Test 3: DistAdaptiveGroupNorm
# ═══════════════════════════════════════════════════════════════════════════════

@testset "DistAdaptiveGroupNorm" begin
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if nprocs >= 4
        P = create_cartesian_topology(comm, (2, 2, 1))

        if P.active
            C, G, B = 4, 2, 2
            cond_dim = 8
            global_sp = (8, 8, 4)

            gn = DistGroupNorm(P, C, G)
            layer = DistAdaptiveGroupNorm(gn, cond_dim)

            @testset "construction" begin
                @test layer.cond_dim == cond_dim
                @test layer.groupnorm.num_channels == C
                @test layer.groupnorm.num_groups == G
            end

            using Random
            rng = MersenneTwister(99)
            x_global = randn(rng, Float64, global_sp..., C, B)
            local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
            x_local = x_global[local_ranges..., :, :]

            cond = randn(MersenneTwister(55), Float64, cond_dim, B)

            ps = (
                gamma_weight = zeros(Float64, C, cond_dim),
                gamma_bias   = ones(Float64, C, 1),
                beta_weight  = zeros(Float64, C, cond_dim),
                beta_bias    = zeros(Float64, C, 1),
            )

            @testset "forward shape" begin
                y = DomainDecomposition.dist_adaptive_groupnorm_forward(x_local, cond, ps, layer)
                @test size(y) == size(x_local)
            end

            @testset "conditioning affects output" begin
                cond1 = ones(Float64, cond_dim, B)
                cond2 = 2.0 .* ones(Float64, cond_dim, B)
                ps_varied = (
                    gamma_weight = randn(MersenneTwister(11), Float64, C, cond_dim),
                    gamma_bias   = ones(Float64, C, 1),
                    beta_weight  = randn(MersenneTwister(22), Float64, C, cond_dim),
                    beta_bias    = zeros(Float64, C, 1),
                )
                y1 = DomainDecomposition.dist_adaptive_groupnorm_forward(x_local, cond1, ps_varied, layer)
                y2 = DomainDecomposition.dist_adaptive_groupnorm_forward(x_local, cond2, ps_varied, layer)
                @test !(y1 ≈ y2)
            end

            @testset "identity conditioning recovers GroupNorm" begin
                # γ_weight=0, γ_bias=1 → γ_cond=1; β_weight=0, β_bias=0 → β_cond=0
                ps_id = (
                    gamma_weight = zeros(Float64, C, cond_dim),
                    gamma_bias   = ones(Float64, C, 1),
                    beta_weight  = zeros(Float64, C, cond_dim),
                    beta_bias    = zeros(Float64, C, 1),
                )
                y_adagn = DomainDecomposition.dist_adaptive_groupnorm_forward(x_local, cond, ps_id, layer)

                # Plain GroupNorm with identity affine
                gn_ps = (scale=ones(Float64, C), bias=zeros(Float64, C))
                y_gn, _ = gn(x_local, gn_ps, NamedTuple())

                @test y_adagn ≈ y_gn atol=1e-12
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Test 4: DistLinear
# ═══════════════════════════════════════════════════════════════════════════════

@testset "DistLinear" begin
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if nprocs >= 4
        P = create_cartesian_topology(comm, (4,))

        if P.active
            in_features, out_features, B = 8, 6, 3

            layer = DistLinear(P, in_features, out_features)

            @testset "construction" begin
                @test layer.in_features == in_features
                @test layer.out_features == out_features
                @test layer.use_bias == true
            end

            using Random
            # Build full weight and input (same on all ranks for verification)
            rng = MersenneTwister(200)
            W_full = randn(rng, Float64, out_features, in_features)
            x_full = randn(MersenneTwister(201), Float64, in_features, B)
            bias = randn(MersenneTwister(202), Float64, out_features)

            # Each rank gets a column slice of input and weight
            local_sizes = balanced_decomposition(in_features, 4)
            offsets = cumsum([0; local_sizes[1:end-1]])
            my_start = offsets[P.rank + 1] + 1
            my_end = my_start + local_sizes[P.rank + 1] - 1

            x_local = x_full[my_start:my_end, :]
            W_local = W_full[:, my_start:my_end]

            ps = (weight=W_local, bias=bias)
            st = NamedTuple()

            @testset "Lux interface + shape" begin
                y, st2 = layer(x_local, ps, st)
                @test st2 === st
                @test size(y) == (out_features, B)
            end

            @testset "distributed vs serial" begin
                y_dist, _ = layer(x_local, ps, st)

                if rank == 0
                    y_serial = W_full * x_full .+ reshape(bias, :, 1)
                    @test y_dist ≈ y_serial atol=1e-10
                end
            end

            @testset "no-bias variant" begin
                layer_nb = DistLinear(P, in_features, out_features; bias=false)
                @test layer_nb.use_bias == false
                ps_nb = (weight=W_local,)
                y, _ = layer_nb(x_local, ps_nb, st)
                @test size(y) == (out_features, B)
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Test 5: DistSkipConnection
# ═══════════════════════════════════════════════════════════════════════════════

@testset "DistSkipConnection" begin
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if nprocs >= 4
        @testset "same partition (no repartition)" begin
            P = create_cartesian_topology(comm, (2, 2, 1))

            if P.active
                global_sp = (8, 8, 4)
                C_enc, C_dec, B = 4, 6, 1

                skip = DistSkipConnection(P, P, global_sp, comm)
                @test skip.repartition_info === nothing

                local_ranges = subtensor_indices(global_sp, P.dims, P.coords)
                local_sp = Tuple(length.(local_ranges))

                using Random
                enc = randn(MersenneTwister(301), Float64, local_sp..., C_enc, B)
                dec = randn(MersenneTwister(302), Float64, local_sp..., C_dec, B)

                y = DomainDecomposition.dist_skip_forward(enc, dec, skip)

                @test size(y) == (local_sp..., C_enc + C_dec, B)
                @test y[:, :, :, 1:C_enc, :] ≈ enc
                @test y[:, :, :, C_enc+1:end, :] ≈ dec
            end
        end

        @testset "different partitions (with repartition)" begin
            # Use 5D partitions so repartition indices match 5D tensors.
            # Both partitions must have prod(dims)=4 so all ranks participate
            # in repartition (collective operation).
            C_enc, C_dec, B = 4, 6, 1
            P_enc5 = create_cartesian_topology(comm, (2, 2, 1, 1, 1))
            P_dec5 = create_cartesian_topology(comm, (4, 1, 1, 1, 1))
            global_5d = (8, 8, 4, C_enc, B)

            skip = DistSkipConnection(P_enc5, P_dec5, global_5d, comm)
            @test skip.repartition_info !== nothing

            if P_enc5.active
                enc_ranges = subtensor_indices(global_5d, P_enc5.dims, P_enc5.coords)
                enc = fill(Float64(P_enc5.rank + 1), Tuple(length.(enc_ranges))...)
            else
                enc = zeros(Float64, 0, 0, 0, C_enc, B)
            end

            if P_dec5.active
                dec_ranges = subtensor_indices(global_5d, P_dec5.dims, P_dec5.coords)
                dec_sp = Tuple(length(dec_ranges[d]) for d in 1:3)
                dec = randn(MersenneTwister(303), Float64, dec_sp..., C_dec, B)
            else
                dec = zeros(Float64, 0, 0, 0, C_dec, B)
            end

            # All ranks must call dist_skip_forward (repartition is collective)
            y = DomainDecomposition.dist_skip_forward(enc, dec, skip)
            if P_dec5.active
                @test size(y, 4) == C_enc + C_dec
                @test size(y, 5) == B
                # Decoder portion should be unchanged
                @test y[:, :, :, C_enc+1:end, :] ≈ dec
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Test 6: DistDownsample
# ═══════════════════════════════════════════════════════════════════════════════

@testset "DistDownsample" begin
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if nprocs >= 4
        # Both partitions must have prod=4 so all ranks stay active
        # (repartition + conv are collective/must run on non-empty tensors)
        src_dims = (2, 2, 1, 1, 1)
        dst_dims = (4, 1, 1, 1, 1)
        P_src = create_cartesian_topology(comm, src_dims)
        P_dst = create_cartesian_topology(comm, dst_dims)

        C_in, C_out = 2, 4
        B = 1
        global_5d = (8, 8, 4, C_in, B)
        kernel = (2, 2, 2)

        layer = DistDownsample(P_src, P_dst, global_5d, comm, C_in, C_out; kernel_size=kernel)

        @testset "construction" begin
            @test layer.in_channels == C_in
            @test layer.out_channels == C_out
            @test layer.kernel_size == kernel
        end

        using Random
        rng = MersenneTwister(400)
        weight = randn(rng, Float64, kernel..., C_in, C_out)
        bias = zeros(Float64, C_out)

        x_global = randn(MersenneTwister(401), Float64, global_5d...)

        if P_src.active
            src_ranges = subtensor_indices(global_5d, P_src.dims, P_src.coords)
            x_local = x_global[src_ranges...]
        else
            x_local = zeros(Float64, 0, 0, 0, C_in, B)
        end

        ps = (weight=weight, bias=bias)
        st = NamedTuple()

        @testset "Lux interface + output shape" begin
            y, st2 = layer(x_local, ps, st)
            @test st2 === st
            if P_dst.active
                # After repartition + strided conv: output spatial = local_dst_spatial / kernel
                dst_ranges = subtensor_indices(global_5d, P_dst.dims, P_dst.coords)
                dst_local_sp = Tuple(length(dst_ranges[d]) for d in 1:3)
                @test size(y, 1) == dst_local_sp[1] ÷ kernel[1]
                @test size(y, 2) == dst_local_sp[2] ÷ kernel[2]
                @test size(y, 3) == dst_local_sp[3] ÷ kernel[3]
                @test size(y, 4) == C_out
                @test size(y, 5) == B
            end
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Test 7: DistUpsample
# ═══════════════════════════════════════════════════════════════════════════════

@testset "DistUpsample" begin
    MPI.Initialized() || MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)

    if nprocs >= 4
        # Both partitions have prod=4 so all ranks participate
        src_dims = (4, 1, 1, 1, 1)
        dst_dims = (2, 2, 1, 1, 1)
        P_src = create_cartesian_topology(comm, src_dims)
        P_dst = create_cartesian_topology(comm, dst_dims)

        C_in, C_out = 4, 2
        B = 1
        fine_sp = (8, 8, 4)
        # global_5d for repartition is the OUTPUT shape after transposed conv
        global_5d = (fine_sp..., C_out, B)
        kernel = (2, 2, 2)

        layer = DistUpsample(P_src, P_dst, global_5d, comm, C_in, C_out; kernel_size=kernel)

        @testset "construction" begin
            @test layer.in_channels == C_in
            @test layer.out_channels == C_out
            @test layer.kernel_size == kernel
        end

        using Random
        # Weight for transposed conv: (kW, kH, kD, C_out, C_in)
        rng = MersenneTwister(500)
        weight = randn(rng, Float64, kernel..., C_out, C_in)
        bias = zeros(Float64, C_out)

        # Each src rank has a local coarse tile: spatial = global_src_spatial / src_dims
        # The src partition holds the INPUT to upsample, which is the coarse tensor.
        # For repartition: global_5d is the OUTPUT shape. Input is coarser.
        # Each src rank's local input spatial = (global_5d[1:3] .÷ kernel) per-rank
        if P_src.active
            # Coarse global spatial = fine / kernel = (4, 4, 2)
            coarse_global_5d = (fine_sp[1] ÷ kernel[1], fine_sp[2] ÷ kernel[2], fine_sp[3] ÷ kernel[3], C_in, B)
            src_ranges = subtensor_indices(coarse_global_5d, P_src.dims, P_src.coords)
            local_shape_src = Tuple(length.(src_ranges))
            x_local = randn(MersenneTwister(501 + P_src.rank), Float64, local_shape_src...)
        else
            x_local = zeros(Float64, 0, 0, 0, C_in, B)
        end

        ps = (weight=weight, bias=bias)
        st = NamedTuple()

        @testset "Lux interface + output shape" begin
            y, st2 = layer(x_local, ps, st)
            @test st2 === st
            if P_dst.active
                # After transposed conv + repartition to dst
                dst_ranges = subtensor_indices(global_5d, P_dst.dims, P_dst.coords)
                expected_local = Tuple(length.(dst_ranges))
                @test size(y) == expected_local
            end
        end
    end
end
