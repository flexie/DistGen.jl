using Test
using NNlib
using DomainDecomposition

# ═══════════════════════════════════════════════════════════════════════════════
# Test: Magnitude-Preserving Operations (no MPI needed — single process)
# ═══════════════════════════════════════════════════════════════════════════════

@testset "mp_silu" begin
    x = randn(Float32, 4, 4, 4, 2, 1)
    y = mp_silu(x)
    @test size(y) == size(x)
    # mp_silu(0) = 0
    @test mp_silu(0.0f0) ≈ 0.0f0 atol=1e-6
    # mp_silu should scale silu by 1/0.596
    @test mp_silu(1.0f0) ≈ NNlib.swish(1.0f0) / 0.596f0 atol=1e-6
end

@testset "mp_add" begin
    x = randn(Float32, 4, 4, 4, 2, 1)
    # mp_add(x, x; t=0.5) should return x (identity when inputs equal)
    y = mp_add(x, x; t=0.5f0)
    # (0.5*x + 0.5*x) / sqrt(0.25 + 0.25) = x / sqrt(0.5) = x * sqrt(2)
    # Actually: num = x, den = sqrt(0.5), so y = x / sqrt(0.5) = x * sqrt(2)
    # Wait, let me recalculate:
    # num = (1-t)*x + t*x = x
    # den = sqrt((1-t)^2 + t^2) = sqrt(0.5)
    # y = x / sqrt(0.5) ≈ x * 1.4142
    @test y ≈ x ./ sqrt(0.5f0)

    # With t=0: mp_add(x, res) = x (pure first input)
    res = randn(Float32, 4, 4, 4, 2, 1)
    y0 = mp_add(x, res; t=0.0f0)
    @test y0 ≈ x

    # With t=1: mp_add(x, res) = res (pure second input)
    y1 = mp_add(x, res; t=1.0f0)
    @test y1 ≈ res
end

@testset "mp_cat" begin
    a = randn(Float32, 4, 4, 4, 3, 1)
    b = randn(Float32, 4, 4, 4, 5, 1)
    y = mp_cat(a, b; t=0.5f0, dim=4)
    @test size(y) == (4, 4, 4, 8, 1)

    # Equal-sized inputs with t=0.5
    a2 = randn(Float32, 4, 4, 4, 4, 1)
    b2 = randn(Float32, 4, 4, 4, 4, 1)
    y2 = mp_cat(a2, b2; t=0.5f0, dim=4)
    @test size(y2) == (4, 4, 4, 8, 1)
end

@testset "pixel_norm" begin
    x = randn(Float32, 4, 4, 4, 8, 2)
    y = pixel_norm(x; dim=4)
    @test size(y) == size(x)

    # Each spatial/batch position should have per-channel L2 norm ≈ sqrt(C)
    C = 8
    for b in 1:2, k in 1:4, j in 1:4, i in 1:4
        vec = y[i, j, k, :, b]
        @test sqrt(sum(vec .^ 2)) ≈ sqrt(Float32(C)) atol=0.1
    end
end

@testset "normalize_weight" begin
    # 3×3×3 conv weight: (kW, kH, kD, C_in, C_out)
    w = randn(Float32, 3, 3, 3, 4, 8)
    wn = normalize_weight(w)
    @test size(wn) == size(w)

    # Each output filter should have norm ≈ sqrt(numel / fan_out)
    fan_out = 8
    numel = length(w)
    expected_norm = sqrt(Float32(numel) / Float32(fan_out))
    for i in 1:fan_out
        filter_norm = sqrt(sum(wn[:, :, :, :, i] .^ 2))
        @test filter_norm ≈ expected_norm atol=0.01
    end

    # 2D weight: (in_features, out_features) — last dim = fan_out (NNlib convention)
    w2 = randn(Float32, 6, 10)
    wn2 = normalize_weight(w2)
    fan_out2 = 10  # last dim
    numel2 = length(w2)
    expected2 = sqrt(Float32(numel2) / Float32(fan_out2))
    for j in 1:fan_out2
        @test sqrt(sum(wn2[:, j] .^ 2)) ≈ expected2 atol=0.01
    end
end

@testset "mp_fourier_embedding" begin
    using Random
    rng = MersenneTwister(42)
    half_dim = 8
    freqs = randn(rng, Float32, half_dim)
    t = randn(Float32, 3)  # batch of 3

    emb = mp_fourier_embedding(t, freqs)
    @test size(emb) == (2 * half_dim, 3)

    # First half should be sin, second half cos, both scaled by sqrt(2)
    angles = freqs * t' .* Float32(2π)
    @test emb[1:half_dim, :] ≈ sin.(angles) .* sqrt(2.0f0)
    @test emb[half_dim+1:end, :] ≈ cos.(angles) .* sqrt(2.0f0)

    # Different time values should give different embeddings
    t2 = t .+ 1.0f0
    emb2 = mp_fourier_embedding(t2, freqs)
    @test !(emb ≈ emb2)
end
