# Magnitude-preserving operations from Karras et al. (2024)
#
# All operations are pointwise/local — no MPI communication needed.
# AD is handled automatically by ChainRulesCore (no custom rrules).

"""
    mp_silu(x)

Magnitude-preserving SiLU activation (Section 2.5 of Karras et al.).
Scales silu output to approximately preserve input magnitude.

    mp_silu(x) = silu(x) / 0.596
"""
mp_silu(x) = NNlib.swish(x) / eltype(x)(0.596)

"""
    mp_add(x, res; t=0.3f0)

Magnitude-preserving addition (Equation 88).
Weighted sum normalized to preserve expected magnitude.

    mp_add(x, res; t) = ((1-t)*x + t*res) / sqrt((1-t)^2 + t^2)
"""
function mp_add(x::AbstractArray{T}, res::AbstractArray{T}; t::Real = T(0.3)) where {T}
    t = T(t)
    num = (one(T) - t) .* x .+ t .* res
    den = sqrt((one(T) - t)^2 + t^2)
    return num ./ den
end

"""
    mp_cat(a, b; t=0.5f0, dim=4)

Magnitude-preserving concatenation (Equation 103).
Weights and normalizes the two tensors before concatenation to preserve magnitude.
"""
function mp_cat(a::AbstractArray{T}, b::AbstractArray{T}; t::Real = T(0.5), dim::Int = 4) where {T}
    t = T(t)
    Na = size(a, dim)
    Nb = size(b, dim)
    C = sqrt(T(Na + Nb) / ((one(T) - t)^2 + t^2))
    a_scaled = a .* ((one(T) - t) / sqrt(T(Na)))
    b_scaled = b .* (t / sqrt(T(Nb)))
    return C .* cat(a_scaled, b_scaled; dims=dim)
end

"""
    pixel_norm(x; dim=4, eps=1f-4)

Pixel normalization (Equation 30): L2 normalize per channel, scale by sqrt(C).
No learnable parameters.

    pixel_norm(x) = x / ||x||_2 * sqrt(C)

where C is the size along `dim`.
"""
function pixel_norm(x::AbstractArray{T}; dim::Int = 4, eps::Real = T(1e-4)) where {T}
    C = T(size(x, dim))
    # L2 normalize along dim, then scale by sqrt(C)
    norm_sq = sum(x .^ 2; dims=dim)
    norm = sqrt.(norm_sq .+ eps)
    return x ./ norm .* sqrt(C)
end

"""
    normalize_weight(w; eps=1f-4)

Forced weight normalization (Algorithm 1 in Karras et al.).
Normalizes each output filter to unit L2 norm, then scales by sqrt(numel / fan_out).

For a weight tensor of shape (k1, k2, k3, C_in, C_out), the output (last) dimension
is the fan_out dimension. Each slice w[:,:,:,:,i] is normalized.
"""
function normalize_weight(w::AbstractArray{T}; eps::Real = T(1e-4)) where {T}
    # Convention: for NNlib conv weights (k1,k2,k3,C_in,C_out), the last dim is output.
    # For linear weights (out, in), the first dim is output.
    # We follow the NNlib convention: last dim = output (fan_out).
    # Reshape to (fan_in_total, fan_out) and normalize each column.
    fan_out = size(w, ndims(w))
    numel = length(w)
    w_2d = reshape(w, :, fan_out)
    norms = sqrt.(sum(w_2d .^ 2; dims=1) .+ eps)
    w_normed = w_2d ./ norms
    w_normed = w_normed .* sqrt(T(numel) / T(fan_out))
    return reshape(w_normed, size(w))
end

"""
    mp_fourier_embedding(t, freqs)

Magnitude-preserving Fourier embedding for time/noise conditioning.

`t`: (batch,) — scalar conditioning values (e.g., log(sigma)/4)
`freqs`: (half_dim,) — random Fourier frequencies (fixed, not learned)

Returns: (2 * half_dim, batch) — sin/cos features scaled by sqrt(2)
"""
function mp_fourier_embedding(t::AbstractVector{T}, freqs::AbstractVector{T}) where {T}
    # t: (batch,), freqs: (half_dim,)
    # angles = freqs * t' * 2pi → (half_dim, batch)
    angles = freqs * t' .* T(2π)
    # Concatenate sin and cos, scale by sqrt(2)
    return vcat(sin.(angles), cos.(angles)) .* sqrt(T(2))
end
