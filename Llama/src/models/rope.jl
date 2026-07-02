export precompute_rope_freqs, apply_rotary_embeddings

function precompute_rope_freqs(head_dim::Int, max_seq_len::Int; theta::Float32 = 10_000f0)
    @assert iseven(head_dim) "head_dim must be even"
    inv_freq = 1.0f0 ./ (theta .^ (Float32.(0:2:(head_dim - 1)) ./ head_dim))  # (head_dim/2,)
    pos = Float32.(0:(max_seq_len - 1))                                     # (max_seq_len,)
    freqs = inv_freq * pos'                                                 # (head_dim/2, max_seq_len)
    return cos.(freqs), sin.(freqs)   # each (head_dim/2, max_seq_len)
end

# x: (head_dim, n_heads, seq_len, batch)
function apply_rotary_embeddings(x::AbstractArray{T, 4}, cosf, sinf) where {T}
    head_dim, n_heads, seq_len, batch = size(x)
    half = head_dim ÷ 2

    x1 = @view x[1:half, :, :, :]
    x2 = @view x[(half + 1):end, :, :, :]

    c = reshape(cosf, half, 1, seq_len, 1)
    s = reshape(sinf, half, 1, seq_len, 1)

    rotated1 = x1 .* c .- x2 .* s
    rotated2 = x2 .* c .+ x1 .* s

    return vcat(rotated1, rotated2)
end
