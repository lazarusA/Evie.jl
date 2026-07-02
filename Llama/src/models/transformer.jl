export RMSNorm, FeedForward, TransformerBlock

struct RMSNorm{W} <: Lux.AbstractLuxLayer
    dim::Int
    eps::Float32
end
RMSNorm(dim::Int; eps::Float32 = 1.0f-5) = RMSNorm{Nothing}(dim, eps)

Lux.initialparameters(rng::AbstractRNG, m::RMSNorm) = (weight = ones(Float32, m.dim),)
Lux.initialstates(rng::AbstractRNG, m::RMSNorm) = NamedTuple()

function (m::RMSNorm)(x, ps, st)
    # x: (dim, seq_len, batch) — normalize over dim 1, matching how LayerNorm
    # was wired for Whisper (dims=1), not Lux's Colon default.
    ms = mean(abs2, x; dims = 1)
    x_normed = x ./ sqrt.(ms .+ m.eps)
    return reshape(ps.weight, m.dim, 1, 1) .* x_normed, st
end

struct FeedForward{W1, W2, W3} <: Lux.AbstractLuxLayer
    w1::W1   # gate
    w2::W2   # down
    w3::W3   # up
end

function FeedForward(dim::Int; multiple_of::Int = 256, ffn_dim_multiplier::Maybe{Float64} = nothing)
    hidden = 2 * (4 * dim) ÷ 3
    hidden = ffn_dim_multiplier === nothing ? hidden : floor(Int, hidden * ffn_dim_multiplier)
    hidden = multiple_of * cld(hidden, multiple_of)
    return FeedForward(
        Dense(dim => hidden; use_bias = false),
        Dense(hidden => dim; use_bias = false),
        Dense(dim => hidden; use_bias = false),
    )
end

Lux.initialparameters(rng::AbstractRNG, m::FeedForward) =
    (w1 = Lux.initialparameters(rng, m.w1), w2 = Lux.initialparameters(rng, m.w2), w3 = Lux.initialparameters(rng, m.w3))
Lux.initialstates(rng::AbstractRNG, m::FeedForward) =
    (w1 = Lux.initialstates(rng, m.w1), w2 = Lux.initialstates(rng, m.w2), w3 = Lux.initialstates(rng, m.w3))

function (m::FeedForward)(x, ps, st)
    g, st_w1 = m.w1(x, ps.w1, st.w1)
    u, st_w3 = m.w3(x, ps.w3, st.w3)
    h = NNlib.swish.(g) .* u
    y, st_w2 = m.w2(h, ps.w2, st.w2)
    return y, (w1 = st_w1, w2 = st_w2, w3 = st_w3)
end

struct TransformerBlock{A, N1, F, N2} <: Lux.AbstractLuxLayer
    attention::A
    attn_norm::N1
    feed_forward::F
    ffn_norm::N2
end

function TransformerBlock(dim, n_heads, n_kv_heads; norm_eps = 1.0f-5)
    return TransformerBlock(
        GroupedQueryAttention(dim, n_heads, n_kv_heads),
        RMSNorm(dim; eps = norm_eps),
        FeedForward(dim),
        RMSNorm(dim; eps = norm_eps),
    )
end

Lux.initialparameters(rng::AbstractRNG, m::TransformerBlock) = (
    attention = Lux.initialparameters(rng, m.attention),
    attn_norm = Lux.initialparameters(rng, m.attn_norm),
    feed_forward = Lux.initialparameters(rng, m.feed_forward),
    ffn_norm = Lux.initialparameters(rng, m.ffn_norm),
)
Lux.initialstates(rng::AbstractRNG, m::TransformerBlock) = (
    attention = Lux.initialstates(rng, m.attention),
    attn_norm = Lux.initialstates(rng, m.attn_norm),
    feed_forward = Lux.initialstates(rng, m.feed_forward),
    ffn_norm = Lux.initialstates(rng, m.ffn_norm),
)

function (m::TransformerBlock)(x, cache, start_pos, cosf, sinf, ps, st)
    y, st_n1 = m.attn_norm(x, ps.attn_norm, st.attn_norm)
    y, st_attn = m.attention(y, cache, start_pos, cosf, sinf, ps.attention, st.attention)
    x = x .+ y

    y, st_n2 = m.ffn_norm(x, ps.ffn_norm, st.ffn_norm)
    y, st_ff = m.feed_forward(y, ps.feed_forward, st.feed_forward)
    x = x .+ y

    return x, (attention = st_attn, attn_norm = st_n1, feed_forward = st_ff, ffn_norm = st_n2)
end
