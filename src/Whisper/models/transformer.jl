module Transformer

using Lux
using Random
using ..Attention

export TransformerBlock, SequentialWithContext

struct TransformerBlock{A, N, CA, CN, F, LN} <: Lux.AbstractLuxLayer
    attention::A
    norm1::N
    cross_attention::CA
    norm_cross::CN
    feedforward::F
    norm2::LN
end

struct SequentialWithContext{L} <: Lux.AbstractLuxLayer
    layers::L
end

function TransformerBlock(d_model, n_heads; cross_attention = false)

    return TransformerBlock(
        MultiHeadSelfAttention(d_model, n_heads),
        LayerNorm(d_model),
        cross_attention ? MultiHeadSelfAttention(d_model, n_heads) : NoOpLayer(),
        cross_attention ? LayerNorm(d_model) : NoOpLayer(),
        Chain(
            Dense(d_model => 4d_model, gelu),
            Dense(4d_model => d_model)
        ),
        LayerNorm(d_model)
    )
end

function Lux.initialparameters(rng::AbstractRNG, m::TransformerBlock)
    return (
        attention = Lux.initialparameters(rng, m.attention),
        norm1 = Lux.initialparameters(rng, m.norm1),
        cross_attention = Lux.initialparameters(rng, m.cross_attention),
        norm_cross = Lux.initialparameters(rng, m.norm_cross),
        feedforward = Lux.initialparameters(rng, m.feedforward),
        norm2 = Lux.initialparameters(rng, m.norm2),
    )
end

function Lux.initialstates(rng::AbstractRNG, m::TransformerBlock)
    return (
        attention = Lux.initialstates(rng, m.attention),
        norm1 = Lux.initialstates(rng, m.norm1),
        cross_attention = Lux.initialstates(rng, m.cross_attention),
        norm_cross = Lux.initialstates(rng, m.norm_cross),
        feedforward = Lux.initialstates(rng, m.feedforward),
        norm2 = Lux.initialstates(rng, m.norm2),
    )
end

# cross-attention present
function _cross_attn(m::TransformerBlock, x, ps, st, context)
    y, st_cn = m.norm_cross(x, ps.norm_cross, st.norm_cross)
    y, st_ca = m.cross_attention(y, ps.cross_attention, st.cross_attention; context)
    return x .+ y, st_ca, st_cn
end

# no cross-attention
function _cross_attn(m::TransformerBlock{A, N, NoOpLayer, NoOpLayer}, x, ps, st, context) where {A, N}
    return x, st.cross_attention, st.norm_cross
end

function (m::TransformerBlock)(x, ps, st; context = nothing, mask = nothing)
    # Self-attention block (pre-norm)
    y, st_n1 = m.norm1(x, ps.norm1, st.norm1)
    y, st_attn = m.attention(y, ps.attention, st.attention; mask)
    x = x .+ y

    # Cross-attention block
    x, st_ca, st_cn = _cross_attn(m, x, ps, st, context)

    # Feedforward block (pre-norm)
    y, st_n2 = m.norm2(x, ps.norm2, st.norm2)
    y, st_ff = m.feedforward(y, ps.feedforward, st.feedforward)

    return x .+ y, (
            attention = st_attn,
            norm1 = st_n1,
            cross_attention = st_ca,
            norm_cross = st_cn,
            feedforward = st_ff,
            norm2 = st_n2,
        )
end

function SequentialWithContext(layers::Vector)
    names = Tuple(Symbol(:layer_, i) for i in eachindex(layers))
    return SequentialWithContext(NamedTuple{names}(layers))
end

function Lux.initialparameters(rng::AbstractRNG, m::SequentialWithContext)
    return NamedTuple{keys(m.layers)}(
        Lux.initialparameters(rng, l) for l in values(m.layers)
    )
end

function Lux.initialstates(rng::AbstractRNG, m::SequentialWithContext)
    return NamedTuple{keys(m.layers)}(
        Lux.initialstates(rng, l) for l in values(m.layers)
    )
end

function (m::SequentialWithContext)(x, ps, st; context = nothing, mask = nothing)
    st_new = st
    for name in keys(m.layers)
        x, st_i = m.layers[name](x, ps[name], st[name]; context, mask)
        st_new = merge(st_new, (; name => st_i))
    end
    return x, st_new
end

end
