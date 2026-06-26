module Transformer

using Lux
using ..Attention

struct TransformerBlock{A, N, F, CN} <: Lux.AbstractLuxLayer
    attention::A
    norm1::N
    cross_attention
    norm_cross
    feedforward::F
    norm2::CN
end

function TransformerBlock(d_model, n_heads; cross_attention = false)

    return TransformerBlock(
        MultiHeadSelfAttention(d_model, n_heads),
        LayerNorm(d_model),

        cross_attention ?
            MultiHeadSelfAttention(d_model, n_heads) :
            NoOpLayer(),

        cross_attention ?
            LayerNorm(d_model) :
            NoOpLayer(),

        Chain(
            Dense(d_model => 4d_model, gelu),
            Dense(4d_model => d_model)
        ),

        LayerNorm(d_model)
    )
end

function (m::TransformerBlock)(x, ps, st; context = nothing, mask = nothing)

    y, st_n1 = m.norm1(x, ps.norm1, st.norm1)
    y, st_attn = m.attention(y, ps.attention, st.attention; mask)
    x = x .+ y

    if !(m.cross_attention isa NoOpLayer)
        y, st_crossnorm = m.norm_cross(x, ps.norm_cross, st.norm_cross)
        y, st_cross = m.cross_attention(y, ps.cross_attention, st.cross_attention; context)
        x = x .+ y
    else
        st_cross = st.cross_attention
        st_crossnorm = st.norm_cross
    end

    y, st_n2 = m.norm2(x, ps.norm2, st.norm2)
    y, st_ff = m.feedforward(y, ps.feedforward, st.feedforward)

    return x .+ y, (
            attention = st_attn,
            norm1 = st_n1,
            cross_attention = st_cross,
            norm_cross = st_crossnorm,
            feedforward = st_ff,
            norm2 = st_n2,
        )
end

end
