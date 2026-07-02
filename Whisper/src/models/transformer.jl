export TransformerBlock

struct TransformerBlock{A, N, CA, CN, F, LN} <: LuxCore.AbstractLuxContainerLayer{(:attention, :norm1, :cross_attention, :norm_cross, :feedforward, :norm2)}
    attention::A
    norm1::N
    cross_attention::CA
    norm_cross::CN
    feedforward::F
    norm2::LN
end

function TransformerBlock(d_model, n_heads; cross_attention = false)

    return TransformerBlock(
        MultiHeadSelfAttention(d_model, n_heads),
        LayerNorm((d_model,); dims = 1),
        cross_attention ? MultiHeadSelfAttention(d_model, n_heads) : NoOpLayer(),
        cross_attention ? LayerNorm((d_model,); dims = 1) : NoOpLayer(),
        Chain(
            Dense(d_model => 4d_model, gelu),
            Dense(4d_model => d_model)
        ),
        LayerNorm((d_model,); dims = 1)
    )
end

function _cross_attn(m::TransformerBlock, x, ps, st, context)
    y, st_cn = m.norm_cross(x, ps.norm_cross, st.norm_cross)
    y, st_ca = m.cross_attention(y, ps.cross_attention, st.cross_attention; context)
    return x .+ y, st_ca, st_cn
end

function _cross_attn(m::TransformerBlock{A, N, NoOpLayer, NoOpLayer}, x, ps, st, context) where {A, N}
    return x, st.cross_attention, st.norm_cross
end

function (m::TransformerBlock)(x, ps, st; context = nothing, mask = nothing)
    y, st_n1 = m.norm1(x, ps.norm1, st.norm1)
    y, st_attn = m.attention(y, ps.attention, st.attention; mask)
    x = x .+ y

    x, st_ca, st_cn = _cross_attn(m, x, ps, st, context)

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
