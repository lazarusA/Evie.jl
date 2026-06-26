module Attention

using Lux
using NNlib

struct MultiHeadSelfAttention{Q, K, V, O} <: Lux.AbstractLuxLayer
    query::Q
    key::K
    value::V
    out::O
    n_heads::Integer
end

function MultiHeadSelfAttention(d_model, n_heads)
    return MultiHeadSelfAttention(
        Dense(d_model => d_model),
        Dense(d_model => d_model; use_bias = false),
        Dense(d_model => d_model),
        Dense(d_model => d_model),
        n_heads
    )
end

function (m::MultiHeadSelfAttention)(x, ps, st; context = nothing, mask = nothing)
    src = isnothing(context) ? x : context

    q, st_q = m.query(x, ps.query, st.query)
    k, st_k = m.key(src, ps.key, st.key)
    v, st_v = m.value(src, ps.value, st.value)

    y = dot_product_attention(q, k, v; mask, nheads = m.n_heads)

    out, st_out = m.output(y, ps.output, st.output)

    return out, (query = st_q, key = st_k, value = st_v, output = st_out)
end

end
