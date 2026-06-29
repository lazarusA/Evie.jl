export MultiHeadSelfAttention

struct MultiHeadSelfAttention{Q, K, V, O} <: Lux.AbstractLuxLayer
    query::Q
    key::K
    value::V
    out::O
    n_heads::Int
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

function Lux.initialparameters(rng::AbstractRNG, m::MultiHeadSelfAttention)
    return (
        query = Lux.initialparameters(rng, m.query),
        key = Lux.initialparameters(rng, m.key),
        value = Lux.initialparameters(rng, m.value),
        out = Lux.initialparameters(rng, m.out),
    )
end

function Lux.initialstates(rng::AbstractRNG, m::MultiHeadSelfAttention)
    return (
        query = Lux.initialstates(rng, m.query),
        key = Lux.initialstates(rng, m.key),
        value = Lux.initialstates(rng, m.value),
        out = Lux.initialstates(rng, m.out),
    )
end

function (m::MultiHeadSelfAttention)(x, ps, st; context = nothing, mask = nothing)
    # @info "MultiHeadSelfAttention" is_cross=!isnothing(context) mask_is_nothing=isnothing(mask) mask_size=(isnothing(mask) ? nothing : size(mask))
    src = isnothing(context) ? x : context

    q, st_q = m.query(x, ps.query, st.query)
    k, st_k = m.key(src, ps.key, st.key)
    v, st_v = m.value(src, ps.value, st.value)

    y, _ = dot_product_attention(q, k, v; mask, nheads = m.n_heads)

    out, st_out = m.out(y, ps.out, st.out)

    return out, (query = st_q, key = st_k, value = st_v, out = st_out)
end
