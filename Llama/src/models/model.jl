export LlamaModel, LlamaCache

struct LlamaModel{E, L, N, O} <: Lux.AbstractLuxLayer
    token_embedding::E
    blocks::L
    norm::N
    output::O
    n_heads::Int
    n_layers::Int
    max_seq_len::Int
end

function LlamaModel(;
        vocab_size, dim, n_layers, n_heads, n_kv_heads = n_heads,
        max_seq_len, norm_eps = 1.0f-5,
    )
    return LlamaModel(
        Embedding(vocab_size => dim),
        Tuple(TransformerBlock(dim, n_heads, n_kv_heads; norm_eps) for _ in 1:n_layers),
        RMSNorm(dim; eps = norm_eps),
        Dense(dim => vocab_size; use_bias = false),
        n_heads, n_layers, max_seq_len,
    )
end

function Lux.initialparameters(rng::AbstractRNG, m::LlamaModel)
    return (
        token_embedding = Lux.initialparameters(rng, m.token_embedding),
        blocks = Tuple(Lux.initialparameters(rng, b) for b in m.blocks),
        norm = Lux.initialparameters(rng, m.norm),
        output = Lux.initialparameters(rng, m.output),
    )
end

function Lux.initialstates(rng::AbstractRNG, m::LlamaModel)
    return (
        token_embedding = Lux.initialstates(rng, m.token_embedding),
        blocks = Tuple(Lux.initialstates(rng, b) for b in m.blocks),
        norm = Lux.initialstates(rng, m.norm),
        output = Lux.initialstates(rng, m.output),
    )
end

# Holds one KVCache per layer
struct LlamaCache{C}
    layers::C
end

function LlamaCache(m::LlamaModel, head_dim::Int, n_kv_heads::Int, max_batch::Int)
    return LlamaCache(Tuple(KVCache(head_dim, n_kv_heads, m.max_seq_len, max_batch) for _ in 1:m.n_layers))
end

# tokens: (seq_len, batch), 1-based already-shifted ids matching Lux.Embedding convention
function (m::LlamaModel)(tokens, cache::LlamaCache, start_pos::Int, cosf_full, sinf_full, ps, st)
    seq_len = size(tokens, 1)
    x, st_te = m.token_embedding(tokens, ps.token_embedding, st.token_embedding)

    cosf = @view cosf_full[:, start_pos:(start_pos + seq_len - 1)]
    sinf = @view sinf_full[:, start_pos:(start_pos + seq_len - 1)]

    st_blocks = ()
    for (i, block) in enumerate(m.blocks)
        x, st_b = block(x, cache.layers[i], start_pos, cosf, sinf, ps.blocks[i], st.blocks[i])
        st_blocks = (st_blocks..., st_b)
    end

    x, st_n = m.norm(x, ps.norm, st.norm)
    logits, st_o = m.output(x, ps.output, st.output)

    return logits, (token_embedding = st_te, blocks = st_blocks, norm = st_n, output = st_o)
end
