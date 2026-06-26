module Embeddings

using Lux

export PositionEmbedding, TokenEmbedding

struct PositionEmbedding{E} <: Lux.AbstractLuxLayer
    embedding::E
end

function PositionEmbedding(n_positions, d_model)
    return PositionEmbedding(Embedding(n_positions => d_model))
end

function (m::PositionEmbedding)(x, ps, st)
    pos = 1:size(x, 1)
    emb, st_emb = m.embedding(pos, ps.embedding, st.embedding)
    return x .+ emb, (embedding = st_emb,)
end

struct TokenEmbedding{E} <: Lux.AbstractLuxLayer
    embedding::E
end

function TokenEmbedding(n_vocab, d_model)
    return TokenEmbedding(Embedding(n_vocab => d_model))
end

function (m::TokenEmbedding)(x, ps, st)
    emb, st_emb = m.embedding(x, ps.embedding, st.embedding)
    return emb, (embedding = st_emb,)
end

end
