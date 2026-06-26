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
    pos = reshape(1:size(x, 2), :)
    emb, st = m.embedding(pos, ps.embedding, st.embedding)

    return x .+ emb, st
end

struct TokenEmbedding{E} <: Lux.AbstractLuxLayer
    embedding::E
end

end
