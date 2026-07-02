export TokenEmbedding, PositionEmbedding

struct TokenEmbedding{E} <: LuxCore.AbstractLuxContainerLayer{(:embedding,)}
    embedding::E
    function TokenEmbedding{E}(embedding::E) where {E}
        return new{E}(embedding)
    end
end

function TokenEmbedding(n_vocab::Integer, d_model::Integer)
    emb = Embedding(n_vocab => d_model)
    return TokenEmbedding{typeof(emb)}(emb)
end

function (m::TokenEmbedding)(x, ps, st)
    emb, st_emb = m.embedding(x, ps.embedding, st.embedding)
    return emb, (embedding = st_emb,)
end

struct PositionEmbedding{E} <: LuxCore.AbstractLuxContainerLayer{(:embedding,)}
    embedding::E
    dim::Int
    function PositionEmbedding{E}(embedding::E, dim::Int) where {E}
        return new{E}(embedding, dim)
    end
end

function PositionEmbedding(n_positions::Integer, d_model::Integer; dim::Int = 1)
    emb = Embedding(n_positions => d_model)
    return PositionEmbedding{typeof(emb)}(emb, dim)
end

function (m::PositionEmbedding)(x, ps, st)
    pos = 1:size(x, m.dim)
    emb, st_emb = m.embedding(pos, ps.embedding, st.embedding)
    return x .+ reshape(emb, size(emb, 1), size(emb, 2), 1), (embedding = st_emb,)
end
