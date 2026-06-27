module Decoder

using Lux
using Random
using NNlib
using ..Transformer
using ..Embeddings
using ..Masks

export WhisperDecoder

struct WhisperDecoder{T, P, L, N} <: Lux.AbstractLuxLayer
    token_embedding::T
    position_embedding::P
    layers::L
    norm::N
end

function WhisperDecoder(; n_vocab, d_model, n_layers, n_heads, max_positions)
    return WhisperDecoder(
        TokenEmbedding(n_vocab, d_model),
        PositionEmbedding(max_positions, d_model; dim = 2),
        SequentialWithContext(
            [TransformerBlock(d_model, n_heads; cross_attention = true) for _ in 1:n_layers]
        ),
        LayerNorm((d_model,))
    )
end

function Lux.initialparameters(rng::AbstractRNG, m::WhisperDecoder)
    return (
        token_embedding = Lux.initialparameters(rng, m.token_embedding),
        position_embedding = Lux.initialparameters(rng, m.position_embedding),
        layers = Lux.initialparameters(rng, m.layers),
        norm = Lux.initialparameters(rng, m.norm),
    )
end

function Lux.initialstates(rng::AbstractRNG, m::WhisperDecoder)
    return (
        token_embedding = Lux.initialstates(rng, m.token_embedding),
        position_embedding = Lux.initialstates(rng, m.position_embedding),
        layers = Lux.initialstates(rng, m.layers),
        norm = Lux.initialstates(rng, m.norm),
    )
end

function (m::WhisperDecoder)(tokens, encoder_out, ps, st)
    # Token + positional embeddings
    x, st_te = m.token_embedding(tokens, ps.token_embedding, st.token_embedding)
    x, st_pe = m.position_embedding(x, ps.position_embedding, st.position_embedding)

    # Causal mask for self-attention
    mask = causal_mask(size(tokens, 1))

    # Transformer layers — context and mask threaded via SequentialWithContext
    x, st_ly = m.layers(x, ps.layers, st.layers; context = encoder_out, mask)

    # Final norm
    x, st_n = m.norm(x, ps.norm, st.norm)

    # Tied output projection
    logits = ps.token_embedding.embedding.weight' ⊠ x

    return logits, (
            token_embedding = st_te,
            position_embedding = st_pe,
            layers = st_ly,
            norm = st_n,
        )
end

end
