export WhisperDecoder

struct WhisperDecoder{T, P, L, N} <: LuxCore.AbstractLuxContainerLayer{(:token_embedding, :position_embedding, :layers, :norm)}
    token_embedding::T
    position_embedding::P
    layers::L
    norm::N
end

function WhisperDecoder(; n_vocab, d_model, n_layers, n_heads, max_positions)
    return WhisperDecoder(
        TokenEmbedding(n_vocab, d_model),
        PositionEmbedding(max_positions, d_model; dim = 2),
        TransformerStack(
            [TransformerBlock(d_model, n_heads; cross_attention = true) for _ in 1:n_layers]
        ),
        LayerNorm((d_model,); dims = 1)
    )
end

function (m::WhisperDecoder)(tokens, encoder_out, ps, st)
    x, st_te = m.token_embedding(tokens .+ Int32(1), ps.token_embedding, st.token_embedding)
    x, st_pe = m.position_embedding(x, ps.position_embedding, st.position_embedding)

    mask = causal_mask(size(tokens, 1))
    x, st_ly = m.layers(x, ps.layers, st.layers; context = encoder_out, mask)

    x, st_n = m.norm(x, ps.norm, st.norm)
    logits = ps.token_embedding.embedding.weight' ⊠ x

    return logits, (
            token_embedding = st_te,
            position_embedding = st_pe,
            layers = st_ly,
            norm = st_n,
        )
end

causal_mask(n::Int) = triu(ones(Bool, n, n))
