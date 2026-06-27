struct WhisperModel{E, D} <: Lux.AbstractLuxLayer
    encoder::E
    decoder::D
end

function WhisperModel(;
        n_mels, d_model, n_layers_enc, n_heads_enc, max_positions_enc,
        n_vocab, n_layers_dec, n_heads_dec, max_positions_dec
    )
    encoder = WhisperEncoder(;
        n_mels, d_model,
        n_layers = n_layers_enc,
        n_heads = n_heads_enc,
        max_positions = max_positions_enc
    )
    decoder = WhisperDecoder(;
        n_vocab, d_model,
        n_layers = n_layers_dec,
        n_heads = n_heads_dec,
        max_positions = max_positions_dec
    )
    return WhisperModel(encoder, decoder)
end

function (m::WhisperModel)(mel, tokens, ps, st; mask = nothing)
    enc, st_enc = m.encoder(mel, ps.encoder, st.encoder)
    dec, st_dec = m.decoder(tokens, enc, ps.decoder, st.decoder; mask)
    return dec, (encoder = st_enc, decoder = st_dec)
end
