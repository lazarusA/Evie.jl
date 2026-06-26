module Whisper

using Lux

struct WhisperModel{E, D} <: Lux.AbstractLuxLayer
    encoder::E
    decoder::D
end

function (m::WhisperModel)(mel, tokens, ps, st; mask = nothing)
    enc, st_enc = m.encoder(mel, ps.encoder, st.encoder)
    dec, st_dec = m.decoder(tokens, enc, ps.decoder, st.decoder; mask)

    return dec, (encoder = st_enc, decoder = st_dec)
end

include("./models/attention.jl")
include("./models/transformer.jl")
include("./models/encoder.jl")
include("./models/decoder.jl")
include("./models/embeddings.jl")
include("./models/masks.jl")

export WhisperModel

end
