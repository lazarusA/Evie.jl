module Encoder

using Lux
using ..Transformer

struct WhisperEncoder{C, L, N, P} <: Lux.AbstractLuxLayer
    frontend::C
    layers::L
    norm::N
    position::P
end

function WhisperEncoder(; n_mels, d_model, n_layers, n_heads, max_positions)
    return WhisperEncoder(
        Chain(
            Conv((3,), n_mels => d_model, activation = gelu, pad = SamePad()),
            Conv((3,), d_model => d_model, activation = gelu, stride = 2, pad = SamePad())
        ),
        Chain(
            [TransformerBlock(d_model, n_heads) for _ in 1:n_layers]...
        ),
        LayerNorm(d_model),
        Embedding(max_positions => d_model)
    )
end

end
