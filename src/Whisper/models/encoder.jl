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
            Conv((3,), n_mels => d_model, gelu; pad = 1),
            Conv((3,), d_model => d_model, gelu; pad = 1, stride = 2)
        ),
        Chain(
            [TransformerBlock(d_model, n_heads) for _ in 1:n_layers]...
        ),
        LayerNorm(d_model),
        Embedding(max_positions => d_model)
    )
end
function (m::WhisperEncoder)(x, ps, st)
    x, st_fe = m.frontend(x, ps.frontend, st.frontend)
    x = permutedims(x, (2, 1, 3))
    emb, st_pos = m.position(1:size(x, 2), ps.position, st.position)
    x = x .+ emb
    x, st_ly = m.layers(x, ps.layers, st.layers)
    x, st_n = m.norm(x, ps.norm, st.norm)
    return x, (frontend = st_fe, layers = st_ly, norm = st_n, position = st_pos)
end

end
