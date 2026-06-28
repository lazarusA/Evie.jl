export WhisperEncoder

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
        SequentialWithContext(
            [TransformerBlock(d_model, n_heads) for _ in 1:n_layers]
        ),
        LayerNorm((d_model,)),
        PositionEmbedding(max_positions, d_model; dim = 2)
    )
end
function Lux.initialparameters(rng::AbstractRNG, m::WhisperEncoder)
    return (
        frontend = Lux.initialparameters(rng, m.frontend),
        layers = Lux.initialparameters(rng, m.layers),
        norm = Lux.initialparameters(rng, m.norm),
        position = Lux.initialparameters(rng, m.position),
    )
end

function Lux.initialstates(rng::AbstractRNG, m::WhisperEncoder)
    return (
        frontend = Lux.initialstates(rng, m.frontend),
        layers = Lux.initialstates(rng, m.layers),
        norm = Lux.initialstates(rng, m.norm),
        position = Lux.initialstates(rng, m.position),
    )
end

function (m::WhisperEncoder)(x, ps, st)
    x, st_fe = m.frontend(x, ps.frontend, st.frontend)
    x = permutedims(x, (2, 1, 3))
    x, st_pos = m.position(x, ps.position, st.position)
    x, st_ly = m.layers(x, ps.layers, st.layers)
    x, st_n = m.norm(x, ps.norm, st.norm)
    return x, (frontend = st_fe, layers = st_ly, norm = st_n, position = st_pos)
end
