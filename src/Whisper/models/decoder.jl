struct WhisperDecoder <: Lux.AbstractLuxLayer
    token_embedding
    position_embedding
    layers
    norm
end
