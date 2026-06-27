module Whisper

using Accessors
using Lux
using NNlib
using Random
using Downloads
using Pickle
using DataStructures

include("./models/attention.jl")
include("./models/transformer.jl")
include("./models/embeddings.jl")
include("./models/masks.jl")
include("./models/encoder.jl")
include("./models/decoder.jl")

using .Masks
using .Embeddings
using .Attention
using .Transformer
using .Encoder
using .Decoder

include("./models/model.jl")

include("./weights/registry.jl")
include("./weights/download.jl")
include("./weights/loader.jl")
include("./weights/mapping.jl")

export WhisperModel, download_weights, load_checkpoint, load_model 

end
