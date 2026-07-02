module Llama

const Maybe{T} = Union{Nothing, T}

using Accessors: Accessors, @set
using Distributions: Categorical
using JSON3: JSON3
using Lux: Lux, Chain, Dense, Embedding, NoOpLayer
using NNlib: NNlib, dot_product_attention, softmax, swish
using Pickle: Pickle
using PythonCall: PythonCall, Py, pyimport, pyconvert
using Random: Random, AbstractRNG
using Statistics: mean

include("./models/rope.jl")
include("./models/attention.jl")
include("./models/transformer.jl")
include("./models/model.jl")

# include("./weights/registry.jl")
# include("./weights/mapping.jl")
# include("./inference/tokenizer.jl")
# include("./inference/generate.jl")

# export load_model, load_checkpoint
# export consolidated_paths, params_path, tokenizer_path
# export generate

export load_gguf_model, sample, sampleObs

function load_gguf_model end
function sample end
function sampleObs end

end # module Llama
