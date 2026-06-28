module Whisper

using Accessors
using Lux
using NNlib
using Random
using Downloads
using Pickle
using DataStructures: OrderedDict
using Distributions: Categorical
using LinearAlgebra: tril
using Base64
using Printf
using FileIO
using LibSndFile
using SampledSignals

include("./models/attention.jl")
include("./models/transformer.jl")
include("./models/embeddings.jl")
include("./models/encoder.jl")
include("./models/decoder.jl")
include("./models/model.jl")

include("./weights/registry.jl")
include("./weights/mapping.jl")

# tokens / transcribe
include("./inference/tokenizer.jl")
include("./inference/transcribe.jl")

# audio
include("./audio/samples.jl")
include("./audio/preprocess.jl")

export WhisperModel, download_weights, load_checkpoint, load_model, load_vocab_file

end
