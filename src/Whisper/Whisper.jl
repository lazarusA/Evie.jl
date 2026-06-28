module Whisper

using AbstractFFTs: AbstractFFTs, fftfreq, rfft
using Accessors: Accessors, @set
using Base64: Base64
using DataStructures: OrderedDict
using Distributions: Categorical
using Downloads: Downloads
using FFTW: FFTW
using FileIO: FileIO
using LibSndFile: LibSndFile
using LinearAlgebra: triu
using Lux: Lux, Chain, Conv, Dense, LayerNorm, NoOpLayer
using NNlib: NNlib, dot_product_attention, gelu, softmax, ⊠
using Pickle: Pickle
using Printf: Printf, @sprintf
using Random: Random, AbstractRNG
using SampledSignals: SampledSignals, SampleBuf, SampleBufSink, SampleBufSource, nchannels, nframes, samplerate
using Statistics: Statistics

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
