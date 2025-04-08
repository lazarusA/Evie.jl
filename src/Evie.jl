module Evie
using Makie
using Suppressor
using Makie.FileIO
using PortAudio, SampledSignals, FFTW
using DataStructures: CircularBuffer, isfull, empty!
using Whisper # https://github.com/lazarusA/Whisper.jl.git
using Llama2: load_gguf_model, sample, sampleObs # https://github.com/lazarusA/Llama2.jl.git
export load_gguf_model, sample, sampleObs

include("listen.jl")
include("plot_spectrogram.jl")
include("transcribe_stream.jl")

end
