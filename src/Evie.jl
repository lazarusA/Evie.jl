module Evie
using Makie
using Makie.FileIO
using PortAudio, SampledSignals, FFTW
using DataStructures: CircularBuffer
using Whisper
# Write your package code here.
include("listen.jl")
include("plot_spectrogram.jl")
include("transcribe_stream.jl")

end
