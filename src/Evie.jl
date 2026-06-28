module Evie
using FFTW: FFTW, fft, ifft
using FileIO: FileIO
using Makie: Makie, @lift, Axis, DataAspect, Figure, Observable, Vec2f, hidedecorations!, hidespines!, limits!, lines!, scatter!, text!
using PortAudio: PortAudio, PortAudioStream
using SampledSignals
using SampledSignals: SampledSignals, Hz, SampleBuf, domain
using Suppressor: Suppressor
using DataStructures: CircularBuffer, isfull, empty!, OrderedDict
include("Whisper/Whisper.jl")

using Llama2: load_gguf_model, sample, sampleObs # https://github.com/lazarusA/Llama2.jl.git
export load_gguf_model, sample, sampleObs

include("listen.jl")
include("plot_spectrogram.jl")
include("transcribe_stream.jl")

end
