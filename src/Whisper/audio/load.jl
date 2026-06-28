using FileIO
using LibSndFile
using SampledSignals

const SAMPLE_RATE = 16_000   # Whisper expects 16kHz

function load_audio(path::String)
    s = FileIO.load(path)
    n_frames = round(Int, nframes(s) * (SAMPLE_RATE / samplerate(s)))  # nframes not length
    sout = SampleBuf(Float32, SAMPLE_RATE, n_frames, nchannels(s))
    write(SampleBufSink(sout), SampleBufSource(s))

    if nchannels(sout) == 1
        return vec(sout.data)
    elseif nchannels(sout) == 2
        sd = sout.data
        return Float32[(sd[i, 1] + sd[i, 2]) / 2.0f0 for i in 1:size(sd, 1)]
    else
        error("Unsupported number of channels: $(nchannels(sout))")
    end
end
