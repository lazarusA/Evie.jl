export live_transcribe!
export resampleStream
function liveTranscribe!(buf_att, model_att, txt_att::Observable)
    audio_data = resampleStream(buf_att)
    txt_query = transcribe(model_att, audio_data)
    txt_att[] = txt_query
end

function resampleStream(stream_sample)
    sout = SampleBuf(Float32, 16000, round(Int, length(stream_sample)*(16000/samplerate(stream_sample))), nchannels(stream_sample))
    write(SampleBufSink(sout), SampleBufSource(stream_sample)) # Resample
    return sout.data
end

function transcribe_async!(stream, model_att, done)
    @async while !done
        buf_att=read(stream, 1.25s)
        live_transcribe!(buf_att, model_att, txt_att)
    end
end