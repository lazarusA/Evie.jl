export liveTranscribe
export resampleStream
function liveTranscribe(buf_att, model_att)
    audio_data = resampleStream(buf_att)
    return transcribe(model_att, audio_data)
end

function resampleStream(stream_sample)
    sout = SampleBuf(Float32, 16000, round(Int, length(stream_sample)*(16000/samplerate(stream_sample))), nchannels(stream_sample))
    write(SampleBufSink(sout), SampleBufSource(stream_sample)) # Resample
    return sout.data
end

function transcribe_async!(c_buf, model_att, done)
    @async while !done
        if isfull(c_buf)
            c_buf_sampled = SampleBuf(Array(c_buf), 48000)
            liveTranscribe!(c_buf_sampled, model_att)
            empty!(c_buf) # reset the buffer
        end
    end
end