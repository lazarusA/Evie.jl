function live_transcribe!(buf_att, model_att, txt_att::Observable)
    audio_data = resample_stream(buf_att)
    txt_query = transcribe(model_att, audio_data)
    txt_att[] = txt_query
end
function transcribe_async!(stream, model_att, done)
    @async while !done
        buf_att=read(stream, 1.25s)
        live_transcribe!(buf_att, model_att, txt_att)
    end
end