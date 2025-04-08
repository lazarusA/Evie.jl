export liveTranscribe
export resampleStream
export loadWhisperModel
export startEvie

function liveTranscribe(circ_buf, ctx, wparams)
    c_buf_sampled = SampleBuf(Array(circ_buf), 48000)
    audio_data = resampleStream(c_buf_sampled)
    return transcribeEvie(ctx, wparams, audio_data)
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

function loadWhisperModel(model_att)
    local ctx, wparams
    @suppress begin
        ctx = Whisper.whisper_init_from_file(model_att)
        wparams = Whisper.whisper_full_default_params(Whisper.LibWhisper.WHISPER_SAMPLING_GREEDY)
    end
    return ctx, wparams
end
function transcribeEvie(ctx, wparams, data)
    # Run the heavy computation in a separate thread
    ret = Threads.@spawn begin
        @suppress begin
            Whisper.whisper_full_parallel(ctx, wparams, data, length(data), 1)
        end
    end
    # Wait for the result
    ret = fetch(ret)
    
    if ret != 0
        error("Error running whisper model: $ret")
    end
    
    n_segments = Whisper.whisper_full_n_segments(ctx)
    result = ""
    for i in 0:(n_segments - 1)
        txt = Whisper.whisper_full_get_segment_text(ctx, i)
        result = result * unsafe_string(txt)
    end
    
    return result
end

function startEvie()
    fs, _buf, _ = initFsBuf()  # initialize the audio stream
    audio_obs = Observable(_buf)
    audio_buf = CircularBuffer{Float32}(1024*52) # ≈ 1.109s, circular buffer for the audio
    speech_obs = Observable("[ Silence ]") # text input
    btn_label = Observable("⬤") # button label
    plotSpectrogram(audio_obs, fs, speech_obs, btn_label) 
    return (; audio_obs, audio_buf, speech_obs, btn_label)
end