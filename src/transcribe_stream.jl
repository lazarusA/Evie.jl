export liveTranscribe
export startEvie

function liveTranscribe(circ_buf, model_att)
    c_buf_sampled = SampleBuf(Array(circ_buf), 48000)
    return transcribe(model_att, c_buf_sampled)
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