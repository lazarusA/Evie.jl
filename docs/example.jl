using Evie, GLMakie
using DataStructures
using Whisper
using Llama2
using SampledSignals
using GLMakie.FileIO

fs, _buf, buf = initFsBuf()
buf_obs = Observable(_buf)
circ_buf = CircularBuffer{Float32}(1024*52) # ≈ 1.109s
txt_obs = Observable("[ Silence ]")

with_theme(theme_black()) do
     # backgroundcolor=:ghostwhite
    plt_spectra = plotSpectrogram(buf_obs, fs, txt_obs; marker=:circle,
        colormap=:Hiroshige)
end

model_att = joinpath(@__DIR__, "models/ggml-base.en.bin")

listenToMe(1.2, buf_obs, txt_obs, circ_buf, model_att;
    transcribe_text=true) # Q. What is love?


if isfull(circ_buf)
    c_buf_sampled = SampleBuf(Array(circ_buf), 48000)
    liveTranscribe(c_buf_sampled, model_att)
end
txt_out = liveTranscribe(c_buf_sampled, model_att)

txt_obs[] = txt_out

# connect to Llama2
llama_model = joinpath(@__DIR__, "models/llama-2-7b-chat.Q4_K_S.gguf")
model_llama = load_gguf_model(llama_model);

sample(model_llama, txt_query[]; temperature=0.7f0)

