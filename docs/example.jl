using Evie, GLMakie
using DataStructures
using Whisper
using Llama2
using SampledSignals
using GLMakie.FileIO

fs, _buf, buf = initFsBuf()
buf_obs = Observable(_buf)
c_buf = CircularBuffer{Float32}(1024*52) # ≈ 1.1s
txt_query = Observable("[ Silence ]")

with_theme(theme_dark()) do
     # backgroundcolor=:ghostwhite
    plt_spectra = plotSpectrogram(buf_obs, fs; marker=:circle,
        colormap=:Hiroshige)
end

listenToMe(1.2, buf_obs, c_buf) # Q. What is love?
c_buf_sampled = SampleBuf(Array(c_buf), 48000)

model_att = joinpath(@__DIR__, "models/ggml-base.en.bin")

if isfull(c_buf)
    liveTranscribe!(c_buf_sampled, model_att, txt_query)
end

txt_query[]

# connect to Llama2
llama_model = joinpath(@__DIR__, "models/llama-2-7b-chat.Q4_K_S.gguf")
model_llama = load_gguf_model(llama_model);

sample(model_llama, txt_query[]; temperature=0.7f0)

