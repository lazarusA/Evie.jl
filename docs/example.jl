using Evie, GLMakie
using DataStructures
using SampledSignals
using GLMakie.FileIO

fs, _buf, buf = initFsBuf()
buf_obs = Observable(_buf)
circ_buf = CircularBuffer{Float32}(1024*52) # ≈ 1.109s
txt_obs = Observable("[ Silence ]")

with_theme(theme_dark()) do
    plt_spectra = plotSpectrogram(buf_obs, fs, txt_obs;
        marker=:circle, colormap=:Hiroshige)
end

model_att = joinpath(@__DIR__, "models/ggml-base.en.bin")

listenToMe(1.2, buf_obs, txt_obs, circ_buf, model_att;
    transcribe_text=true) # Q. What is love?

# connect to Llama2
using Llama2
llama_model = joinpath(@__DIR__, "models/llama-2-7b-chat.Q4_K_S.gguf")
model_llama = load_gguf_model(llama_model);

sample(model_llama, "how to sum two numbers in Julia?"; temperature=0.7f0) # txt_obs[]

output_prompt=[]
sampleObs(model_llama, "What is love?", output_prompt; temperature=0.7f0) # txt_obs[]


