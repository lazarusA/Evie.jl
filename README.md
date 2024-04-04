# Evie

[![Build Status](https://github.com/lazarusA/Evie.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lazarusA/Evie.jl/actions/workflows/CI.yml?query=branch%3Amain)

An offline AI assistant

## Donwload models

> [!IMPORTANT]
> Transcribe with Whisper.jl
> For transcribing download: https://huggingface.co/ggerganov/whisper.cpp/blob/main/ggml-base.en.bin

> [!IMPORTANT]
> For the AI LLM: Download: https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/blob/main/llama-2-7b-chat.Q4_K_S.gguf

## Package dependencies:
For now, add this ones

```sh
julia > add https://github.com/lazarusA/Whisper.jl.git#main
```

```sh
julia > add https://github.com/lazarusA/Llama2.jl.git#master
```

in addition to the ones in the Project.toml files.

## Listening

```julia
using Evie, GLMakie
using DataStructures
using SampledSignals
using GLMakie.FileIO

fs, _buf, buf = initFsBuf()
buf_obs = Observable(_buf)
circ_buf = CircularBuffer{Float32}(1024*52) # ≈ 1.109s
txt_obs = Observable("[ Silence ]")

with_theme(theme_black()) do
    plt_spectra = plotSpectrogram(buf_obs, fs, txt_obs;
        marker=:circle, colormap=:Hiroshige)
end

model_att = joinpath(@__DIR__, "models/ggml-base.en.bin")

listenToMe(1.2, buf_obs, txt_obs, circ_buf, model_att;
    transcribe_text=true) # Q. What is love?

# connect to Llama2
llama_model = joinpath(@__DIR__, "models/llama-2-7b-chat.Q4_K_S.gguf")
model_llama = load_gguf_model(llama_model);

sample(model_llama, "What is love?"; temperature=0.7f0) # txt_obs[]

output_prompt=[]
sampleObs(model_llama, "What is love?", output_prompt; temperature=0.7f0) # txt_obs[]

```