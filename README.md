# Evie

[![Build Status](https://github.com/lazarusA/Evie.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/lazarusA/Evie.jl/actions/workflows/CI.yml?query=branch%3Amain)

An offline AI assistant

## Listening

```julia
using Evie, GLMakie
using DataStructures
using Whisper
using Llama2

fs, _buf, buf = initFsBuf()
buf_obs = Observable(_buf)
c_buf = CircularBuffer{Float32}(1024*52) # ≈ 1.1s
txt_query = Observable("[ Silence ]")

with_theme(theme_dark()) do
     # backgroundcolor=:ghostwhite
    plt_spectra = plotSpectrogram(buf_obs, fs; marker=:circle,
        colormap=:Hiroshige)
end

listenToMe(10, buf_obs, c_buf) # test with some music or your own voice
```