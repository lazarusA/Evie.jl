using Evie
using Lux
using Downloads
using FileIO, LibSndFile
using Statistics

s_audio = Evie.Whisper.SAMPLE_URLS

cache = joinpath(homedir(), "Documents/Evie.jl/docs", "models")
file = Evie.Whisper.download_sample("gb1.ogg"; cache)

name = "tiny.en"
cache = joinpath(homedir(), "Documents/Evie.jl/docs", "models")

# Load model
model, ps, st = Evie.Whisper.load_model(name; cache);
# Same mel input to both models
mel_test = ones(Float32, 3000, 80, 1) .* 0.5f0

# Lux
enc_lux, _ = model.encoder(mel_test, ps.encoder, st.encoder)
@info "Lux encoder output" mean = Statistics.mean(enc_lux) std = Statistics.std(enc_lux)

# Flux — load the same checkpoint
import Flux
flux_model = FluxWhisper.WHISPER("base")  # or whichever model you're using
enc_flux = flux_model.encoder(permutedims(mel_test, (1, 2, 3)))
@info "Flux encoder output" mean = Statistics.mean(enc_flux) std = Statistics.std(enc_flux)
