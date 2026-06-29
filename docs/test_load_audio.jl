using Evie
using Lux
using Downloads
using FileIO, LibSndFile
using Statistics

s_audio = Evie.Whisper.SAMPLE_URLS

cache = joinpath(homedir(), "Documents/Evie.jl/docs", "models")
file = Evie.Whisper.download_sample("gb1.ogg"; cache)
waveform = Evie.Whisper.load_audio(file);
@info "Waveform" length=length(waveform) min=minimum(waveform) max=maximum(waveform) mean=Statistics.mean(waveform)

mel = Evie.Whisper.prep_audio(waveform; debug=true);
@info "Mel input to model" size=size(mel) min=minimum(mel) max=maximum(mel) mean=Statistics.mean(mel)

# Flux version path
waveform_flux, sr = FileIO.load(file)  # returns Matrix{Float32}
@info "Flux waveform" size=size(waveform_flux) sr=sr

# Your version
waveform_lux = Evie.Whisper.load_audio(file)
@info "Lux waveform" length=length(waveform_lux)

SAMPLE_RATE = 16_000
@info "Resampling check" begin
    s = FileIO.load(file)
    original_sr   = samplerate(s)
    original_len  = nframes(s)
    resampled_len = length(waveform)
    expected_len  = round(Int, original_len * (SAMPLE_RATE / original_sr))
    (; original_sr, original_len, resampled_len, expected_len)
end

mel = Evie.Whisper.load_audio(file)
mel_debug = Evie.Whisper.prep_audio(mel; debug = true)


# l2 = FileIO.load(file)

name = "tiny.en"
cache = joinpath(homedir(), "Documents/Evie.jl/docs", "models")

# Load model
model, ps, st = Evie.Whisper.load_model(name; cache);

# Check that weights are non-trivial
@info "Weight value sanity" begin
    q_w  = ps.encoder.layers.layer_1.attention.query.weight
    ln_w = ps.encoder.norm.scale
    te_w = ps.decoder.token_embedding.embedding.weight
    pe_w = ps.encoder.position.embedding.weight
    (
        query_mean   = Statistics.mean(abs.(q_w)),
        query_std    = Statistics.std(q_w),
        ln_scale_mean = Statistics.mean(ln_w),   # should be ~1.0 for LayerNorm
        token_emb_std = Statistics.std(te_w),
        pos_emb_std   = Statistics.std(pe_w),
    )
end

# After generating first ".", check if position embeddings are correct
# tokens should be [sot, en, transcribe, notimestamps, 13]
# positions 1:5 should be used

@info "Position embedding check" begin
    pos1 = ps.decoder.position_embedding.embedding.weight[:, 1]  # position 1
    pos5 = ps.decoder.position_embedding.embedding.weight[:, 5]  # position 5
    pos_similarity = LinearAlgebra.dot(pos1, pos5) / (norm(pos1) * norm(pos5))
    (; pos1_norm=norm(pos1), pos5_norm=norm(pos5), pos_similarity)
end

@info "Token embedding check" begin
    emb_period = ps.decoder.token_embedding.embedding.weight[:, 14]   # token 13, 1-based
    emb_comma  = ps.decoder.token_embedding.embedding.weight[:, 12]   # token 11 ","
    emb_space  = ps.decoder.token_embedding.embedding.weight[:, 2]    # token 1
    (
        period_norm = norm(emb_period),
        comma_norm  = norm(emb_comma),
        similarity  = LinearAlgebra.dot(emb_period, emb_comma) / (norm(emb_period) * norm(emb_comma)),
    )
end

# Load tokenizer
vocab_file = Evie.Whisper.load_vocab_file(; multilingual = false)
vocab = Evie.Whisper.load_vocab(vocab_file);
tokenizer = Evie.Whisper.BPETokenizer(vocab);

waveform = Evie.Whisper.load_audio(file)   # Vector{Float32} at 16kHz
mel = Evie.Whisper.prep_audio(waveform) # (3000, 80, 1)
tokens = Evie.Whisper.transcribe(mel, ps, st, model, tokenizer)
text = Evie.Whisper.decode(tokenizer, Int64.(tokens));
@info "Transcription: $text"

file = Evie.Whisper.download_sample("gb1.ogg"; cache)
waveform = Evie.Whisper.load_audio(file)
mel = Evie.Whisper.prep_audio(waveform; debug = true)

@info "Waveform length: $(length(waveform)) samples = $(length(waveform) / 16000)s"
@info "Waveform range: min=$(minimum(waveform)) max=$(maximum(waveform))"

mel = Evie.Whisper.prep_audio(waveform)
@info "Mel shape: $(size(mel))"
@info "Mel range: min=$(minimum(mel)) max=$(maximum(mel))"
@info "Mel mean: $(Statistics.mean(mel))"

waveform = Evie.Whisper.load_audio(file)
mel = Evie.Whisper.prep_audio(waveform; debug = true)

#
url = Evie.Whisper.MODELS[name]
file = Evie.Whisper.download_weights(name, url, cache)
checkpoint = Evie.Whisper.load_checkpoint(file)
# Raw checkpoint weight for token embedding
raw = checkpoint["model_state_dict"]["decoder.token_embedding.weight"]
# PyTorch shape: (n_vocab, d_model) = (51864, 384)
# After to_emb (transpose): (384, 51864)
# So weight[:, token_id+1] should give the embedding for that token

# Verify by checking if similar tokens have similar embeddings in the RAW weight
raw_period = raw[14, :]   # token 13, row in PyTorch (n_vocab, d_model)
raw_comma  = raw[12, :]   # token 11
raw_sim = LinearAlgebra.dot(raw_period, raw_comma) / (norm(raw_period) * norm(raw_comma))
@info "Raw weight similarity period/comma" raw_sim

# And check your loaded version
loaded_period = ps.decoder.token_embedding.embedding.weight[:, 14]
loaded_comma  = ps.decoder.token_embedding.embedding.weight[:, 12]
loaded_sim = LinearAlgebra.dot(loaded_period, loaded_comma) / (norm(loaded_period) * norm(loaded_comma))
@info "Loaded weight similarity period/comma" loaded_sim

# Are the values actually the same?
@info "Values match raw vs loaded" isapprox(Float32.(raw[14, :]), loaded_period, atol=1e-5)
