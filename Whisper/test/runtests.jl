using Whisper
using Lux
using Random

# Step 1 — Download and load checkpoint (raw PyTorch weights)
# Start with tiny.en — smallest model, fastest to download (~75MB)
name = "tiny.en"
cache = joinpath(@__DIR__, "models")
isdir(cache) || mkdir(cache)

url = Whisper.MODELS[name]
file = Whisper.download_weights(name, url, cache)

@info "Weights file: $file"

# Step 2 — Load raw checkpoint and inspect dims
checkpoint = Whisper.load_checkpoint(file)
dims = checkpoint["dims"]
@info "Model dims: $dims"

# Step 3 — Construct model from dims
model = Whisper.WhisperModel(;
    n_mels = dims["n_mels"],
    d_model = dims["n_audio_state"],
    n_layers_enc = dims["n_audio_layer"],
    n_heads_enc = dims["n_audio_head"],
    max_positions_enc = dims["n_audio_ctx"],
    n_vocab = dims["n_vocab"],
    n_layers_dec = dims["n_text_layer"],
    n_heads_dec = dims["n_text_head"],
    max_positions_dec = dims["n_text_ctx"]
);
@info "Model constructed"

# Step 4 — Initialize parameters and states
rng = Random.default_rng()
ps, st = Lux.setup(rng, model);
@info "Parameters initialized"

# Step 5 — Inspect ps keys to verify layer naming before mapping
@info "Encoder frontend keys: $(keys(ps.encoder.frontend))"
@info "Encoder layer1 keys:   $(keys(ps.encoder.layers.layers.layer_1))"
@info "Decoder layer1 keys:   $(keys(ps.decoder.layers.layers.layer_1))"

# Step 6 — Map weights
ps, st = Whisper.map_weights(model, checkpoint);
@info "Weights mapped successfully"

# Step 7 — Smoke test: run a dummy forward pass
mel = randn(Float32, 3000, 80, 1);
tokens = reshape(Int32[1, 2, 3], 3, 1)
out, st2 = model(mel, tokens, ps, st);
@info "Forward pass output shape: $(size(out))"

# audio test

# Load tokenizer
vocab_file = Whisper.load_vocab_file(; multilingual = false)
vocab = Whisper.load_vocab(vocab_file);
tokenizer = Whisper.BPETokenizer(vocab);

s_audio = Whisper.SAMPLE_URLS
file = Whisper.download_sample("gb1.ogg"; cache)
waveform = Whisper.load_audio(file)   # Vector{Float32} at 16kHz
mel = Whisper.prep_audio(waveform) # (3000, 80, 1)
tokens = Whisper.transcribe(mel, ps, st, model, tokenizer)
text = Whisper.decode(tokenizer, Int64.(tokens));
@info "Transcription: $text"
