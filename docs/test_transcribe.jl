using Evie

# Synthetic audio — 30 seconds at 16kHz → 3000 mel frames
# log mel spectrogram
mel = randn(Float32, 3000, 80, 1)

name = "tiny.en"
cache = joinpath(homedir(), "Documents/Evie.jl/docs", "models")

# Load model
model, ps, st = Evie.Whisper.load_model(name; cache);

# Load tokenizer
vocab_file = Evie.Whisper.load_vocab_file(; multilingual = false)
vocab = Evie.Whisper.load_vocab(vocab_file);
tokenizer = Evie.Whisper.BPETokenizer(vocab);

# Run inference
tokens = Evie.Whisper.transcribe(mel, ps, st, model, tokenizer)
@info "pos embed weight" size(ps.encoder.position.embedding.weight)
@info "token embed weight" size(ps.decoder.position_embedding.embedding.weight)

# Decode to text
text = Evie.Whisper.decode(tokenizer, tokens)
@info "Transcription: $text"

# weights
name = "tiny.en"
cache = joinpath(homedir(), "Documents/Evie.jl/docs", "models")
isdir(cache) || mkdir(cache)

url = Evie.Whisper.MODELS[name]
file = Evie.Whisper.download_weights(name, url, cache)

checkpoint = Evie.Whisper.load_checkpoint(file)
# Check conv weight shapes before and after to_conv
raw_conv1 = checkpoint["model_state_dict"]["encoder.conv1.weight"];
@info "conv1 raw shape" size(raw_conv1);   # expect (d_model, n_mels, 3) in PyTorch = (out, in, kernel)

converted = Evie.Whisper.to_conv(raw_conv1);
@info "conv1 converted shape" size(converted)  # Lux Conv expects (kernel, in, out) = (3, n_mels, d_model)

using Evie.Whisper: to_emb, to_matrix
# Check what Lux actually expects
@info "conv1 param shape in ps" size(ps.encoder.frontend.layer_1.weight)

raw_q = checkpoint["model_state_dict"]["encoder.blocks.0.attn.query.weight"];
@info "query raw shape" size(raw_q)

converted_q = to_matrix(raw_q);
@info "query converted shape" size(converted_q)

@info "query param shape in ps" size(ps.encoder.layers.layer_1.attention.query.weight);

raw_tok = checkpoint["model_state_dict"]["decoder.token_embedding.weight"];
@info "token_embedding raw" size(raw_tok)
@info "token_embedding converted" size(to_emb(raw_tok))
@info "token_embedding in ps" size(ps.decoder.token_embedding.embedding.weight)

raw_pos_enc = checkpoint["model_state_dict"]["encoder.positional_embedding"];
@info "encoder pos_embedding raw" size(raw_pos_enc)
@info "encoder pos_embedding converted" size(to_emb(raw_pos_enc))
@info "encoder pos_embedding in ps" size(ps.encoder.position.embedding.weight)

raw_pos_dec = checkpoint["model_state_dict"]["decoder.positional_embedding"];
@info "decoder pos_embedding raw" size(raw_pos_dec)
@info "decoder pos_embedding converted" size(to_emb(raw_pos_dec))
@info "decoder pos_embedding in ps" size(ps.decoder.position_embedding.embedding.weight)
