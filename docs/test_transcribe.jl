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
vocab = Evie.Whisper.Vocabulary.load_vocab(vocab_file);
tokenizer = Evie.Whisper.Tokenizer.BPETokenizer(vocab);

# Run inference
tokens = Evie.Whisper.transcribe(mel, ps, st, model, tokenizer)

# Decode to text
text = Evie.Whisper.Tokenizer.decode(tokenizer, tokens)
@info "Transcription: $text"
