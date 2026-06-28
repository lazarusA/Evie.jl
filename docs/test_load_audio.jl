using Evie
using Lux
using Downloads
using FileIO, LibSndFile

s_audio = Evie.Whisper.SAMPLE_URLS

cache = joinpath(homedir(), "Documents/Evie.jl/docs", "models")
file = Evie.Whisper.download_sample("gb1.ogg"; cache)

mel = Evie.Whisper.load_audio(file)
mel_debug = Evie.Whisper.prep_audio(mel; debug = true)


# l2 = FileIO.load(file)

name = "tiny.en"
cache = joinpath(homedir(), "Documents/Evie.jl/docs", "models")

# Load model
model, ps, st = Evie.Whisper.load_model(name; cache);

# Load tokenizer
vocab_file = Evie.Whisper.load_vocab_file(; multilingual = false)
vocab = Evie.Whisper.Vocabulary.load_vocab(vocab_file);
tokenizer = Evie.Whisper.Tokenizer.BPETokenizer(vocab);

waveform = Evie.Whisper.load_audio(file)   # Vector{Float32} at 16kHz
mel = Evie.Whisper.prep_audio(waveform) # (3000, 80, 1)
tokens = Evie.Whisper.transcribe(mel, ps, st, model, tokenizer)
text = Evie.Whisper.Tokenizer.decode(tokenizer, Int64.(tokens));
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
