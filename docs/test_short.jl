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

# Load tokenizer
vocab_file = Evie.Whisper.load_vocab_file(; multilingual = false);
vocab = Evie.Whisper.load_vocab(vocab_file);
tokenizer = Evie.Whisper.BPETokenizer(vocab);

# waveform = Evie.Whisper.load_audio(file)   # Vector{Float32} at 16kHz
waveform = load_audio(file);
mel = Evie.Whisper.prep_audio(Float32.(waveform)) # (3000, 80, 1)
tokens = Evie.Whisper.transcribe(mel, ps, st, model, tokenizer) # temperature = 0.35f0

txt_full = Evie.Whisper.transcribe_file(file, ps, st, model, tokenizer)

# @info "Full file"
text = Evie.Whisper.decode(tokenizer, Int64.(tokens));
@info "Transcription: $text"

# tokens at step 1: [sot, en, transcribe, notimestamps]  → 4 tokens
# tokens at step 2: [sot, en, transcribe, notimestamps, 13]  → 5 tokens
# Does the embedding actually change?

tokens_step1 = Int32[50257, 50258, 50358, 50362]
tokens_step2 = Int32[50257, 50258, 50358, 50362, 13]

ctx1 = reshape(tokens_step1, :, 1)
ctx2 = reshape(tokens_step2, :, 1)

x1, _ = model.decoder.token_embedding(ctx1 .+ Int32(1), ps.decoder.token_embedding, st.decoder.token_embedding)
x2, _ = model.decoder.token_embedding(ctx2 .+ Int32(1), ps.decoder.token_embedding, st.decoder.token_embedding)

@info "Token embeddings" size1 = size(x1) size2 = size(x2) last_token_step2 = Evie.Whisper.decode(tokenizer, 13)

x1p, _ = model.decoder.position_embedding(x1, ps.decoder.position_embedding, st.decoder.position_embedding)
x2p, _ = model.decoder.position_embedding(x2, ps.decoder.position_embedding, st.decoder.position_embedding)

@info "After position embedding" x1p_last = x1p[:, end, 1] x2p_last = x2p[:, end, 1]
@info "Are last positions same?" same = isapprox(x1p[:, end, 1], x2p[:, end, 1])

@info "Suspicious token embeddings" begin
    en_id = tokenizer.vocab.special_tokens["<|en|>"]          # 50258
    transcribe_id = tokenizer.vocab.special_tokens["<|transcribe|>"] # 50358

    emb_en = ps.decoder.token_embedding.embedding.weight[:, en_id + 1]
    emb_transcribe = ps.decoder.token_embedding.embedding.weight[:, transcribe_id + 1]

    (
        en_id = en_id,
        transcribe_id = transcribe_id,
        same = isapprox(emb_en, emb_transcribe),
        en_norm = LinearAlgebra.norm(emb_en),
        tr_norm = LinearAlgebra.norm(emb_transcribe),
        similarity = LinearAlgebra.dot(emb_en, emb_transcribe) / (LinearAlgebra.norm(emb_en) * LinearAlgebra.norm(emb_transcribe)),
    )
end

@info "Raw checkpoint suspicious tokens" begin
    raw_en = Float32.(raw_tok[50259, :])  # 50258 + 1, 1-based
    raw_transcribe = Float32.(raw_tok[50359, :])  # 50358 + 1, 1-based
    raw_eot = Float32.(raw_tok[50257, :])  # 50256 + 1, 1-based
    (
        en_tr_same = isapprox(raw_en, raw_transcribe),
        en_norm = LinearAlgebra.norm(raw_en),
        tr_norm = LinearAlgebra.norm(raw_transcribe),
        similarity = LinearAlgebra.dot(raw_en, raw_transcribe) / (LinearAlgebra.norm(raw_en) * LinearAlgebra.norm(raw_transcribe)),
        eot_norm = LinearAlgebra.norm(raw_eot),
    )
end

#! debug audio

waveform = Evie.Whisper.load_audio(file)
@info "Audio structure" begin
    total_seconds = length(waveform) / 16_000
    # Energy per 10-second window
    window = 16_000 * 10
    n_windows = length(waveform) ÷ window
    energies = [Statistics.mean(waveform[((i - 1) * window + 1):(i * window)] .^ 2) for i in 1:n_windows]
    (; total_seconds, n_windows, energies)
end

# What does the first chunk's waveform look like?
chunks_wav = [waveform[((i - 1) * 480_000 + 1):min(i * 480_000, length(waveform))] for i in 1:7]
energies = [Statistics.mean(c .^ 2) for c in chunks_wav]
@info "Chunk energies" energies

function load_audio(path::String)
    return mktempdir() do tmp
        out = joinpath(tmp, "audio.flac")
        run(`ffmpeg -nostdin -threads 0 -i $path -ac 1 -ar 16000 -f flac $out -y`)
        s, sr = FileIO.load(out)
        @assert sr == 16_000
        return vec(Array(s))
    end
end

using FFMPEG

function load_audio_ff(path::String)
    return mktempdir() do tmp
        out = joinpath(tmp, "audio.flac")
        FFMPEG.exe(`-nostdin -threads 0 -i $path -ac 1 -ar 16000 -f flac $out -y`)
        s = FileIO.load(out)
        return vec(Array(s))
    end
end


# only 10 seconds
waveform = load_audio(file);

# Just the first 30 seconds
chunk1 = Float32.(waveform[1:480_000])
mel1 = Evie.Whisper.prep_audio(chunk1)

@info "Chunk 1 mel" size = size(mel1) min = minimum(mel1) max = maximum(mel1)

# Energy of first vs last 5 seconds of chunk 1
first_5s = waveform[1:80_000]
last_5s = waveform[400_001:480_000]
@info "Chunk 1 energy" begin
    e_first = Statistics.mean(first_5s .^ 2)
    e_last = Statistics.mean(last_5s .^ 2)
    (; e_first, e_last)
end

# Try transcribing JUST the first 10 seconds padded to 30
chunk_10s = vcat(waveform[1:160_000], zeros(Float32, 320_000))
mel_10s = Evie.Whisper.prep_audio(chunk_10s)
tokens = Evie.Whisper.transcribe(mel_10s, ps, st, model, tokenizer)
@info "First 10s only" Evie.Whisper.decode(tokenizer, tokens; include_specials = false)
