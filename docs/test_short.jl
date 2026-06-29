using Evie
using Lux
using Downloads
using FileIO, LibSndFile
using Statistics

s_audio = Evie.Whisper.SAMPLE_URLS

cache = joinpath(homedir(), "Documents/Evie.jl/docs", "models")
file = Evie.Whisper.download_sample("gb1.ogg"; cache)

@info "Audio check" begin
    SAMPLE_RATE = 16_000
    waveform = Evie.Whisper.load_audio(file) 
    duration_seconds = length(waveform) / SAMPLE_RATE
    (; path=file, duration_seconds, min=minimum(waveform), max=maximum(waveform))
end

name = "tiny.en"
cache = joinpath(homedir(), "Documents/Evie.jl/docs", "models")

# Load model
model, ps, st = Evie.Whisper.load_model(name; cache);

# Load tokenizer
vocab_file = Evie.Whisper.load_vocab_file(; multilingual = false)
vocab = Evie.Whisper.load_vocab(vocab_file);
tokenizer = Evie.Whisper.BPETokenizer(vocab);

waveform = Evie.Whisper.load_audio(file)   # Vector{Float32} at 16kHz
mel = Evie.Whisper.prep_audio(waveform) # (3000, 80, 1)
tokens = Evie.Whisper.transcribe(mel, ps, st, model, tokenizer; temperature=0.2f0)
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

@info "Token embeddings" size1=size(x1) size2=size(x2) last_token_step2=Evie.Whisper.decode(tokenizer, 13)

x1p, _ = model.decoder.position_embedding(x1, ps.decoder.position_embedding, st.decoder.position_embedding)
x2p, _ = model.decoder.position_embedding(x2, ps.decoder.position_embedding, st.decoder.position_embedding)

@info "After position embedding" x1p_last=x1p[:, end, 1] x2p_last=x2p[:, end, 1]
@info "Are last positions same?" same=isapprox(x1p[:, end, 1], x2p[:, end, 1])

@info "Suspicious token embeddings" begin
    en_id        = tokenizer.vocab.special_tokens["<|en|>"]          # 50258
    transcribe_id = tokenizer.vocab.special_tokens["<|transcribe|>"] # 50358

    emb_en        = ps.decoder.token_embedding.embedding.weight[:, en_id + 1]
    emb_transcribe = ps.decoder.token_embedding.embedding.weight[:, transcribe_id + 1]

    (
        en_id        = en_id,
        transcribe_id = transcribe_id,
        same         = isapprox(emb_en, emb_transcribe),
        en_norm      = LinearAlgebra.norm(emb_en),
        tr_norm      = LinearAlgebra.norm(emb_transcribe),
        similarity   = LinearAlgebra.dot(emb_en, emb_transcribe) / (LinearAlgebra.norm(emb_en) * LinearAlgebra.norm(emb_transcribe)),
    )
end

@info "Raw checkpoint suspicious tokens" begin
    raw_en        = Float32.(raw_tok[50259, :])  # 50258 + 1, 1-based
    raw_transcribe = Float32.(raw_tok[50359, :])  # 50358 + 1, 1-based
    raw_eot       = Float32.(raw_tok[50257, :])  # 50256 + 1, 1-based
    (
        en_tr_same    = isapprox(raw_en, raw_transcribe),
        en_norm       = LinearAlgebra.norm(raw_en),
        tr_norm       = LinearAlgebra.norm(raw_transcribe),
        similarity    = LinearAlgebra.dot(raw_en, raw_transcribe) / (LinearAlgebra.norm(raw_en) * LinearAlgebra.norm(raw_transcribe)),
        eot_norm      = LinearAlgebra.norm(raw_eot),
    )
end