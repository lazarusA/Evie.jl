export transcribe
export transcribe_file

function transcribe_file(path, ps, st, model, tokenizer; max_tokens = 224, rng = Random.MersenneTwister(42))
    waveform = load_audio(path)
    chunks = prep_audio_chunks(waveform)

    all_tokens = Vector{Vector{Int64}}()
    for (i, mel) in enumerate(chunks)
        tokens = transcribe(mel, ps, st, model, tokenizer; max_tokens, rng)
        text = decode(tokenizer, tokens; include_specials = false)
        @info "Chunk $i/$(length(chunks))" text
        push!(all_tokens, tokens)
    end

    texts = [decode(tokenizer, t; include_specials = false) for t in all_tokens]
    return strip(join(texts, " "))
end

function transcribe(
        mel, ps, st, model, tokenizer;
        max_tokens::Int = 224,
        rng::Random.AbstractRNG = Random.MersenneTwister(42),
    )
    local tokens
    for temperature in (0.0f0, 0.2f0, 0.4f0, 0.6f0, 0.8f0, 1.0f0)
        tokens = _transcribe(mel, ps, st, model, tokenizer; max_tokens, temperature, rng)
        text = decode(tokenizer, tokens; include_specials = false)
        is_degenerate(tokens, text, max_tokens) || return tokens
    end
    return tokens
end

function _transcribe(
        mel, ps, st, model, tokenizer;
        max_tokens::Int,
        temperature::Float32,
        rng::Random.AbstractRNG,
    )
    enc, _ = model.encoder(mel, ps.encoder, st.encoder)

    sot_id = tokenizer.vocab.special_tokens["<|startoftranscript|>"]
    eot_id = tokenizer.vocab.special_tokens["<|endoftext|>"]
    not_id = tokenizer.vocab.special_tokens["<|notimestamps|>"]

    tokens = Int64[sot_id]
    if haskey(tokenizer.vocab.special_tokens, "<|en|>")
        push!(tokens, tokenizer.vocab.special_tokens["<|en|>"])
        push!(tokens, tokenizer.vocab.special_tokens["<|transcribe|>"])
    end
    push!(tokens, not_id)

    suppress_ids = Set{Int64}(values(tokenizer.vocab.special_tokens))
    delete!(suppress_ids, eot_id)

    for _ in 1:max_tokens
        ctx = reshape(Int32.(tokens), :, 1)
        logits, _ = model.decoder(ctx, enc, ps.decoder, st.decoder)
        last_logits = logits[:, end, 1]

        for id in suppress_ids
            idx = id + 1
            1 <= idx <= length(last_logits) && (last_logits[idx] = -Inf32)
        end

        next_idx = if temperature ≈ 0.0f0
            argmax(last_logits)
        else
            probs = softmax(Float64.(last_logits) ./ Float64(temperature))
            probs ./= sum(probs)
            rand(rng, Categorical(probs))
        end

        next_id = next_idx - 1
        push!(tokens, Int64(next_id))
        next_id == eot_id && break
    end

    return tokens
end

function is_degenerate(tokens::Vector{Int64}, text::String, max_tokens::Int)
    # Hit token limit without EOT
    length(tokens) >= max_tokens && return true
    # Too short to be real transcription
    length(strip(text)) < 10 && return true
    # Check for repeating n-gram in token ids
    n = length(tokens)
    for window in (3, 4, 5, 6, 8)
        n >= window * 2 || continue
        for i in 1:(n - window * 2 + 1)
            tokens[i:(i + window - 1)] == tokens[(i + window):(i + window * 2 - 1)] && return true
        end
    end
    # Check for repeating substrings in decoded text (character-safe)
    chars = collect(text)
    nc = length(chars)
    for window in (8, 12, 16, 20)
        nc >= window * 2 || continue
        for i in 1:(nc - window * 2 + 1)
            chars[i:(i + window - 1)] == chars[(i + window):(i + window * 2 - 1)] && return true
        end
    end
    return false
end
