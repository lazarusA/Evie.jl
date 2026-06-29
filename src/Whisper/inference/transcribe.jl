export transcribe

function transcribe(
        mel, ps, st, model, tokenizer;
        max_tokens::Int = 224,
        temperature::Float32 = 0.0f0,
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

    # All special tokens are suppressed during generation except EOT
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
            rand(Categorical(probs))
        end

        next_id = next_idx - 1  # convert 1-based Julia index → 0-based token id
        push!(tokens, Int64(next_id))
        next_id == eot_id && break
    end

    return tokens
end
