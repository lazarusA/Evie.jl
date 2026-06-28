using NNlib

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

    # 0-based token ids (as stored in vocab)
    tokens = Int64[sot_id]
    if haskey(tokenizer.vocab.special_tokens, "<|en|>")
        push!(tokens, tokenizer.vocab.special_tokens["<|en|>"])
        push!(tokens, tokenizer.vocab.special_tokens["<|transcribe|>"])
    end
    push!(tokens, not_id)

    # Collect special token ids to suppress (0-based)
    suppress_ids = Set{Int64}(values(tokenizer.vocab.special_tokens))
    delete!(suppress_ids, eot_id)

    for _ in 1:max_tokens
        ctx = reshape(Int32.(tokens), :, 1)
        logits, _ = model.decoder(ctx, enc, ps.decoder, st.decoder)
        last_logits = logits[:, end, 1]

        # Suppress special tokens
        # logits index is 1-based, token ids are 0-based → add 1
        for id in suppress_ids
            idx = id + 1
            if 1 <= idx <= length(last_logits)
                last_logits[idx] = -Inf32
            end
        end

        next_idx = if temperature ≈ 0.0f0
            argmax(last_logits)        # 1-based index
        else
            probs = softmax(last_logits ./ temperature)
            rand(Categorical(Float64.(probs)))
        end

        # Convert back to 0-based token id
        next_id = next_idx - 1

        push!(tokens, Int64(next_id))
        next_id == eot_id && break
    end

    return tokens
end
