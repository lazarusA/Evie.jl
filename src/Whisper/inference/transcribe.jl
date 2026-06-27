function transcribe(
        mel, ps, st, model, tokenizer;
        max_tokens::Int = 224,
        temperature::Float32 = 0.0f0
    )
    # Run encoder once — reuse for all decoder steps
    enc, st_enc = model.encoder(mel, ps.encoder, st.encoder)

    # Start with [startoftranscript] token
    sot_id = tokenizer.vocab.special_tokens["<|startoftranscript|>"]
    eot_id = tokenizer.vocab.special_tokens["<|endoftext|>"]
    tokens = Int32[sot_id]

    for _ in 1:max_tokens
        # Build token input — (seq_len, 1) for batch=1
        ctx = reshape(tokens, :, 1)

        # Decoder forward pass
        logits, _ = model.decoder(ctx, enc, ps.decoder, st.decoder)

        # Take logits at last position only — (n_vocab,)
        last_logits = logits[:, end, 1]

        # Sample next token
        next_id = if temperature ≈ 0.0f0
            # Greedy — pick highest logit
            argmax(last_logits)
        else
            # Temperature sampling
            probs = softmax(last_logits ./ temperature)
            rand(Categorical(Float64.(probs)))
        end

        # Append and check for end
        push!(tokens, Int32(next_id))
        next_id == eot_id && break
    end

    return tokens
end
