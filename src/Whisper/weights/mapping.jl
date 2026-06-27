using Lux
using Random
using Accessors

to_vec(x) = collect(Float32.(vec(x)))
to_matrix(x) = collect(Float32.(x))
to_conv(x) = collect(Float32.(permutedims(x, (3, 2, 1))[end:-1:1, :, :]))
to_emb(x) = collect(Float32.(transpose(x)))
to_ln(x) = collect(Float32.(reshape(vec(x), :, 1, 1)))

function map_weights(model::WhisperModel, checkpoint::Dict)
    ps, st = Lux.setup(Random.default_rng(), model)
    state = copy(checkpoint["model_state_dict"])

    ps = map_encoder(ps, state)
    ps = map_decoder(ps, state)

    isempty(state) || @warn "Unmapped weights remain: $(keys(state))"

    return ps, st
end

function map_encoder(ps, state)
    ps = @set ps.encoder.frontend.layer_1.weight = to_conv(pop!(state, "encoder.conv1.weight"))
    ps = @set ps.encoder.frontend.layer_1.bias = to_vec(pop!(state, "encoder.conv1.bias"))

    ps = @set ps.encoder.frontend.layer_2.weight = to_conv(pop!(state, "encoder.conv2.weight"))
    ps = @set ps.encoder.frontend.layer_2.bias = to_vec(pop!(state, "encoder.conv2.bias"))

    ps = @set ps.encoder.position.embedding.weight = to_emb(pop!(state, "encoder.positional_embedding"))

    for i in 1:length(keys(ps.encoder.layers))
        ps = map_encoder_block(ps, state, Symbol(:layer_, i), "encoder.blocks.$(i - 1).")
    end

    ps = @set ps.encoder.norm.scale = to_ln(pop!(state, "encoder.ln_post.weight"))
    ps = @set ps.encoder.norm.bias = to_ln(pop!(state, "encoder.ln_post.bias"))

    return ps
end

function map_encoder_block(ps, state, layer_name, prefix)
    # Self-attention
    ps = @set ps.encoder.layers[layer_name].attention.query.weight = to_matrix(pop!(state, prefix * "attn.query.weight"))
    ps = @set ps.encoder.layers[layer_name].attention.query.bias = to_vec(pop!(state, prefix * "attn.query.bias"))
    ps = @set ps.encoder.layers[layer_name].attention.key.weight = to_matrix(pop!(state, prefix * "attn.key.weight"))
    ps = @set ps.encoder.layers[layer_name].attention.value.weight = to_matrix(pop!(state, prefix * "attn.value.weight"))
    ps = @set ps.encoder.layers[layer_name].attention.value.bias = to_vec(pop!(state, prefix * "attn.value.bias"))
    ps = @set ps.encoder.layers[layer_name].attention.out.weight = to_matrix(pop!(state, prefix * "attn.out.weight"))
    ps = @set ps.encoder.layers[layer_name].attention.out.bias = to_vec(pop!(state, prefix * "attn.out.bias"))

    # Self-attention norm
    ps = @set ps.encoder.layers[layer_name].norm1.scale = to_ln(pop!(state, prefix * "attn_ln.weight"))
    ps = @set ps.encoder.layers[layer_name].norm1.bias = to_ln(pop!(state, prefix * "attn_ln.bias"))

    # Feedforward
    ps = @set ps.encoder.layers[layer_name].feedforward.layer_1.weight = to_matrix(pop!(state, prefix * "mlp.0.weight"))
    ps = @set ps.encoder.layers[layer_name].feedforward.layer_1.bias = to_vec(pop!(state, prefix * "mlp.0.bias"))
    ps = @set ps.encoder.layers[layer_name].feedforward.layer_2.weight = to_matrix(pop!(state, prefix * "mlp.2.weight"))
    ps = @set ps.encoder.layers[layer_name].feedforward.layer_2.bias = to_vec(pop!(state, prefix * "mlp.2.bias"))

    # Feedforward norm
    ps = @set ps.encoder.layers[layer_name].norm2.scale = to_ln(pop!(state, prefix * "mlp_ln.weight"))
    ps = @set ps.encoder.layers[layer_name].norm2.bias = to_ln(pop!(state, prefix * "mlp_ln.bias"))

    return ps
end

# ── Decoder ───────────────────────────────────────────────────────────────────
function map_decoder(ps, state)
    ps = @set ps.decoder.token_embedding.embedding.weight = to_emb(pop!(state, "decoder.token_embedding.weight"))
    ps = @set ps.decoder.position_embedding.embedding.weight = to_emb(pop!(state, "decoder.positional_embedding"))

    for i in 1:length(keys(ps.decoder.layers))
        ps = map_decoder_block(ps, state, Symbol(:layer_, i), "decoder.blocks.$(i - 1).")
    end

    ps = @set ps.decoder.norm.scale = to_ln(pop!(state, "decoder.ln.weight"))
    ps = @set ps.decoder.norm.bias = to_ln(pop!(state, "decoder.ln.bias"))

    return ps
end

function map_decoder_block(ps, state, layer_name, prefix)
    # Self-attention
    ps = @set ps.decoder.layers[layer_name].attention.query.weight = to_matrix(pop!(state, prefix * "attn.query.weight"))
    ps = @set ps.decoder.layers[layer_name].attention.query.bias = to_vec(pop!(state, prefix * "attn.query.bias"))
    ps = @set ps.decoder.layers[layer_name].attention.key.weight = to_matrix(pop!(state, prefix * "attn.key.weight"))
    ps = @set ps.decoder.layers[layer_name].attention.value.weight = to_matrix(pop!(state, prefix * "attn.value.weight"))
    ps = @set ps.decoder.layers[layer_name].attention.value.bias = to_vec(pop!(state, prefix * "attn.value.bias"))
    ps = @set ps.decoder.layers[layer_name].attention.out.weight = to_matrix(pop!(state, prefix * "attn.out.weight"))
    ps = @set ps.decoder.layers[layer_name].attention.out.bias = to_vec(pop!(state, prefix * "attn.out.bias"))

    # Self-attention norm
    ps = @set ps.decoder.layers[layer_name].norm1.scale = to_ln(pop!(state, prefix * "attn_ln.weight"))
    ps = @set ps.decoder.layers[layer_name].norm1.bias = to_ln(pop!(state, prefix * "attn_ln.bias"))

    # Cross-attention
    ps = @set ps.decoder.layers[layer_name].cross_attention.query.weight = to_matrix(pop!(state, prefix * "cross_attn.query.weight"))
    ps = @set ps.decoder.layers[layer_name].cross_attention.query.bias = to_vec(pop!(state, prefix * "cross_attn.query.bias"))
    ps = @set ps.decoder.layers[layer_name].cross_attention.key.weight = to_matrix(pop!(state, prefix * "cross_attn.key.weight"))
    ps = @set ps.decoder.layers[layer_name].cross_attention.value.weight = to_matrix(pop!(state, prefix * "cross_attn.value.weight"))
    ps = @set ps.decoder.layers[layer_name].cross_attention.value.bias = to_vec(pop!(state, prefix * "cross_attn.value.bias"))
    ps = @set ps.decoder.layers[layer_name].cross_attention.out.weight = to_matrix(pop!(state, prefix * "cross_attn.out.weight"))
    ps = @set ps.decoder.layers[layer_name].cross_attention.out.bias = to_vec(pop!(state, prefix * "cross_attn.out.bias"))

    # Cross-attention norm
    ps = @set ps.decoder.layers[layer_name].norm_cross.scale = to_ln(pop!(state, prefix * "cross_attn_ln.weight"))
    ps = @set ps.decoder.layers[layer_name].norm_cross.bias = to_ln(pop!(state, prefix * "cross_attn_ln.bias"))

    # Feedforward
    ps = @set ps.decoder.layers[layer_name].feedforward.layer_1.weight = to_matrix(pop!(state, prefix * "mlp.0.weight"))
    ps = @set ps.decoder.layers[layer_name].feedforward.layer_1.bias = to_vec(pop!(state, prefix * "mlp.0.bias"))
    ps = @set ps.decoder.layers[layer_name].feedforward.layer_2.weight = to_matrix(pop!(state, prefix * "mlp.2.weight"))
    ps = @set ps.decoder.layers[layer_name].feedforward.layer_2.bias = to_vec(pop!(state, prefix * "mlp.2.bias"))

    # Feedforward norm
    ps = @set ps.decoder.layers[layer_name].norm2.scale = to_ln(pop!(state, prefix * "mlp_ln.weight"))
    ps = @set ps.decoder.layers[layer_name].norm2.bias = to_ln(pop!(state, prefix * "mlp_ln.bias"))

    return ps
end
