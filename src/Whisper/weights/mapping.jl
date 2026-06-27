using Pickle
using Random
using Lux

function map_weights(model::WhisperModel, checkpoint::Dict)
    ps, st = Lux.setup(Random.default_rng(), model)
    @show ps, st

    # dims = checkpoint["dims"]
    # state = checkpoint["model_state_dict"]

    # ps = map_encoder(ps, state)
    # ps = map_decoder(ps, state)

    # @assert isempty(state) "Unmapped weights remain: $(keys(state))"

    return ps, st
end

function map_encoder(ps, state)
    # Conv layers — need permutation + flip (cross-correlation to convolution)
    ps = @set ps.encoder.frontend.layer_1.weight =
        permutedims(pop!(state, "encoder.conv1.weight"), (3, 2, 1))[end:-1:1, :, :]
    ps = @set ps.encoder.frontend.layer_1.bias =
        pop!(state, "encoder.conv1.bias")

    ps = @set ps.encoder.frontend.layer_2.weight =
        permutedims(pop!(state, "encoder.conv2.weight"), (3, 2, 1))[end:-1:1, :, :]
    ps = @set ps.encoder.frontend.layer_2.bias =
        pop!(state, "encoder.conv2.bias")

    # Positional embedding
    ps = @set ps.encoder.position.embedding.weight =
        transpose(pop!(state, "encoder.positional_embedding"))

    # Transformer blocks
    for i in eachindex(ps.encoder.layers)
        name = Symbol(:layer, i)
        ps = map_transformer_block(ps, state, name,
            "encoder.blocks.$(i-1).", :encoder, false)
    end

    # Final norm
    ps = @set ps.encoder.norm.scale = pop!(state, "encoder.ln_post.weight")
    ps = @set ps.encoder.norm.bias  = pop!(state, "encoder.ln_post.bias")

    return ps
end

function map_decoder(ps, state)
    # Token embedding
    ps = @set ps.decoder.token_embedding.embedding.weight =
        transpose(pop!(state, "decoder.token_embedding.weight"))

    # Positional embedding
    ps = @set ps.decoder.position_embedding.embedding.weight =
        transpose(pop!(state, "decoder.positional_embedding"))

    # Transformer blocks
    for i in eachindex(ps.decoder.layers)
        name = Symbol(:layer, i)
        ps = map_transformer_block(ps, state, name,
            "decoder.blocks.$(i-1).", :decoder, true)
    end

    # Final norm
    ps = @set ps.decoder.norm.scale = pop!(state, "decoder.ln.weight")
    ps = @set ps.decoder.norm.bias  = pop!(state, "decoder.ln.bias")

    return ps
end

function map_transformer_block(ps, state, layer_name, prefix, component, cross_attention)
    # Self-attention
    ps = @set ps[component].layers[layer_name].attention.query.weight =
        pop!(state, prefix * "attn.query.weight")
    ps = @set ps[component].layers[layer_name].attention.query.bias =
        pop!(state, prefix * "attn.query.bias")
    ps = @set ps[component].layers[layer_name].attention.key.weight =
        pop!(state, prefix * "attn.key.weight")
    ps = @set ps[component].layers[layer_name].attention.value.weight =
        pop!(state, prefix * "attn.value.weight")
    ps = @set ps[component].layers[layer_name].attention.value.bias =
        pop!(state, prefix * "attn.value.bias")
    ps = @set ps[component].layers[layer_name].attention.out.weight =
        pop!(state, prefix * "attn.out.weight")
    ps = @set ps[component].layers[layer_name].attention.out.bias =
        pop!(state, prefix * "attn.out.bias")

    # Self-attention norm
    ps = @set ps[component].layers[layer_name].norm1.scale =
        pop!(state, prefix * "attn_ln.weight")
    ps = @set ps[component].layers[layer_name].norm1.bias =
        pop!(state, prefix * "attn_ln.bias")

    # Cross-attention (decoder only)
    if cross_attention
        ps = @set ps[component].layers[layer_name].cross_attention.query.weight =
            pop!(state, prefix * "cross_attn.query.weight")
        ps = @set ps[component].layers[layer_name].cross_attention.query.bias =
            pop!(state, prefix * "cross_attn.query.bias")
        ps = @set ps[component].layers[layer_name].cross_attention.key.weight =
            pop!(state, prefix * "cross_attn.key.weight")
        ps = @set ps[component].layers[layer_name].cross_attention.value.weight =
            pop!(state, prefix * "cross_attn.value.weight")
        ps = @set ps[component].layers[layer_name].cross_attention.value.bias =
            pop!(state, prefix * "cross_attn.value.bias")
        ps = @set ps[component].layers[layer_name].cross_attention.out.weight =
            pop!(state, prefix * "cross_attn.out.weight")
        ps = @set ps[component].layers[layer_name].cross_attention.out.bias =
            pop!(state, prefix * "cross_attn.out.bias")

        # Cross-attention norm
        ps = @set ps[component].layers[layer_name].norm_cross.scale =
            pop!(state, prefix * "cross_attn_ln.weight")
        ps = @set ps[component].layers[layer_name].norm_cross.bias =
            pop!(state, prefix * "cross_attn_ln.bias")
    end

    # Feedforward
    ps = @set ps[component].layers[layer_name].feedforward.layer_1.weight =
        pop!(state, prefix * "mlp.0.weight")
    ps = @set ps[component].layers[layer_name].feedforward.layer_1.bias =
        pop!(state, prefix * "mlp.0.bias")
    ps = @set ps[component].layers[layer_name].feedforward.layer_2.weight =
        pop!(state, prefix * "mlp.2.weight")
    ps = @set ps[component].layers[layer_name].feedforward.layer_2.bias =
        pop!(state, prefix * "mlp.2.bias")

    # Feedforward norm
    ps = @set ps[component].layers[layer_name].norm2.scale =
        pop!(state, prefix * "mlp_ln.weight")
    ps = @set ps[component].layers[layer_name].norm2.bias =
        pop!(state, prefix * "mlp_ln.bias")

    return ps
end