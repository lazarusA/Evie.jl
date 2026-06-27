using Pickle
using Random
using Lux

function load_checkpoint(file::String)
    return Pickle.Torch.THload(file)
end

function load_model(name::String; cache::String = joinpath(homedir(), ".cache", "Whisper.jl"))
    isdir(cache) || mkdir(cache)

    url  = MODELS[name]
    file = download_weights(name, url, cache)

    checkpoint = load_checkpoint(file)
    dims       = checkpoint["dims"]

    @info "Loaded checkpoint with dims: $dims"

    model = WhisperModel(;
        n_mels            = dims["n_mels"],
        d_model           = dims["n_audio_state"],
        n_layers_enc      = dims["n_audio_layer"],
        n_heads_enc       = dims["n_audio_head"],
        max_positions_enc = dims["n_audio_ctx"],
        n_vocab           = dims["n_vocab"],
        n_layers_dec      = dims["n_text_layer"],
        n_heads_dec       = dims["n_text_head"],
        max_positions_dec = dims["n_text_ctx"]
    )

    ps, st = map_weights(model, checkpoint)

    return model, ps, st
end