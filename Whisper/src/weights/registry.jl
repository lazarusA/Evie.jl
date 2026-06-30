const MODELS = Dict(
    "tiny.en" => "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "base" => "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small" => "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "large-v3" => "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt"
)

const VOCAB_URLS = Dict(
    "multilingual" => "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/multilingual.tiktoken",
    "gpt2" => "https://raw.githubusercontent.com/openai/whisper/main/whisper/assets/gpt2.tiktoken"
)

function load_vocab_file(; multilingual::Bool = false, cache::String = joinpath(homedir(), ".cache", "Whisper.jl"))
    isdir(cache) || mkdir(cache)
    name = multilingual ? "multilingual" : "gpt2"
    file = joinpath(cache, name * ".tiktoken")
    if !isfile(file)
        @info "Downloading vocab file: $file"
        Downloads.download(VOCAB_URLS[name], file)
    end
    return file
end

function download_weights(name::String, url::String, cache::String)
    file = joinpath(cache, name * ".pt")
    if !isfile(file)
        @info "Downloading $name weights from $url"
        Downloads.download(url, file)
    else
        @info "Found cached weights at $file"
    end
    return file
end
