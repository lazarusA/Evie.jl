# weights/vocab.jl
using Downloads

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
