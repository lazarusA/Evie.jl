const SAMPLE_URLS = Dict(
    "gb0.oga" => "https://upload.wikimedia.org/wikipedia/commons/2/22/George_W._Bush%27s_weekly_radio_address_%28November_1%2C_2008%29.oga",
    "gb1.ogg" => "https://upload.wikimedia.org/wikipedia/commons/1/1f/George_W_Bush_Columbia_FINAL.ogg",
    "hp0.ogg" => "https://upload.wikimedia.org/wikipedia/en/d/d4/En.henryfphillips.ogg",
    "mm1.wav" => "https://cdn.openai.com/whisper/draft-20220913a/micro-machines.wav",
    "es1.ogg" => "https://upload.wikimedia.org/wikipedia/commons/c/c1/La_contaminacion_del_agua.ogg",
)

const SAMPLE_CHECKSUMS = Dict(
    "gb0.oga" => "b844a36b9b0c0d777c64f1d62356bf0b6cad6a0f753627f5a7e7abd17c843f0c",
    "gb1.ogg" => "97a6384767e2fc3fb27c7593831aa19115d909fcdb85a1e389359ecc4b92a1e8",
    "hp0.ogg" => "753014d9f365a3d49989aecb3d2416f2aa6644909fc2d6289e5b986ee1324472",
    "mm1.wav" => "37de21902b32aa2fc147ccbfdcc0566cc7061fffb2c0b10874f05147c0b9de0f",
    "es1.ogg" => "43ee99686d75fd2976128450cec95a621a70a99b4dbd1c224fb9b35c6549daae",
)

function download_sample(name::String; cache::String = joinpath(homedir(), ".cache", "Whisper.jl", "samples"))
    isdir(cache) || mkpath(cache)
    url = SAMPLE_URLS[name]
    file = joinpath(cache, basename(url))
    if !isfile(file)
        @info "Downloading sample: $name"
        Downloads.download(url, file)
    end
    return file
end
