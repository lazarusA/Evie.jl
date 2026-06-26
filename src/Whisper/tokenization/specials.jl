const LANGUAGES = Dict(
    "en" => "english",
    "de" => "german",
    "fr" => "french",
    "es" => "spanish"
)

const SPECIALS = vcat(
    ["<|endoftext|>", "<|startoftranscript|>"],
    ["<|$(v)|>" for v in values(LANGUAGES)],
    ["<|translate|>", "<|transcribe|>"]
)
