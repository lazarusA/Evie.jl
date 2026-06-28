using Printf

const LANGUAGES = OrderedDict(
    "en" => "english",
    "de" => "german",
    "fr" => "french",
    "es" => "spanish"
    # add more as needed
)

const SPECIALS = vcat(
    ["<|endoftext|>", "<|startoftranscript|>"],
    ["<|$lang|>" for lang in keys(LANGUAGES)],      # keys = language codes
    ["<|translate|>", "<|transcribe|>"],
    ["<|startoflm|>", "<|startofprev|>"],
    ["<|nospeech|>", "<|notimestamps|>"],
    ["<|$(@sprintf("%.2f", i * 0.02))|>" for i in 0:1500],  # timestamp tokens
)
